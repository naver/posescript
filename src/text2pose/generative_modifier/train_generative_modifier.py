##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import math
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.option import get_args_parser
from text2pose.trainer import GenericTrainer
from text2pose.generative_modifier.model_generative_modifier import FeedbackGenerator
from text2pose.generative_modifier.evaluate_generative_modifier import get_evaluation_model_paths, get_evaluation_models, compute_eval_metrics
from text2pose.data import PoseFix, PoseMix
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.data_augmentations import DataAugmentation

import text2pose.config as config
import text2pose.utils_logging as logging


################################################################################


class ModifierGenerationTrainer(GenericTrainer):

	def __init__(self, args, path_to_pretrained_pose_encoder_ckpt=None):
		super(ModifierGenerationTrainer, self).__init__(args)
		self.path_to_pretrained_pose_encoder_ckpt = path_to_pretrained_pose_encoder_ckpt


	def load_dataset(self, split, caption_index, tokenizer_name=None):
		
		if tokenizer_name is None: tokenizer_name = get_tokenizer_name(self.args.text_decoder_name)
		data_size = self.args.data_size if split=="train" else None

		if "posefix" in self.args.dataset:
			d = PoseFix(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size)
		elif "posemix" in self.args.dataset:
			# NOTE: if specifying data_size: only the first loaded data items
			# will be considered (since PoseFix is loaded before PoseScript, if
			# data_size < the size of PoseFix, no PoseScript data will be
			# loaded)
			d = PoseMix(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size)
		else:
			raise NotImplementedError
		return d


	def init_model(self):
		print('Load model')
		self.model = FeedbackGenerator(text_decoder_name=self.args.text_decoder_name,
								encoder_latentD=self.args.latentD,
								comparison_latentD=self.args.comparison_latentD,
								decoder_latentD=self.args.decoder_latentD,
								decoder_nhead=self.args.decoder_nhead,
								decoder_nlayers=self.args.decoder_nlayers,
								comparison_module_mode=self.args.comparison_module_mode,
								transformer_mode=self.args.transformer_mode,
								num_body_joints=self.args.num_body_joints)
		self.model.to(self.device)

		# load pretrained weights for the pose encoder
		if self.path_to_pretrained_pose_encoder_ckpt:
			model_dict = self.model.state_dict()
			# get the weights from the pretrained model only for the entries of the current model that are related to the pose encoder
			pretrained_pose_encoder_weight_dict = torch.load(self.path_to_pretrained_pose_encoder_ckpt)['model']
			pretrained_pose_encoder_weight_dict = {k: v for k, v in pretrained_pose_encoder_weight_dict.items() if f'pose_encoder.{k}' in model_dict}
			# overwrite entries of the current model with the imported weights
			model_dict.update(pretrained_pose_encoder_weight_dict) 
			self.model.load_state_dict(model_dict)
			print("Initialize pose encoder weights from", self.path_to_pretrained_pose_encoder_ckpt)
	

	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': [p for p in self.model.pose_encoder.parameters() if p.requires_grad] + \
									[p for p in self.model.comparison_module.parameters() if p.requires_grad], 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': [p for p in self.model.text_decoder.parameters() if p.requires_grad] + \
		      						[p for p in self.model.modalityInputAdapter.parameters() if p.requires_grad], 'lr': self.args.lr*self.args.lrtextmul})
		return param_groups


	def init_other_training_elements(self):
		self.data_augmentation_module = DataAugmentation(self.args, mode="posefix", tokenizer_name=get_tokenizer_name(self.args.text_decoder_name), nb_joints=self.args.num_body_joints)


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(epoch=epoch, is_training=True)
		return train_stats
	

	def validation_epoch(self, epoch):
		val_stats = {}
		if self.args.val_every and (epoch+1)%self.args.val_every==0:
			val_stats.update(self.one_epoch(epoch=epoch, is_training=False))
		if self.args.val_every and (epoch+1)%(self.args.val_every*10)==0:
			val_stats.update(self.validate(epoch=epoch))
		return val_stats
	

	def one_epoch(self, epoch, is_training):

		self.model.train(is_training)

		# define loggers
		metric_logger = logging.MetricLogger(delimiter="  ")
		if is_training:
			prefix, sstr = '', 'train'
			metric_logger.add_meter(f'{sstr}_lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
		else:
			prefix, sstr = '[val] ', 'val'
		header = f'{prefix}Epoch: [{epoch}]'
		
		# define dataloader & other elements
		if is_training:
			data_loader = self.data_loader_train
		if not is_training:
			data_loader = self.data_loader_val

		# iterate over the batches
		for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):

			# get data
			poses_A = item['poses_A'].to(self.device)
			poses_B = item['poses_B'].to(self.device)
			caption_tokens = item['caption_tokens'].to(self.device)
			caption_lengths = item['caption_lengths'].to(self.device)
			caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate within the batch, based on the longest text 
			
			# online random augmentations
			posescript_poses = item['poses_A_ids'] == config.PID_NAN
			poses_A, caption_tokens, caption_lengths, poses_B = self.data_augmentation_module(poses_A, caption_tokens, caption_lengths, poses_B, posescript_poses)
			
			# forward + compute loss
			with torch.set_grad_enabled(is_training):
				output = self.model(poses_A, caption_tokens, caption_lengths, poses_B)

			loss = output["loss"]
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)

			# training step
			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			# format data for logging
			scalars = [('loss', loss_value)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				scalars += [('lr', lr_value)]
			else: # validation
				scalars += [('fake_loss', output['fake_loss'])]
				# add texts to get insights about the good progress of the training
				with torch.inference_mode():
					if data_iter_step == 0: # only for the first batch
						sample_texts, _ = self.model.generate_text(poses_A, poses_B)

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, texts=('modifier', sample_texts) if not is_training else None, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)

		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


	def validate(self, epoch):

		self.model.eval()
		initial_tokenizer_name = get_tokenizer_name(self.args.text_decoder_name)

		# load evaluation models
		pose_model_path, fid_version, textret_model_path, _ = get_evaluation_model_paths(self.args.pose_generative_model, self.args.fid, self.args.textret_model)
		pose_model, textret_model, tokenizer_name, tokenizer_name_textret_model = get_evaluation_models(initial_tokenizer_name, self.device,
									self.args.pose_generative_model, pose_model_path,
									self.args.textret_model, textret_model_path)

		# get dataset (with the tokenizer adapted to the evaluation models)
		if tokenizer_name != initial_tokenizer_name:
			dataset_val = self.load_dataset(split='val', caption_index='deterministic-mix', tokenizer_name=tokenizer_name)
		else:
			dataset_val = self.data_loader_val.dataset

		# compute metrics
		metrics = compute_eval_metrics(self.model, dataset_val, self.device,
										pose_model=pose_model,
										fid_version=fid_version,
										textret_model=textret_model,
										tokenizer_name_textret_model=tokenizer_name_textret_model)
		
		# log
		self.add_data_to_log_writer(epoch, 'val', scalars=[('validation', metrics)], should_log_data=True)
		print(f"[val] Epoch: [{epoch}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in metrics.items()) )
		return metrics		


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	path_to_pretrained_pose_encoder_ckpt = None
	ModifierGenerationTrainer(args, path_to_pretrained_pose_encoder_ckpt)()