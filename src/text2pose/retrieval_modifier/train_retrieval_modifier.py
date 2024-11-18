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
from text2pose.retrieval_modifier.model_retrieval_modifier import PairText
from text2pose.retrieval_modifier.evaluate_retrieval_modifier import compute_eval_metrics
from text2pose.loss import BBC, symBBC
from text2pose.data import  PoseFix, PoseMix, PoseScript
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.data_augmentations import DataAugmentation

import text2pose.config as config
import text2pose.utils_logging as logging


################################################################################


class PairTextTrainer(GenericTrainer):

	def __init__(self, args):
		super(PairTextTrainer, self).__init__(args, retrieval_trainer=True)

	
	def load_dataset(self, split, caption_index, tokenizer_name=None):
		
		if tokenizer_name is None: tokenizer_name = get_tokenizer_name(self.args.text_encoder_name)
		data_size = self.args.data_size if split=="train" else None

		if "posefix" in self.args.dataset:
			d = PoseFix(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size)
		elif "posemix" in self.args.dataset:
			# NOTE: if specifying data_size: only the first loaded data items
			# will be considered (since PoseFix is loaded before PoseScript, if
			# data_size < the size of PoseFix, no PoseScript data will be
			# loaded)
			d = PoseMix(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size)
		elif "posescript" in self.args.dataset:
			d = PoseScript(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size, posefix_format=True)
		else:
			raise NotImplementedError
		return d


	def init_model(self):
		print('Load model')
		self.model = PairText(text_encoder_name=self.args.text_encoder_name,
							transformer_topping=self.args.transformer_topping,
							latentD=self.args.latentD,
							num_body_joints=self.args.num_body_joints)
		self.model.to(self.device)


	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': self.model.pose_encoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.pose_mlp.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': [p for k,p in self.model.text_encoder.named_parameters() if 'pretrained_text_encoder.' not in k], 'lr': self.args.lr*self.args.lrtextmul})
		param_groups.append({'params': [self.model.loss_weight]})
		return param_groups


	def init_optimizer(self):
		assert self.args.optimizer=='Adam'
		param_groups = self.get_param_groups()
		self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr)


	def init_lr_scheduler(self):
		self.lr_scheduler = None
		if self.args.lr_scheduler == "stepLR":
			self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
													step_size=self.args.lr_step,
													gamma=self.args.lr_gamma,
													last_epoch=-1)


	def init_other_training_elements(self):
		self.data_augmentation_module = DataAugmentation(self.args, mode="posefix", tokenizer_name=get_tokenizer_name(self.args.text_encoder_name), nb_joints=self.args.num_body_joints)


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(epoch=epoch, is_training=True)
		return train_stats
	

	def validation_epoch(self, epoch):
		val_stats = {}
		if self.args.val_every and (epoch+1)%self.args.val_every==0:
			val_stats = self.validate(epoch=epoch)
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

			# forward; compute scores
			with torch.set_grad_enabled(is_training):
				poses_features, texts_features = self.model(poses_A, caption_tokens, caption_lengths, poses_B)
				score_t2p = texts_features.mm(poses_features.t()) * self.model.loss_weight
			
			# compute loss
			if self.args.retrieval_loss == "BBC":
				loss = BBC(score_t2p)
			elif self.args.retrieval_loss == "symBBC":
				loss = symBBC(score_t2p)
			else:
				raise NotImplementedError

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

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)

		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


	def validate(self, epoch):

		self.model.eval()

		recalls, loss_value = compute_eval_metrics(self.model, self.data_loader_val.dataset, self.device, compute_loss=True)
		val_stats = {"loss": loss_value}
		val_stats.update(recalls)

		# log
		self.add_data_to_log_writer(epoch, 'val', scalars=[('loss', loss_value), ('validation', recalls)], should_log_data=True)
		print(f"[val] Epoch: [{epoch}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in val_stats.items()) )
		return val_stats	


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	PairTextTrainer(args)()