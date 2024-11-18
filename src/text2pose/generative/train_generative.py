##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import math
import roma
from functools import reduce
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.option import get_args_parser
from text2pose.trainer import GenericTrainer
from text2pose.generative.model_generative import CondTextPoser
from text2pose.data import PoseScript, PoseFix
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.data_augmentations import DataAugmentation
from text2pose.loss import laplacian_nll, gaussian_nll

import text2pose.config as config
import text2pose.utils as utils
import text2pose.utils_logging as logging


################################################################################


class PoseGenerationTrainer(GenericTrainer):

	def __init__(self, args):
		super(PoseGenerationTrainer, self).__init__(args)


	def load_dataset(self, split, caption_index, tokenizer_name=None):
		
		if tokenizer_name is None: tokenizer_name = get_tokenizer_name(self.args.text_encoder_name)
		data_size = self.args.data_size if split=="train" else None

		if "posescript" in self.args.dataset:
			d = PoseScript(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size)
		elif "posefix" in self.args.dataset:
			d = PoseFix(version=self.args.dataset, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=self.args.num_body_joints, data_size=data_size, posescript_format=True)
		else:
			raise NotImplementedError
		return d


	def init_model(self):
		print('Load model')
		self.model = CondTextPoser(text_encoder_name=self.args.text_encoder_name,
								   transformer_topping=self.args.transformer_topping,
								   latentD=self.args.latentD,
								   num_body_joints=self.args.num_body_joints)
		self.model.to(self.device)


	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': self.model.pose_encoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.pose_decoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.text_encoder.parameters(), 'lr': self.args.lr*self.args.lrtextmul}) 
		param_groups.append({'params': [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot]})
		return param_groups


	def init_other_training_elements(self):
		self.init_fid(name_in_batch="pose")
		self.init_body_model()
		self.data_augmentation_module = DataAugmentation(self.args, mode="posescript", tokenizer_name=get_tokenizer_name(self.args.text_encoder_name), nb_joints=self.args.num_body_joints)


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(
			epoch=epoch,
			is_training=True
		)
		return train_stats
	

	def validation_epoch(self, epoch):
		if self.args.val_every and (epoch+1)%self.args.val_every==0: # and lr_scheduler is None
			return self.one_epoch(
							epoch=epoch,
							is_training=False
						)
		return {}
	

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
			self.fid.reset_gen_features()

		# iterate over the batches
		for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):


			# --- forward pass

			# get data
			poses = item['pose'].to(self.device)
			caption_tokens = item['caption_tokens'].to(self.device)
			caption_lengths = item['caption_lengths'].to(self.device)
			caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate within the batch, based on the longest text 

			# online random augmentations
			poses, caption_tokens, caption_lengths = self.data_augmentation_module(poses, caption_tokens, caption_lengths)
			
			# forward
			with torch.set_grad_enabled(is_training):
				output = self.model(poses, caption_tokens, caption_lengths)


			# --- loss computation

			bs = poses.size(0)
			losses = {}

			# get body poses for loss computations
			with torch.set_grad_enabled(is_training):
				bm_rec = self.body_model(**utils.pose_data_as_dict(output['pose_body_pose']))
			with torch.no_grad():
				bm_orig = self.body_model(**utils.pose_data_as_dict(poses))

			# (reconstruction terms)
			losses[f'v2v'] = torch.mean(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v))
			losses[f'jts'] = torch.mean(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts))
			losses[f'rot'] = torch.mean(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), self.model.decsigma_rot))

			# (KL term)
			losses['kldpt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
			
			# (KL regularization terms)
			bs = poses.size(0)
			n_z = torch.distributions.normal.Normal(
				loc=torch.zeros((bs, self.model.latentD), device=self.device, requires_grad=False),
				scale=torch.ones((bs, self.model.latentD), device=self.device, requires_grad=False))
			losses['kldnp'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], n_z), dim=[1])) if self.args.wloss_kldnpmul else torch.tensor(0.0)
			losses['kldnt'] = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['t_z'], n_z), dim=[1])) if self.args.wloss_kldntmul else torch.tensor(0.0)
				
			# (total loss)
			loss = losses['v2v'] * self.args.wloss_v2v + \
				losses['jts'] * self.args.wloss_jts + \
				losses['rot'] * self.args.wloss_rot + \
				losses['kldpt'] * self.args.wloss_kld + \
				losses['kldnp'] * self.args.wloss_kldnpmul * self.args.wloss_kld + \
				losses['kldnt'] * self.args.wloss_kldntmul * self.args.wloss_kld
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)


			# --- other computations (elbo, fid...)

			# (elbos)
			# normalization is a bit different than for the losses
			elbos = {}
			elbos['v2v'] = (-torch.sum(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v), dim=[1,2]) - losses['kldpt']).sum().detach().item() # (batch_size, nb_vertices, 3): first sum over the coeffs, substract the kld, then sum over the batch
			elbos['jts'] = (-torch.sum(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts), dim=[1,2]) - losses['kldpt']).sum().detach().item() # (batch_size, nb_joints, 3): first sum over the coeffs, substract the kld, then sum over the batch
			elbos['rot'] = (-torch.sum(gaussian_nll(output['pose_body_matrot_pose'].view(-1,config.NB_INPUT_JOINTS,3,3), roma.rotvec_to_rotmat(poses.view(-1,config.NB_INPUT_JOINTS,3)), self.model.decsigma_rot), dim =[1,2,3]) - losses['kldpt']).sum().detach().item() # (batch_size, nb_joints, 3, 3): first sum over the coeffs, substract the kld, then sum over the batch
			# normalize, by the batch size and the number of coeffs
			prod = lambda li: reduce(lambda x, y: x*y, li, 1)
			v2v_reweight, jts_reweight, rot_reweight  = [prod(s.shape[1:]) for s in [bm_orig.v, bm_orig.Jtr, output[f'pose_body_matrot_pose']]]
			elbos = {
				'v2v': elbos['v2v'] / (bs * v2v_reweight),
				'jts': elbos['jts'] / (bs * jts_reweight),
				'rot': elbos['rot'] / (bs * rot_reweight)}

			# (fid)
			if not is_training:
				self.fid.add_gen_features( output['pose_body_text'] )


			# --- training step

			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()


			# --- logging

			# format data for logging
			scalars = [('loss', loss_value), ('loss', losses), ('elbo', elbos)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				decsigma = {k:v for k,v in zip(['v2v', 'jts', 'rot'], [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot])}
				scalars += [('lr', lr_value), ('decsigma', decsigma)]

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)


		# computations at the end of the epoch

		# (fid) - must wait to have computed all the features
		if not is_training:
			scalars = [(self.fid.sstr(), self.fid.compute())]
			metric_logger.add_meter(f'{sstr}_{self.fid.sstr()}', logging.SmoothedValue(window_size=1, fmt='{value:.3f}'))
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)


		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	PoseGenerationTrainer(args)()