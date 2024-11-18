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
import roma
from functools import reduce
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.option import get_args_parser
from text2pose.trainer import GenericTrainer
from text2pose.generative_B.model_generative_B import PoseBGenerator
from text2pose.data import PoseFix, PoseMix, PoseScript, PoseStream
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.data_augmentations import DataAugmentation
from text2pose.loss import laplacian_nll, gaussian_nll

import text2pose.config as config
import text2pose.utils as utils
import text2pose.utils_logging as logging

# visualization details
LOGGED_POSE_COLORS = {"original":"grey", "reconstructed":"green", "generated":"blue"}
config.meshviewer_size = 200 # plot small images in tensorboard; this line must happen before the import of text2pose.utils_visu
import text2pose.utils_visu as utils_visu


################################################################################


class PoseEditingTrainer(GenericTrainer):

	def __init__(self, args):
		super(PoseEditingTrainer, self).__init__(args)


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


	def init_dataloaders(self):
		super().init_dataloaders()
		if self.args.pose_stream:
			print('Load auxiliary training dataset (pose stream)')
			dataset_posestream_train = PoseStream(split='train', num_body_joints=self.args.num_body_joints)
			print(dataset_posestream_train, len(dataset_posestream_train))
			self.data_loader_posestream_train = torch.utils.data.DataLoader(
				dataset_posestream_train, sampler=None, shuffle=True,
				batch_size=self.args.batch_size,
				num_workers=self.args.num_workers,
				pin_memory=True,
				drop_last=True,
			)


	def init_model(self):
		print('Load model')
		self.model = PoseBGenerator(text_encoder_name=self.args.text_encoder_name,
							transformer_topping=self.args.transformer_topping,
							latentD=self.args.latentD,
							num_body_joints=self.args.num_body_joints,
							special_text_latentD=self.args.special_text_latentD,
							correction_module_mode=self.args.correction_module_mode)
		self.model.to(self.device)


	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': self.model.pose_encoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.pose_decoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.pose_decoder_distribution_layer.parameters(), 'lr': self.args.lr*self.args.lrposemul})
		param_groups.append({'params': self.model.text_encoder.parameters(), 'lr': self.args.lr*self.args.lrtextmul}) 
		param_groups.append({'params': [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot] + \
										list(self.model.correction_module.parameters()) + \
										list(self.model.pose_text_fusion_distribution_layer.parameters())})
		return param_groups


	def init_other_training_elements(self):
		self.init_fid(name_in_batch="poses_B")
		self.init_body_model()
		self.data_augmentation_module = DataAugmentation(self.args, mode="posefix", tokenizer_name=get_tokenizer_name(self.args.text_encoder_name), nb_joints=self.args.num_body_joints)


	def training_epoch(self, epoch):
		if self.args.pose_stream:
			train_stats_pose_stream = self.one_epoch(
				epoch=epoch,
				is_training=True,
				pose_stream=True
			)
		train_stats = self.one_epoch(
			epoch=epoch,
			is_training=True
		)
		if self.args.pose_stream: train_stats.update(**train_stats_pose_stream)
		return train_stats
	

	def validation_epoch(self, epoch):
		if self.args.val_every and (epoch+1)%self.args.val_every==0: # and lr_scheduler is None
			return self.one_epoch(
							epoch=epoch,
							is_training=False
						)
		return {}
	

	def one_epoch(self, epoch, is_training, pose_stream=False):

		self.model.train(is_training)

		# define loggers
		metric_logger = logging.MetricLogger(delimiter="  ")
		if is_training:
			prefix, sstr = '', 'train'
			metric_logger.add_meter(f'{sstr}_lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
			if pose_stream:
				prefix, sstr = '[pose_stream] ', 'trainPoseStream'
		else:
			prefix, sstr = '[val] ', 'val'
		header = f'{prefix}Epoch: [{epoch}]'
		
		# define dataloader & other elements
		if is_training:
			data_loader = self.data_loader_train if not pose_stream else self.data_loader_posestream_train
		if not is_training:
			data_loader = self.data_loader_val
			self.fid.reset_gen_features()

		# iterate over the batches
		for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):


			# --- forward pass

			# [regular stream]
			if not pose_stream:

				# get data
				poses_A = item['poses_A'].to(self.device)
				poses_B = item['poses_B'].to(self.device)
				caption_tokens = item['caption_tokens'].to(self.device)
				caption_lengths = item['caption_lengths'].to(self.device)
				caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate within the batch, based on the longest text

				# online random augmentations
				posescript_poses = item['poses_A_ids'] == config.PID_NAN
				poses_A, caption_tokens, caption_lengths, poses_B = self.data_augmentation_module(poses_A, caption_tokens, caption_lengths, poses_B, posescript_poses)
				
				# forward
				with torch.set_grad_enabled(is_training):
					output = self.model(poses_A, caption_tokens, caption_lengths, poses_B)

			# [pose stream]
			else:

				# get data
				poses_B = item['pose'].to(self.device)
				# forward
				with torch.set_grad_enabled(is_training):
					output = self.model.forward_autoencoder(poses_B)


			# --- loss computation

			# get body poses for loss computations
			with torch.set_grad_enabled(is_training):
				bm_rec = self.body_model(**utils.pose_data_as_dict(output['pose_body_pose']))
			with torch.no_grad():
				bm_orig = self.body_model(**utils.pose_data_as_dict(poses_B))

			# (term initialization)
			bs = poses_B.size(0)
			losses = {k: torch.zeros(bs) for k in ['v2v', 'jts', 'rot', 'kldft', 'kldft_training', 'kldnp']}

			# (normalization terms: number of coefficients)
			prod = lambda li: reduce(lambda x, y: x*y, li, 1)
			v2v_reweight, jts_reweight, rot_reweight  = [prod(s.shape[1:]) for s in [bm_orig.v, bm_orig.Jtr, output[f'pose_body_matrot_pose']]]

			# (reconstruction terms; sum over coeffs)
			losses['v2v'] = torch.sum(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v), dim=[1,2]) # size (batch_size)
			losses['jts'] = torch.sum(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts), dim=[1,2]) # size (batch_size)
			losses['rot'] = torch.sum(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,config.NB_INPUT_JOINTS,3,3), roma.rotvec_to_rotmat(poses_B.view(-1,config.NB_INPUT_JOINTS,3)), self.model.decsigma_rot), dim=[1,2,3]) # size (batch_size)

			# (KL term between modalities' distributions)
			if not pose_stream:
				losses['kldft'] = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['q_f']), dim=[1]) # size (batch_size)
				losses['kldft_training'] = losses['kldft'].clamp(min=self.args.kld_epsilon) if self.args.kld_epsilon else losses['kldft'] # size (batch_size)
				# NOTE: `kldft_training` is only to be used at training time;
				# the clamping helps to prevent model collapse

			# (KL regularization term)
			n_z = torch.distributions.normal.Normal(
				loc=torch.zeros((bs, self.model.latentD), device=self.device, requires_grad=False),
				scale=torch.ones((bs, self.model.latentD), device=self.device, requires_grad=False))
			losses['kldnp'] = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], n_z), dim=[1]) # size (batch_size)

			# (last arrangements)
			# at validation, compute the "correct/actual" loss value, to compare
			# values across experiments
			wloss_kld = self.args.wloss_kld if is_training else 1.0
			kld_loss = losses['kldft_training'] if is_training else losses['kldft']
			
			# (total loss computation)
			loss = self.args.wloss_v2v * (losses['v2v'] + wloss_kld * kld_loss) / v2v_reweight + \
					self.args.wloss_jts * (losses['jts'] + wloss_kld * kld_loss) / jts_reweight + \
					self.args.wloss_rot * (losses['rot'] + wloss_kld * kld_loss) / rot_reweight + \
					self.args.wloss_kldnpmul * losses['kldnp'] #  regularization
			loss = torch.mean(loss)
			# sanity check
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)
			
			# (prepare loss terms for logging)
			for k, v in losses.items():
				losses[k] = torch.mean(v)


			# --- other computations (elbo, fid...)

			# (elbos)
			# normalization is a bit different than for the losses
			if not pose_stream:
				elbos = {}
				elbos['v2v'] = (-torch.sum(laplacian_nll(bm_orig.v, bm_rec.v, self.model.decsigma_v2v), dim=[1,2]) - losses['kldft']).sum().detach().item() # (batch_size, nb_vertices, 3): first sum over the coeffs, substract the kld, then sum over the batch
				elbos['jts'] = (-torch.sum(laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, self.model.decsigma_jts), dim=[1,2]) - losses['kldft']).sum().detach().item() # (batch_size, nb_joints, 3): first sum over the coeffs, substract the kld, then sum over the batch
				elbos['rot'] = (-torch.sum(gaussian_nll(output['pose_body_matrot_pose'].view(-1,config.NB_INPUT_JOINTS,3,3), roma.rotvec_to_rotmat(poses_B.view(-1,config.NB_INPUT_JOINTS,3)), self.model.decsigma_rot), dim =[1,2,3]) - losses['kldft']).sum().detach().item() # (batch_size, nb_joints, 3, 3): first sum over the coeffs, substract the kld, then sum over the batch
				# normalize, by the batch size and the number of coeffs
				elbos = {
					'v2v': elbos['v2v'] / (bs * v2v_reweight),
					'jts': elbos['jts'] / (bs * jts_reweight),
					'rot': elbos['rot'] / (bs * rot_reweight)}

			# (fid)
			if not is_training: # ==> [regular stream]
				self.fid.add_gen_features( output['pose_body_fusion'] )


			# --- training step

			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()


			# --- logging

			# format data for logging
			scalars = [('loss', loss_value), ('loss', losses)]
			images = None # init & default
			if not pose_stream:
				scalars += [('elbo', elbos)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				decsigma = {k:v for k,v in zip(['v2v', 'jts', 'rot'], [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot])}
				scalars += [('lr', lr_value), ('decsigma', decsigma)]
			else: # validation
				# add images to get insights about the good progress of the training
				with torch.inference_mode():
					if data_iter_step == 0: # only for the first batch
						bm_gen = self.body_model(**utils.pose_data_as_dict(output['pose_body_fusion']))
						bm_all_log = {"original":bm_orig,
									"reconstructed":bm_rec,
									"generated":bm_gen}
						img_batch = []
						nb_examples = 3 # must be < batch size
						for i in range(nb_examples):
							for bm_type in bm_all_log:
								img_batch += utils_visu.image_from_body_vertices(utils_visu.c2c(bm_all_log[bm_type].v[i]), utils_visu.c2c(self.body_model.f), color=LOGGED_POSE_COLORS[bm_type])
						images = ('B_poses', nb_examples, img_batch)

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, images=images, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
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
	
	PoseEditingTrainer(args)()