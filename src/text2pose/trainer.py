##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import time, datetime
import numpy as np
import json
from pathlib import Path
import shutil
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.option import get_output_dir
from text2pose.fid import FID


################################################################################


class GenericTrainer():

	def __init__(self, args):
		super(GenericTrainer, self).__init__()

		# define setting
		self.args = args
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		cudnn.benchmark = True

		# fix the seed for reproducibility
		seed = args.seed
		torch.manual_seed(seed)
		np.random.seed(seed)

		# create path to saving directory
		if args.output_dir=='':
			args.output_dir = get_output_dir(args)
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
		print(args.output_dir)
		# define checkpoint
		self.ckpt_fname = os.path.join(self.args.output_dir, 'checkpoint_last.pth')


	# METHODS TO DEFINE IN CHILD CLASSES
	############################################################################

	def init_model(self):
		"""
		Must define self.model.
		"""
		raise NotImplementedError


	def get_param_groups(self):
		"""
		Must return a list of param groups.
		"""
		raise NotImplementedError


	def init_other_training_elements(self):
		"""
		Must initialize as attributes other elements necessary at training or
		validation time, in particular the data augmentation module. It can be
		used to include the initialization of elements like the FID or the body
		model. This method should not return anything.
		"""
	

	def load_dataset(self, split, caption_index):
		"""
		Must return a dataset.
		"""
		raise NotImplementedError
	

	def training_epoch(self):
		"""
		Must return a dict of training statistics.
		"""
		raise NotImplementedError
	

	def validation_epoch(self, epoch):
		"""
		Must return a dict of validation statistics.
		"""
		raise NotImplementedError


	# GENERIC METHODS
	############################################################################


	# TRAINING STATUS ----------------------------------------------------------


	def check_training_status(self):
		if os.path.isfile(self.ckpt_fname):
			ckpt = torch.load(self.ckpt_fname, 'cpu')
			if ckpt["epoch"] == self.args.epochs - 1:
				print(f'Training already done ({self.args.epochs} epochs: {os.path.basename(self.ckpt_fname)})!')
				return
			

	def start_or_resume_training(self):
		if os.path.isfile(self.ckpt_fname): # resume training
			print("Resume training. Load weights from last checkpoint.")
			ckpt = torch.load(self.ckpt_fname, 'cpu')
			self.start_epoch = ckpt['epoch'] + 1
			self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
			self.model.load_state_dict(ckpt['model'])
			self.optimizer.load_state_dict(ckpt['optimizer'])
		else:
			self.start_epoch = 0
			self.best_val_loss = float('inf')
			if self.args.pretrained: # load pretrained weights
				pretrained_path = (config.shortname_2_model_path[self.args.pretrained]).format(seed=self.args.seed)
				print(f'Loading pretrained model: {pretrained_path}')
				ckpt = torch.load(pretrained_path, 'cpu')
				self.model.load_state_dict(ckpt['model'])


	def has_model_improved(self, val_stats):
		if val_stats and val_stats['val_loss'] < self.best_val_loss:
			self.best_val_loss = val_stats['val_loss']
			return True
		return False
	

	def save_checkpoint(self, save_best_model, epoch):
		tosave = {'epoch': epoch,
				'model': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'args': self.args,
				'best_val_loss': self.best_val_loss}
		# overwrite the current checkpoint after each epoch
		torch.save(tosave, self.ckpt_fname)
		# overwrite the "best" checkpoint each time the model improves
		if save_best_model:
			shutil.copyfile(self.ckpt_fname, os.path.join(self.args.output_dir, 'checkpoint_best.pth'))
		# save an independent checkpoint every required time step
		if epoch and epoch % self.args.saving_ckpt_step == 0:
			shutil.copyfile(self.ckpt_fname, os.path.join(self.args.output_dir, 'checkpoint_{:d}.pth'.format(epoch)))
		

	# INITIALIZATIONS ----------------------------------------------------------


	def init_dataloaders(self):
		print('Load training dataset')
		dataset_train = self.load_dataset('train', caption_index='rand')
		print(dataset_train, len(dataset_train))
		self.data_loader_train = torch.utils.data.DataLoader(
			dataset_train, sampler=None, shuffle=True,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			pin_memory=True,
			drop_last=True,
		)
		print('Load validation dataset')
		dataset_val = self.load_dataset('val', caption_index='deterministic-mix')
		print(dataset_val, len(dataset_val))
		self.data_loader_val = torch.utils.data.DataLoader(
			dataset_val, sampler=None, shuffle=False,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			pin_memory=True,
			drop_last=False
		)


	def init_optimizer(self):
		assert self.args.optimizer=='Adam'
		param_groups = self.get_param_groups()
		self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr, weight_decay=self.args.wd)


	# (other optional initializations) -----------------------------------------
	# (to be added, if need be, in the overriden version of
	# self.init_other_training_elements)


	def init_body_model(self):
		print('Initialize body model')
		self.body_model = BodyModel(model_type = config.POSE_FORMAT,
									bm_fname = config.NEUTRAL_BM,
									num_betas = config.n_betas)
		self.body_model.eval()
		self.body_model.to(self.device)
	

	def init_fid(self, name_in_batch, seed=1):
		# NOTE: default seed=1 for consistent evaluation
		# prepare fid on the val set
		print('Preparing FID')
		self.fid = FID((self.args.fid, seed), device=self.device, name_in_batch=name_in_batch)
		self.fid.extract_real_features(self.data_loader_val)


	# TRAINING PROCESS ---------------------------------------------------------


	def print_trainable_params(self):
		l = []
		for n, p in self.model.named_parameters():
			if p.requires_grad == True:
				l.append(n)
		print("Trainable parameters:", l)


	def add_data_to_log_writer(self, epoch, sstr, scalars=None, images=None, texts=None, is_training=None, data_iter_step=1, total_steps=1, should_log_data=None):
		"""
		Args:
		    scalars: list of tuples (variable name, value) or
					(variable name, dict{sub_variable_name:sub_variable_value})
			images: tuple (variable name, number of examples, tensor of size BHWC)
			texts: tuple (variable name, list of texts)
			should_log_data: (None|True), use True to force log
		"""
		if hasattr(self, "log_writer"):
			if should_log_data is None:
				# compute condition for logging
				should_log_data = (data_iter_step==total_steps-1) or data_iter_step%self.args.log_step==0
				should_log_data = should_log_data and is_training
				should_log_data = should_log_data or (not is_training and data_iter_step==total_steps-1)
			if self.log_writer and should_log_data: 
				# use epoch_1000x as the x-axis in tensorboard to calibrate different
				# curves when batch size changes
				epoch_1000x = int((data_iter_step / total_steps + epoch) * 1000)
				# log scalar values
				for log_name, log_values in scalars:
					if type(log_values) is dict:
						for k, v in log_values.items():
							self.log_writer.add_scalar(f'{sstr}/{log_name}_{k}', v.item() if type(v) is torch.Tensor else v, epoch_1000x)
					else: # scalar value
						self.log_writer.add_scalar(f'{sstr}/{log_name}', log_values.item() if type(log_values) is torch.Tensor else log_values, epoch_1000x)
				# log images
				if images is not None:
					log_name, nb_examples, images = images
					x = len(images) / nb_examples # number of pictures per example
					images = torch.tensor(np.array(images)).permute(0, 3, 1, 2) # convert to torch tensor, and set to BCHW
					grid = make_grid(images, nrow=x, normalize=False) # nrow is actually the number of columns
					self.log_writer.add_image(log_name, grid, epoch_1000x)
				# log texts
				if texts is not None:
					log_name, texts = texts
					for i, t in enumerate(texts):
						self.log_writer.add_text(f'{sstr}/{log_name}_{i}', t, epoch_1000x)


	def add_data_to_metric_logger(self, metric_logger, sstr, scalars):
		"""
		scalars: list of tuples (variable name, value) or (variable name, dict{sub_variable_name:sub_variable_value})
		"""
		d = {}
		for log_name, log_values in scalars:
			if type(log_values) is dict:
				d.update({f'{sstr}_{log_name}_{k}': v.item() if type(v) is torch.Tensor else v for k,v in log_values.items()})
			else: # scalar value
				d[f'{sstr}_{log_name}'] = log_values.item() if type(log_values) is torch.Tensor else log_values
		metric_logger.update( **d )


	def __call__(self):

		# safety checks
		self.check_training_status()
		
		# initializations
		self.init_dataloaders()
		self.init_model()
		self.init_optimizer()
		self.init_other_training_elements()
		
		# load previous ckpt & log parameter names
		self.start_or_resume_training()
		self.print_trainable_params()

		# init tensorboard
		self.log_writer = SummaryWriter(log_dir=self.args.output_dir)
	
		# start training process
		start_time = time.time()
		for epoch in range(self.start_epoch, self.args.epochs): 

			# training
			train_stats = self.training_epoch(epoch)

			# validation
			val_stats = self.validation_epoch(epoch)
			
			# save data
			# -- (ckpt)
			save_best_model = self.has_model_improved(val_stats)
			self.save_checkpoint(save_best_model, epoch)
			# -- (tensorboard)
			self.log_writer.flush()
			# -- (regular logs)
			log_stats = {'epoch': epoch, 'time':time.time(), 'lr': self.optimizer.param_groups[0]["lr"]}
			log_stats.update(**train_stats)
			log_stats.update(**val_stats)
			with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
				f.write(json.dumps(log_stats) + "\n")

		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('Training time {}'.format(total_time_str))