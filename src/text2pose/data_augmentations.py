##############################################################
## text2pose                                                ##
## Copyright (c) 2023, 2024                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from copy import deepcopy

import text2pose.config as config
from text2pose.encoders.tokenizers import Tokenizer
from text2pose.posescript.utils import ALL_JOINT_NAMES


class PoseFlip():

	def __init__(self, nb_joints=22):
		super(PoseFlip, self).__init__()

		# get joint names (depends on the case)
		if nb_joints == 21:
			# all main joints, without the root
			joint_names = ALL_JOINT_NAMES[1:22]
		elif nb_joints == 22:
			# all main joints, with the root
			joint_names = ALL_JOINT_NAMES[:22]
		elif nb_joints == 52:
			joint_names = ALL_JOINT_NAMES[:]
		else:
			raise NotImplementedError

		# build joint correspondance indices
		n2i = {n:i for i, n in enumerate(joint_names)}
		l2r_j_id = {i:n2i[n.replace("left", "right")] for n,i in n2i.items() if "left" in n} # joint index correspondance between left and right
		self.left_joint_inds = torch.tensor(list(l2r_j_id.keys()))
		self.right_joint_inds = torch.tensor(list(l2r_j_id.values()))

	def flip_pose_data_LR(self, pose_data):
		"""
		pose_data: shape (batch_size, nb_joint, 3)
		"""
		l_data = deepcopy(pose_data[:,self.left_joint_inds])
		r_data = deepcopy(pose_data[:,self.right_joint_inds])
		pose_data[:,self.left_joint_inds] = r_data
		pose_data[:,self.right_joint_inds] = l_data
		pose_data[:,:, 1:3] *= -1
		return pose_data
	
	def __call__(self, pose_data):
		return self.flip_pose_data_LR(pose_data.clone())


def DataAugmentation(args, mode, tokenizer_name=None, nb_joints=config.NB_INPUT_JOINTS):
	# --- define process
	if mode == "posefix":
		return PosefixDataAugmentation(args, tokenizer_name, nb_joints)
	elif mode == "posescript":
		return PosescriptDataAugmentation(args, tokenizer_name, nb_joints)
	else:
		raise NotImplementedError


class GenericDataAugmentation():
	def __init__(self, args, tokenizer_name=None, nb_joints=config.NB_INPUT_JOINTS):
		super(GenericDataAugmentation, self).__init__()

		self.args = args
		
		# --- initialize data augmentation tools
		if tokenizer_name and (self.args.apply_LR_augmentation or self.args.copy_augmentation):
			self.tokenizer = Tokenizer(tokenizer_name)
		
		if tokenizer_name and self.args.copy_augmentation:
			empty_text_tokens = self.tokenizer("")
			self.empty_text_length = len(empty_text_tokens) # account for BOS & EOS tokens
			self.empty_text_tokens = torch.cat( (empty_text_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-self.empty_text_length, dtype=empty_text_tokens.dtype) ), dim=0)
		
		if self.args.apply_LR_augmentation:
			self.pose_flip = PoseFlip(nb_joints)


class PosescriptDataAugmentation(GenericDataAugmentation):
	def __init__(self, args, tokenizer_name=None, nb_joints=config.NB_INPUT_JOINTS):
		super(PosescriptDataAugmentation, self).__init__(args, tokenizer_name=tokenizer_name, nb_joints=nb_joints)

	def __call__(self, poses, caption_tokens=None, caption_lengths=None):

		batch_size = poses.size(0) # beware of incomplete batches!

		# random L/R flip
		if self.args.apply_LR_augmentation:
			flippable = torch.rand(batch_size) < 0.5 # completely random flip
			if hasattr(self, "tokenizer"):
				caption_tokens, caption_lengths, actually_flipped = self.tokenizer.flip(caption_tokens, flippable)
			else:
				actually_flipped = flippable
			poses[actually_flipped] = self.pose_flip(poses[actually_flipped])

		return poses, caption_tokens, caption_lengths
	

class PosefixDataAugmentation(GenericDataAugmentation):
	def __init__(self, args, tokenizer_name=None, nb_joints=config.NB_INPUT_JOINTS):
		super(PosefixDataAugmentation, self).__init__(args, tokenizer_name=tokenizer_name, nb_joints=nb_joints)

	def __call__(self, poses_A, caption_tokens=None, caption_lengths=None, poses_B=None, posescript_poses=None):
		
		batch_size = poses_A.size(0) # beware of incomplete batches!
		
		# random L/R flip
		if self.args.apply_LR_augmentation:
			flippable = torch.rand(batch_size) < 0.5 # completely random flip
			if hasattr(self, "tokenizer"):
				caption_tokens, caption_lengths, actually_flipped = self.tokenizer.flip(caption_tokens, flippable)
			else:
				actually_flipped = flippable
			poses_A[actually_flipped] = self.pose_flip(poses_A[actually_flipped])
			poses_B[actually_flipped] = self.pose_flip(poses_B[actually_flipped])

		# remove text cue: learn to copy pose A
		if self.args.copy_augmentation > 0:
			change = torch.rand(batch_size) < self.args.copy_augmentation # change at most a proportion of args.copy_augmentation poses
			if self.args.copyB2A:
				copy_B2A = torch.ones(batch_size).bool()
			else:
				# in the case of PoseScript data, A is "0"; therefore, for such
				# elements, we must copy B to A
				copy_B2A = posescript_poses # (batch_size)
			copy_A2B = ~copy_B2A
			poses_A[copy_B2A*change] = deepcopy(poses_B[copy_B2A*change])
			poses_B[copy_A2B*change] = deepcopy(poses_A[copy_A2B*change])
			# empty text
			if hasattr(self, "tokenizer"):
				caption_tokens[change] = self.empty_text_tokens[:caption_tokens.shape[1]] # by default, `empty_text_tokens` is very long
				caption_lengths[change] = self.empty_text_length
				caption_tokens = caption_tokens[:,:caption_lengths.max()] # update truncation, the longest text may have changed

		return poses_A, caption_tokens, caption_lengths, poses_B