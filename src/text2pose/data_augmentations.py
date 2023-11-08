##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from copy import deepcopy

from text2pose.encoders.tokenizers import Tokenizer
from text2pose.posescript.captioning import ALL_JOINT_NAMES


def DataAugmentation(args, mode, tokenizer_name):
		# --- define process
	if mode == "posefix":
		return PosefixDataAugmentation(args, tokenizer_name)
	elif mode == "posescript":
		return PosescriptDataAugmentation(args, tokenizer_name)
	else:
		raise NotImplementedError


class GenericDataAugmentation():
	def __init__(self, args, tokenizer_name):
		super(GenericDataAugmentation, self).__init__()

		self.args = args
		
		# --- initialize data augmentation tools
		if self.args.apply_LR_augmentation or self.args.copy_augmentation:
			self.tokenizer = Tokenizer(tokenizer_name)
		
		if self.args.copy_augmentation:
			empty_text_tokens = self.tokenizer("")
			self.empty_text_length = len(empty_text_tokens) # account for BOS & EOS tokens
			self.empty_text_tokens = torch.cat( (empty_text_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-self.empty_text_length, dtype=empty_text_tokens.dtype) ), dim=0)
		
		if self.args.apply_LR_augmentation:
			n2i = {n:i for i, n in enumerate(ALL_JOINT_NAMES)}
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


class PosescriptDataAugmentation(GenericDataAugmentation):
	def __init__(self, args, tokenizer_name):
		super(PosescriptDataAugmentation, self).__init__(args, tokenizer_name)

	def __call__(self, poses, caption_tokens, caption_lengths):

		batch_size = poses.size(0) # beware of incomplete batches!

		# random L/R flip
		if self.args.apply_LR_augmentation:
			flippable = torch.rand(batch_size) < 0.5 # completely random flip
			caption_tokens, caption_lengths, actually_flipped = self.tokenizer.flip(caption_tokens, flippable)
			poses[actually_flipped] = self.flip_pose_data_LR(poses.clone()[actually_flipped])

		return poses, caption_tokens, caption_lengths
	

class PosefixDataAugmentation(GenericDataAugmentation):
	def __init__(self, args, tokenizer_name):
		super(PosefixDataAugmentation, self).__init__(args, tokenizer_name)

	def __call__(self, poses_A, caption_tokens, caption_lengths, poses_B, posescript_poses):
		
		batch_size = poses_A.size(0) # beware of incomplete batches!
		
		# random L/R flip
		if self.args.apply_LR_augmentation:
			flippable = torch.rand(batch_size) < 0.5 # completely random flip
			caption_tokens, caption_lengths, actually_flipped = self.tokenizer.flip(caption_tokens, flippable)
			poses_A[actually_flipped] = self.flip_pose_data_LR(poses_A.clone()[actually_flipped])
			poses_B[actually_flipped] = self.flip_pose_data_LR(poses_B.clone()[actually_flipped])

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
			caption_tokens[change] = self.empty_text_tokens[:caption_tokens.shape[1]] # by default, `empty_text_tokens` is very long
			caption_lengths[change] = self.empty_text_length
			caption_tokens = caption_tokens[:,:caption_lengths.max()] # update truncation, the longest text may have changed

		return poses_A, caption_tokens, caption_lengths, poses_B