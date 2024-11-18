##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from torch import nn

import text2pose.config as config
from text2pose.encoders.tokenizers import Tokenizer, get_text_encoder_or_decoder_module_name, get_tokenizer_name
from text2pose.encoders.modules import ConCatModule, L2Norm
from text2pose.encoders.pose_encoder_decoder import PoseEncoder
from text2pose.encoders.text_encoders import TextEncoder, TransformerTextEncoder


class PairText(nn.Module):
	def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512,
			  num_body_joints=config.NB_INPUT_JOINTS,
			  text_encoder_name='distilbertUncased', transformer_topping=None):
		super(PairText, self).__init__()

		self.latentD = latentD

		# Define pose encoder
		self.pose_encoder = PoseEncoder(num_neurons=num_neurons,
								num_neurons_mini=num_neurons_mini,
								latentD=latentD,
								num_body_joints=num_body_joints,
								role="retrieval")

		# Define text encoder
		self.text_encoder_name = text_encoder_name
		module_ref = get_text_encoder_or_decoder_module_name(text_encoder_name)
		if module_ref in ["glovebigru"]:
			self.text_encoder = TextEncoder(self.text_encoder_name, latentD=latentD, role="retrieval")
		elif module_ref in ["glovetransf", "distilbertUncased"]:
			self.text_encoder = TransformerTextEncoder(self.text_encoder_name, latentD=latentD, topping=transformer_topping, role="retrieval")
		else:
			raise NotImplementedError

		# Define projecting layers
		self.pose_mlp = nn.Sequential(
			ConCatModule(),
			nn.Linear(2 * latentD, 2 * latentD),
			nn.LeakyReLU(),
			nn.Linear(2 * latentD, latentD),
			nn.LeakyReLU(),
			nn.Linear(latentD, latentD),
			nn.LeakyReLU(),
			L2Norm()
		)

		# Loss temperature
		self.loss_weight = torch.nn.Parameter( torch.FloatTensor((10,)) )
		self.loss_weight.requires_grad = True

	def forward(self, poses_A, captions, caption_lengths, poses_B):
		embed_AB = self.encode_pose_pair(poses_A, poses_B)
		text_embs = self.encode_text(captions, caption_lengths)
		return embed_AB, text_embs

	def encode_raw_text(self, raw_text):
		if not hasattr(self, 'tokenizer'):
			self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
		tokens = self.tokenizer(raw_text).to(device=self.loss_weight.device)
		length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
		text_embs = self.text_encoder(tokens.view(1, -1), length)
		return text_embs
	
	def encode_pose_pair(self, poses_A, poses_B):
		embed_poses_A = self.pose_encoder(poses_A)
		embed_poses_B = self.pose_encoder(poses_B)
		embed_AB = self.pose_mlp([embed_poses_A, embed_poses_B])
		return embed_AB
	
	def encode_text(self, captions, caption_lengths):
		return self.text_encoder(captions, caption_lengths)