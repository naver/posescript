##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch.nn as nn

import text2pose.config as config
from text2pose.encoders.tokenizers import get_text_encoder_or_decoder_module_name
from text2pose.encoders.pose_encoder_decoder import PoseEncoder
from text2pose.encoders.text_decoders import TransformerTextDecoder, ModalityInputAdapter

class DescriptionGenerator(nn.Module):

	def __init__(self, num_neurons=512, encoder_latentD=32, decoder_latentD=512,
			  	num_body_joints=config.NB_INPUT_JOINTS,
			  	decoder_nhead=8, decoder_nlayers=4, text_decoder_name="",
				transformer_mode="crossattention"):
		super(DescriptionGenerator, self).__init__()

		# Define pose encoder
		self.pose_encoder = PoseEncoder(num_neurons=num_neurons, latentD=encoder_latentD, num_body_joints=num_body_joints, role="retrieval")

		# Define modality input adaptor
		self.modalityInputAdapter = ModalityInputAdapter(inlatentD=encoder_latentD,
						  								outlatentD=decoder_latentD)

		# Define text decoder
		self.text_decoder_name = text_decoder_name
		self.transformer_mode = transformer_mode
		module_ref = get_text_encoder_or_decoder_module_name(text_decoder_name)
		if module_ref == "transformer":
			self.text_decoder = TransformerTextDecoder(self.text_decoder_name,
														nhead=decoder_nhead,
														nlayers=decoder_nlayers,
														decoder_latentD=decoder_latentD,
														transformer_mode=transformer_mode)
		else:
			raise NotImplementedError
		

	def encode_pose(self, pose_body):
		return self.pose_encoder(pose_body)

	def decode_text(self, z, captions, caption_lengths, train=False):
		return self.text_decoder(z, captions, caption_lengths, train=train)

	def forward(self, poses, captions, caption_lengths):
		z = self.encode_pose(poses)
		z = self.modalityInputAdapter(z)
		decoded = self.decode_text(z, captions, caption_lengths, train=True)
		return dict(z=z, **decoded)

	def generate_text(self, poses):
		z = self.encode_pose(poses)
		z = self.modalityInputAdapter(z)
		decoded_texts, likelihood_scores = self.text_decoder.generate_greedy(z)
		return decoded_texts, likelihood_scores