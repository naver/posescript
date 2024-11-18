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
from text2pose.encoders.modules import TIRG
from text2pose.encoders.pose_encoder_decoder import PoseEncoder
from text2pose.encoders.text_decoders import TransformerTextDecoder, ModalityInputAdapter


class FeedbackGenerator(nn.Module):

	def __init__(self, num_neurons=512, encoder_latentD=32, comparison_latentD=32,
					num_body_joints=config.NB_INPUT_JOINTS,
			  		decoder_latentD=512, decoder_nhead=8, decoder_nlayers=4, text_decoder_name="",
					comparison_module_mode="tirg", transformer_mode="crossattention"):
		super(FeedbackGenerator, self).__init__()

		# Define pose encoder
		self.pose_encoder = PoseEncoder(num_neurons=num_neurons, latentD=encoder_latentD, num_body_joints=num_body_joints, role="retrieval")

		# Define fusing module
		self.comparison_module = ComparisonModule(inlatentD=encoder_latentD,
					    						  outlatentD=comparison_latentD,
												  mode=comparison_module_mode)

		# Define modality input adaptor
		self.modalityInputAdapter = ModalityInputAdapter(inlatentD=comparison_latentD,
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

	def fuse_input_poses(self, embed_poses_A, embed_poses_B):
		z = self.comparison_module(embed_poses_A, embed_poses_B)
		z = self.modalityInputAdapter(z)
		return z

	def forward(self, poses_A, captions, caption_lengths, poses_B):
		z_a = self.encode_pose(poses_A)
		z_b = self.encode_pose(poses_B)
		z = self.fuse_input_poses(z_a, z_b)
		decoded = self.decode_text(z, captions, caption_lengths, train=True)
		return dict(z=z, **decoded)

	def generate_text(self, poses_A, poses_B):
		z_a = self.encode_pose(poses_A)
		z_b = self.encode_pose(poses_B)
		z = self.fuse_input_poses(z_a, z_b)
		decoded_texts, likelihood_scores = self.text_decoder.generate_greedy(z)
		return decoded_texts, likelihood_scores


class ComparisonModule(nn.Module):
	"""
	Given two poses A and B, compute an embedding representing the result from
	the comparison of the two poses.
	"""

	def __init__(self, inlatentD, outlatentD, mode="tirg"):
		super(ComparisonModule, self).__init__()

		self.inlatentD = inlatentD
		self.outlatentD = outlatentD
		self.mode = mode

		if mode == "tirg":
			self.tirg = TIRG(input_dim=[inlatentD, inlatentD], output_dim=outlatentD, out_l2_normalize=False)
			self.forward = self.forward_tirg		
		else:
			print(f"Name for the mode of the comparison module is unknown (provided {mode}).")
			raise NotImplementedError


	def forward_tirg(self, pose_A_embeddings, pose_B_embeddings):
		return self.tirg.query_compositional_embedding(pose_B_embeddings, pose_A_embeddings)