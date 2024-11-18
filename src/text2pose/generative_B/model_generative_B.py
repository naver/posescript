##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import torch.nn as nn

from human_body_prior.models.vposer_model import NormalDistDecoder

import text2pose.config as config
from text2pose.encoders.tokenizers import Tokenizer, get_text_encoder_or_decoder_module_name, get_tokenizer_name
from text2pose.encoders.modules import ConCatModule, TIRG
from text2pose.data import T_POSE
from text2pose.encoders.pose_encoder_decoder import PoseDecoder, PoseEncoder
from text2pose.encoders.text_encoders import TextEncoder, TransformerTextEncoder


class PoseBGenerator(nn.Module):

	def __init__(self, num_neurons=512, latentD=32, num_body_joints=config.NB_INPUT_JOINTS, special_text_latentD=None,
					text_encoder_name='distilbertUncased', transformer_topping=None,
					correction_module_mode="tirg"):
		super(PoseBGenerator, self).__init__()

		self.latentD = latentD # access needed in train file
		special_text_latentD = special_text_latentD if special_text_latentD else latentD

		# Define pose auto-encoder
		self.pose_encoder = PoseEncoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints, role="modifier")
		self.pose_decoder = PoseDecoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints)
		self.pose_decoder_distribution_layer = NormalDistDecoder(latentD, latentD)

		# Define text encoder
		self.text_encoder_name = text_encoder_name
		module_ref = get_text_encoder_or_decoder_module_name(text_encoder_name)
		if module_ref in ["glovebigru"]:
			self.text_encoder = TextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=special_text_latentD, role="modifier")
		elif module_ref in ["glovetransf", "distilbertUncased"]:
			self.text_encoder = TransformerTextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=special_text_latentD, topping=transformer_topping, role="modifier")
		else:
			raise NotImplementedError
		
		# Define fusing module
		self.correction_module = CorrectionModule(latentD=latentD, text_latentD=special_text_latentD, mode=correction_module_mode)

		# Define module to get distribution parameters
		self.pose_text_fusion_distribution_layer = NormalDistDecoder(latentD, latentD) # must be of same dimension as the pose encoder/decoder latent dimention

		# Define learned loss parameters
		self.decsigma_v2v = nn.Parameter( torch.zeros(1) ) # logsigma
		self.decsigma_jts = nn.Parameter( torch.zeros(1) ) # logsigma
		self.decsigma_rot = nn.Parameter( torch.zeros(1) ) # logsigma


	# FORWARD METHODS ----------------------------------------------------------


	def encode_pose(self, pose_body):
		return self.pose_encoder(pose_body)

	def encode_text(self, captions, caption_lengths):
		return self.text_encoder(captions, caption_lengths)

	def decode_pose(self, z):
		return self.pose_decoder(z)

	def fuse_input_pose_and_text(self, embed_poses_A, embed_texts):
		x = self.correction_module(embed_poses_A, embed_texts)
		x = self.pose_text_fusion_distribution_layer(x)
		return x

	def forward_autoencoder(self, poses_B):
		# encode all elements
		embed_poses_B = self.encode_pose(poses_B)
		# get distribution of pose B
		q_z = self.pose_decoder_distribution_layer(embed_poses_B)
		# sample
		q_z_sample = q_z.rsample()
		ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}
		ret.update({'q_z': q_z})
		return ret

	def forward(self, poses_A, captions, caption_lengths, poses_B):
		# encode all elements
		embed_texts = self.encode_text(captions, caption_lengths)
		embed_poses_A = self.encode_pose(poses_A)
		embed_poses_B = self.encode_pose(poses_B)
		# get distribution of pose B
		q_z = self.pose_decoder_distribution_layer(embed_poses_B)
		# fuse pose A & text; get distribution of the fusion
		q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)
		# sample
		q_f_sample = q_f.rsample()
		q_z_sample = q_z.rsample()
		ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}
		ret.update({f"{k}_fusion":v for k,v in self.decode_pose(q_f_sample).items()})
		ret.update({'q_f': q_f, 'q_z': q_z})
		return ret
	

	# SAMPLE METHODS -----------------------------------------------------------


	def sample_nposes(self, poses_A, captions, caption_lengths, n=1, **kwargs):
		embed_poses_A = self.encode_pose(poses_A)
		embed_texts = self.encode_text(captions, caption_lengths)
		q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)
		z = q_f.sample( [n] ).permute(1,0,2).flatten(0,1)
		decode_results = self.decode_pose(z)
		return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}

	def sample_str_nposes(self, pose_A, s="", n=1): # expect pose_A to be of format (1, nb_joints, 3)
		device = self.decsigma_v2v.device
		# encode the input pose & text to sample a pose conditioned on it
		if not hasattr(self, 'tokenizer'):
			self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
		tokens = self.tokenizer(s).to(device=device)
		return self.sample_nposes(pose_A, tokens.view(1, -1), torch.tensor([ len(tokens) ], dtype=tokens.dtype), n=n)

	def sample_str_meanposes(self, pose_A, s=""): # expect pose_A to be of format (1, nb_joints, 3)
		device = self.decsigma_v2v.device
		# encode the input pose & text to sample a pose conditioned on it
		if not hasattr(self, 'tokenizer'):
			self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
		tokens = self.tokenizer(s).to(device=device)
		caption_length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
		embed_poses_A = self.encode_pose(pose_A)
		embed_texts = self.encode_text(tokens.view(1, -1), caption_length)
		q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)
		return self.decode_pose(q_f.mean.view(1, -1))
	

	# METHODS FOR SPECIAL EVALUATION -------------------------------------------


	def special_eval_forward(self, poses_A, captions, caption_lengths, poses_B):
		### begin as usual
		# encode all elements
		embed_texts = self.encode_text(captions, caption_lengths)
		embed_poses_A = self.encode_pose(poses_A)
		embed_poses_B = self.encode_pose(poses_B)
		# get distribution of pose B
		q_z = self.pose_decoder_distribution_layer(embed_poses_B)
		# sample
		q_z_sample = q_z.rsample()
		ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}

		### but compute q_f differently
		q_f = self.special_eval_fuse_input_pose_and_text(captions, caption_lengths, embed_poses_A, embed_texts)

		### end as usual
		q_f_sample = q_f.rsample()
		ret.update({f"{k}_fusion":v for k,v in self.decode_pose(q_f_sample).items()})
		ret.update({'q_f': q_f, 'q_z': q_z})
		return ret

	def special_eval_sample_nposes(self, poses_A, captions, caption_lengths, n=1, **kwargs):
		### begin as usual
		embed_poses_A = self.encode_pose(poses_A)
		embed_texts = self.encode_text(captions, caption_lengths)

		### but compute q_f differently
		q_f = self.special_eval_fuse_input_pose_and_text(captions, caption_lengths, embed_poses_A, embed_texts)

		### end as usual
		z = q_f.sample( [n] ).permute(1,0,2).flatten(0,1)
		decode_results = self.decode_pose(z)
		return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}

	def special_eval_fuse_input_pose_and_text(self, captions, caption_lengths, embed_poses_A, embed_texts):
		# NOTE: some class attributes are expected to be initialized externally
		# before this method is called.
		# Such attributes include:
		# 	- self.special_eval_setting
		# 	- self.empty_text_tokens, self.empty_text_length

		if self.special_eval_setting == "pose_similarity":
			q_f = self.pose_decoder_distribution_layer(embed_poses_A)

		elif self.special_eval_setting == "pose_only":
			captions = captions[:,self.empty_text_length] # truncate to final length
			captions = self.empty_text_tokens.unsqueeze(0).repeat(captions.shape[0], 1) # expand to (batch_size, length)
			caption_lengths = torch.ones_like(caption_lengths) * self.empty_text_length
	
			embed_texts = self.encode_text(captions, caption_lengths)
			q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)

		elif self.special_eval_setting == "text_only":
			poses_A = T_POSE.repeat(captions.shape[0], 1, 1).to(embed_poses_A.device) # expand to get size (batch_size, NB_INPUT_JOINTS, 3)
			embed_poses_A = self.encode_pose(poses_A)
			q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)

		elif self.special_eval_setting == "regular": # equivalent to self.forward()
			q_f = self.fuse_input_pose_and_text(embed_poses_A, embed_texts)

		return q_f


class CorrectionModule(nn.Module):
	"""
	Given a pose A and the modifier m, compute an embedding representing the
	result from the modification of A by m.
	"""

	def __init__(self, latentD, text_latentD, mode="tirg"):
		super(CorrectionModule, self).__init__()

		self.mode = mode

		if self.mode == "tirg":
			self.tirg = TIRG(input_dim=[latentD, text_latentD], output_dim=latentD, out_l2_normalize=False)
			self.forward = self.tirg.query_compositional_embedding
		elif self.mode == "concat-mlp":
			self.sequential = nn.Sequential(
				ConCatModule(),
				nn.BatchNorm1d(latentD+text_latentD),
				nn.ReLU(),
				nn.Linear(latentD+text_latentD, 2 * latentD),
				nn.ReLU(),
				nn.Linear(2 * latentD, latentD)
			)
			self.forward = self.process_concat_sequential
		else:
			print(f"Name for the mode of the correction module is unknown (provided {mode}).")
			raise NotImplementedError

	def process_concat_sequential(self, pose_embeddings, modifier_embeddings):
		return self.sequential((pose_embeddings, modifier_embeddings))