##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from text2pose.encoders.modules import PositionalEncoding
from text2pose.encoders.tokenizers import Tokenizer, get_tokenizer_name, get_text_encoder_or_decoder_module_name


################################################################################
## Text decoders
################################################################################

class TextDecoder(nn.Module):
	def __init__(self):
		super(TextDecoder, self).__init__()

		self.min_decoded_length = 10 # minimum number of tokens to be predicted
		self.max_decoded_length = 200 # maximum number of tokens to be predicted

		self.tokenizer = None # to define in subclasse!

	def generate_greedy(self, z):
		# Input:
		# 	z: size (batch_size, *, latentD)
		# Output:
		# 	- list of texts of size (batch_size)
		# 	- and a tensor of scores of size (batch_size)
		
		# initializations
		device = z.device
		batch_size = z.shape[0]
		captions = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.tokenizer.bos_token_id # decoded tokens
		caption_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
		likelihood = torch.zeros(batch_size, device=device)
		ongoing = torch.ones(batch_size, dtype=torch.bool, device=device) # keep track of captions that are still in the process of being decoded
		
		# iteratively decode the text (until the maximum length is reached)
		for i in range(1, self.max_decoded_length):

			# get token probabilities
			pred = self.forward(z, captions, caption_lengths, train=False)["logits"] # iterative prediction, output of size (batch_size, sequence length, vocab_size)
			pred = F.softmax(pred, dim=-1) # get probabilities
			# prevent choosing the EOS token until the caption is long enough
			if caption_lengths[0].item() < self.min_decoded_length:
				pred[:,-1,self.tokenizer.eos_token_id] = 0
			pred = torch.log(pred) # log-likelihood

			# determine next token using greedy search (token with the highest probability)
			p, inds = pred.topk(1)
			next_item = inds[:,-1]
			likelihood[ongoing] += p[:,-1].view(-1)[ongoing] # get total likelihood; likelihood can only change for uncompleted captions

			# concatenate previous input with predicted best word
			captions = torch.cat((captions, next_item), dim=1)
			captions[~ongoing,-1] = self.tokenizer.pad_token_id # completed captions can only be appended the PAD token
			caption_lengths[ongoing] += 1 # length does not change for completed captions

			# update the list of ongoing captions (ie. captions that can still be completed)
			ongoing = torch.logical_and(ongoing, ~(next_item == self.tokenizer.eos_token_id).view(-1))
			if (~ongoing).all():
				# stop decoding if all captions were completed
				break

		# compute final scores
		scores = likelihood/caption_lengths

		return self.postprocess_generated_text(captions), scores


	def postprocess_generated_text(self, token_ids):
		# token_ids is a tensor of size (batch_size, number of tokens in the longest text)
		generated = self.tokenizer.batch_decode(token_ids)
		return generated


class TransformerTextDecoder(TextDecoder):

	def __init__(self, text_decoder_name, decoder_latentD=512,
				nlayers=4, nhead=4, dim_feedforward=1024, activation="gelu", dropout=0.1,
				transformer_mode="crossattention"):
		super(TransformerTextDecoder, self).__init__()

		module_ref = get_text_encoder_or_decoder_module_name(text_decoder_name)
		assert module_ref == "transformer", f"Text decoder: module {module_ref} is not implemented."

		# get special token ids & vocab info
		tokenizer_name = get_tokenizer_name(text_decoder_name)
		self.tokenizer = Tokenizer(tokenizer_name)
		self.vocab_size = len(self.tokenizer)

		# define common layers to both modes (prompt|crossattention)
		# 1) token embeddings
		self.decoder_latentD = decoder_latentD
		self.embedding = nn.Embedding(self.vocab_size, decoder_latentD) # encode the expected target tokens
		# 2) positional encoding 
		self.positional_encoding = PositionalEncoding(d_model=decoder_latentD, dropout=0.1)
		# 3) token prediction layer (vocabulary size)
		self.final_layer = nn.Linear(decoder_latentD, self.vocab_size) # predict target tokens

		# define how the pose data is injected (prompt|crossattention)
		self.transformer_mode = transformer_mode

		if transformer_mode == "crossattention":

			# transformer with cross-attention
			transformer_decoder_layers = nn.TransformerDecoderLayer(d_model=decoder_latentD,
														nhead=nhead,
														dim_feedforward=dim_feedforward,
														dropout=dropout,
														activation=activation)
			self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layers, nlayers)
			self.forward = self.forward_cross_attention
			
			# define positional encoding to distinguish between the different
			# encoder-from tokens, to be provided as encoder hidden state
			self.encoder_positional_encoding = PositionalEncoding(d_model=decoder_latentD, dropout=0.1)
		
		elif transformer_mode == "prompt":

			# transformer without cross-attention
			transformer_decoder_layers = nn.TransformerEncoderLayer(d_model=decoder_latentD,
														nhead=nhead,
														dim_feedforward=dim_feedforward,
														dropout=dropout,
														activation=activation)
			self.transformer_decoder = nn.TransformerEncoder(transformer_decoder_layers, nlayers)
			self.forward = self.forward_prompt

		else:
			raise NotImplementedError

		# define loss criterion
		self.loss_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
	

	def forward_prompt(self, z, captions, caption_lengths, train=False):
		# Input:
		# 	- z: (batch_size, encoder_sequence_length, model_dim)
		# 	- captions, caption_lengths: ground truth captions & lengths, expected to be generated
		# [prompt: <pose data> <bos> <text data> <eos>]
		
		device = z.device
		batch_size, nb_pose_tokens, hidden_state_dim = z.shape
		
		# Prepare attention masks
		# pose tokens and caption tokens must be attended
		src_padding_mask = ~self.get_padding_mask(nb_pose_tokens+caption_lengths).to(device) # shape (batch_size, input_sequence_length)
		mask = self.get_tgt_mask(nb_pose_tokens+max(caption_lengths)).to(device) # shape (input_sequence_length, input_sequence_length)
		mask[:nb_pose_tokens+1,:nb_pose_tokens+1] = True
		mask = ~ mask

		# Forward with prompting
		# embed caption input ids
		caption_embeds = self.embedding(captions) * math.sqrt(self.decoder_latentD)
		# reshape to (sequence length, batch_size, decoder_latentD)
		caption_embeds = caption_embeds.permute(1, 0, 2)
		z = z.permute(1, 0, 2)
		# concatenate input tokens (prompt pose tokens, then the caption tokens)
		inputs_embeds = torch.cat([z, caption_embeds], dim=0)
		# apply positional encoding
		inputs_embeds = self.positional_encoding(inputs_embeds)

		# decode word queries, using pose tokens for prompting
		x = self.transformer_decoder(src=inputs_embeds,
									mask=mask,
									src_key_padding_mask=src_padding_mask)
		x = self.final_layer(x)

		# reshape to (batch_size, sequence length, vocab_size)
		x = x.permute(1, 0, 2)

		# Prepare output
		ret = dict(logits=x[:,nb_pose_tokens:]) # remove pose logits
		# compute loss	
		if train:
			# shift the tokens by one so the model is trained to predict next token
			get_target = lambda a: a[:,1:].contiguous().view(-1)
			# (shift the tokens also to ignore the pose tokens)
			predicted = x[:,nb_pose_tokens:-1].contiguous().view(-1, self.vocab_size)
			# (NOTE: the pose tokens are not in the initial captions!)
			loss = self.loss_criterion(predicted, get_target(captions))
			# compute loss for wrong associations as a mean of comparison
			shuffling = (torch.arange(batch_size).to(device)+batch_size//2)%batch_size
			fake_loss = self.loss_criterion(predicted, get_target(captions[shuffling]))
			# output
			ret.update(dict(loss=loss, fake_loss=fake_loss))

		return ret


	def forward_cross_attention(self, z, captions, caption_lengths, train=False):
		# Input:
		# 	- z: (batch_size, encoder_sequence_length, model_dim)
		# 	- captions, caption_lengths: ground truth captions & lengths, expected to be generated
		
		device = z.device

		# Prepare encoder data (pose data)
		# reshape to (encoder_sequence_length, batch_size, *)
		z = z.permute(1, 0, 2)
		# add positional encoding if there are several pose tokens
		if z.shape[0] > 1: # encoder_sequence_length
			# the positional encoding module takes input of shape (sequence_length, batch_size, model_dim)
			z = self.encoder_positional_encoding(z)

		# Prepare attention masks
		tgt_padding_mask = ~self.get_padding_mask(caption_lengths).to(device) # shape (batch_size, target_sequence_length)
		tgt_mask = ~self.get_tgt_mask(max(caption_lengths)).to(device) # shape (target_sequence_length, target_sequence_length)

		# Forward
		# embed token queries
		inputs_embeds = self.embedding(captions) * math.sqrt(self.decoder_latentD)
		# reshape to (sequence length, batch_size, decoder_latentD) & apply PE
		inputs_embeds = inputs_embeds.permute(1, 0, 2)
		inputs_embeds = self.positional_encoding(inputs_embeds)

		# decode word queries, using the input embedding z for memory
		x = self.transformer_decoder(memory=z,
									tgt=inputs_embeds,
									tgt_mask=tgt_mask,
									tgt_key_padding_mask=tgt_padding_mask)
		x = self.final_layer(x)

		# reshape to (batch_size, sequence length, vocab_size)
		x = x.permute(1, 0, 2)

		# Prepare output
		ret = dict(logits=x)
		# compute loss	
		if train:
			# shift the tokens by one so the model is trained to predict next token
			get_target = lambda a: a[:,1:].contiguous().view(-1)
			predicted = x[:,:-1].contiguous().view(-1, self.vocab_size)
			loss = self.loss_criterion(predicted, get_target(captions))
			# compute loss for wrong associations as a mean of comparison
			batch_size = x.shape[0]
			shuffling = (torch.arange(batch_size).to(device)+batch_size//2)%batch_size
			fake_loss = self.loss_criterion(predicted, get_target(captions[shuffling]))
			# output
			ret.update(dict(loss=loss, fake_loss=fake_loss))

		return ret


	def get_padding_mask(self, caption_lengths):
		"""
		Get caption mask based on the expected caption length (get False at
		padding positions). Return torch tensor of size (len(caption_lengths),
		max(caption_lengths)).
	
		For instance, 
		given: tensor([4,2,3])
		return: tensor([[ True,  True,  True,  True],
						[ True,  True, False, False],
						[ True,  True,  True, False]]
		"""
		max_length = max(caption_lengths)
		mask = torch.arange(max_length).expand(len(caption_lengths), max_length).to(caption_lengths.device) < caption_lengths.unsqueeze(1)
		return mask


	def get_tgt_mask(self, max_length):
		"""
		Get progressive masking of words for the provided length (get False at
		non-accessed positions). Return torch tensor of size (max_length,
		max_length).

		For instance,
		given: max_length = 3
		return: tensor([[ True, False, False],
						[ True,  True, False],
						[ True,  True,  True]])
		"""
		mask = torch.tril(torch.ones((max_length, max_length))).bool()
		return mask
	

class ModalityInputAdapter(nn.Module):
	"""
	Converts the output of the other modality's encoder into something feedable
	to the text decoder).
	"""

	def __init__(self, inlatentD, outlatentD):
		super(ModalityInputAdapter, self).__init__()

		# define a linear layer to adapt the dimension of the input embedding so
		# it can be used as 'encoder last hidden state' input to the text
		# decoder model
		self.encoder_hidden_state_adaptor = nn.Linear(inlatentD, outlatentD)

	def forward(self, z):

		# ensure z can be viewed as a sequence of tokens
		if len(z.shape) == 2:
			# here z is of shape (batch_size, inlatentD), because each
			# "sequence" output by the previous module is composed of only one
			# element
			z = z.unsqueeze(1)
		
		# project to (*, hidden_state_dim)
		z = self.encoder_hidden_state_adaptor(z)

		return z