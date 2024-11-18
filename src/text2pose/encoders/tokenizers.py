##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
# import nltk # moved to a place where it is required for sure
import torch
import random
from copy import deepcopy
from transformers import AutoTokenizer, CLIPTokenizer

import text2pose.config as config
from text2pose.vocab import load_vocab_from_ref, side_flipping_correspondances, word_list_for_flipping_correspondances


################################################################################
## L/R FLIPPING RULES (mirroring augmentation)
################################################################################

# NOTE: one could probably get away with all this R/L flipping rules by
# preprocessing the texts offline, determining the grammatical status of each
# word and defining rules based on that; but if we want to do R/L flip online,
# we need this.

# NOTE: these rules exist because we can't just flip all "right" occurences to
# substring "left". For instance:
#   'the hand is right beside the knee',
#   'the torso is upright',
#   'the knee is bent at a right angle' ...
# Potentially, if there is a spelling mistake on the word "lift" (then changed
# to "left"), flipping "left" to "right" could be wrong. However, we will assume
# that there is a very small amount of such mistakes and thus will not directly
# tackle them.

# NOTE: left-right flipping correspondances are defined in file vocab.py.

words_blocking_sentence_flip_wordsVersion = ["o'clock", "clock", "x-axis"] # potentially complicated (but also rare) cases
# ==> do NOT flip 'right' to 'left' when the annotation contains one of those words

determiners_wordsVersion = ["your", "their", "his", "her", "the", "a", "an", "this", "these", "those"] # do not include "them"/"him" ("her" is super rare, so hopefully it being already in the list as "his" female counterpart is fine)
need_determiner_for_local_R2L_flip_wordsVersion = ['about', 'before', 'after', 'off', 'between', 'behind', 'in', 'under', 'next', 'below', 'above', 'across', 'away', 'near', 'by', 'at']
# ==> do NOT flip 'right' to 'left' when followed by one of those words, UNLESS
# the previous word is a determiner
# ie. [determiner, "right", word] ==> flip
# 	  [anything else than a determiner, "right", word] ==> do not flip
# (found only 1 case that did not respect the rule in PoseScript-H2 (due to bad grammar))

previous_word_preventing_local_R2L_flip_wordsVersion = ['is', 'are', "it's", "isn't", "aren't", "not", "be"]
# ==> do NOT flip IF previous word is one of those words
# Did not find any cases where 'right' would need to be flipped when preceded by
# one of those words (found 3 problematic cases in PoseScript-H2)
# In such cases, 'right' is often followed by one of the adverbs listed previously.

next_word_preventing_local_R2L_flip_wordsVersion = ['angle', 'angles']
# ==> do NOT flip 'right' to 'left' when followed by one word of this list
# Other word candidates were: ["spot", "position", "way"]
# ... meant for cases where 'right' is used in ways that are difficult to easily
# distinguish automatically; however, this would lead to mistakes, one way or
# the other. All in all, it seems better to process those as regular cases (in
# particular, "position" is used very often without any "right" word attached,
# so it could be unfair to make a text unflippable because of that (as in
# `words_blocking_sentence_flip`)); basically, these cases, all together, are
# very rare (<10), and tend in majority to require a R2L flip ==> allow that
# flip and accept to have a few mistakes

# NOTE: potential special cases x=('one', 'to'): did not find any cases in the
# annotations, where flipping "right" to "left" before "{x}" would be
# problematic AND not already tackled by another rule
# NOTE: 'to' is a hard one; most of the time, the rule of the determiner works, 
# but there are often sentences like 'Turn to the right' ...


################################################################################
## DECODING POLISHING RULES
################################################################################

decoding_polishing_rules_rm_spaces = ["counter - clockwise", "anti - clockwise", "mid - ", " - width", "upside - down", "sit - up", "push - up", "criss - cross", "hip - hop", "cross - legged", "semi - ", "tippy - ", "tip - ", " - height", " - shape", " - hand", "in - between", "ready - to - ", " - wise"]


################################################################################
## TOKENIZERS
################################################################################

def get_tokenizer_name(text_encoder_or_decoder_name):
	ref = text_encoder_or_decoder_name.split("_")
	if len(ref) == 2: return ref[1] # vocab reference
	return ref[0]


def get_text_encoder_or_decoder_module_name(text_encoder_or_decoder_name):
	ref = text_encoder_or_decoder_name.split("_")
	if len(ref) == 2: return ref[0]
	return ref[0]


def Tokenizer(text_encoder_or_decoder_name):
	tokenizer_name = get_tokenizer_name(text_encoder_or_decoder_name)
	if tokenizer_name in ["distilbertUncased", "clip"]:
		return TransformTokenizer(tokenizer_name)
	elif "voc" in tokenizer_name:
		import nltk # only needed if using this kind of tokenizer
		return BaseTokenizer(tokenizer_name)
	else:
		raise NotImplementedError


class GenericTokenizer():
	def __init__(self):
		super(GenericTokenizer, self).__init__()
		
		# must be overriden with the correct token ids
		
		self.pad_token_id = None # int
		self.bos_token_id = None # int
		self.eos_token_id = None # int
		self.unk_token_id = None # int
		self.max_tokens = None # int
		
		self.sfc_dict = None # dict
		self.sfc_list = None # list
		self.token_id_for_right = None # int
		self.words_blocking_sentence_flip = None # list
		self.determiners = None # list
		self.need_determiner_for_local_R2L_flip = None # list
		self.previous_word_preventing_local_R2L_flip = None # list
		self.next_word_preventing_local_R2L_flip = None # list


	def __call__(self, text):
		raise NotImplementedError


	def __len__(self):
		raise NotImplementedError


	def token2ids(self, token):
		"""
		Args: str
		Return: int or list of int, depending on the case
		"""
		raise NotImplementedError


	def decode(self, token_ids):
		raise NotImplementedError


	def batch_decode(self, token_ids):
		raise NotImplementedError
	

	def polish_decode(self, text):
		case_replace = [x.replace(' ', '') for x in decoding_polishing_rules_rm_spaces]
		def process(t):
			for k, case in enumerate(decoding_polishing_rules_rm_spaces):
				t = t.replace(case, case_replace[k])
			return t.strip()
		if type(text) is str: return process(text)
		if type(text) is list:
			for i in range(len(text)):
				text[i] = process(text[i])
		return text

	
	def get_actual_length(self, t):
		"""
		Get the actual length of each element of t.
		"""
		lengths = t.shape[1] - (t == self.pad_token_id).sum(dim=1)
		if self.pad_token_id == self.eos_token_id:
			lengths += 1 # must keep one slot for the EOS token
		if self.pad_token_id == self.bos_token_id:
			lengths += 1 # must keep one slot for the BOS token
		return lengths


	def assemble_tokenized_texts(self, caption_tokens_list):
		"""
		Create a minimal (in terms of padding tokens) tensor containing the 
		token IDs for each of the provided texts.

		Args: list of batch_size tensors of varying size containing token IDs
		Return:
		    torch.tensor or size (batch_size, max_length) completed with a
				minimum number of padding tokens
			torch.tensor of size (batch_size) giving the actual length of each
				text
		"""
		device = caption_tokens_list[0].device
		# normalize with padding tokens
		caption_tokens_list = [torch.cat( (caption_tokens, self.pad_token_id * torch.ones( self.max_tokens-len(caption_tokens), dtype=caption_tokens.dtype, device=device) ), dim=0) for caption_tokens in caption_tokens_list]
		caption_tokens = torch.stack(caption_tokens_list)
		# get actual length
		caption_lengths = self.get_actual_length(caption_tokens).to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate
		return caption_tokens, caption_lengths


	def assemble_raw_texts(self, text_list):
		caption_tokens_list = [self.__call__(c) for c in text_list]
		return self.assemble_tokenized_texts(caption_tokens_list)


	def is_safely_flippable(self, t):
		"""
		t: torch.tensor of size (batch_size, length) containing token IDs
		"""
		# a text is flippable if it contains none of the tokens from self.words_blocking_sentence_flip
		return ~ sum(t == b for b in self.words_blocking_sentence_flip).any(1)


	def flip(self, t, flippable=None):
		"""
		Args:
		t: torch.tensor of size (batch_size, length) containing token IDs
		flippable: torch.tensor of size (batch_size) of bool type indicating
			which elements should be L/R-flipped; use None to have all flippable
		    elements flipped (determined flippable on account of
		    self.words_blocking_sentence_flip)
		
		Returns:
		t: (same but L/R flipped)
		t_len: torch.tensor giving the true length (ie. w/o padding) of each
			text after flipping
		flippable: bool tensor indicated which element has been eventually
			flipped; this is an updated version of the provided input, that
			accounts for safely flippable elements
		"""

		batch_size, length = t.shape
		device = t.device
		
		# only flip elements that are required AND that can be safely flipped
		if flippable is None:
			flippable = torch.ones(batch_size).bool()
		flippable = torch.logical_and(flippable.to(device), self.is_safely_flippable(t))

		# initialize conversion masks: compute all changing masks on the original
		# text, and apply them at the end (prevent changing a word twice)
		sfc_mask = torch.zeros(batch_size, length, len(self.sfc_list)).bool().to(device)
		for i, sfc in enumerate(self.sfc_list): # process all the side referencing words

			# 'right' leads to some special cases, as it does not always refer to
			# the side 'right', treat it differently
			if sfc == self.token_id_for_right:
				# --- CASE 1
				# find `right` words
				# first assume that they are all refencing to the side
				# ==> allow R/L flip in all cases
				test1 = t==sfc

				# --- CASE 2
				# if 'right' is followed by one of `need_determiner_for_local_R2L_flip`,
				# one needs to check that 'right' is preceded by one of the `determiners`
				# to allow the R/L flip: allow flip for all elements but those that
				# do not meet this constraint
				test2_a = t[:,:-1]==sfc # position of the word `right`
				test2_b = sum(t[:,1:]==k for k in self.need_determiner_for_local_R2L_flip).bool() # position of the studied following word
				test2_c = sum(t[:,:-2]==k for k in self.determiners).bool() # position of the determiner
				# check that all the stars are aligned
				test2 = torch.logical_and(test2_a, test2_b)
				test2 = torch.logical_or(~test2[:,1:], test2_c) # relation (a,b) ==> c
				# give the mask its original shape back
				test2 = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=device),
									test2,
									torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)

				# --- CASE 3
				# if 'right' is preceded by one of the `previous_word_preventing_local_R2L_flip`
				# do not allow the R/L flip ; in such cases, 'right' is often
				# followed by one of the adverbs from CASE 1, or refers to the
				# meaning of 'correct': allow flip for all elements but those that
				# do not meet this constraint
				test3_a = t[:,1:]==sfc # position of the word `right`
				test3_b = sum(t[:,:-1]==k for k in self.previous_word_preventing_local_R2L_flip).bool() # position of the studied previous word
				# check that all the stars are aligned
				test3 = ~torch.logical_and(test3_a, test3_b)
				# give the mask its original shape back
				test3 = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=device),
									test3], dim=1)
				
				# --- CASE 4
				# if 'right' is followed by one of the `next_word_preventing_local_R2L_flip`
				# do not allow the R/L flip ; an example of such cases is the 
				# famous 'right angle': allow flip for all eelemnts but thos that
				# do not meet  this constraint
				test4_a = t[:,:-1]==sfc # position of the word `right`
				test4_b = sum(t[:,1:]==k for k in self.next_word_preventing_local_R2L_flip).bool() # position of the studied following word
				# check that all the stars are aligned
				test4 = ~torch.logical_and(test4_a, test4_b)
				# give the mask its original shape back
				test4 = torch.cat([test4,
					   				torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1)

				# --- COMBINE RESULTS
				sfc_mask[:,:,i] = torch.logical_and(torch.logical_and(torch.logical_and(test4, test3), test2), test1)

			# regular cases
			else:
				sfc_mask[:,:,i] = t==sfc

		# apply changes
		for i, sfc in enumerate(self.sfc_list):
			changed_token_id = self.sfc_dict[sfc]
			try:
				t[flippable] = torch.where(sfc_mask[:,:,i], changed_token_id, t)[flippable] # replace tokens for flippable elements only
			except TypeError:
				# type(changed_token_id) is list:
				changed_token_id = random.sample(changed_token_id, k=1)[0]
				t[flippable] = torch.where(sfc_mask[:,:,i], changed_token_id, t)[flippable] # replace tokens for flippable elements only

		return t, self.get_actual_length(t), flippable
		

class BaseTokenizer(GenericTokenizer):
	def __init__(self, tokenizer_name):
		super(BaseTokenizer, self).__init__()

		vocab_ref = tokenizer_name
		self.vocab = load_vocab_from_ref(vocab_ref)
		
		self.pad_token_id = self.vocab('<pad>')
		self.bos_token_id = self.vocab('<start>')
		self.eos_token_id = self.vocab('<end>')
		self.unk_token_id = self.vocab('<unk>')
		self.max_tokens = config.MAX_TOKENS
		# the following makes it possible to perform online L/R flipping
		self.prepare_for_meta_data_for_LRflipping()

	def __call__(self, text):
		tokens = nltk.tokenize.word_tokenize(text.lower()) # keeps punctuations
		tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
		tokens = torch.tensor(tokens).long()
		return tokens
	
	def __len__(self):
		return len(self.vocab)

	def prepare_for_meta_data_for_LRflipping(self):
		# NOTE: assumes that all the side flipping correspondance words are in
		# the vocabulary
		# convert words from side_flipping_correspondances into token ids
		process_w = lambda w: self.token2ids(w) if type(w) is str else [self.token2ids(ww) for ww in w]
		self.sfc_dict = {process_w(k):process_w(v) for k,v in side_flipping_correspondances.items()}
		# consider both ways for conversion (L --> R and R --> L)
		sfc_dict_reverse = {v:k for k,v in self.sfc_dict.items() if type(v) is int}
		sfc_dict_reverse.update({v[i]:k for k,v in self.sfc_dict.items() if type(v) is list for i in range(len(v))})
		self.sfc_dict.update(sfc_dict_reverse)
		# flatten all the elements from self.sfc_dict into a single list
		self.sfc_list = list(self.sfc_dict.keys())
		# convert other lists of words to token ids
		self.token_id_for_right = self.token2ids("right")
		self.words_blocking_sentence_flip = torch.tensor([self.token2ids(w) for w in words_blocking_sentence_flip_wordsVersion])
		self.determiners = [self.token2ids(w) for w in determiners_wordsVersion]
		self.need_determiner_for_local_R2L_flip = [self.token2ids(w) for w in need_determiner_for_local_R2L_flip_wordsVersion]
		self.previous_word_preventing_local_R2L_flip = [self.token2ids(w) for w in previous_word_preventing_local_R2L_flip_wordsVersion]
		self.next_word_preventing_local_R2L_flip = [self.token2ids(w) for w in next_word_preventing_local_R2L_flip_wordsVersion]

	def token2ids(self, token):
		"""
		Args: str
		Return: int
		"""
		return self.vocab(token) # always one token id

	def decode(self, token_ids):
		"""token_ids: list containing words IDs, for one text"""
		# convert token ids into words
		token_ids = token_ids.tolist()
		t = " ".join([self.vocab.idx2word[tid] for tid in token_ids])
		# remove special tokens
		for special_token in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
			t = t.replace(self.vocab.idx2word[special_token], "")
		t = t.strip()
		t = t.replace(" ,", ",").replace(" .", ".") # remove spaces before punctuation
		t = ". ".join([p[0].upper() + p[1:] for p in t.split(". ")]) # use capital letters at the beginning of sentences
		return t # NOTE: self.polish_decode would be applied here if necessary

	def batch_decode(self, token_ids):
		"""token_ids: list of list contraining words IDs"""
		return [self.decode(t) for t in token_ids]


class TransformTokenizer(GenericTokenizer):
	def __init__(self, tokenizer_name):
		super(TransformTokenizer, self).__init__()

		self.tokenizer_name = tokenizer_name
		self.cased_tokenizer = False
		if tokenizer_name == "distilbertUncased":
			self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.TRANSFORMER_CACHE_DIR, "distilbert-base-uncased"))
		elif tokenizer_name == "gpt2":
			self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.TRANSFORMER_CACHE_DIR, "gpt2"), add_prefix_space=True) # be sure that the same word is always encoded the same way, no matter if it is at the beginning of the sentence or not (makes things simpler for L/R augmentation)
			self.cased_tokenizer = True
		elif tokenizer_name == "clip":
			self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(config.TRANSFORMER_CACHE_DIR, "openai/clip-vit-base-patch32"))
		else:
			raise NotImplementedError

		# define required token ids
		if tokenizer_name == "gpt2":
			print("# NOTE: GPT2 does not have a padding token " +
				"(this is not a problem as we use the attention mask to inform about padding later on). " +
				"Using the EOS token id as substitution for the padding token id, to stay coherent with the rest of the code.")
				# NOTE: cannot use the vocabulary size instead of the EOS token
				# id: this triggers an error because the data is out of range of
				# nn.Embedding (see
				# https://github.com/pytorch/pytorch/issues/72696)
			self.pad_token_id = self.tokenizer.eos_token_id 
		else:
			self.pad_token_id = self.tokenizer.pad_token_id
		self.bos_token_id = 101 if tokenizer_name in ["distilbertUncased"] else self.tokenizer.bos_token_id
		self.eos_token_id = 102 if tokenizer_name in ["distilbertUncased"] else self.tokenizer.eos_token_id
		self.unk_token_id = self.tokenizer.unk_token_id
		self.max_tokens = self.tokenizer.model_max_length

		# the following makes it possible to perform online L/R flipping
		self.prepare_for_meta_data_for_LRflipping()

	def __call__(self, text):
		x = self.tokenizer(text, truncation=True, return_tensors="pt")["input_ids"][0]
		if self.tokenizer_name in ["gpt2"]:
			# add BOS & EOS tokens, which are not added automatically
			x = torch.cat((torch.tensor([self.bos_token_id]), x, torch.tensor([self.eos_token_id])))
		return x
	
	def __len__(self):
		return len(self.tokenizer)

	def prepare_for_meta_data_for_LRflipping(self):
		# NOTE: as pretrained tokenizers tend to tokenize some of the side
		# flipping correspondance words into several tokens, we will first need
		# to convert them into a unique temporary token, to run the flipping
		# algorithm, then convert them back to the actual tokenizer ids
		# (see the self.flip method)

		# deepcopy global variables to avoid any problems due to potential
		# modification (when eg. considering capitalized words)
		global word_list_for_flipping_correspondances, words_blocking_sentence_flip_wordsVersion, determiners_wordsVersion, need_determiner_for_local_R2L_flip_wordsVersion, previous_word_preventing_local_R2L_flip_wordsVersion, next_word_preventing_local_R2L_flip_wordsVersion, side_flipping_correspondances
		side_flipping_correspondances_loc = deepcopy(side_flipping_correspondances)
		word_list_for_flipping_correspondances_loc = deepcopy(word_list_for_flipping_correspondances)
		words_blocking_sentence_flip_wordsVersion_loc = deepcopy(words_blocking_sentence_flip_wordsVersion)
		determiners_wordsVersion_loc = deepcopy(determiners_wordsVersion)
		need_determiner_for_local_R2L_flip_wordsVersion_loc = deepcopy(need_determiner_for_local_R2L_flip_wordsVersion)
		previous_word_preventing_local_R2L_flip_wordsVersion_loc = deepcopy(previous_word_preventing_local_R2L_flip_wordsVersion)
		next_word_preventing_local_R2L_flip_wordsVersion_loc = deepcopy(next_word_preventing_local_R2L_flip_wordsVersion)

		# Add capitalized word versions in meta-data if necessary
		if self.cased_tokenizer:
			capitalize_w = lambda w: w.capitalize() if type(w) is str else [ww.capitalize() for ww in w]
			side_flipping_correspondances_loc.update({capitalize_w(k):capitalize_w(v) for k,v in side_flipping_correspondances_loc.items()})
			# do not consider need_determiner_for_local_R2L_flip_wordsVersion_loc as those are supposed to be at least in 3rd position (not at the beginning of the sentence ==> not capitalized)
			word_list_for_flipping_correspondances_loc += capitalize_w(word_list_for_flipping_correspondances_loc)
			words_blocking_sentence_flip_wordsVersion_loc += capitalize_w(words_blocking_sentence_flip_wordsVersion_loc)
			determiners_wordsVersion_loc += capitalize_w(determiners_wordsVersion_loc)
			previous_word_preventing_local_R2L_flip_wordsVersion_loc += capitalize_w(previous_word_preventing_local_R2L_flip_wordsVersion_loc)
			next_word_preventing_local_R2L_flip_wordsVersion_loc += capitalize_w(next_word_preventing_local_R2L_flip_wordsVersion_loc)

		# Build meta data to perform many-to-one/one-to-many token ids
		# conversion (from actual tokenizer ids to temporary new "flip" ids that
		# will be used only to perform L/R flipping)
		offset = len(self.tokenizer) # size of the full vocabulary; offset to define new ids (the flip ids)
		words_of_interest = word_list_for_flipping_correspondances_loc \
							+ words_blocking_sentence_flip_wordsVersion_loc \
							+ determiners_wordsVersion_loc \
							+ need_determiner_for_local_R2L_flip_wordsVersion_loc \
							+ previous_word_preventing_local_R2L_flip_wordsVersion_loc \
							+ next_word_preventing_local_R2L_flip_wordsVersion_loc
		self.token_word_2_flipid = {} # will replace self.token2ids in the regular case (cf. BaseTokenizer)
		self.flipid_2_token_ids = {} # converts regular tokenizer (list of) ids to flip ids
		for w in words_of_interest:
			self.token_word_2_flipid[w] = offset
			self.flipid_2_token_ids[offset] = tuple(self.token2ids(w))
			assert self.unk_token_id not in self.flipid_2_token_ids[offset], "One of the words necessary to perform proper L/R flip augmentation is unknwown to this tokenizer. If the 'unknown' token is used in its place, weird behavior to be expected, if not an error. Modify the code to treat such cases."
			offset += 1
		# because of potential overlaps between token id sequences, we need to
		# process the larger sequences first
		# for instance, in the case 'anti-clockwise'='anti'+'-'+'clockwise', if
		# 'clockwise' is processed before 'anti-clockwise', L/R flipping would
		# yield: 'anti-anti-clockwise' 
		self.flipid_2_token_ids = dict(sorted(self.flipid_2_token_ids.items(), key=lambda item: -len(item[1])))

		# Now, convert utilitary words for L/R flipping rules to (singular
		# temporary) ids
		# convert words from side_flipping_correspondances_loc into token (flip)ids
		process_w = lambda w: self.token_word_2_flipid[w] if type(w) is str else [self.token_word_2_flipid[ww] for ww in w]
		self.sfc_dict = {process_w(k):process_w(v) for k,v in side_flipping_correspondances_loc.items()}
		# consider both ways for conversion (L --> R and R --> L)
		sfc_dict_reverse = {v:k for k,v in self.sfc_dict.items() if type(v) is int}
		sfc_dict_reverse.update({v[i]:k for k,v in self.sfc_dict.items() if type(v) is list for i in range(len(v))})
		self.sfc_dict.update(sfc_dict_reverse)
		# flatten all the elements from self.sfc_dict into a single list
		self.sfc_list = list(self.sfc_dict.keys())
		# convert other lists of words to token ids
		self.token_id_for_right = self.token_word_2_flipid["right"] # even in the cased version, because special flipping rules apply on 'right' as a middle sentence word
		self.words_blocking_sentence_flip = torch.tensor([self.token_word_2_flipid[w] for w in words_blocking_sentence_flip_wordsVersion_loc])
		self.determiners = [self.token_word_2_flipid[w] for w in determiners_wordsVersion_loc]
		self.need_determiner_for_local_R2L_flip = [self.token_word_2_flipid[w] for w in need_determiner_for_local_R2L_flip_wordsVersion_loc]
		self.previous_word_preventing_local_R2L_flip = [self.token_word_2_flipid[w] for w in previous_word_preventing_local_R2L_flip_wordsVersion_loc]
		self.next_word_preventing_local_R2L_flip = [self.token_word_2_flipid[w] for w in next_word_preventing_local_R2L_flip_wordsVersion_loc]

	def token2ids(self, token):
		"""
		Args: str
		Return: list of int
		"""
		# NOTE: self.tokenizer.convert_tokens_to_ids(token) assumes the token
		# will give only one token id; but sometimes, it should be tokenized
		# into several token ids; for instance, "counter-clockwise"
		# would be 3 known token ids, but processed by convert_tokens_to_ids(.),
		# it would become 1 token id (corresponding to the token "unknown").
		# ==> rather use __call__
		return self.__call__(token)[1:-1].tolist() # remove BOS/EOS

	def decode(self, token_ids, skip_special_tokens=True):
		t = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens) # convert token ids into words, remove BOS & ENS tokens
		if not self.cased_tokenizer:
			return self.polish_decode(". ".join([p[0].upper() + p[1:] for p in t.split(". ")])) # use capital letters at the beginning of sentences
		return self.polish_decode(t)

	def batch_decode(self, token_ids):
		t_list = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
		if not self.cased_tokenizer:
			return self.polish_decode([". ".join([p[0].upper() + p[1:] for p in t.split(". ")]) for t in t_list]) # use capital letters at the beginning of sentences
		return self.polish_decode(t_list)

	def many_to_one_flip_ids_conversion(self, t):
		batch_size, length = t.shape
		device = t.device
		for flipid, token_ids in self.flipid_2_token_ids.items():
			new_t = []
			# convert tuple to tensor
			token_ids = torch.tensor(token_ids).to(device)
			size = len(token_ids)
			# pass a sliding window on the tensor to search, and get True in positions where the tuple was found
			x = (t.unfold(1,size,1) == token_ids).all(2) # NOTE: unfold(dimension along which to run the sliding window, size of the window, step between each slice)
			x = x.nonzero(as_tuple=True) # get indices where the subsequence were found: row_indices, column_indices
			
			# let's look at the batch elements one by one
			for i in range(batch_size):
				# get indices (along dim = 1) where the subsequence was found
				inds = x[1][x[0] == i]
				# slice the sequence
				slices = torch.zeros(2*len(inds), device=device).long() # number of different spans to slice (-1, as the last one will be computed automatically)
				slices[::2] = inds # span of the original sequence
				slices[1::2] = inds + size # span with the found subsequence
				spans = torch.hsplit(t[i], slices.tolist())
				# form the new sequence
				new_spans = [None for _ in range(len(spans))]
				new_spans[::2] = spans[::2] # keep the parts of the original subsequence
				new_spans[1::2] = [torch.tensor([flipid], device=device) for _ in range(len(inds))] # replace the subsequence of token ids with the new flip id
				new_t.append(torch.cat(new_spans).to(device))
			
			# normalize to iterate over the next pair of flipid & token ids subsquence
			t, t_len = self.assemble_tokenized_texts(new_t)

		return t, t_len

	def one_to_many_flip_ids_conversion(self, t):
		batch_size = t.shape[0]
		device = t.device
		# process all the flip ids at once, as there is only one value involved
		flipids_all_tensor = torch.tensor(list(self.flipid_2_token_ids.keys())).view(-1,1).to(device)

		# let's look at the batch elements one by one
		new_t = []
		for i in range(batch_size):
			# find the indices of the flip ids
			flipid_loc = t[i].view(1,-1) == flipids_all_tensor # row j gives the positions of flip id j in t[i]
			inds = flipid_loc.any(0).nonzero().view(-1)
			# slice the sequence
			slices = torch.zeros(2*len(inds), device=device).long() # number of different spans to slice (-1, as the last one will be computed automatically)
			slices[::2] = inds # span of the original sequence
			slices[1::2] = inds + 1 # span with the found flip id
			spans = torch.hsplit(t[i], slices.tolist())
			# form the new sequence
			new_spans = [None for _ in range(len(spans))]
			new_spans[::2] = spans[::2] # keep the parts of the original subsequence
			# get the found flip ids
			flipids = [flipids_all_tensor[flipid_loc[:,ind]].item() for ind in inds]
			new_spans[1::2] = [torch.tensor(self.flipid_2_token_ids[flipid], device=device) for flipid in flipids] # replace the flip id with the subsequence of token ids
			new_t.append(torch.cat(new_spans).to(device))

		# normalize
		t, t_len = self.assemble_tokenized_texts(new_t)
		
		return t, t_len

	def flip(self, t, flippable=None):
		# 1) many-to-one token id conversion
		t, _ = self.many_to_one_flip_ids_conversion(t)
		# 2) perform flipping on unique token ids
		t, _, flipped = super(TransformTokenizer, self).flip(t, flippable)
		# 3) one-to-many token id reverse conversion
		t, t_len = self.one_to_many_flip_ids_conversion(t) # due to word-piece tokenization, the length may have changed after flipping
		return t, t_len, flipped