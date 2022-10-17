##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os
import random
import pickle
import json
from copy import deepcopy
from tqdm import tqdm 
import torch

import text2pose.config as config
import text2pose.utils as utils


################################################################################
## TOKENIZERS
################################################################################

import nltk


def Tokenizer(text_encoder_name):
    if text_encoder_name.split("_")[0] in ["glovebigru"]:
        return BaseTokenizer(text_encoder_name)
    else:
        raise NotImplementedError


class BaseTokenizer():
    def __init__(self, text_encoder_name):

        vocab_ref = text_encoder_name.split("_")[1]
        vocab_file = os.path.join(config.POSESCRIPT_LOCATION, config.vocab_files[vocab_ref])
        assert os.path.isfile(vocab_file), f"Vocab file not found ({vocab_file})."
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = 0
        self.max_tokens = config.MAX_TOKENS

    def __call__(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower()) # keeps punctuations
        tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        tokens = torch.tensor(tokens).long()
        return tokens

    def token2id(self, token):
        return self.vocab(token)


################################################################################
## POSESCRIPT DATASET
################################################################################

class PoseScript(object):

    def __init__(self, version="posescript-H1", split='train',
                    text_encoder_name='glovebigru_vocA1H1', caption_index='rand',
                    cache=True, generated_pose_samples_path=None):

        # NOTE: generated_pose_samples_path should be None if training on
        # original poses, otherwise it should be a path to a .pth file
        # containing the generated poses for each data point of a given split.
        # In the filepath, there must be a '{data_version}' and a '{split}'
        # fields to substitute.
        
        self.version = version
        self.split = split
        assert type(caption_index) is int or caption_index in ['deterministic-mix', 'rand']
        self.caption_index = caption_index
        self.use_generated_pose_samples = True if generated_pose_samples_path else False

        self.cache = cache
        if cache:
            cache_file = config.cache_file_path.format(data_version=version, split=split, tokenizer=text_encoder_name)
            # create the cache file if it does not exist
            if not os.path.isfile(cache_file):
                print(f'Caching data [{version} version][{text_encoder_name} tokenization][{split} split]')
                d = PoseScript(version=version, split=split, text_encoder_name=text_encoder_name, cache=False)
                self._data_cache = []
                for index in tqdm(range(len(d))):
                    pose = d.get_pose(index)
                    caption_list = d.get_all_captions(index)
                    caption_tokens_list = [d.tokenizer(caption) for caption in caption_list]
                    caption_length_list = [len(caption_tokens) for caption_tokens in caption_tokens_list]
                    # padd tokenized captions
                    caption_tokens_list = [torch.cat( (caption_tokens, d.tokenizer.pad_token_id * torch.ones( d.tokenizer.max_tokens-len(caption_tokens), dtype=caption_tokens.dtype) ), dim=0) for caption_tokens in caption_tokens_list]
                    self._data_cache.append( (pose, caption_tokens_list, caption_length_list, d.dataIDs[index]) )
                # save data
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self._data_cache, f)
                print('done')
            else:
                # load data from cache
                with open(cache_file, 'rb') as f:
                    self._data_cache = pickle.load(f)
        else:
            # Load data
            self._load_data()
            # Define tokenizer
            self.tokenizer = Tokenizer(text_encoder_name)

        # load generated pose samples
        if self.use_generated_pose_samples:
            print('Using generated poses samples.')
            self.pose_samples = torch.load(generated_pose_samples_path.format(data_version=self.version, split=self.split)) # tensor, size (dataset_size, ncaptions, nsamples, njoints, 3)

    def _load_data(self):

        self.dataIDs = utils.read_posescript_json(f"{self.split}_ids.json") # split dependent
        self.dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")
        
        # load automatic captions
        if "posescript-A" in self.version:
            self.captions = {data_id: [] for data_id in self.dataIDs}
            for caption_file in config.caption_files[self.version]:
                capts = utils.read_posescript_json(caption_file)
                for data_id in self.dataIDs:
                    self.captions[data_id].append(capts[str(data_id)])
        
        # load human-written captions
        elif "posescript-H" in self.version:
            capts = utils.read_posescript_json(config.caption_files[self.version]) # expecting only one file
            dataIDs = [] # to be updated with data actually available in PoseScript-H
            self.captions = {}
            for data_id in self.dataIDs:
                if str(data_id) in capts:
                    self.captions[data_id] = [capts[str(data_id)]]
                    dataIDs.append(data_id)
            self.dataIDs = dataIDs

    def get_pose(self, index):
        # load pose data
        pose_info = self.dataID_2_pose_info[str(self.dataIDs[index])]
        pose = utils.get_pose_data_from_file(pose_info)
        pose = pose.reshape(-1, 3) # (njoints, 3)
        return pose

    def get_generated_pose(self, index, cidx=None):
        # self.pose_samples is of size (dataset_size, ncaptions, nsamples, njoints, 3)
        if cidx is None: cidx = self.get_caption_index(self.pose_samples.shape[1], index)
        s = random.randint(0, self.pose_samples.shape[2]-1)
        return self.pose_samples[index][cidx][s] # (njoints, 3)
        
    def get_all_captions(self, index):
        data_id = self.dataIDs[index] if not self.cache else self._data_cache[index][-1]
        return self.captions[data_id]
        
    def get_caption(self, index, cidx=None):
        caption_list = self.get_all_captions(index)
        if cidx is None: cidx = self.get_caption_index(len(caption_list), index)
        return caption_list[cidx]
        
    def get_caption_index(self, n, index):
        if self.caption_index=='deterministic-mix':
            return index % n
        elif self.caption_index=='rand':
            return random.randint(0, n-1)
        elif self.caption_index < n:
            return self.caption_index
        raise ValueError
       
    def __len__(self):
        return len(self.dataIDs) if not self.cache else len(self._data_cache)
        
    def __getitem__(self, index, cidx=None):

        if self.cache: 
            pose, caption_tokens_list, caption_length_list, _ = deepcopy(self._data_cache[index])
            cidx = cidx if cidx else self.get_caption_index(len(caption_tokens_list), index)
            caption_tokens = caption_tokens_list[cidx]
            caption_lengths = caption_length_list[cidx]
        else:
            pose = None if self.use_generated_pose_samples else self.get_pose(index)
            caption = self.get_caption(index, cidx=cidx)
            caption_tokens = self.tokenizer(caption)
            caption_lengths = len(caption_tokens)

        if self.use_generated_pose_samples:
            # overwrite variable content
            pose = self.get_generated_pose(index) # let cidx at None, to still select among generated samples for every automatic caption when training on human-written captions

        item = dict(pose=pose, caption_tokens=caption_tokens, caption_lengths=caption_lengths)
        return item