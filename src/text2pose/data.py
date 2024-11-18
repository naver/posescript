##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import glob
import random
import pickle
from copy import deepcopy
from tqdm import tqdm
import torch

import text2pose.config as config
import text2pose.utils as utils
from text2pose.encoders.tokenizers import Tokenizer


################################################################################
## HELPERS
################################################################################

# function to convert dataset-specific data IDs into global data IDs
globalize_data_ID = lambda d, prefix: [f'{prefix}_{l}' for l in d] if type(d)==list else {f'{prefix}_{k}':v for k,v in d.items()}

posescript_prefix = "PS"
posefix_prefix = "PF"

# empty pose
T_POSE = torch.zeros(config.NB_INPUT_JOINTS, 3)
T_POSE[:1] = torch.tensor([1, 0, 0]) * torch.pi/2 # normalized rotation
PID_NAN = config.PID_NAN


################################################################################
## DATASET STRUCTURE
################################################################################

class GenericDataset():
    def __init__(self, version, split, tokenizer_name, caption_index, num_body_joints=config.NB_INPUT_JOINTS, cache=True, data_size=None):
        super(GenericDataset, self).__init__()

        self.tokenizer_name = tokenizer_name
        self.version = version
        self.split = split
        assert type(caption_index) is int or caption_index in ['deterministic-mix', 'rand']
        self.caption_index = caption_index
        self.num_body_joints = num_body_joints
        self.cache = cache
        self.data_size = data_size
        if self.data_size:
            print(f'Considering only the first {self.data_size} loaded items (before dataloader shuffling).')

        if cache:
            cache_file = config.cache_file_path[self.__class__.__name__.lower()].format(data_version=version, split=split, tokenizer=tokenizer_name)
            print("Cache file:", cache_file)
            # in the case where tokenizer_name is None, try loading an
            # existing cache, based on another tokenizer_name
            if tokenizer_name is None:
                candidate_cache_file = config.cache_file_path[self.__class__.__name__.lower()].format(data_version=version, split=split, tokenizer="*")
                tmp = glob.glob(candidate_cache_file)
                if len(tmp):
                    cache_file = tmp[0]
                    print(f"Loading dataset with tokenizer_name=None; used the following saved data cache: {cache_file}")
            # create the cache file if it does not exist
            if not os.path.isfile(cache_file):
                print(f'Caching data [{version} version][{tokenizer_name} tokenization][{split} split]')
                d = self.__class__(version=version, split=split, tokenizer_name=tokenizer_name, cache=False)
                self._data_cache = []
                for index in tqdm(range(len(d))):
                    self._data_cache.append( d._element_to_cache(index) )
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
            self.init_tokenizer()
    
    def _load_data(self):
        raise NotADirectoryError

    def _element_to_cache(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.data_size: return self.data_size
        return len(self.dataIDs) if not self.cache else len(self._data_cache)

    def init_tokenizer(self):
        if not hasattr(self, "tokenizer") and self.tokenizer_name is not None:
            self.tokenizer = Tokenizer(self.tokenizer_name)

    def get_all_captions(self, index):
        raise NotImplementedError

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
    
    def get_minimum_number_of_caption(self):
        """
        Return the minimum number of texts per item.
        """
        if self.cache:
            numbers = [len(self._data_cache[index][1]) for index in range(self.__len__())]
        else:
            numbers = [len(self.get_all_captions(index)) for index in range(self.__len__())]
        return min(numbers)


################################################################################
## POSE-ONLY DATASET
################################################################################

class PoseStream(GenericDataset): # description-based dataset

    def __init__(self, version="100k", split='train', tokenizer_name=None, caption_index=0, num_body_joints=config.NB_INPUT_JOINTS, cache=True, data_size=None): # must keep some arguments due to class inheritance
        super(PoseStream, self).__init__(version=version, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=num_body_joints, cache=cache, data_size=data_size)
        assert self.version == "100k", f"PoseStream not implemented for version: {self.version}"


    def _load_data(self):
        # load data info
        self.dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
        # get split ids
        self.dataIDs = utils.read_json(config.file_posescript_split % self.split)


    def _element_to_cache(self, index):
        pose = self.get_pose(index)
        return (pose, self.dataIDs[index])
    

    def get_pose(self, index):
        # load pose data
        pose_info = self.dataID_2_pose_info[str(self.dataIDs[index])]
        pose = utils.get_pose_data_from_file(pose_info)
        pose = pose.reshape(-1, 3) # (njoints, 3)
        return pose
        

    def __getitem__(self, index, cidx=None):

        if self.cache: 
            pose, data_ids = deepcopy(self._data_cache[index])
        else:
            pose = self.get_pose(index)
            data_ids = self.dataIDs[index]
            
        item = dict(pose=pose[:self.num_body_joints], data_ids=data_ids, indices=index)
        return item


################################################################################
## POSESCRIPT DATASET
################################################################################

def load_posescript(caption_files, split):

    # initialize split ids & caption data
    dataIDs = utils.read_json(config.file_posescript_split % split) # split dependent
    captions = {data_id: [] for data_id in dataIDs}

    # load available caption data
    for caption_file in caption_files:
        capts = utils.read_json(caption_file)
        for data_id_str, c in capts.items():
            try:
                captions[int(data_id_str)] += c if type(c) is list else [c]
            except KeyError:
                # this caption is not part of the split
                pass

    # clean dataIDs and captions: remove access to unavailable
    # data (when no caption is provided); necessary step if a smaller data
    # set is loaded (eg. PoseScript-H is smaller than PoseScript-A)
    dataIDs = [data_id for data_id in dataIDs if len(captions[data_id])]
    captions = {data_id:captions[data_id] for data_id in dataIDs}

    duplicates = 0
    for t in captions.values():
        duplicates += 1 if len(t) > 1 else 0
    print(f"[PoseScript] Loaded {len(dataIDs)} poses in {split} split (found {duplicates} with more than 1 annotation).")

    return dataIDs, captions


def get_all_posescript_descriptions(caption_files):
    """
    Load all descriptions in the provided caption files, independently of the
    split, and format data to have correspondances between pose IDs and a list 
    of descriptions.
    """
    captions = {}
    for caption_file in caption_files:
        capts = utils.read_json(caption_file)
        for data_id_str, c in capts.items():
            data_id = int(data_id_str)
            captions[data_id] = captions.get(data_id, []) + (c if type(c) is list else [c])
    return captions


class PoseScript(GenericDataset): # description-based dataset

    def __init__(self, version="posescript-H2", split='train',
                    tokenizer_name='vocPSA2H2', caption_index='rand',
                    num_body_joints=config.NB_INPUT_JOINTS,
                    cache=True, data_size=None, generated_pose_samples_path=None, posefix_format=False):
        super(PoseScript, self).__init__(version=version, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=num_body_joints, cache=cache, data_size=data_size)

        # NOTE: generated_pose_samples_path should be None if training on
        # original poses, otherwise it should be a path to a .pth file
        # containing the generated poses for each data point of a given split.
        # In the filepath, there must be a '{data_version}' and a '{split}'
        # fields to substitute.
        self.use_generated_pose_samples = True if generated_pose_samples_path else False

        # load generated pose samples
        if self.use_generated_pose_samples:
            print('Using generated poses samples.')
            self.pose_samples = torch.load(generated_pose_samples_path.format(data_version=self.version, split=self.split)) # tensor, size (dataset_size, ncaptions, nsamples, njoints, 3)

        # will update item format if using the posefix format
        self.posefix_format = posefix_format


    def _load_data(self):
        # load data info
        self.dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
        # get split ids & caption data
        self.dataIDs, self.captions = load_posescript(config.caption_files[self.version][1], self.split)


    def _element_to_cache(self, index):
        pose = self.get_pose(index)
        if self.tokenizer_name:
            caption_list = self.get_all_captions(index)
            caption_tokens_list = [self.tokenizer(caption) for caption in caption_list]
            caption_length_list = [len(caption_tokens) for caption_tokens in caption_tokens_list]
            # padd tokenized captions
            caption_tokens_list = [torch.cat( (caption_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-len(caption_tokens), dtype=caption_tokens.dtype) ), dim=0) for caption_tokens in caption_tokens_list]
            return (pose, caption_tokens_list, caption_length_list, self.dataIDs[index])
        else:
            return (pose, None, None, self.dataIDs[index])
    

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
        return self.captions[self.dataIDs[index]]
        

    def __getitem__(self, index, cidx=None):

        if self.cache: 
            pose, caption_tokens_list, caption_length_list, data_ids = deepcopy(self._data_cache[index])
            cidx = cidx if cidx else self.get_caption_index(len(caption_tokens_list), index)
            caption_tokens = caption_tokens_list[cidx]
            caption_lengths = caption_length_list[cidx]
        else:
            pose = None if self.use_generated_pose_samples else self.get_pose(index)
            data_ids = self.dataIDs[index]
            captions = self.get_caption(index, cidx=cidx)
            if self.tokenizer_name is None: # will yield the raw texts
                caption_tokens = captions
                caption_lengths = 0
            else:
                caption_tokens = self.tokenizer(captions)
                caption_lengths = len(caption_tokens)
                # padding
                caption_tokens = torch.cat( (caption_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-caption_lengths, dtype=caption_tokens.dtype) ), dim=0)

        if self.use_generated_pose_samples:
            # overwrite variable content
            pose = self.get_generated_pose(index) # let cidx at None, to still select among generated samples for every automatic caption when training on human-written captions

        item = dict(pose=pose[:self.num_body_joints], caption_tokens=caption_tokens, caption_lengths=caption_lengths, data_ids=data_ids, indices=index)

        if self.posefix_format:
            # adapt item to PoseFix format: add pose A, update indices...
            poses_A, pidA = T_POSE[:self.num_body_joints], PID_NAN
            item.update(dict(poses_A=poses_A, poses_B=item.pop("pose"), poses_A_ids=pidA, poses_B_ids=item["data_ids"], data_ids=f"{posescript_prefix}_{item['data_ids']}"))

        return item


################################################################################
## POSEFIX DATASET
################################################################################

def load_posefix(caption_files, split, dataID_2_pose_info):
    
    # initialize split ids (pair ids) ; split dependent
    dataIDs = utils.read_json(config.file_posefix_split % (split, 'in'))
    dataIDs += utils.read_json(config.file_posefix_split % (split, 'out'))
    
    # load pose pairs (pairs of pose ids)
    pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)

    # initialize triplet data
    triplets = {data_id: {"pose_A": pose_pairs[data_id][0],
                          "pose_B": pose_pairs[data_id][1],
                          "modifier": []}
                    for data_id in dataIDs}
    for t,v in triplets.items(): triplets[t]["in-sequence"] = dataID_2_pose_info[str(v["pose_A"])][1] == dataID_2_pose_info[str(v["pose_B"])][1]

    # load available modifiers
    for triplet_file in caption_files:
        annotations = utils.read_json(triplet_file)
        for data_id_str, c in annotations.items():
            try:
                triplets[int(data_id_str)]["modifier"] += c if type(c) is list else [c]
            except KeyError:
                # this annotation is not part of the split
                pass

    # clean dataIDs and triplets: remove access to unavailable
    # data (when no annotation was performed); necessary step if a smaller
    # data set was loaded
    dataIDs = [data_id for data_id in dataIDs if triplets[data_id]["modifier"]]
    triplets = {data_id:triplets[data_id] for data_id in dataIDs}
    
    duplicates = 0
    for t in triplets.values():
        duplicates += 1 if len(t["modifier"]) > 1 else 0
    print(f"[PoseFix] Loaded {len(dataIDs)} pairs in {split} split (found {duplicates} with more than 1 annotation).")
    
    return dataIDs, triplets


def get_all_posefix_triplets(caption_files):
    """
    Load all modifiers in the provided caption files, independently of the
    split, and format data to have triplets giving the IDs of poses A & B, a
    list of modifiers and a boolean for in-sequence information.
    """

    # load pose & pair information
    dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
    pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)

    # build triplet data
    triplets = {}
    for triplet_file in caption_files:
        annotations = utils.read_json(triplet_file)
        for data_id_str, c in annotations.items():
            data_id = int(data_id_str)
            # add element if new
            if data_id not in triplets:
                triplets[data_id] = {
                    "pose_A": pose_pairs[data_id][0],
                    "pose_B": pose_pairs[data_id][1],
                    "modifier": []
                }
            # add annotations
            triplets[data_id]["modifier"] += c if type(c) is list else [c]
    for t,v in triplets.items(): triplets[t]["in-sequence"] = dataID_2_pose_info[str(v["pose_A"])][1] == dataID_2_pose_info[str(v["pose_B"])][1]
    
    return triplets


class PoseFix(GenericDataset): # modifier-based dataset

    def __init__(self, version="posefix-H", split='train', 
                    tokenizer_name='vocPFAHPP', caption_index='rand',
                    num_body_joints=config.NB_INPUT_JOINTS,
                    cache=True, data_size=None, posescript_format=False):
        super(PoseFix, self).__init__(version=version, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=num_body_joints, cache=cache, data_size=data_size)

        # will update item format if using the posescript format
        self.posescript_format = posescript_format


    def _load_data(self):
        # load data info
        self.dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
        # get split ids & triplet data
        self.dataIDs, self.triplets = load_posefix(config.caption_files[self.version][1], self.split, self.dataID_2_pose_info)


    def _element_to_cache(self, index):
        pose_A, pose_A_id, pose_B, pose_B_id = self.get_poses_AB(index)
        if self.tokenizer_name:
            captions = self.get_all_captions(index)
            caption_tokens_list = [self.tokenizer(c) for c in captions]
            caption_length_list = [len(caption_tokens) for caption_tokens in caption_tokens_list]
            # padd tokenized captions
            caption_tokens_list = [torch.cat( (caption_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-len(caption_tokens), dtype=caption_tokens.dtype) ), dim=0) for caption_tokens in caption_tokens_list]
            return (pose_A, pose_B, caption_tokens_list, caption_length_list, self.dataIDs[index], pose_A_id, pose_B_id)
        else:
            return (pose_A, pose_B, None, None, self.dataIDs[index], pose_A_id, pose_B_id)


    def get_all_captions(self, index):
        return self.triplets[self.dataIDs[index]]["modifier"]
                

    def get_pose(self, index, pose_type, applied_rotation=None, output_rotation=False):
        # get pose id
        pose_id = self.triplets[self.dataIDs[index]][pose_type]
        # load pose data
        pose_info = self.dataID_2_pose_info[str(pose_id)]
        ret = utils.get_pose_data_from_file(pose_info,
                                            applied_rotation=applied_rotation,
                                            output_rotation=output_rotation)
        # reshape pose to (njoints, 3)
        if output_rotation:
            return ret[0].reshape(-1, 3), int(pose_id), ret[1] # rotation
        else:
            return ret.reshape(-1, 3), int(pose_id)
            

    def get_poses_AB(self, index):
        if self.triplets[self.dataIDs[index]]["in-sequence"]:
            pA, pidA, rA = self.get_pose(index, pose_type="pose_A", output_rotation=True)
            pB, pidB = self.get_pose(index, pose_type="pose_B", applied_rotation=rA)
        else:
            pA, pidA = self.get_pose(index, pose_type="pose_A")
            pB, pidB = self.get_pose(index, pose_type="pose_B")
        return pA, pidA, pB, pidB
        

    def __getitem__(self, index, cidx=None):

        if self.cache: 
            poses_A, poses_B, caption_tokens_list, caption_lengths_list, data_ids, poses_A_ids, poses_B_ids = deepcopy(self._data_cache[index])
            cidx = cidx if cidx else self.get_caption_index(len(caption_tokens_list), index)
            caption_tokens = caption_tokens_list[cidx]
            caption_lengths = caption_lengths_list[cidx]
        else:
            poses_A, poses_A_ids, poses_B, poses_B_ids = self.get_poses_AB(index)
            data_ids = self.dataIDs[index]
            captions = self.get_caption(index, cidx=cidx)
            if self.tokenizer_name is None: # will yield the raw texts
                caption_tokens = captions
                caption_lengths = 0
            else:
                caption_tokens = self.tokenizer(captions)
                caption_lengths = len(caption_tokens)
                # padding
                caption_tokens = torch.cat( (caption_tokens, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-caption_lengths, dtype=caption_tokens.dtype) ), dim=0)

        item = dict(poses_A=poses_A[:self.num_body_joints], poses_B=poses_B[:self.num_body_joints], caption_tokens=caption_tokens, caption_lengths=caption_lengths, poses_A_ids=poses_A_ids, poses_B_ids=poses_B_ids, data_ids=data_ids, indices=index)
        
        if self.posescript_format:
            item.update(dict(pose=item.pop("poses_B"), data_ids=item.pop("poses_B_ids")))

        return item


################################################################################
## MIXED DATASET (PoseScript+PoseFix)
################################################################################

# NOTE: cannot reuse the cache from PoseScript & PoseFix, as they don't
# share the same vocabulary
class PoseMix(PoseFix): # also a modifier-based dataset

    def __init__(self, version="posemix-PSH2-PFH", split='train',
                    tokenizer_name='vocMixPSA2H2PFAH', caption_index='rand',
                    num_body_joints=config.NB_INPUT_JOINTS,
                    cache=True, data_size=None):
        super(PoseMix, self).__init__(version=version, split=split, tokenizer_name=tokenizer_name, caption_index=caption_index, num_body_joints=num_body_joints, cache=cache, data_size=data_size)


    def _load_data(self):

        # load data info
        self.dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
        
        # initialization
        self.dataIDs = []
        self.triplets = {}

        # 1) load posefix
        captions_files_posefix = [f for f in config.caption_files[self.version][1] if config.POSEFIX_LOCATION in f]
        dataIDs_posefix, triplets_posefix = load_posefix(captions_files_posefix, self.split, self.dataID_2_pose_info)
        # include posefix data
        self.dataIDs += globalize_data_ID(dataIDs_posefix, posefix_prefix)
        self.triplets.update(globalize_data_ID(triplets_posefix, posefix_prefix))

        # 2) load posescript
        captions_files_posescript = [f for f in config.caption_files[self.version][1] if config.POSESCRIPT_LOCATION in f]
        dataIDs_posescript, captions_posescript = load_posescript(captions_files_posescript, self.split)
        # convert posescript data to triplet format
        triplets_posescript = {
            dataID: {
                "pose_A": None,
                "pose_B": dataID,
                "modifier": caption,
            } for dataID, caption in captions_posescript.items()
        }
        # include posescript data
        self.dataIDs += globalize_data_ID(dataIDs_posescript, posescript_prefix)
        self.triplets.update(globalize_data_ID(triplets_posescript, posescript_prefix))


    def get_poses_AB(self, index):
        data_id = self.dataIDs[index]
        if posefix_prefix in data_id:
            return super(PoseMix, self).get_poses_AB(index)
        elif posescript_prefix in data_id:
            # sets pose A to the T-pose and its corresponding ID to 'NaN'
            pA, pidA = T_POSE, PID_NAN
            pB, pidB = self.get_pose(index, pose_type="pose_B")
            return pA, pidA, pB, pidB


if __name__ == '__main__':
    # building all caches
    for split in ["train", "val", "test"]:
        for v in ["A2", "H2"]:
            dataset = PoseScript(version=f"posescript-{v}", split=split, tokenizer_name="vocPSA2H2", cache=True)
            dataset = PoseScript(version=f"posescript-{v}", split=split, tokenizer_name="distilbertUncased", cache=True)
        for v in ["A", "H", "HPP"]:
            dataset = PoseFix(version=f"posefix-{v}", split=split, tokenizer_name="vocPFAHPP", cache=True)
            dataset = PoseFix(version=f"posefix-{v}", split=split, tokenizer_name="distilbertUncased", cache=True)