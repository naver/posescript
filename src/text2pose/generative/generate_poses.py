##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np

import text2pose.config as config
from text2pose.data import PoseScript
from text2pose.vocab import Vocabulary # needed
from text2pose.generative.evaluate_generative import load_model


parser = argparse.ArgumentParser(description='Parameters to generate poses corresponding to each caption.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--n_generate', type=int, default=5, help="Number of poses to generate for a given caption.")
args = parser.parse_args()


### INPUT
################################################################################

model_path = args.model_path
n_generate = args.n_generate

splits = ['train', 'test', 'val']
device = torch.device('cuda:0')
save_path = config.generated_pose_path % os.path.dirname(model_path)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


### GENERATE POSES
################################################################################

# load model
model, text_encoder_name = load_model(model_path, device)
dataset_version = torch.load(model_path, 'cpu')['args'].dataset

# create saving directory
if not os.path.isdir(os.path.dirname(save_path)):
    os.mkdir(os.path.dirname(save_path))

# generate poses
for s in splits:

    # check that the poses were not already generated
    filepath = save_path.format(data_version=dataset_version, split=s)
    assert not os.path.isfile(filepath), "Poses already generated!"

    d = PoseScript(version=dataset_version, split=s, text_encoder_name=text_encoder_name)
    ncaptions = len(d._data_cache[0][1]) # list of captions for the first pose
    output = torch.empty( (len(d), ncaptions, n_generate, model.pose_decoder.num_joints, 3), dtype=torch.float32)
    
    for index in tqdm(range(len(d))):
        # look at each available caption in turn
        for cidx in range(ncaptions):
            item = d.__getitem__(index, cidx=cidx)
            caption_tokens = item['caption_tokens'].to(device).unsqueeze(0)
            caption_lengths = torch.tensor([item['caption_lengths']]).to(device)
            caption_tokens = caption_tokens[:,:caption_lengths.max()]
            with torch.no_grad():
                genposes = model.sample_text_nposes(caption_tokens, caption_lengths, n=n_generate)['pose_body'][0,...]
                output[index,cidx,...] = genposes

    # save
    torch.save(output, filepath)
    print(filepath)
