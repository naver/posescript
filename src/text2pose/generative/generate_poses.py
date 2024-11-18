##############################################################
## text2pose                                                ##
## Copyright (c) 2022                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np

import text2pose.config as config
from text2pose.data import PoseScript
from text2pose.generative.evaluate_generative import load_model


parser = argparse.ArgumentParser(description='Parameters to generate poses corresponding to each caption.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--n_generate', type=int, default=5, help="Number of poses to generate for a given caption.")
args = parser.parse_args()


### INPUT
################################################################################

device = torch.device('cuda:0')
save_path = config.generated_pose_path % os.path.dirname(args.model_path)
splits = ['train', 'test', 'val']

torch.manual_seed(42)
np.random.seed(42)


### GENERATE POSES
################################################################################

# load model
model, tokenizer_name = load_model(args.model_path, device)
dataset_version = torch.load(args.model_path, 'cpu')['args'].dataset

# create saving directory
if not os.path.isdir(os.path.dirname(save_path)):
    os.mkdir(os.path.dirname(save_path))

# generate poses
for s in splits:

    # check that the poses were not already generated
    filepath = save_path.format(data_version=dataset_version, split=s)
    assert not os.path.isfile(filepath), "Poses already generated!"

    d = PoseScript(version=dataset_version, split=s, tokenizer_name=tokenizer_name, num_body_joints=model.pose_decoder.num_body_joints)
    ncaptions = config.caption_files[dataset_version][0]
    output = torch.empty( (len(d), ncaptions, args.n_generate, model.pose_decoder.num_body_joints, 3), dtype=torch.float32)
    
    for index in tqdm(range(len(d))):
        # look at each available caption in turn
        for cidx in range(ncaptions):
            item = d.__getitem__(index, cidx=cidx)
            caption_tokens = item['caption_tokens'].to(device).unsqueeze(0)
            caption_lengths = torch.tensor([item['caption_lengths']]).to(device)
            caption_tokens = caption_tokens[:,:caption_lengths.max()]
            with torch.no_grad():
                genposes = model.sample_nposes(caption_tokens, caption_lengths, n=args.n_generate)['pose_body'][0,...]
                output[index,cidx,...] = genposes

    # save
    torch.save(output, filepath)
    print(filepath)