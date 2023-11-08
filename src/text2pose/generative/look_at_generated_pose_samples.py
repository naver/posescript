##############################################################
## text2pose                                                ##
## Copyright (c) 2022                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
import os
import argparse
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.utils_visu as utils_visu
from text2pose.data import PoseScript


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model that generated the pose samples to visualize.')
parser.add_argument('--dataset_version', type=str, help='Dataset version (depends on the model)')
parser.add_argument('--split', type=str, help='Split')
args = parser.parse_args()


### SETUP
################################################################################

@st.cache_resource
def setup(args):

	# setup data
	generated_pose_path = config.generated_pose_path % os.path.dirname(args.model_path)
	generated_pose_path = generated_pose_path.format(data_version=args.dataset_version, split=args.split)
	
	dataset = PoseScript(version=args.dataset_version, split=args.split,
						cache=False, generated_pose_samples_path=generated_pose_path)

	# setup body model
	body_model = BodyModel(model_type = config.POSE_FORMAT,
                       bm_fname = config.NEUTRAL_BM,
                       num_betas = config.n_betas)
	body_model.eval()
	body_model.to('cpu')

	return generated_pose_path, dataset, body_model


generated_pose_path, dataset, body_model = setup(args)


### VISUALIZE
################################################################################

st.write(f"**Dataset:** {args.dataset_version}")
st.write(f"**Split:** {args.split}")
st.write(f"**Using pose samples from:** {generated_pose_path}")

# get input pose index & caption index
st.write("**Choose a data point:**")
index = st.number_input("Index in split:", 0, len(dataset.pose_samples)-1)
cidx = st.number_input("Caption index:", 0, len(dataset.pose_samples[0])-1)

# display description
st.write("**Description:** "+dataset.captions[dataset.dataIDs[index]][cidx])

# render poses
nb_samples = dataset.pose_samples.shape[2]
img_original = utils_visu.image_from_pose_data(dataset.get_pose(index).view(1, -1), body_model, color='blue')
imgs_sampled = utils_visu.image_from_pose_data(dataset.pose_samples[index, cidx].view(nb_samples, -1), body_model, color='green')

# display original pose
st.write("**Original pose:**")
st.image(img_original[0])

# display generated pose samples
st.write("**Generated pose samples for this description:**")
cols = st.columns(nb_samples)
for i in range(nb_samples):
	cols[i].image(imgs_sampled[i])