##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# $ streamlit run demo_app.py -- --model_path <model_dir_path> --n_generate <n_generate>

import streamlit as st
import argparse
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.vocab import Vocabulary # needed
from text2pose.generative.evaluate_generative import load_model
import text2pose.utils_visu as utils_visu


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--n_generate', type=int, default=12, help="Number of poses to generate.")
args = parser.parse_args()


### INPUT
################################################################################

model_path = args.model_path
n_generate = args.n_generate

nb_cols = 4
nb_rows = n_generate//nb_cols
margin_img = 80

device = 'cpu'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


### SETUP
################################################################################

@st.cache
def setup(model_path):

	# load model
	model, _ = load_model(model_path, device)

	# setup body model
	body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas)
	body_model.eval()
	body_model.to(device)

	return model, body_model


model, body_model = setup(model_path)


### GENERATE
################################################################################

st.write(f"Using model: {model_path}")

# get input description
description = st.text_area("Pose description:",
					placeholder="The person is...",
					height=None, max_chars=None)

# generate poses from text
with torch.no_grad():
	gen_pose_data = model.sample_str_nposes(description, n=n_generate)['pose_body'][0,...].view(n_generate, -1)

# render poses
imgs = utils_visu.image_from_pose_data(gen_pose_data, body_model)

# display images
st.write("**Generated poses for this description:**")
cols = st.columns(nb_cols)
for i in range(n_generate):
	cols[i%nb_cols].image(imgs[i][margin_img:-margin_img,margin_img:-margin_img]) # remove white margin to make poses look larger