##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# $ streamlit run demo_retrieval.py -- --model_path <model_path> --split <split> --n_retrieve <n_retrieve>

import streamlit as st
import argparse
from tqdm import tqdm
import torch
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.retrieval.evaluate_retrieval import load_model
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--split', type=str, default="test", help="Split in which poses are retrieved.")
parser.add_argument('--n_retrieve', type=int, default=12, help="Number of poses to retrieve.")
args = parser.parse_args()


### INPUT
################################################################################

model_path = args.model_path
split_for_research = args.split
n_retrieve = args.n_retrieve

nb_cols = 4
nb_rows = n_retrieve//nb_cols
margin_img = 80

device = 'cpu'
batch_size = 32

data_version = "posescript-A1" # to consider all available poses
print(f"Load poses from {data_version}.")


### SETUP
################################################################################

@st.cache
def setup(model_path, split_for_research):

	# load model
	model, text_encoder_name = load_model(model_path, device)
	
	# pre-compute pose features
	dataset = PoseScript(version=data_version, split=split_for_research, text_encoder_name=text_encoder_name)
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	poses_features = torch.zeros(len(dataset), model.latentD)
	for i, batch in tqdm(enumerate(data_loader)):
		poses = batch['pose'].to(device)
		with torch.inference_mode():
			pfeat = model.pose_encoder(poses)
			poses_features[i*batch_size:i*batch_size+len(poses)] = pfeat

	# prepare pose identifiers within the studied split
	dataIDs = utils.read_posescript_json(f"{split_for_research}_ids.json")
	dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")

	# setup body model
	body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas)
	body_model.eval()
	body_model.to(device)

	return model, poses_features, dataIDs, dataID_2_pose_info, body_model


model, poses_features, dataIDs, dataID_2_pose_info, body_model= setup(model_path, split_for_research)


### RETRIEVE
################################################################################

st.write(f"Using model: {model_path}")

# get input description
description = st.text_area("Pose description:",
					placeholder="The person is...",
					height=None, max_chars=None)

# encode text
with torch.no_grad():
	text_feature = model.encode_raw_text(description)

# rank poses by relevance and get their pose id
scores = text_feature.view(1, -1).mm(poses_features.t())[0]
_, indices_rank = scores.sort(descending=True)
relevant_pose_ids = [dataIDs[i] for i in indices_rank[:n_retrieve]]

# get corresponding pose data
all_pose_data = []
for pose_id in relevant_pose_ids:
	pose_info = dataID_2_pose_info[str(pose_id)]
	all_pose_data.append(utils.get_pose_data_from_file(pose_info))
all_pose_data = torch.cat(all_pose_data)

# render poses
imgs = utils_visu.image_from_pose_data(all_pose_data, body_model, color="blue")

# display images
st.write(f"**Retrieved poses for this description [{split_for_research} split]:**")
cols = st.columns(nb_cols)
for i in range(n_retrieve):
	cols[i%nb_cols].image(imgs[i][margin_img:-margin_img,margin_img:-margin_img]) # remove white margin to make poses look larger