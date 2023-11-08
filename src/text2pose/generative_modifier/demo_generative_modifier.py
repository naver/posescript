##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
import argparse
import torch

import text2pose.config as config
import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.data as data
import text2pose.utils_visu as utils_visu
from text2pose.generative_modifier.evaluate_generative_modifier import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_paths', nargs='+', type=str, help='Path to the models to be compared.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help="Checkpoint to choose if model path is incomplete.")
args = parser.parse_args()


### INPUT
################################################################################

posefix_data_version = "posefix-H"
posescript_data_version = "posescript-H2"


### SETUP
################################################################################

# --- layout
st.markdown("""
			<style>
			.smallfont {
				font-size:10px !important;
			}
			</style>
			""", unsafe_allow_html=True)

# --- data
available_splits = ['train', 'val', 'test']
models, _, body_model = demo.setup_models(args.model_paths, args.checkpoint, load_model)
dataID_2_pose_info, triplet_data = demo.setup_posefix_data(posefix_data_version)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)
_, captions = demo.setup_posescript_data(posescript_data_version)


### MAIN APP
################################################################################

# define query input interface
cols_query = st.columns(3)
split_for_research = cols_query[0].selectbox('Split:', tuple(available_splits), index=available_splits.index('test'))
query_type = cols_query[1].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[2].number_input("Split index or ID:", 0)
st.markdown("""---""")

# get query data
pair_ID, pid_A, pid_B, pose_A_data, pose_B_data, pose_A_img, pose_B_img, default_modifier = demo.get_posefix_datapoint(number, query_type, split_for_research, triplet_data, pose_pairs, dataID_2_pose_info, body_model)

# show query data
st.write(f"**Query data:**")
cols_input = st.columns([1,1,2])
# (enable PoseScript mode: description only)
no_pose_A = cols_input[2].checkbox("PoseScript mode")
if no_pose_A:
	pose_A_data = data.T_POSE.view(1, -1)
	pose_A_img = utils_visu.image_from_pose_data(pose_A_data, body_model, color="grey", add_ground_plane=True)
	pose_A_img = demo.process_img(pose_A_img[0])
	pose_B_data, pose_B_img, default_description = demo.get_posescript_datapoint_from_pid(pid_B, captions, dataID_2_pose_info, body_model)
# (actually show)
cols_input[0].image(pose_A_img, caption="T-pose" if no_pose_A else "Pose A")
cols_input[1].image(pose_B_img, caption="Annotated pose" if no_pose_A else "Annotated pose B")
txt = default_description if no_pose_A else default_modifier
if txt:
	cols_input[2].write("Annotated text:")
	cols_input[2].write(f"_{txt}_")
else:
	cols_input[2].write("_(Not annotated.)_")

# generate text
st.markdown("""---""")
st.write("**Text generation:**")
for i, model in enumerate(models):

	with torch.no_grad():
		texts, scores = model.generate_text(pose_A_data.view(1, -1, 3), pose_B_data.view(1, -1, 3)) # (1, njoints, 3)

	if len(models) > 1:
		cols = st.columns(2)
		cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
		cols[1].write(texts[0])
		st.markdown("""---""")
	else:
		st.write(texts[0])
		st.markdown("""---""")
		st.write(f"_Results obtained with model: {args.model_paths[0]}_")