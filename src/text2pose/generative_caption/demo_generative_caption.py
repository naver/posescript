
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

import text2pose.demo as demo
from text2pose.generative_caption.evaluate_generative_caption import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_paths', nargs='+', type=str, help='Paths to the models to be compared.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
args = parser.parse_args()


### INPUT
################################################################################

data_version = "posescript-H2"


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
dataID_2_pose_info, captions = demo.setup_posescript_data(data_version)


### MAIN APP
################################################################################

# define query input interface
cols_query = st.columns(3)
split_for_research = cols_query[0].selectbox('Split:', tuple(available_splits), index=available_splits.index('test'))
query_type = cols_query[1].selectbox("Query type:", ('Split index', 'ID'), index=1)
number = cols_query[2].number_input("Split index or ID:", 0)
st.markdown("""---""")

# get query data
pose_ID, pose_data, pose_img, default_description = demo.get_posescript_datapoint(number, query_type, split_for_research, captions, dataID_2_pose_info, body_model)

# show query data
cols_input = st.columns(2)
cols_input[0].image(pose_img, caption="Annotated pose")
if default_description:
	cols_input[1].write("Annotated text:")
	cols_input[1].write(f"_{default_description}_")
else:
	cols_input[1].write("_(Not annotated.)_")

# generate text
st.markdown("""---""")
st.write("**Text generation:**")
for i, model in enumerate(models):

	with torch.no_grad():
		texts, scores = model.generate_text(pose_data.view(1, -1, 3)) # (1, njoints, 3)
	
	if len(models) > 1:
		cols = st.columns(2)
		cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
		cols[1].write(texts[0])
		st.markdown("""---""")
	else:
		st.write(texts[0])
		st.markdown("""---""")
		st.write(f"_Results obtained with model: {args.model_paths[0]}_")