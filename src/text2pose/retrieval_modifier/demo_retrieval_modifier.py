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
import text2pose.utils_visu as utils_visu
from text2pose.retrieval_modifier.evaluate_retrieval_modifier import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
parser.add_argument('--n_retrieve', type=int, default=12, help="Number of elements to retrieve.")
args = parser.parse_args()

args.n_retrieve = 12

### INPUT
################################################################################

data_version_annotations = "posefix-H" # defines what annotations to use as query examples
data_version_poses_collection = "posefix-A" # defines the set of poses to rank


### SETUP
################################################################################

# --- data
available_splits = ['train', 'val', 'test']
model, tokenizer_name, body_model = demo.setup_models([args.model_path], args.checkpoint, load_model)
model, tokenizer_name = model[0], tokenizer_name[0]
dataID_2_pose_info, triplet_data = demo.setup_posefix_data(data_version_annotations)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)


### MAIN APP
################################################################################

# define query input interface: split selection
cols_query = st.columns(3)
split_for_research = cols_query[0].selectbox('Split:', tuple(available_splits), index=available_splits.index('test'))

# precompute features
dataIDs = demo.setup_posefix_split(split_for_research)
pair_dataIDs, pairs_features = demo.precompute_posefix_pair_features(data_version_poses_collection, split_for_research, model)
text_dataIDs, text_features = demo.precompute_text_features(data_version_annotations, split_for_research, model, tokenizer_name)

# define query input interface: example selection
query_type = cols_query[1].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[2].number_input("Split index or ID:", 0)
st.markdown("""---""")

# get query data
pair_ID, pid_A, pid_B, pose_A_data, pose_B_data, pose_A_img, pose_B_img, default_modifier = demo.get_posefix_datapoint(number, query_type, split_for_research, triplet_data, pose_pairs, dataID_2_pose_info, body_model)

# show query data
cols_input = st.columns([1,1,2])
cols_input[0].image(pose_A_img, caption="Pose A")
cols_input[1].image(pose_B_img, caption="Pose B")
if default_modifier:
	cols_input[2].write("Annotated text:")
	cols_input[2].write(f"_{default_modifier}_")
else:
	cols_input[2].write("_(Not annotated.)_")

# get retrieval direction
dt2p = "Text-2-Pair"
dp2t = "Pair-2-Text"
retrieval_direction = st.radio("Retrieval direction:", [dt2p, dp2t])

# TEXT-2-PAIR
if retrieval_direction == dt2p:

	# get input modifier
	modifier = cols_input[2].text_area("Pose modifier:",
							placeholder="Move your right arm... lift your left leg...",
							value=default_modifier,
							height=None, max_chars=None)
	# encode text
	with torch.no_grad():
		text_feature = model.encode_raw_text(modifier)

	# rank poses by relevance and get their pose id
	scores = text_feature.view(1, -1).mm(pairs_features.t())[0]
	_, indices_rank = scores.sort(descending=True)
	relevant_pair_ids = [pair_dataIDs[i] for i in indices_rank[:args.n_retrieve]]

	# get corresponding pair data and render the pairs as images
	imgs = []
	for pair_id in relevant_pair_ids:
		ret_pid_A, ret_pid_B = pose_pairs[pair_id]
		pose_A_info = dataID_2_pose_info[str(ret_pid_A)]
		pose_A_data, rA = utils.get_pose_data_from_file(pose_A_info, output_rotation=True)
		pose_B_info = dataID_2_pose_info[str(ret_pid_B)]
		pose_B_data = utils.get_pose_data_from_file(pose_B_info, applied_rotation=rA if pose_A_info[1]==pose_B_info[1] else None)
		imgs.append(utils_visu.image_from_pair_data(pose_A_data, pose_B_data, body_model, add_ground_plane=True))

	# display images
	st.markdown("""---""")
	st.write(f"**Retrieved pairs for this modifier [{split_for_research} split]:**")
	cols = st.columns(demo.nb_cols)
	for i in range(args.n_retrieve):
		cols[i%demo.nb_cols].image(demo.process_img(imgs[i]))

# PAIR-2-TEXT
elif retrieval_direction == dp2t:

	# rank texts by relevance and get their id
	pair_index = pair_dataIDs.index(pair_ID)
	scores = pairs_features[pair_index].view(1, -1).mm(text_features.t())[0]
	_, indices_rank = scores.sort(descending=True)
	relevant_pair_ids = [text_dataIDs[i] for i in indices_rank[:args.n_retrieve]]

	# get corresponding text data (the text features were obtained using the first text)
	texts = [triplet_data[pair_id]['modifier'][0] for pair_id in relevant_pair_ids]

	# display texts
	st.markdown("""---""")
	st.write(f"**Retrieved modifiers for this pair [{split_for_research} split]:**")
	for i in range(args.n_retrieve):
		st.write(f"**({i+1})** {texts[i]}")

st.markdown("""---""")
st.write(f"_Results obtained with model: {args.model_path}_")