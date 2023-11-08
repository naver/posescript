##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
import argparse
import torch

import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu
from text2pose.retrieval.evaluate_retrieval import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
parser.add_argument('--n_retrieve', type=int, default=12, help="Number of elements to retrieve.")
args = parser.parse_args()


### INPUT
################################################################################

data_version_annotations = "posescript-H2" # defines what annotations to use as query examples
data_version_poses_collection = "posescript-A2" # defines the set of poses to rank


### SETUP
################################################################################

# --- data
available_splits = ['train', 'val', 'test']
model, tokenizer_name, body_model = demo.setup_models([args.model_path], args.checkpoint, load_model)
model, tokenizer_name = model[0], tokenizer_name[0]
dataID_2_pose_info, captions = demo.setup_posescript_data(data_version_annotations)


### MAIN APP
################################################################################

# define query input interface: split selection
cols_query = st.columns(3)
split_for_research = cols_query[0].selectbox('Split:', tuple(available_splits), index=available_splits.index('test'))

# precompute features
dataIDs = demo.setup_posescript_split(split_for_research)
pose_dataIDs, poses_features = demo.precompute_posescript_pose_features(data_version_poses_collection, split_for_research, model)
text_dataIDs, text_features = demo.precompute_text_features(data_version_annotations, split_for_research, model, tokenizer_name)

# define query input interface: example selection
query_type = cols_query[1].selectbox("Query type:", ('Split index', 'ID'))
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

# get retrieval direction
dt2p = "Text-2-Pose"
dp2t = "Pose-2-Text"
retrieval_direction = st.radio("Retrieval direction:", [dt2p, dp2t])

# TEXT-2-POSE
if retrieval_direction == dt2p:

	# get input description
	description = cols_input[1].text_area("Pose description:",
							value=default_description,
							placeholder="The person is...",
							height=None, max_chars=None)

	# encode text
	with torch.no_grad():
		text_feature = model.encode_raw_text(description)

	# rank poses by relevance and get their pose id
	scores = text_feature.view(1, -1).mm(poses_features.t())[0]
	_, indices_rank = scores.sort(descending=True)
	relevant_pose_ids = [pose_dataIDs[i] for i in indices_rank[:args.n_retrieve]]

	# get corresponding pose data
	all_pose_data = []
	for pose_id in relevant_pose_ids:
		pose_info = dataID_2_pose_info[str(pose_id)]
		all_pose_data.append(utils.get_pose_data_from_file(pose_info))
	all_pose_data = torch.cat(all_pose_data)

	# render poses
	imgs = utils_visu.image_from_pose_data(all_pose_data, body_model, color="blue", add_ground_plane=True)

	# display images
	st.markdown("""---""")
	st.write(f"**Retrieved poses for this description [{split_for_research} split]:**")
	cols = st.columns(demo.nb_cols)
	for i in range(args.n_retrieve):
		cols[i%demo.nb_cols].image(demo.process_img(imgs[i]))

# POSE-2-TEXT
elif retrieval_direction == dp2t:

	# rank texts by relevance and get their id
	pose_index = pose_dataIDs.index(pose_ID)
	scores = poses_features[pose_index].view(1, -1).mm(text_features.t())[0]
	_, indices_rank = scores.sort(descending=True)
	relevant_pose_ids = [text_dataIDs[i] for i in indices_rank[:args.n_retrieve]]

	# get corresponding text data (the text features were obtained using the first text)
	texts = [captions[pose_id][0] for pose_id in relevant_pose_ids]

	# display texts
	st.markdown("""---""")
	st.write(f"**Retrieved descriptions for this pose [{split_for_research} split]:**")
	for i in range(args.n_retrieve):
		st.write(f"**({i+1})** {texts[i]}")

st.markdown("""---""")
st.write(f"_Results obtained with model: {args.model_path}_")