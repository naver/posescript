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
import numpy as np

import text2pose.config as config
import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.data as data
import text2pose.utils_visu as utils_visu
from text2pose.generative_B.evaluate_generative_B import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_paths', nargs='+', type=str, help='Paths to the models to be compared.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help="Checkpoint to choose if model path is incomplete.")
parser.add_argument('--n_generate', type=int, default=12, help="Number of poses to generate (number of samples); if considering only one model.")
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

# correct the number of generated sample depending on the setting
if len(args.model_paths) > 1:
    n_generate = 4
else:
    n_generate = args.n_generate

# --- data
available_splits = ['train', 'val', 'test']
models, _, body_model = demo.setup_models(args.model_paths, args.checkpoint, load_model)
dataID_2_pose_info, triplet_data = demo.setup_posefix_data(posefix_data_version)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)
_, captions = demo.setup_posescript_data(posescript_data_version)

# --- seed
torch.manual_seed(42)
np.random.seed(42)


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

# get input modifier
modifier = cols_input[2].text_area("Pose modifier:",
                        placeholder="Move your right arm... lift your left leg...",
                        value=txt,
                        height=None, max_chars=None)

analysis = cols_input[2].checkbox('Analysis') # whether to show the reconstructed pose and the mean sample pose in addition of some samples

# generate pose B
if analysis:

    st.markdown("""---""")
    st.write("**Generated poses** (*The reconstructed B pose is shown in green; the mean pose in red; and samples in blue.*):")
    n_generate = 2
    nb_cols = 2 + n_generate # reconstructed pose + mean sample pose + n_generate sample poses: all must fit in one row, for each studied model

    for i, model in enumerate(models):
        with torch.no_grad():
            rec_pose_data = model.forward_autoencoder(pose_B_data)['pose_body_pose'].view(1, -1)
            gen_pose_data_mean = model.sample_str_meanposes(pose_A_data, modifier)['pose_body'].view(1, -1)
            gen_pose_data_samples = model.sample_str_nposes(pose_A_data, modifier, n=n_generate)['pose_body'][0,...].view(n_generate, -1)

        # render poses
        imgs = utils_visu.image_from_pose_data(rec_pose_data, body_model, color='green', add_ground_plane=True, two_views=60)
        imgs += utils_visu.image_from_pose_data(gen_pose_data_mean, body_model, color='red', add_ground_plane=True, two_views=60)
        imgs += utils_visu.image_from_pose_data(gen_pose_data_samples, body_model, color='blue', add_ground_plane=True, two_views=60)

        # display images
        cols = st.columns(nb_cols+1) # +1 to display model info
        cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
        for i in range(nb_cols):
            cols[i%nb_cols+1].image(demo.process_img(imgs[i]))
        st.markdown("""---""")

else:

    st.markdown("""---""")
    st.write("**Generated poses:**")

    for i, model in enumerate(models):
        with torch.no_grad():
            gen_pose_data_samples = model.sample_str_nposes(pose_A_data, modifier, n=n_generate)['pose_body'][0,...].view(n_generate, -1)

        # render poses
        imgs = utils_visu.image_from_pose_data(gen_pose_data_samples, body_model, color='blue', add_ground_plane=True, two_views=60)

        # display images
        if len(models) > 1:
            cols = st.columns(n_generate+1) # +1 to display model info
            cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
            for i in range(n_generate):
                cols[i%n_generate+1].image(demo.process_img(imgs[i]))
            st.markdown("""---""")
        else:
            cols = st.columns(demo.nb_cols)
            for i in range(n_generate):
                cols[i%demo.nb_cols].image(demo.process_img(imgs[i]))
            st.markdown("""---""")
            st.write(f"_Results obtained with model: {args.model_paths[0]}_")