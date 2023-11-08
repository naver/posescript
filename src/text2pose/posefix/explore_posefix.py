##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# $ streamlit run posefix/explore_posefix.py

import streamlit as st
import random

import text2pose.config as config
import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu


### INPUT
################################################################################

version_human = 'posefix-H'
version_paraphrases = 'posefix-PP'
version_auto = 'posefix-A'


### SETUP
################################################################################

dataID_2_pose_info, triplet_data_human = demo.setup_posefix_data(version_human)
_, triplet_data_pp = demo.setup_posefix_data(version_paraphrases)
_, triplet_data_auto = demo.setup_posefix_data(version_auto)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)
body_model = demo.setup_body_model()


### DISPLAY DATA
################################################################################

# get input pair id
pair_ID = st.number_input("Pair ID:", 0, len(pose_pairs))
if st.button('Look at a random pose!'):
	pair_ID = random.randint(0, len(pose_pairs))
	st.write(f"Looking at pair ID: **{pair_ID}**")

# display information about the pair
pid_A, pid_B = pose_pairs[pair_ID]
in_sequence = dataID_2_pose_info[str(pid_A)][1] == dataID_2_pose_info[str(pid_B)][1]
st.markdown(f"<b>{'In' if in_sequence else 'Out-of'}-sequence</b> pair (<b><font color='#b2b2b2'>pose A</font></b> to <b><font color='#6666ff'>pose B</font></b>).", unsafe_allow_html=True)

# load pose data
pose_A_info = dataID_2_pose_info[str(pid_A)]
pose_A_data, rA = utils.get_pose_data_from_file(pose_A_info, output_rotation=True)
pose_B_info = dataID_2_pose_info[str(pid_B)]
pose_B_data = utils.get_pose_data_from_file(pose_B_info, applied_rotation=rA if in_sequence else None)

# render the pair under the desired viewpoint, and display it
view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))
pair_img = utils_visu.image_from_pair_data(pose_A_data, pose_B_data, body_model, viewpoint=viewpoint, add_ground_plane=True)
st.image(demo.process_img(pair_img))

# display text annotations
if pair_ID in triplet_data_human:
    for k,m in enumerate(triplet_data_human[pair_ID]["modifier"]):
        st.markdown(f"<b><font color='#5fd0eb'>Human-written modifier n°{k+1}</font></b>", unsafe_allow_html=True)
        st.write(f"_{m.strip()}_")
if pair_ID in triplet_data_pp:
    for k,m in enumerate(triplet_data_pp[pair_ID]["modifier"]):
        st.markdown(f"<b><font color='#7dcc65'>Paraphrase n°{k+1}</font></b>", unsafe_allow_html=True)
        st.write(f"_{m.strip()}_")
if pair_ID in triplet_data_auto:
    for k,m in enumerate(triplet_data_auto[pair_ID]["modifier"]):
        st.markdown(f"<b><font color='orange'>Automatic modifier n°{k+1}</font></b>", unsafe_allow_html=True)
        st.write(f"_{m.strip()}_")
else:
    st.write("This pair ID was not annotated.")