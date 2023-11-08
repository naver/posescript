##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# $ streamlit run posescript/explore_posescript.py

import streamlit as st

import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu


### INPUT
################################################################################

version_human = 'posescript-H2'
version_auto = 'posescript-A2'


### SETUP
################################################################################

dataID_2_pose_info, captions_human = demo.setup_posescript_data(version_human)
_, captions_auto = demo.setup_posescript_data(version_auto)
body_model = demo.setup_body_model()


### DISPLAY DATA
################################################################################

# get input pose id
dataID = st.number_input("Pose ID:", 0, len(dataID_2_pose_info)-1)

# display information about the pose
pose_info = dataID_2_pose_info[str(dataID)]
st.write(f"Pose from the **{pose_info[0]}** dataset, **frame {pose_info[2]}** of sequence *{pose_info[1]}*")

# load pose data
pose_data = utils.get_pose_data_from_file(pose_info)

# render the pose under the desired viewpoint, and display it
view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))
img = utils_visu.image_from_pose_data(pose_data, body_model, viewpoints=[viewpoint], add_ground_plane=True)[0] # 1 viewpoint
st.image(demo.process_img(img))

# display captions
if dataID in captions_human:
    for k,c in enumerate(captions_human[dataID]):
        st.markdown(f"<b><font color='purple'>Human-written description n°{k+1}</font></b>", unsafe_allow_html=True)
        st.write(captions_human[dataID][k])
if dataID in captions_auto:
    for k,c in enumerate(captions_auto[dataID]):
        st.markdown(f"<b><font color='orange'>Automatic description n°{k+1}</font></b>", unsafe_allow_html=True)
        st.write(captions_auto[dataID][k])
else:
    st.write("This pose ID was not annotated.")