##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# $ streamlit run explore_posescript.py

import streamlit as st
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu


### INPUT
################################################################################

margin_img = 80 # crop white margin on each side of the produced image
viewpoints = [[], (45, (0, 1, 0)), (90, (0, 1, 0))] # view when the body is rotated of resp. 0, 45 et 90 degrees around the y axis ([] means front view)


### SETUP
################################################################################

@st.cache
def setup():

    # utils
    def update(d1, d2, new_key):
        for k in d2:
            if k not in d1:
                d1[k] = {}
            d1[k].update({new_key:d2[k]})
        return d1

    # gather descriptions
    caption_keys = ["human", "A", "B", "C", "D", "E", "F"]
    caption_files = ["human3893.json"]+[f"automatic_{letter}.json" for letter in caption_keys[1:]]
    captions = {}
    for cap_file, key in zip(caption_files, caption_keys):
        captions = update(captions, utils.read_posescript_json(cap_file), key)

    # get pose information
    dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")

    # setup body model
    device = 'cpu' # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas)
    body_model.eval()
    body_model.to(device)

    return dataID_2_pose_info, captions, body_model


dataID_2_pose_info, captions, body_model = setup()


### DISPLAY DATA
################################################################################

# get input pose id
dataID = st.number_input("Pose ID:", 0, len(dataID_2_pose_info)-1)
dataID = str(dataID)

# display information about the pose
pose_info = dataID_2_pose_info[dataID]
st.write(f"Pose from the **{pose_info[0]}** dataset, **frame {pose_info[2]}** of file *{pose_info[1]}*")

# load pose data and render the pose under different viewpoints
pose_data = utils.get_pose_data_from_file(pose_info)
imgs = utils_visu.image_from_pose_data(pose_data, body_model, viewpoints=viewpoints, color="blue")

# display images
cols = st.columns(len(viewpoints))
for i, col in enumerate(cols):
    col.image(imgs[i][margin_img:-margin_img,margin_img:-margin_img]) # remove white margin to make poses look larger

# display captions
if dataID in captions:
    for k in captions[dataID]:
        caption_title = "Human-written caption" if k=="human" else f"Automatic caption - {k}"
        color = 'purple' if k=="human" else 'orange'
        st.markdown(f"<b><font color='{color}'>{caption_title}</font></b>", unsafe_allow_html=True)
        st.write(captions[dataID][k])
