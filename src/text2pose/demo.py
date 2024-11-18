##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# Utilitary functions for the demo applications.
#
# NOTE:
# * Using a decorator on the functions defined here, and importing these
#   functions in another file run with streamlit will work as intended.
# * Unless this file is executed directly, all `st.something` commands here will
#   be automatically disabled by streamlit. Only the `st.something` commands
#   from the main executed file will work.
# ==> check that `st.` is only used for decorators.

import streamlit as st
import torch
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu
import text2pose.data as data


### LAYOUT
################################################################################

nb_cols = 4 # for a nice visualization
margin_img = 80


def process_img(img):
    return img[margin_img:-margin_img,margin_img:-margin_img] # remove white margin to make poses look larger


### SETUP: load data, models...
################################################################################

DEVICE = 'cpu'


@st.cache_resource
def setup_models(model_paths, checkpoint, _load_model_func, device=None):
    
    device = device if device else DEVICE

    # load models
    models = []
    tokenizer_names = []
    for i, mp in enumerate(model_paths):
        if ".pth" not in mp:
            mp = mp + f"/checkpoint_{checkpoint}.pth"
            print(f"Checkpoint not specified (model {i}). Using {checkpoint} checkpoint.")

        m, ten = _load_model_func(mp, device)
        models.append(m)
        tokenizer_names.append(ten)

    # setup body model
    body_model = setup_body_model()

    return models, tokenizer_names, body_model


@st.cache_resource
def setup_body_model():
    body_model = BodyModel(model_type = config.POSE_FORMAT,
                       bm_fname = config.NEUTRAL_BM,
                       num_betas = config.n_betas)
    body_model.eval()
    body_model.to(DEVICE)
    return body_model


@st.cache_data
def setup_posescript_data(data_version):
    # prepare pose identifiers & caption data
    dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
    captions = data.get_all_posescript_descriptions(config.caption_files[data_version][1])
    return dataID_2_pose_info, captions


@st.cache_data
def setup_posescript_split(split_for_research):
    dataIDs = utils.read_json(config.file_posescript_split % split_for_research)
    return dataIDs


@st.cache_data
def setup_posefix_data(data_version):
    # prepare pose identifiers & triplet data
    dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
    triplet_data = data.get_all_posefix_triplets(config.caption_files[data_version][1])
    return dataID_2_pose_info, triplet_data


@st.cache_data
def setup_posefix_split(split_for_research):
    # prepare pose identifiers within the studied split
    dataIDs = utils.read_json(config.file_posefix_split % (split_for_research, 'in'))
    dataIDs += utils.read_json(config.file_posefix_split % (split_for_research, 'out'))
    return dataIDs


@st.cache_data
def precompute_posescript_pose_features(data_version, split_for_research, _model):
    
    batch_size = 32
    
    # create dataset
    dataset = data.PoseScript(version=data_version, split=split_for_research, tokenizer_name=None, num_body_joints=_model.pose_encoder.num_body_joints)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None, shuffle=False,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # infer pose features
    poses_features = torch.zeros(len(dataset), _model.latentD)
    pose_dataIDs = torch.zeros(len(dataset))
    for i, batch in tqdm(enumerate(data_loader)):
        poses = batch['pose'].to(DEVICE)
        with torch.inference_mode():
            pfeat = _model.pose_encoder(poses)
            poses_features[i*batch_size:i*batch_size+len(poses)] = pfeat
        pose_dataIDs[i*batch_size:i*batch_size+len(pfeat)] = batch['data_ids']
               
    return pose_dataIDs.to(torch.int).tolist(), poses_features


@st.cache_data
def precompute_posefix_pair_features(data_version, split_for_research, _model):
    
    batch_size = 32
    
    # create dataset
    dataset = data.PoseFix(version=data_version, split=split_for_research, tokenizer_name=None, num_body_joints=_model.pose_encoder.num_body_joints)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None, shuffle=False,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # infer pose features
    pairs_features = torch.zeros(len(dataset), _model.latentD)
    pair_dataIDs = torch.zeros(len(dataset))
    for i, batch in tqdm(enumerate(data_loader)):
        poses_A = batch['poses_A'].to(DEVICE)
        poses_B = batch['poses_B'].to(DEVICE)
        with torch.inference_mode():
            pfeat = _model.encode_pose_pair(poses_A, poses_B)
            pairs_features[i*batch_size:i*batch_size+len(pfeat)] = pfeat
        pair_dataIDs[i*batch_size:i*batch_size+len(pfeat)] = batch['data_ids']
               
    return pair_dataIDs.to(torch.int).tolist(), pairs_features


@st.cache_data
def precompute_text_features(data_version, split_for_research, _model, tokenizer_name):
    
    batch_size = 32
    
    # create dataset
    if "posescript" in data_version:
        dataset = data.PoseScript(version=data_version, split=split_for_research, tokenizer_name=tokenizer_name, caption_index=0, num_body_joints=_model.pose_encoder.num_body_joints)
    elif "posefix" in data_version:
        dataset = data.PoseFix(version=data_version, split=split_for_research, tokenizer_name=tokenizer_name, caption_index=0, num_body_joints=_model.pose_encoder.num_body_joints)
    else: raise ValueError
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=None, shuffle=False,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # infer pose features
    texts_features = torch.zeros(len(dataset), _model.latentD)
    text_dataIDs = torch.zeros(len(dataset))
    for i, batch in tqdm(enumerate(data_loader)):
        caption_tokens = batch['caption_tokens'].to(DEVICE)
        caption_lengths = batch['caption_lengths'].to(DEVICE)
        caption_tokens = caption_tokens[:,:caption_lengths.max()]
        with torch.inference_mode():
            tfeat = _model.encode_text(caption_tokens, caption_lengths)
            texts_features[i*batch_size:i*batch_size+len(tfeat)] = tfeat
        text_dataIDs[i*batch_size:i*batch_size+len(tfeat)] = batch['data_ids']
               
    return text_dataIDs.to(torch.int).tolist(), texts_features


### Handle data
################################################################################

def get_posescript_datapoint(number, query_type, split_for_research, captions, dataID_2_pose_info, body_model):
    
    # get pose ID from query input
    dataIDs = setup_posescript_split(split_for_research)
    if query_type == "Split index":
        pose_ID = dataIDs[number]
    elif query_type == "ID":
        pose_ID = number

    # get query data
    pose_data, pose_img, default_description = get_posescript_datapoint_from_pid(pose_ID, captions, dataID_2_pose_info, body_model)

    return pose_ID, pose_data, pose_img, default_description


def get_posescript_datapoint_from_pid(pose_ID, captions, dataID_2_pose_info, body_model):

    pose_info = dataID_2_pose_info[str(pose_ID)]
    pose_data = utils.get_pose_data_from_file(pose_info)
    pose_img = utils_visu.image_from_pose_data(pose_data, body_model, color="blue", add_ground_plane=True)
    default_description = captions[pose_ID][0] if pose_ID in captions else "" # # if available, yield the first annotation
    
    return pose_data, pose_img, default_description


def get_posefix_datapoint(number, query_type, split_for_research, triplet_data, pose_pairs, dataID_2_pose_info, body_model):

    # get pair ID from query input
    dataIDs = setup_posefix_split(split_for_research)
    if query_type == "Split index":
        pair_ID = dataIDs[number]
    elif query_type == "ID":
        pair_ID = number

    # get query data
    pid_A, pid_B = pose_pairs[pair_ID]
    default_modifier = triplet_data[pair_ID]["modifier"][0] if pair_ID in triplet_data else "" # if available, yield the first annotation
    in_sequence = dataID_2_pose_info[str(pid_A)][1] == dataID_2_pose_info[str(pid_B)][1]

    # get pose A
    pose_A_info = dataID_2_pose_info[str(pid_A)]
    pose_A_data, rA = utils.get_pose_data_from_file(pose_A_info, output_rotation=True)
    pose_A_img = utils_visu.image_from_pose_data(pose_A_data, body_model, color="grey", add_ground_plane=True, two_views=60)
    pose_A_img = process_img(pose_A_img[0])

    # get pose B
    pose_B_info = dataID_2_pose_info[str(pid_B)]
    pose_B_data = utils.get_pose_data_from_file(pose_B_info, applied_rotation=rA if in_sequence else None)
    pose_B_img = utils_visu.image_from_pose_data(pose_B_data, body_model, color="purple", add_ground_plane=True, two_views=60)
    pose_B_img = process_img(pose_B_img[0])

    return pair_ID, pid_A, pid_B, pose_A_data, pose_B_data, pose_A_img, pose_B_img, default_modifier