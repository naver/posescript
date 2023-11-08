##############################################################
## text2pose                                                ##
## Copyright (c) 2022                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
import math
import torch
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.utils as utils


### INPUT
################################################################################

device = 'cpu'


### SETUP
################################################################################

# setup body model
body_model = BodyModel(model_type = config.POSE_FORMAT,
                       bm_fname = config.NEUTRAL_BM,
                       num_betas = config.n_betas)
body_model.eval()
body_model.to(device)

# load data
dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)

# rotation transformation to apply so that the coordinates correspond to what we
# actually visualize (ie. from front view)
rotX = lambda theta: torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])

def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
    return rotMat(theta_rad).mm(values.t()).t()


### COMPUTE COORDINATES
################################################################################

coords = []

# compute all joint coordinates
for dataID in tqdm(range(len(dataID_2_pose_info))):
    
    # load pose data
    pose_info = dataID_2_pose_info[str(dataID)]
    pose = utils.get_pose_data_from_file(pose_info)

    # infer coordinates
    with torch.no_grad():
        j = body_model(**utils.pose_data_as_dict(pose)).Jtr
    j = j.detach().cpu()[0]
    j = transf(rotX, -90, j)

    # store data
    coords.append(j.view(1, -1, 3))
coords = torch.cat(coords)

# save
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, f"ids_2_coords_correct_orient_adapted{config.version_suffix}.pt")
torch.save(coords, save_filepath)
print("Save coordinates at", save_filepath)