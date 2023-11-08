##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
import torch
import roma

import text2pose.config as config
import text2pose.utils as utils


### SETUP
################################################################################

# load data
dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)


### COMPUTE ROTATION CHANGES
################################################################################

rad2deg = lambda x: x*180/torch.pi

# compute rotation changes
rotation_changes = [[0.0, 0.0, 0.0] for _ in pose_pairs]
for pairID in tqdm(range(len(pose_pairs))):

    pidA, pidB = pose_pairs[pairID]
    pose_info_A = dataID_2_pose_info[str(pidA)]
    pose_info_B = dataID_2_pose_info[str(pidB)]

    # 1) normalize the change in rotation (ie. get the difference in
    #    rotation between pose A and pose B)
    pose_data_A, R_norm = utils.get_pose_data_from_file(pose_info_A, output_rotation=True)
    pose_data_B = utils.get_pose_data_from_file(pose_info_B, applied_rotation=R_norm if pose_info_A[1] == pose_info_B[1] else None)
    
    # 2) get the change of rotation (angle in degree) of B wrt A
    # The angle should be positive if turning left (clockwise); and negative
    # otherwise (when turning right)
    r = roma.rotvec_composition((pose_data_B[0:1,:3], roma.rotvec_inverse(pose_data_A[0:1,:3])))
    r = rad2deg(r[0]).tolist() # convert to degrees
    r = [r[0], r[2], -r[1]] # reorient (x,y,z) where x is oriented towards the right and y points up
    rotation_changes[pairID] = r

# save
save_filepath = os.path.join(config.POSEFIX_LOCATION, f"ids_2_rotation_change{config.version_suffix}.json")
utils.write_json(rotation_changes, save_filepath, pretty=True)
print("Save global rotation changes at", save_filepath)