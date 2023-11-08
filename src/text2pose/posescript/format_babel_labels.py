##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
import json
import numpy as np
import pickle
from tabulate import tabulate

import text2pose.config as config
import text2pose.utils as utils


### SETUP
################################################################################

# load BABEL
l_babel_dense_files = ['train', 'val', 'test']
l_babel_extra_files = ['extra_train', 'extra_val']

babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(os.path.join(config.BABEL_LOCATION, file+'.json')))
    
for file in l_babel_extra_files:
    babel[file] = json.load(open(os.path.join(config.BABEL_LOCATION, file+'.json')))    

# load PoseScript
dataID_2_pose_info = utils.read_json(config.file_pose_id_2_dataset_sequence_and_frame_index)

# AMASS/BABEL path adaptation
amass_to_babel_subdir = {
    'ACCAD': 'ACCAD/ACCAD',
    'BMLhandball': '', # not available
    'BMLmovi': 'BMLmovi/BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub/BioMotionLab_NTroje',
    'CMU': 'CMU/CMU',
    'DFaust_67': 'DFaust67/DFaust_67',
    'DanceDB': '', # not available
    'EKUT': 'EKUT/EKUT',
    'Eyes_Japan_Dataset': 'EyesJapanDataset/Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva/HumanEva',
    'KIT': 'KIT/KIT',
    'MPI_HDM05': 'MPIHDM05/MPI_HDM05',
    'MPI_Limits': 'MPILimits/MPI_Limits',
    'MPI_mosh': 'MPImosh/MPI_mosh',
    'SFU': 'SFU/SFU',
    'SSM_synced': 'SSMsynced/SSM_synced',
    'TCD_handMocap': 'TCDhandMocap/TCD_handMocap',
    'TotalCapture': 'TotalCapture/TotalCapture',
    'Transitions_mocap': 'Transitionsmocap/Transitions_mocap',
}


### GET LABELS
################################################################################

def get_babel_label(amass_rel_path, frame_id):

    # get path correspondance in BABEL
    dname = amass_rel_path.split('/')[0]
    bname = amass_to_babel_subdir[dname]
    if bname == '': 
        return '__'+dname+'__'
    babel_rel_path = '/'.join([bname]+amass_rel_path.split('/')[1:])

    # look for babel annotations
    babelfs = []
    for f in babel.keys():
        for s in babel[f].keys():
            if babel[f][s]['feat_p'] == babel_rel_path:
                babelfs.append((f,s))

    if len(babelfs) == 0:
        return None

    # convert frame id to second
    seqdata = np.load(os.path.join(config.AMASS_FILE_LOCATION, amass_rel_path))
    framerate = seqdata['mocap_framerate']
    t = frame_id / framerate

    # read babel annotations
    labels = []
    for f,s in babelfs:
        if not 'frame_ann' in  babel[f][s]:
            continue
        if babel[f][s]['frame_ann'] is None: 
            continue
        babel_annots = babel[f][s]['frame_ann']['labels']
        for i in range(len(babel_annots)):
            if t >= babel_annots[i]['start_t'] and t <= babel_annots[i]['end_t']:
                labels.append( (babel_annots[i]['raw_label'], babel_annots[i]['proc_label'], babel_annots[i]['act_cat']) )

    return labels

# gather labels for all poses in PoseScript that come from AMASS
babel_labels_for_posescript = {}
for dataID in tqdm(dataID_2_pose_info):
    pose_info = dataID_2_pose_info[dataID]
    if pose_info[0] == "AMASS":
        babel_labels_for_posescript[dataID] = get_babel_label(pose_info[1], pose_info[2])

# display some stats
table = []
table.append(['None', sum([v is None for v in babel_labels_for_posescript.values()])])
table.append(['BMLhandball', sum([v=='__BMLhandball__' for v in babel_labels_for_posescript.values()])])
table.append(['DanceDB', sum([v=='__DanceDB__' for v in babel_labels_for_posescript.values()])])
table.append(['0 label', sum([ (isinstance(v,list) and len(v)==0) for v in babel_labels_for_posescript.values()])])
table.append(['None label', sum([ (isinstance(v,list) and len(v)>=1 and v[0][0] is None) for v in babel_labels_for_posescript.values()])])
table.append(['1 label', sum([ (isinstance(v,list) and len(v)==1 and v[0][0] is not None) for v in babel_labels_for_posescript.values()])])
table.append(['>1 label',sum([ (isinstance(v,list) and len(v)>=2 and v[0][0] is not None) for v in babel_labels_for_posescript.values()])])
print(tabulate(table, headers=["Label", "Number of poses"]))

# save
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, f"babel_labels_for_posescript{config.version_suffix}.pkl")
with open(save_filepath, 'wb') as f:
    pickle.dump(babel_labels_for_posescript, f)
print("Saved", save_filepath)