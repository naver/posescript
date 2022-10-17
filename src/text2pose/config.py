##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# This file serves to store global config parameters and paths
# (those may change depending on the user, provided data, trained models...)

# default
import os
MAIN_DIR = os.path.realpath(__file__)
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR)))


################################################################################
# Output dir for experiments
################################################################################

GENERAL_EXP_OUTPUT_DIR = MAIN_DIR + '/experiments'


################################################################################
# Data
################################################################################

POSESCRIPT_LOCATION = MAIN_DIR + '/data/PoseScript/posescript_release'

### pose config
SMPL_BODY_MODEL_PATH = MAIN_DIR + '/data/smplh_amass_body_models'
SMPLH_NEUTRAL_BM = f'{SMPL_BODY_MODEL_PATH}/neutral/model.npz'
n_betas = 16
NB_INPUT_JOINTS = 52


### pose data
AMASS_FILE_LOCATION = MAIN_DIR + '/data/AMASS/smplh/'
supported_datasets = {"AMASS":AMASS_FILE_LOCATION}

BABEL_LOCATION = MAIN_DIR + '/data/BABEL/babel_v1.0_release'

# saved in the directory of the corresponding model
generated_pose_path = '%s/generated_poses/posescript_version_{data_version}_split_{split}_gensamples.pth' # %s is for the model directory (obtainable with shortname_2_model_path)


### text data
MAX_TOKENS = 500 # defined here because it depends on the provided data (only affects the glovebigru configuration)

vocab_files = {
    # IMPORTANT: do not use "_" symbols in the keys of this dictionary
    "vocA1H1": "vocab3893.pkl",
}

caption_files = {
    "posescript-A1": [f"automatic_{x}.json" for x in ["A", "B", "C", "D", "E", "F"]],
    "posescript-H1": "human3893.json",
}


# data cache
cache_file_path = MAIN_DIR + '/dataset_cache/PoseScript_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl'


################################################################################
# Model cache
################################################################################

GLOVE_DIR = MAIN_DIR + '/tools/torch_models/glove' # or None


################################################################################
# Shortnames to checkpoint paths
################################################################################
# Shortnames are used to refer to:
# - pretrained models
# - models that generated pose files
# - retrieval models to serve for fid

# shortnames for models are expected to be the same accross seed values;
# model paths should contain a specific seed_value field instead of the actual seed value
normalize_model_path = lambda model_path, seed_value: "/".join(model_path.split("/")[:-2]) + f"/seed{seed_value}/"+ model_path.split("/")[-1]

# shortname & model paths are stored in shortname_2_model_path.json (which can be updated by some scripts)
with open("shortname_2_model_path.txt", "r") as f:
    # each line has the following format: <shortname><4 spaces><model path with a {seed} field>
    shortname_2_model_path = [l.split("    ") for l in f.readlines() if len(l.strip())]
    shortname_2_model_path = {l[0]:normalize_model_path(l[1].strip(), '{seed}') for l in shortname_2_model_path}

################################################################################
# Evaluation
################################################################################

# NOTE: models used to compute the fid should be specified in `shortname_2_model_path`

k_recall_values = [1, 5, 10]


if __name__=="__main__":
    # if the provided model shortname is registered, return the complete model path (with the provided seed value)
    import sys
    if sys.argv[1] in shortname_2_model_path:
        print(shortname_2_model_path[sys.argv[1]].format(seed=sys.argv[2]))