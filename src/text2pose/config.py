##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
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

POSEFIX_LOCATION = MAIN_DIR + '/data/PoseFix/posefix_release'
POSESCRIPT_LOCATION = MAIN_DIR + '/data/PoseScript/posescript_release'
POSEMIX_LOCATION = MAIN_DIR + '/data/posemix'

version_suffix = "_100k" # to be used for pipeline-related data (coords, rotation change, babel labels)
file_pose_id_2_dataset_sequence_and_frame_index = f"{POSESCRIPT_LOCATION}/ids_2_dataset_sequence_and_frame_index_100k.json" 
file_pair_id_2_pose_ids = f"{POSEFIX_LOCATION}/pair_id_2_pose_ids.json"
file_posescript_split = f"{POSESCRIPT_LOCATION}/%s_ids_100k.json" # %s --> (train|val|test)
file_posefix_split = f"{POSEFIX_LOCATION}/%s_%s_sequence_pair_ids.json" # %s %s --> (train|val|test), (in|out)


### pose config ----------------------------------------------------------------

POSE_FORMAT = 'smplh'
SMPLH_BODY_MODEL_PATH = MAIN_DIR + '/data/smplh_amass_body_models'
NEUTRAL_BM = f'{SMPLH_BODY_MODEL_PATH}/neutral/model.npz'
NB_INPUT_JOINTS = 52 # default value used when initializing modules, unless specified otherwise
n_betas = 16

SMPLX_BODY_MODEL_PATH = MAIN_DIR + '/data/smpl_models' # should contain "smplx/SMPLX_NEUTRAL.(npz|pkl)"

PID_NAN = -99999 # pose fake IDs, used for empty poses

### pose data ------------------------------------------------------------------

AMASS_FILE_LOCATION = MAIN_DIR + f"/data/AMASS/{POSE_FORMAT}/"
supported_datasets = {"AMASS":AMASS_FILE_LOCATION}

BABEL_LOCATION = MAIN_DIR + "/data/BABEL/babel_v1.0_release"

generated_pose_path = '%s/generated_poses/posescript_version_{data_version}_split_{split}_gensamples.pth' # %s is for the model directory (obtainable with shortname_2_model_path)


### text data ------------------------------------------------------------------

MAX_TOKENS = 500 # defined here because it depends on the provided data (only affects the glovebigru configuration)

vocab_files = {
    # IMPORTANT: do not use "_" symbols in the keys of this dictionary
    # IMPORTANT: vocabs overlap, but the order of the tokens is not the same; a
    #            model trained with one vocab can't be finetuned with another
    #            without risk
    "vocPSA2H2": "vocab_posescript_6293_auto100k.pkl",
    "vocPFAHPP": "vocab_posefix_6157_pp4284_auto.pkl",
    "vocMixPSA2H2PFAHPP": "vocab_posemix_PS6193_PF6157pp4284.pkl",
}

caption_files = {
    # <shortname for the dataset>: (<minimum number of texts per item>, <list of files providing the texts>)
    "posescript-A2": (3, [f"{POSESCRIPT_LOCATION}/posescript_auto_100k.json"]),
    "posescript-H2": (1, [f"{POSESCRIPT_LOCATION}/posescript_human_6293.json"]),
    "posefix-A": (3, [f"{POSEFIX_LOCATION}/posefix_auto_135305.json"]),
    "posefix-PP": (1, [f"{POSEFIX_LOCATION}/posefix_paraphrases_4284.json"]), # average is 2 texts/item (min 1, max 6)
    "posefix-H": (1, [f"{POSEFIX_LOCATION}/posefix_human_6157.json"]),
    "posefix-HPP": (1, [f"{POSEFIX_LOCATION}/posefix_human_6157.json", f"{POSEFIX_LOCATION}/posefix_paraphrases_4284.json"]), # 1 text/item at min, because paraphrases are essentially for the train set
    "posemix-PSH2-PFHPP": (1, [f"{POSESCRIPT_LOCATION}/posescript_human_6293.json", f"{POSEFIX_LOCATION}/posefix_human_6157.json", f"{POSEFIX_LOCATION}/posefix_paraphrases_4284.json"]),
}

# data cache
dirpath_cache_dataset = MAIN_DIR + "/dataset_cache"
cache_file_path = {
    "posescript":'%s/PoseScript_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl' % dirpath_cache_dataset,
    "posefix":'%s/PoseFix_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl' % dirpath_cache_dataset,
    "posemix":'%s/PoseMix_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl' % dirpath_cache_dataset,
    "posestream":'%s/PoseStream_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl' % dirpath_cache_dataset,
}


################################################################################
# Model cache
################################################################################

GLOVE_DIR = MAIN_DIR + '/tools/torch_models/glove' # or None
TRANSFORMER_CACHE_DIR = MAIN_DIR + '/tools/huggingface_models'
SELFCONTACT_ESSENTIALS_DIR = MAIN_DIR + '/tools/selfcontact/essentials'


################################################################################
# Shortnames to checkpoint paths
################################################################################
# Shortnames are used to refer to:
# - pretrained models
# - models that generated pose files
# - models used for evaluation (fid, recall, reconstruction, r-precision...)

# shortnames for models are expected to be the same accross seed values;
# model paths should contain a specific seed_value field instead of the actual seed value
normalize_model_path = lambda model_path, seed_value: "/".join(model_path.split("/")[:-2]) + f"/seed{seed_value}/"+ model_path.split("/")[-1]

# shortname & model paths are stored in shortname_2_model_path.json (which can be updated by some scripts)
try:
    with open("shortname_2_model_path.txt", "r") as f:
        # each line has the following format: <shortname><4 spaces><model path with a {seed} field>
        shortname_2_model_path = [l.split("    ") for l in f.readlines() if len(l.strip())]
        shortname_2_model_path = {l[0]:normalize_model_path(l[1].strip(), '{seed}') for l in shortname_2_model_path}
except FileNotFoundError:
    # print("File not found: shortname_2_model_path.txt - Please ensure you are launching operations from the right directory.")
    pass # this file may not even be needed; subsequent errors can be expected otherwise


################################################################################
# Evaluation
################################################################################

# NOTE: models used to compute the fid should be specified in `shortname_2_model_path`

k_recall_values = [1, 5, 10]
nb_sample_reconstruction = 30
k_topk_reconstruction_values = [1, 6] # keep the top-1 and the top-N/4 where N is the nb_sample_reconstruction
k_topk_r_precision = [1,2,3]
r_precision_n_repetitions = 10
sample_size_r_precision = 32


################################################################################
# Visualization settings
################################################################################

meshviewer_size = 1600


if __name__=="__main__":
    import sys
    try:
        # if the provided model shortname is registered, return the complete model path (with the provided seed value)
        if sys.argv[1] in shortname_2_model_path:
            print(shortname_2_model_path[sys.argv[1]].format(seed=sys.argv[2]))
    except IndexError:
        # clean shortname_2_model_path.txt
        update = []
        for k,p in shortname_2_model_path.items():
            update.append(f"{k}    {p.format(seed=0)}\n")
        with open("shortname_2_model_path.txt", "w") as f:
            f.writelines(update)
        print("Cleaned shortname_2_model_path.txt (unique entries with seed 0).")