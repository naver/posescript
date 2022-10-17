#!/bin/bash

##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################


###############################################################################
# SCRIPT ARGUMENTS

while getopts a:c:s:g:j: flag
do
    case "${flag}" in
        a) action=${OPTARG};; # action (train|eval|train-eval)
        c) config=${OPTARG};; # configuration of the experiment
        s) run_id_seed=${OPTARG};; # (optional) seed value
        g) config_generated_pose_samples=${OPTARG};; # (optional) shortname for the generative model that generated the poses to be used for training (contribute to the model configuration)
        j) eval_generated_pose_samples=${OPTARG};; # (optional) shortname for the generative model that generated the poses to be used for evaluation (independent from the model configuration)
    esac
done

# default values
: ${action="train-eval"}
: ${run_id_seed=0}
: ${config_generated_pose_samples=""}
: ${eval_generated_pose_samples=""}


###############################################################################
# CONFIGURATION OF THE EXPERIMENT

# global/default configuration
args=(
    --model 'PoseText'
    --text_encoder_name 'glovebigru_vocA1H1'
    --latentD 512
    --lr_scheduler "stepLR"
    --lr 0.0002 --lr_step 20 --lr_gamma 0.5
    --batch_size 32
    --seed ${run_id_seed}
)

# specific configuration
if [ "$config" == "ret_glovebigru_vocA1H1_dataA1" ]; then
    dataset='posescript-A1'

elif [ "$config" == "ret_glovebigru_vocA1H1_dataH1" ]; then
    dataset='posescript-H1'

elif [ "$config" == "ret_glovebigru_vocA1H1_dataA1ftH1" ]; then
    dataset='posescript-H1'
    args+=(--pretrained "ret_glovebigru_vocA1H1_dataA1")

else
    echo "Provided config (-c ${config}) is unknown."
fi
args+=(--dataset $dataset)

# generated pose samples, if provided, defining the model & its training 
args_genconf=()
if [ ! "$config_generated_pose_samples" == "" ]; then
    args_genconf=(--generated_pose_samples $config_generated_pose_samples)
    config="${config}___${config_generated_pose_samples}"
fi

# generated_pose_samples, if provided, for evaluation
args_geneval=()
if [ ! "$eval_generated_pose_samples" == "" ]; then
	args_geneval=(--generated_pose_samples $eval_generated_pose_samples)
fi

# utils
model_dir_path=$(python option.py "${args[@]}" "${args_genconf[@]}")
model_path="${model_dir_path}/best_model.pth"


###############################################################################
# TRAIN

if [[ "$action" == *"train"* ]]; then

    python retrieval/train_retrieval.py --epochs 500 "${args[@]}" "${args_genconf[@]}"
    
    # store the shortname and path to the retrieval model in config files
    echo "${config}    ${model_path}" >> shortname_2_model_path.txt
fi


###############################################################################
# EVAL QUANTITATIVELY

if [[ "$action" == *"eval"* ]]; then

    python retrieval/evaluate_retrieval.py --model_path $model_path \
    --dataset $dataset --split 'test' \
    "${args_geneval[@]}"
fi

