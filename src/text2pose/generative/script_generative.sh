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

while getopts a:c:s: flag
do
    case "${flag}" in
        a) action=${OPTARG};; # action (train|eval|train-eval)
        c) config=${OPTARG};; # configuration of the experiment
        s) run_id_seed=${OPTARG};; # (optional) seed value
    esac
done

# default values
: ${action="train-eval"}
: ${run_id_seed=0}

###############################################################################
# CONFIGURATION OF THE EXPERIMENT

# global/default configuration
args=(
    --model 'CondTextPoser'
    --text_encoder_name 'glovebigru_vocA1H1'
    --latentD 32
    --wloss_v2v 4 --wloss_rot 2 --wloss_jts 2
    --wloss_kld 0.2 --wloss_kldnpmul 0.02
    --lr 0.0001 --wd 0.0001
    --batch_size 32
    --seed ${run_id_seed}
)
nb_epoch=2000

# specific configuration
if [ "$config" == "gen_glovebigru_vocA1H1_dataA1" ]; then
    dataset='posescript-A1'
    ret_model_for_recall='ret_glovebigru_vocA1H1_dataA1'
    fid='ret_glovebigru_vocA1H1_dataA1'

elif [ "$config" == "gen_glovebigru_vocA1H1_dataH1" ]; then
    dataset='posescript-H1'
    ret_model_for_recall='ret_glovebigru_vocA1H1_dataH1'
    fid='ret_glovebigru_vocA1H1_dataH1'

elif [ "$config" == "gen_glovebigru_vocA1H1_dataA1ftH1" ]; then
    dataset='posescript-H1'
    args+=( --pretrained "gen_glovebigru_vocA1H1_dataA1"
            --lr 1e-05)
    ret_model_for_recall='ret_glovebigru_vocA1H1_dataA1ftH1'
    fid='ret_glovebigru_vocA1H1_dataA1ftH1'

else
    echo "Provided config (-c ${config}) is unknown."
fi
args+=(--dataset $dataset)

# utils
model_dir_path=$(python option.py "${args[@]}")
model_path="${model_dir_path}/checkpoint_$((${nb_epoch}-1)).pth" # used for evaluation


###############################################################################
# TRAIN

if [[ "$action" == *"train"* ]]; then

    echo "Now training the retrieval model ${fid} to compute the FID."
    bash retrieval/script_retrieval.sh -a "train-eval" -s "${run_id_seed}" -c "${fid}"
        
    echo "Now training the generative model."
    python generative/train_generative.py --epochs ${nb_epoch} "${args[@]}" --fid $fid

    # store the shortname and path to the retrieval model in config files
    echo "${config}    ${model_path}" >> shortname_2_model_path.txt

fi


###############################################################################
# EVAL QUANTITATIVELY

if [[ "$action" == *"eval"* ]]; then

    echo "Now computing the FID & ELBO metrics."
    python generative/evaluate_generative.py --dataset $dataset \
    --model_path $model_path --fid $fid --split 'test'

    echo "Now generating pose samples."
    python generative/generate_poses.py --model_path $model_path

    echo "Now computing the mRecall R/G metric (first training a retrieval model on the original poses, then evaluating it on the generated pose samples)."
    bash retrieval/script_retrieval.sh -a "train" -s "${run_id_seed}" -c "${ret_model_for_recall}"
    bash retrieval/script_retrieval.sh -a "eval" -s "${run_id_seed}" -c "${ret_model_for_recall}" -j "${config}"

    echo "Now computing the mRecall G/R metric (first training a retrieval model on the generated pose samples, then evaluating it on original poses)."
    bash retrieval/script_retrieval.sh -a "train" -s "${run_id_seed}" -c "${ret_model_for_recall}" -g "${config}"
    bash retrieval/script_retrieval.sh -a "eval" -s "${run_id_seed}" -c "${ret_model_for_recall}" -g "${config}"

fi
