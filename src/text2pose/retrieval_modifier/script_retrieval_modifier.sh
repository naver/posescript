#!/bin/bash

##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################


##############################################################
# SCRIPT ARGUMENTS

action=$1 # (train|eval|demo)
checkpoint_type="best" # (last|best)

architecture_args=(
    --model PairText
    --latentD 512
    --text_encoder_name 'distilbertUncased' --transformer_topping "avgp"
    # --text_encoder_name 'glovebigru_vocPFAHPP'
)

loss_args=(
    --retrieval_loss 'symBBC'
)

bonus_args=(
)

pretrained="modret_distilbert_dataPFA" # used only if phase=='finetune'


##############################################################
# EXECUTE

# TRAIN
if [[ "$action" == *"train"* ]]; then

    phase=$2 # (pretrain|finetune)
    echo "NOTE: Expecting as argument the training phase. Got: $phase"
    seed=$3
    echo "NOTE: Expecting as argument the seed value. Got: $seed"
    
    # PRETRAIN 
    if [[ "$phase" == *"pretrain"* ]]; then

        python retrieval_modifier/train_retrieval_modifier.py --dataset "posefix-A" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --lr_scheduler "stepLR" --lr 0.00005 --lr_step 400 --lr_gamma 0.5 \
        --log_step 20 --val_every 20 \
        --batch_size 128 --epochs 400 --seed $seed

    # FINETUNE
    elif [[ "$phase" == *"finetune"* ]]; then

        python retrieval_modifier/train_retrieval_modifier.py --dataset "posefix-H" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --apply_LR_augmentation \
        --lr_scheduler "stepLR" --lr 0.00005 --lr_step 40 --lr_gamma 0.5 \
        --batch_size 128 --epochs 50 --seed $seed \
        --pretrained $pretrained

    fi

fi


# EVAL QUANTITATIVELY
if [[ "$action" == *"eval"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one

    for model_path in "${experiments[@]}"
    do
        echo $model_path
        python retrieval_modifier/evaluate_retrieval_modifier.py --dataset "posefix-H" \
        --model_path ${model_path} --checkpoint $checkpoint_type \
        --split test
    done
fi


# EVAL QUALITATIVELY
if [[ "$action" == *"demo"* ]]; then

    experiment=$2 # only one at a time
    streamlit run retrieval_modifier/demo_retrieval_modifier.py -- --model_path $experiment --checkpoint $checkpoint_type

fi