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

action=$1
checkpoint_type="best" # (last|best)

architecture_args=(
    --model PoseBGenerator
    --correction_module_mode "tirg"
    --latentD 32 --special_text_latentD 128
    --text_encoder_name 'distilbertUncased' --transformer_topping "avgp"
    # --text_encoder_name 'glovebigru_vocPFAHPP'
)

loss_args=(
    --wloss_kld 1.0
    --kld_epsilon 10.0
    --wloss_v2v 1.0 --wloss_rot 1.0 --wloss_jts 1.0
)

bonus_args=(
)

fid="ret_distilbert_dataPSA2ftPSH2"

pretrained="b_gen_distilbert_dataPFA" # used only if phase=='finetune'


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

        python generative_B/train_generative_B.py --dataset "posefix-A" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --lr 0.00001 --wd 0.0001 --batch_size 128 --seed $seed \
        --epochs 5000 --log_step 100 --val_every 20 \
        --fid $fid

    # FINETUNE
    elif [[ "$phase" == *"finetune"* ]]; then

        python generative_B/train_generative_B.py --dataset "posefix-HPP" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --apply_LR_augmentation \
        --lrposemul 0.1 --lrtextmul 1 \
        --lr 0.000001 --wd 0.00001 --batch_size 128 --seed $seed \
        --epochs 5000 --log_step 20 --val_every 20 \
        --fid $fid \
        --pretrained $pretrained

    fi

fi


# EVAL QUANTITATIVELY
if [[ "$action" == *"eval"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one

    for model_path in "${experiments[@]}"
    do
        echo $model_path
        python generative_B/evaluate_generative_B.py --dataset "posefix-H" \
        --model_path ${model_path} --checkpoint $checkpoint_type \
        --fid $fid \
        --split test
        # --special_eval
    done
fi


# EVAL QUALITATIVELY
if [[ "$action" == *"demo"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one    
    streamlit run generative_B/demo_generative_B.py -- --model_paths "${experiments[@]}" --checkpoint $checkpoint_type

fi