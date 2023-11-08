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
    --model DescriptionGenerator
    --text_decoder_name "transformer_vocPSA2H2"
    --transformer_mode 'crossattention'
    --decoder_nhead 8 --decoder_nlayers 4
    --latentD 32 --decoder_latentD 512
)

bonus_args=(
)

fid="ret_distilbert_dataPSA2ftPSH2"
pose_generative_model="gen_distilbert_dataPSA2ftPSH2"
textret_model="ret_distilbert_dataPSA2ftPSH2"

pretrained="capgen_CAtransfPSA2H2_dataPSA2" # used only if phase=='finetune'


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

        python generative_caption/train_generative_caption.py --dataset "posescript-A2" \
        "${architecture_args[@]}" \
        "${bonus_args[@]}" \
        --lr 0.0001 --wd 0.0001 --batch_size 128 --seed $seed \
        --epochs 3000 --log_step 100 --val_every 20 \
        --fid $fid --pose_generative_model $pose_generative_model --textret_model $textret_model

    # FINETUNE
    elif [[ "$phase" == *"finetune"* ]]; then

        python generative_caption/train_generative_caption.py --dataset "posescript-H2" \
        "${architecture_args[@]}" \
        "${bonus_args[@]}" \
        --apply_LR_augmentation \
        --lr 0.00001 --wd 0.0001 --batch_size 128 --seed $seed \
        --epochs 2000 --log_step 100 --val_every 20 \
        --fid $fid --pose_generative_model $pose_generative_model --textret_model $textret_model \
        --pretrained $pretrained

    fi

fi


# EVAL QUANTITATIVELY
if [[ "$action" == *"eval"* ]]; then

    experiments=(
        $2
        "GT" "random" "auto_posescript-A2_cap1" # control metrics
    )

    for model_path in "${experiments[@]}"
    do
        echo $model_path
        python generative_caption/evaluate_generative_caption.py --dataset "posescript-H2" \
        --model_path ${model_path} --checkpoint $checkpoint_type \
        --fid $fid --pose_generative_model $pose_generative_model --textret_model $textret_model \
        --split test
    done
fi


# EVAL QUALITATIVELY
if [[ "$action" == *"demo"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one
    streamlit run generative_caption/demo_generative_caption.py -- --model_paths "${experiments[@]}" --checkpoint $checkpoint_type

fi