#!/bin/bash

##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################


##############################################################
# SCRIPT ARGUMENTS

action=$1 # (train|eval|demo)
eval_data_version="H2"
checkpoint_type="best" # (last|best)

architecture_args=(
    --model CondTextPoser
    --latentD 32
    --text_encoder_name 'distilbertUncased' --transformer_topping "avgp"
    # --text_encoder_name 'glovebigru_vocPSA2H2'
)

loss_args=(
    --wloss_v2v 4 --wloss_rot 2 --wloss_jts 2
    --wloss_kld 0.1 --wloss_kldnpmul 0.01 --wloss_kldntmul 0.01
)

bonus_args=(
)

fid="ret_distilbert_dataPSA2ftPSH2"

pretrained="gen_distilbert_dataPSA2" # used only if phase=='finetune'


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

        python generative/train_generative.py --dataset "posescript-A2" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --lr 1e-05 --wd 0.0001 --batch_size 128 --seed $seed \
        --epochs 5000 --log_step 20 --val_every 20 \
        --fid $fid

    # FINETUNE
    elif [[ "$phase" == *"finetune"* ]]; then

        python generative/train_generative.py --dataset "posescript-H2" \
        "${architecture_args[@]}" \
        "${loss_args[@]}" \
        "${bonus_args[@]}" \
        --apply_LR_augmentation \
        --lrposemul 0.1 --lrtextmul 1 \
        --lr 1e-05 --wd 0.0001 --batch_size 128 --seed $seed \
        --epochs 2000 --val_every 10 \
        --fid $fid \
        --pretrained $pretrained

    fi

fi


# EVAL QUANTITATIVELY
if [[ "$action" == *"eval"* ]]; then

    eval_type=$2 # (regular|generate_poses|RG|GRa|GRb)
    echo "NOTE: Expecting as argument the evaluation phase. Got: $eval_type"

    # regular (fid, elbo)
    if [[ "$eval_type" == "regular" ]]; then

        # parse additional input
        model_path=$3
        echo "NOTE: Expecting as argument the path to the model to evaluate. Got: $model_path"

        python generative/evaluate_generative.py \
        --dataset "posescript-$eval_data_version" --split "test" \
        --model_path ${model_path} --checkpoint $checkpoint_type \
        --fid $fid


    # generate poses for R/G & G/R
    elif [[ "$eval_type" == "generate_poses" ]]; then

        # parse additional input
        model_path=$3
        echo "NOTE: Expecting as argument the path to the model for which to generate sample poses. Got: $model_path"
        seed=$4
        echo "NOTE: Expecting as argument the seed value. Got: $seed"

        mp=${model_path::-1}$seed/checkpoint_best.pth
        python generative/generate_poses.py --model_path $mp


    # eval R/G
    elif [[ "$eval_type" == "RG" ]]; then

        # parse additional input
        seed=$3
        echo "NOTE: Expecting as argument the seed value. Got: $seed"
        model_shortname=$4
        echo "NOTE: Expecting as argument the shortname of the generative model to evaluate. Got: $model_shortname"
        retrieval_model_shortname=$5
        echo "NOTE: Expecting as argument the shortname of the retrieval model used for R/G. Got: $retrieval_model_shortname"
    
        # evaluate generated poses with the retrieval model
        retrieval_model_path=$(python config.py $retrieval_model_shortname $seed)
        python retrieval/evaluate_retrieval.py \
        --dataset "posescript-"$eval_data_version --split 'test' \
        --model_path $retrieval_model_path --checkpoint $checkpoint_type \
        --generated_pose_samples $model_shortname


    # eval G/R, 1st step
    elif [[ "$eval_type" == "GRa" ]]; then

        # parse additional input
        seed=$3
        echo "NOTE: Expecting as argument the seed value. Got: $seed"
        model_shortname=$4
        echo "NOTE: Expecting as argument the shortname of the generative model to evaluate. Got: $model_shortname"
        
        # define specificities for the new retrieval model
        args_ret=(
            --model 'PoseText' --latentD 512
            --lr_scheduler "stepLR" --lr 0.0002  --lr_gamma 0.5
            --text_encoder_name 'distilbertUncased' --transformer_topping "avgp"
            # --text_encoder_name 'glovebigru_vocPSA2H2'
            )
        pret='ret_distilbert_dataPSA2'
        if [[ "$eval_data_version" == "A2" ]]; then
            args_ret+=(--dataset "posescript-A2"
                        --lr_step 400
                        --batch_size 512
                        --epochs 1000
                        )
        elif [[ "$eval_data_version" == "H2" ]]; then
            args_ret+=(--dataset "posescript-H2"
                        --lr_step 40
                        --batch_size 32
                        --epochs 200
                        --pret $pret
                        --apply_LR_augmentation
                        )
        fi
        echo "NOTE: Expecting in the script the spec of the new retrieval model to train on the generated poses. Got:" "${args_ret[@]}"

        # train a new retrieval model with the generated poses 
        python retrieval/train_retrieval.py "${args_ret[@]}" --seed $seed \
        --generated_pose_samples $model_shortname

        echo "IMPORTANT: please create an entry in shortname_2_model_path.txt for this newly trained retrieval model (providing a shortname and the path to the model), as it will be needed for the next evaluation step of this generative model ($model_shortname)"


    # eval G/R, 2st step
    elif [[ "$eval_type" == "GRb" ]]; then

        # parse additional input
        seed=$3
        echo "NOTE: Expecting as argument the seed value. Got: $seed"
        spec_retrieval_model_shortname=$4
        echo "NOTE: Expecting as argument the shortname of the retrieval model used for G/R. Got: $spec_retrieval_model_shortname"
    
        # evaluate the retrieval model trained on generated poses, on the original poses
        spec_retrieval_model_path=$(python config.py $spec_retrieval_model_shortname $seed)
        python retrieval/evaluate_retrieval.py \
        --dataset "posescript-"$eval_data_version --split 'test' \
        --model_path $spec_retrieval_model_path --checkpoint $checkpoint_type
    
    fi
fi


# EVAL QUALITATIVELY
if [[ "$action" == *"demo"* ]]; then

    shift; experiments=( "$@" ) # gets all the arguments starting from the 2nd one
    streamlit run generative/demo_generative.py -- --model_paths "${experiments[@]}" --checkpoint $checkpoint_type

fi