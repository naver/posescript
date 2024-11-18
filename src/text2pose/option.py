##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import argparse

import text2pose.config as config

def none_or_int(value):
    if value == 'None' or value is None:
        return None
    return int(value)


def get_args_parser():
    """
    To know which options are taken into account for the studied model/configuration (training etc.), check whether the option is taken into account in the naming of the model, in get_output_dir() 
    """
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--dataset', default='posescript-H2', type=str, help='training dataset')
    parser.add_argument('--data_size', default=None, type=none_or_int, help="(reduced) size of the training data")
    parser.add_argument('--generated_pose_samples', default=None, help="shortname for the model that generated the pose files to be used (full path registered in config.py")
    parser.add_argument('--pose_stream', action="store_true", help="load a large set of pose-only data and feed it to the pose auto-encoder")

    # model architecture
    parser.add_argument('--model', default='CondTextPoser', choices=("PoseText", "PairText", "CondTextPoser", "PoseBGenerator", "DescriptionGenerator", "FeedbackGenerator"), help='name of the model')
    parser.add_argument('--num_body_joints', default=config.NB_INPUT_JOINTS, type=int, help="Number of body joints to consider as input/output to the model.")
    parser.add_argument('--text_encoder_name', default='glovebigru_vocPSA2H2', help='name of the text encoder (should not have any "_" symbol in it; when needed, auxiliary vocab reference (as in config.vocab_files) must be attached to the actual text encoder name after a "_" character')
    parser.add_argument('--text_decoder_name', default='transformer_vocPFAH', help='name of the text decoder (should not have any "_" symbol in it; when needed, auxiliary vocab reference (as in config.vocab_files) must be attached to the actual text decoder name after a "_" character')
    parser.add_argument('--transformer_topping', help='method for obtaining the sentence embedding (transformer-based text encoders)') # "avgp", "augtokens"
    parser.add_argument('--latentD', type=int, help='dimension of the latent space')
    parser.add_argument('--comparison_latentD', type=int, help='dimension of the latent space in which belongs the output of the comparison module (can be a sequence of tokens of such dimension)')
    parser.add_argument('--decoder_latentD', default=768, type=int, help='dimension of the latent space for the decoder')
    parser.add_argument('--decoder_nhead', default=4, type=int, help='number of heads for a transformer')
    parser.add_argument('--decoder_nlayers', default=4, type=int, help='number of layers for a transformer')
    parser.add_argument('--special_text_latentD', type=int, help='dimension of the textual latent space (if specified, overrides `latent_D` for the textual encoder')
    parser.add_argument('--comparison_module_mode', default='tirg', help='module to fuse the embeddings of poses A and poses B to further generate textual feedback')
    parser.add_argument('--correction_module_mode', default='tirg', help='module to fuse the embeddings of poses A and of the modifiers to further generate poses B')
    parser.add_argument('--transformer_mode', default='crossattention', help='how to inject the multimodal data into the text decoder (prompt, crossattention...)')

    # loss
    parser.add_argument('--retrieval_loss', default='BBC', type=str, help='contrastive loss to train the retrieval model')
    parser.add_argument('--wloss_kld', default=0.2, type=float, help='weight for KLD losses')
    parser.add_argument('--kld_epsilon', default=0.0, type=float, help='minimum value for each component of the KLD losses')
    parser.add_argument('--wloss_v2v', default=0.0, type=float, help='weight for the reconstruction loss term: vertice positions')
    parser.add_argument('--wloss_rot', default=0.0, type=float, help='weight for the reconstruction loss term: joint rotations')
    parser.add_argument('--wloss_jts', default=0.0, type=float, help='weight for the reconstruction loss term: joint positions')
    parser.add_argument('--wloss_kldnpmul', default=0.0, type=float, help='weight for KL(Np, N0), if 0, this KL is not used for training')
    parser.add_argument('--wloss_kldntmul', default=0.0, type=float, help='weight for KL(Nt, N0), if 0, this KL is not used for training')
    
    # training (optimization)
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lrtextmul', default=1.0, type=float, help='learning rate multiplier for the text encoder')
    parser.add_argument('--lrposemul', default=1.0, type=float, help='learning rate multiplier for the pose model')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, help="learning rate scheduler")
    parser.add_argument('--lr_step', default=20, type=float, help='step for the learning rate scheduler')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='gamma for the learning rate scheduler')
    # training (data augmentation)
    parser.add_argument('--apply_LR_augmentation', action='store_true', help='randomly flip "right" and "left" in data during training')
    parser.add_argument('--copyB2A', action="store_true", help='when doing copy augmentation, copy pose B to A (conversely to pose A to pose B)')
    parser.add_argument('--copy_augmentation', default=0, type=float, help='randomly empty the text cue, and make the input and expected output pose the same')
    # training (pretraining/finetuning)
    parser.add_argument('--pretrained', default='', type=str, help="shortname for the model to be used as pretrained model (full path registered in config.py)")

    # validation / evaluation
    parser.add_argument('--val_every', default=1, type=int, help='validate every N epochs (only works if the learning scheduler is None; currently used when training the generative model only)')
    parser.add_argument('--fid', type=str, help='version of the fid for the model')
    parser.add_argument('--pose_generative_model', type=str, help='version of the pose model for evaluation of generated text (can be None)')
    parser.add_argument('--textret_model', type=str, help='version of the text-to-pose(s) retrieval model for evaluation of generated text (can be None)')

    # utils
    parser.add_argument('--output_dir', default='', help='automatically defined if empty')
    parser.add_argument('--subdir', default='', type=str, help='intermediate directory to group models on disk (eg. to separate several sets of experiments)')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproduceability')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for the dataloader')
    parser.add_argument('--saving_ckpt_step', default=10000, type=int, help='number of epochs before creating a persistent checkpoint')    
    parser.add_argument('--log_step', default=20, type=int, help='number of batchs before printing and recording the logs')

    return parser


def get_output_dir(args):
    """
    Automatically create a unique reference path based on the selected options if output_dir==''.
    """
    # utils
    add_flag = lambda t, a: t if a else '' # `t` may have a symbol '_' at the beginning or at the end

    # define strings to provide details about different aspects of the model (architecture, training)
    architecture_details = args.model + add_flag(f'_{args.num_body_joints}bodyjts', args.num_body_joints!=52) + \
        f'_textencoder-{args.text_encoder_name}' + \
        add_flag(f'-{args.transformer_topping}', args.text_encoder_name.split("_")[0] not in ["glovebigru"]) + \
        f"_latentD{args.latentD}"
    
    dataset_details = f'train-{args.dataset}' + add_flag(f"_rsize{args.data_size}", args.data_size)
    optimization_details = f'{args.optimizer}_lr{args.lr}' +  add_flag(f'textmul{args.lrtextmul}posemul{args.lrposemul}', args.lrtextmul!=1.0 or args.lrposemul!=1.0)

    # specific to the text-to-pose retrieval model
    if args.model == "PoseText":
        dataset_details += add_flag(f'_gensamples-{args.generated_pose_samples}', args.generated_pose_samples)
        loss_details = args.retrieval_loss
        optimization_details += add_flag(f'_{args.lr_scheduler}_lrstep{args.lr_step}_lrgamma{args.lr_gamma}', args.lr_scheduler=="stepLR")

    # specific to the text-to-pose-pair retrieval model
    elif args.model == "PairText":
        loss_details = args.retrieval_loss
        optimization_details += add_flag(f'_{args.lr_scheduler}_lrstep{args.lr_step}_lrgamma{args.lr_gamma}', args.lr_scheduler=="stepLR")
    
    # specific to the text-to-pose generative model
    elif args.model == "CondTextPoser":
        loss_details = f'wloss_kld{args.wloss_kld}_v2v{args.wloss_v2v}_rot{args.wloss_rot}_jts{args.wloss_jts}_kldnpmul{args.wloss_kldnpmul}_kldntmul{args.wloss_kldntmul}'
        optimization_details += f'_wd{args.wd}'

    # specific to the pose editing model
    elif args.model == "PoseBGenerator":
        dataset_details += add_flag(f'_poseStream', args.pose_stream)
        architecture_details +=  add_flag(f'_tlatentD{args.special_text_latentD}', args.special_text_latentD) + \
                                f'_{args.correction_module_mode}'
        loss_details = f'wloss_kld{args.wloss_kld}_v2v{args.wloss_v2v}_rot{args.wloss_rot}_jts{args.wloss_jts}' + \
                        add_flag(f'_kldnpmul{args.wloss_kldnpmul}', args.wloss_kldnpmul) + \
                        add_flag(f'_kldeps{args.kld_epsilon}', args.kld_epsilon>0)
        optimization_details += f'_wd{args.wd}'

    # specific to the pose-to-text generation model
    elif args.model == "DescriptionGenerator":
        architecture_details = args.model + add_flag(f'_{args.num_body_joints}bodyjts', args.num_body_joints!=52) + \
            f'_textdecoder-{args.text_decoder_name}' + \
            add_flag(f'_mode-{args.transformer_mode}', args.transformer_mode!="crossattention") + \
            add_flag(f'_dlatentD{args.decoder_latentD}', args.decoder_latentD!=768) + \
            add_flag(f'_dnhead{args.decoder_nhead}', args.decoder_nhead!=4) + \
            add_flag(f'_dnlayers{args.decoder_nlayers}', args.decoder_nlayers!=4) + \
            f'_latentD{args.latentD}'
        loss_details = 'crossentropy' # default
        optimization_details += f'_wd{args.wd}'

    # specific to the pose-pair-to-text generation model
    elif args.model == "FeedbackGenerator":
        architecture_details = args.model + add_flag(f'_{args.num_body_joints}bodyjts', args.num_body_joints!=52) + \
            f'_textdecoder-{args.text_decoder_name}' + \
            add_flag(f'_mode-{args.transformer_mode}', args.transformer_mode!="crossattention") + \
            f'_{args.comparison_module_mode}' + \
            add_flag(f'-clatentD{args.comparison_latentD}', args.comparison_latentD!=args.latentD) + \
            add_flag(f'_dlatentD{args.decoder_latentD}', args.decoder_latentD!=768) + \
            add_flag(f'_dnhead{args.decoder_nhead}', args.decoder_nhead!=4) + \
            add_flag(f'_dnlayers{args.decoder_nlayers}', args.decoder_nlayers!=4) + \
            f'_latentD{args.latentD}'
        loss_details = 'crossentropy' # default
        optimization_details += f'_wd{args.wd}'

    training_details = f'B{args.batch_size}_'+ optimization_details + \
                        add_flag('_LRflip', args.apply_LR_augmentation) + \
                        add_flag(f'_copy{args.copy_augmentation}', args.copy_augmentation) + \
                        add_flag('-B2A', args.copyB2A) + \
                        add_flag(f'_pretrained_{args.pretrained}', args.pretrained)

    return os.path.join(config.GENERAL_EXP_OUTPUT_DIR, args.subdir, architecture_details, dataset_details, loss_details, training_details, f'seed{args.seed}')


if __name__=="__main__":
    # return the complete model path based on the provided arguments
    # do not add print anything else, if the output is to provided to a bash variable for further use
    argparser = get_args_parser()
    args = argparser.parse_args()
    if args.output_dir=='':
        args.output_dir = get_output_dir(args)
    print(args.output_dir)