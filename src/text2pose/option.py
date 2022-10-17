##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os
import argparse

import text2pose.config as config

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--dataset', default='posescript-H1', type=str, help='training dataset')
    parser.add_argument('--generated_pose_samples', default=None, help="shortname for the model that generated the pose files to be used (full path registered in config.py")
    
    # model architecture
    parser.add_argument('--model', default='CondTextPoser', choices=("PoseText", "CondTextPoser"), help='name of the model')
    parser.add_argument('--text_encoder_name', default='glovebigru_vocA1H1', help='name of the text encoder (should not have any "_" symbol in it; when needed, auxiliary vocab reference (as in config.vocab_files) must be attached to the actual text encoder name after a "_" character')
    parser.add_argument('--latentD', type=int, help='dimension of the latent space')

    # loss
    parser.add_argument('--wloss_kld', default=0.2, type=float, help='weight for KLD losses')
    parser.add_argument('--wloss_v2v', default=4.0, type=float, help='weight for the reconstruction loss term: vertice positions')
    parser.add_argument('--wloss_rot', default=2.0, type=float, help='weight for the reconstruction loss term: joint rotations')
    parser.add_argument('--wloss_jts', default=2.0, type=float, help='weight for the reconstruction loss term: joint positions')
    parser.add_argument('--wloss_kldnpmul', default=0.0, type=float, help='weight for KL(Np, N0), if 0, this KL is not used for training')
    parser.add_argument('--wloss_kldntmul', default=0.0, type=float, help='weight for KL(Nt, N0), if 0, this KL is not used for training')
    
    # training
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, help="learning rate scheduler")
    parser.add_argument('--lr_step', default=20, type=float, help='step for the learning rate scheduler')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='gamma for the learning rate scheduler')

    parser.add_argument('--pretrained', default='', type=str, help="shortname for the model to be used as pretrained model (full path registered in config.py")

    # validation / evaluation
    parser.add_argument('--fid', type=str, help='version of the fid for the model')
    
    # utils
    parser.add_argument('--output_dir', default='', help='automatically defined if empty')
    parser.add_argument('--subdir', default='', type=str, help='intermediate directory to group models on disk (eg. to separate several sets of experiments)')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproduceability')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for the dataloader')
    parser.add_argument('--saving_ckpt_step', default=500, type=int, help='number of epochs before creating a persistent checkpoint')    
    parser.add_argument('--log_step', default=20, type=int, help='number of batchs before printing and recording the logs')

    return parser


def get_output_dir(args):
    """
    Automatically create a unique reference path based on the selected options if output_dir==''.
    """
    # utils
    add_flag = lambda t, a: t if a else '' # `t` may have a symbol '_' at the beginning or at the end

    # define strings to provide details about different aspects of the model (architecture, training)
    architecture_details = f'{args.model}_textencoder-{args.text_encoder_name}' + f"_latentD{args.latentD}"
    dataset_details = f'train-{args.dataset}'

    # specific to the generative model
    if args.model == "CondTextPoser":
        loss_details = f'wloss_kld{args.wloss_kld}_v2v{args.wloss_v2v}_rot{args.wloss_rot}_jts{args.wloss_jts}_kldnpmul{args.wloss_kldnpmul}_kldntmul{args.wloss_kldntmul}'
        optimization_details = f'{args.optimizer}_lr{args.lr}_wd{args.wd}'

    # specific to the retrieval model
    elif args.model == "PoseText":
        dataset_details += add_flag(f'_gensamples-{args.generated_pose_samples}', args.generated_pose_samples)
        loss_details = 'BBC' # default
        optimization_details = f'{args.optimizer}_lr{args.lr}' + add_flag(f'_{args.lr_scheduler}_lrstep{args.lr_step}_lrgamma{args.lr_gamma}', args.lr_scheduler=="stepLR")
    
    training_details = f'B{args.batch_size}_'+ optimization_details + add_flag(f'_pretrained_{args.pretrained}', args.pretrained)

    return os.path.join(config.GENERAL_EXP_OUTPUT_DIR, args.subdir, architecture_details, dataset_details, loss_details, training_details, f'seed{args.seed}')


if __name__=="__main__":
    # return the complete model path based on the provided arguments
    # do not add print anything else, as the output is provided to a bash variable for further use (training/eval scripts)
    argparser = get_args_parser()
    args = argparser.parse_args()
    if args.output_dir=='':
        args.output_dir = get_output_dir(args)
    print(args.output_dir)