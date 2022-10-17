##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os
import shutil
from pathlib import Path
import time, datetime
import json
import numpy as np

import torch 
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import text2pose.config as config
from text2pose.option import get_args_parser, get_output_dir
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.loss import BBC
from text2pose.utils import MetricLogger, SmoothedValue
from text2pose.retrieval.model_retrieval import PoseText
from text2pose.retrieval.evaluate_retrieval import compute_eval_metrics

os.umask(0x0002)


def main(args):
    
    # check if the model was already trained
    ckpt_fname = os.path.join(args.output_dir, 'checkpoint_last.pth')
    if os.path.isfile(ckpt_fname):
        ckpt = torch.load(ckpt_fname, 'cpu')
        if ckpt["epoch"] == args.epochs - 1:
            print(f'Training already done ({args.epochs} epochs: checkpoint_last.pth)')
            return

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # load data
    generated_pose_samples_path = None # default
    if args.generated_pose_samples:
        generated_pose_samples_model_path = (config.shortname_2_model_path[args.generated_pose_samples]).format(seed=args.seed)
        generated_pose_samples_path = config.generated_pose_path % os.path.dirname(generated_pose_samples_model_path)
    
    print('Load training dataset')
    dataset_train = PoseScript(version=args.dataset, split='train', text_encoder_name=args.text_encoder_name, caption_index='rand', generated_pose_samples_path=generated_pose_samples_path)
    print(dataset_train, len(dataset_train))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print('Load validation dataset')
    dataset_val = PoseScript(version=args.dataset, split='val', text_encoder_name=args.text_encoder_name, caption_index="deterministic-mix", generated_pose_samples_path=generated_pose_samples_path)
    print(dataset_val, len(dataset_val))
    
    # set up models
    print('Load model')
    model = PoseText(text_encoder_name=args.text_encoder_name, latentD=args.latentD)
    model.to(args.device)
    
    # optimizer 
    assert args.optimizer=='Adam'
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # scheduler
    lr_scheduler = None
    if args.lr_scheduler == "stepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												step_size=args.lr_step,
												gamma=args.lr_gamma,
												last_epoch=-1)
   
    # start or resume training
    if os.path.isfile(ckpt_fname): # resume training
        # checkpoint was loaded at the beginning
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt['scheduler'])
        best_score = ckpt['best_score']
        print(f"Resume training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_score = None
        if args.pretrained: # load pretrained weights
            pretrained_path = (config.shortname_2_model_path[args.pretrained]).format(seed=args.seed)
            print(f'Loading pretrained model: {pretrained_path}')
            ckpt = torch.load(pretrained_path, 'cpu')
            model.load_state_dict(ckpt['model'])

    # tensorboard 
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    # training process
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs): 

        # training
        train_stats = one_epoch(
            model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            log_writer=log_writer,
            args=args
        )

        # validation
        val_stats = validate(model,
            dataset=dataset_val,
            device=args.device,
            epoch=epoch,
            log_writer=log_writer)

        # save checkpoint
        tosave = {'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                'args': args,
                'best_score': best_score}
        # ... current checkpoint
        torch.save(tosave, ckpt_fname)
        # ... most recent best model
        if (not best_score) or (val_stats["mRecall"] > best_score):
            best_score = val_stats["mRecall"]
            shutil.copyfile(ckpt_fname, os.path.join(args.output_dir, 'best_model.pth'))

        # tensorboard
        log_writer.flush()

        # save logs
        log_stats = {'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]}
        log_stats.update(**train_stats)
        log_stats.update(**val_stats)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
def one_epoch(model, data_loader, optimizer, epoch, args, log_writer=None):

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    model.train(True)
        
    for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, args.log_step, header)):

        # get data
        poses = item['pose'].to(args.device)
        caption_tokens = item['caption_tokens'].to(args.device)
        caption_lengths = item['caption_lengths'].to(args.device)
        caption_tokens = caption_tokens[:,:caption_lengths.max()] # truncate within the batch, based on the longest text 
        
        # compute scores
        poses_features, texts_features = model(poses, caption_tokens, caption_lengths)
        score_t2p = texts_features.mm(poses_features.t())
        loss = BBC(score_t2p*model.loss_weight)
        loss_value = loss.item()
        
        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loggers
        metric_logger.update( **{'train_loss': loss_value} )
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_writer:
            # We use epoch_1000x as the x-axis in tensorboard.
            # This calibrates different curves when batch size changes.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

def validate(model, dataset, device, epoch=-1, log_writer=None):

    # evaluate model & get metric values
    model.eval()
    recalls, loss_value = compute_eval_metrics(model, dataset, device, compute_loss=True)
    val_stats = {"val_loss": loss_value}
    val_stats.update(recalls)

    # loggers
    if log_writer:
        # We use epoch_1000x as the x-axis in tensorboard.
        # This calibrates different curves when batch size changes.
        epoch_1000x = int((1 + epoch) * 1000)
        log_writer.add_scalar('val/loss', loss_value, epoch_1000x)
        for k,v in recalls.items():
            log_writer.add_scalar(f'val/{k}', v, epoch_1000x)

    print( f"[val] Epoch: [{epoch}]\nStats: " + "  ".join(f"{k}: {v}" for k,v in val_stats.items()) )
    return val_stats


if __name__ == '__main__':
    argparser = get_args_parser()
    args = argparser.parse_args()
    # create path to saving directory
    if args.output_dir=='':
        args.output_dir = get_output_dir(args)
        print(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)