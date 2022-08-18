# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from xml.sax.handler import all_features
import numpy as np
import torch
import datasets
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from datasets import build_dataset
from engine import preprocess_image_features
from engine import encoder_evaluate_swig, decoder_evaluate_swig
from engine import Encoder_train_one_epoch, Decoder_train_one_epoch
from models import build_encoder_model, build_decoder_model
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set Grounded Situation Recognition Transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--encoder_epochs', default=0, type=int)
    parser.add_argument('--decoder_epochs', default=0, type=int)

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--num_enc_layers', default=6, type=int,
                        help="Number of encoding layers in GSRFormer")
    parser.add_argument('--num_dec_layers', default=5, type=int,
                        help="Number of decoding layers in GSRFormer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Loss coefficients
    parser.add_argument('--noun_loss_coef', default=2, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_conf_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=5, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--swig_path', type=str, default="SWiG")
    parser.add_argument('--dev', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    # Etc...
    parser.add_argument('--inference', default=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--encoder_start_epoch', default=0, type=int, metavar='N',
                        help='encoder start epoch')
    parser.add_argument('--decoder_start_epoch', default=0, type=int, metavar='N',
                        help='decoder start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_saved_model', default='GSRFormer/encoder_checkpoint.pth',
                        help='path where saved encoder model is')
    parser.add_argument('--decoder_saved_model', default='GSRFormer/decoder_checkpoint.pth',
                        help='path where saved decoder model is')    
    parser.add_argument('--load_saved_encoder', default=False, type=bool)
    parser.add_argument('--load_saved_decoder', default=False, type=bool)


    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def trim_dataset(dataset, step):
    return Subset(dataset, list(range(0, len(dataset), step)))

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # check dataset
    if args.dataset_file == "swig":
        from datasets.swig import collater
    else:
        assert False, f"dataset {args.dataset_file} is not supported now"

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()
    if not args.test:
        dataset_val = build_dataset(image_set='val', args=args)
    else:
        dataset_test = build_dataset(image_set='test', args=args)
    
    # build model
    encoder_model, encoder_criterion = build_encoder_model(args)
    encoder_model.to(device)
    encoder_model_without_ddp = encoder_model

    if args.load_saved_encoder == True:
        encoder_checkpoint = torch.load(args.encoder_saved_model, map_location='cpu')
        encoder_model.load_state_dict(encoder_checkpoint["encoder_model"])

    if args.distributed:
        encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model, device_ids=[args.gpu])
        encoder_model_without_ddp = encoder_model.module
    num_encoder_parameters = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)

    print('number of encoder parameters:', num_encoder_parameters)

    param_dicts = [
        {"params": [p for n, p in encoder_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in encoder_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    # optimizer & LR scheduler
    encoder_optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, args.lr_drop)
    
    if args.load_saved_encoder == True:
        encoder_optimizer.load_state_dict(encoder_checkpoint["encoder_optimizer"])
        encoder_lr_scheduler.load_state_dict(encoder_checkpoint["encoder_lr_scheduler"])
        args.encoder_start_epoch = encoder_checkpoint["encoder_epoch"] + 1

    # dataset sampler
    if not args.test and not args.dev:
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.dev:
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        elif args.test:
            if args.distributed:
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    output_dir = Path(args.output_dir)
    # dataset loader
    if not args.test and not args.dev:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                    collate_fn=collater, batch_sampler=batch_sampler_train)
        data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                    drop_last=False, collate_fn=collater, sampler=sampler_val)
    else:
        if args.dev:
            data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                        drop_last=False, collate_fn=collater, sampler=sampler_val)
        elif args.test:
            data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers,
                                        drop_last=False, collate_fn=collater, sampler=sampler_test)
    
    # use saved model for evaluation (using dev set or test set)
    """
    if args.dev or args.test:
        checkpoint = torch.load(args.saved_model, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        encoder_model.load_state_dict(checkpoint["encoder_model"])

        if args.dev:
            data_loader = data_loader_val 
        elif args.test:
            data_loader = data_loader_test

        test_stats = evaluate_swig(model, criterion, data_loader, device, args.output_dir)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return None
    """

    # train GSRFormer Encoder
    print("Start training GSRFormer Encoder")
    start_time = time.time()
    max_test_verb_acc_top1 = 43
    for epoch in range(args.encoder_start_epoch, args.encoder_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = Encoder_train_one_epoch(encoder_model, encoder_criterion, data_loader_train, encoder_optimizer, 
                                      device, epoch, args.clip_max_norm)
        encoder_lr_scheduler.step()

        # evaluate
        test_stats = encoder_evaluate_swig(encoder_model, encoder_criterion, data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'encoder_epoch': epoch,
                     'num_encoder_parameters': num_encoder_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'encoder_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'encoder_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'encoder_model': encoder_model_without_ddp.state_dict(),
                                      'encoder_optimizer': encoder_optimizer.state_dict(),
                                      'encoder_lr_scheduler': encoder_lr_scheduler.state_dict(),
                                      'encoder_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Preprocess
    all_features, all_role_tokens = preprocess_image_features(encoder_model, data_loader_train, device, dataset_train)

    # Build Decoder Model
    decoder_model, decoder_criterion = build_decoder_model(all_features, all_role_tokens, args)
    decoder_model.to(device)
    decoder_model_without_ddp = decoder_model

    if args.load_saved_decoder == True:
        decoder_checkpoint = torch.load(args.decoder_saved_model, map_location='cpu')
        decoder_model.load_state_dict(decoder_checkpoint["decoder_model"])

    if args.distributed:
        decoder_model = torch.nn.parallel.DistributedDataParallel(decoder_model, device_ids=[args.gpu])
        decoder_model_without_ddp = decoder_model.module
    num_decoder_parameters = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
    print('number of decoder parameters:', num_decoder_parameters)


    param_dicts = [
        {"params": [p for n, p in decoder_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in decoder_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    decoder_optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, args.lr_drop)

    if args.load_saved_decoder == True:
        decoder_optimizer.load_state_dict(decoder_checkpoint["decoder_optimizer"])
        decoder_lr_scheduler.load_state_dict(decoder_checkpoint["decoder_lr_scheduler"])
        args.decoder_start_epoch = decoder_checkpoint["decoder_epoch"] + 1

    

    # Train GSRFormer Decoder
    print("Start training GSRFormer Decoder")
    start_time = time.time()
    max_test_verb_acc_top1 = 43
    for epoch in range(args.decoder_start_epoch, args.decoder_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = Decoder_train_one_epoch(decoder_model, decoder_criterion, encoder_model,
                                              data_loader_train, decoder_optimizer, 
                                              device, epoch, args.clip_max_norm)
        decoder_lr_scheduler.step()

        # evaluate
        test_stats = decoder_evaluate_swig(decoder_model, decoder_criterion, encoder_model,
                                           data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'decoder_epoch': epoch,
                     'num_decoder_parameters': num_decoder_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'decoder_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'decoder_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'decoder_model': decoder_model_without_ddp.state_dict(),
                                      'decoder_optimizer': decoder_optimizer.state_dict(),
                                      'decoder_lr_scheduler': decoder_lr_scheduler.state_dict(),
                                      'decoder_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GSRFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)