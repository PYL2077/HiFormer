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
import os
from xml.sax.handler import all_features
import numpy as np
import torch
import datasets
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from datasets import build_dataset, build_processed_dataset
from engine import preprocess_neighbors
from engine import leaf_evaluate_swig, evaluate_swig
from engine import leaf_train_one_epoch, root_train_one_epoch
from models import build_leaf_model, build_root_model
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set Grounded Situation Recognition Transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--leaf_batch_size', default=16, type=int)
    parser.add_argument('--root_batch_size', default=4, type=int)
    parser.add_argument('--leaf_epochs', default=0, type=int)
    parser.add_argument('--root_epochs', default=0, type=int)

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--num_enc_layers', default=6, type=int,
                        help="Number of encoding layers in HiFormer")
    parser.add_argument('--num_dec_layers', default=5, type=int,
                        help="Number of decoding layers in HiFormer")
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
    parser.add_argument('--leaf_start_epoch', default=0, type=int, metavar='N',
                        help='leaf start epoch')
    parser.add_argument('--root_start_epoch', default=0, type=int, metavar='N',
                        help='root start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--leaf_saved_model', default='HiFormer/leaf_checkpoint.pth',
                        help='path where saved leaf model is')
    parser.add_argument('--root_saved_model', default='HiFormer/root_checkpoint.pth',
                        help='path where saved root model is')    
    parser.add_argument('--load_saved_leaf', default=False, type=bool)
    parser.add_argument('--load_saved_root', default=False, type=bool)
    parser.add_argument('--preprocess', default=False, type=bool)
    parser.add_argument('--images_per_segment', default=9463, type=int)
    parser.add_argument('--images_per_eval_segment', default=12600, type=int)


    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

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
        from datasets.swig import collater, processed_collater
    else:
        assert False, f"dataset {args.dataset_file} is not supported now"

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)
    # build Leaf Transformer model
    leaf_model, leaf_criterion = build_leaf_model(args)
    leaf_model.to(device)
    leaf_model_without_ddp = leaf_model
    if args.load_saved_leaf == True:
        leaf_checkpoint = torch.load(args.leaf_saved_model, map_location='cpu')
        leaf_model.load_state_dict(leaf_checkpoint["leaf_model"])
    if args.distributed:
        leaf_model = torch.nn.parallel.DistributedDataParallel(leaf_model, device_ids=[args.gpu])
        leaf_model_without_ddp = leaf_model.module
    num_leaf_parameters = sum(p.numel() for p in leaf_model.parameters() if p.requires_grad)
    print('number of Leaf Transformer parameters:', num_leaf_parameters)
    leaf_param_dicts = [
        {"params": [p for n, p in leaf_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in leaf_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    leaf_optimizer = torch.optim.AdamW(leaf_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    leaf_lr_scheduler = torch.optim.lr_scheduler.StepLR(leaf_optimizer, args.lr_drop)
    
    if args.load_saved_leaf == True:
        leaf_optimizer.load_state_dict(leaf_checkpoint["leaf_optimizer"])
        leaf_lr_scheduler.load_state_dict(leaf_checkpoint["leaf_lr_scheduler"])
        args.leaf_start_epoch = leaf_checkpoint["leaf_epoch"] + 1

    # build Root Transformer Model
    root_model, root_criterion = build_root_model(args)
    root_model.to(device)
    root_model_without_ddp = root_model
    if args.load_saved_root == True:
        root_checkpoint = torch.load(args.root_saved_model, map_location='cpu')
        root_model.load_state_dict(root_checkpoint["root_model"])
    if args.distributed:
        root_model = torch.nn.parallel.DistributedDataParallel(root_model, device_ids=[args.gpu])
        root_model_without_ddp = root_model.module
    num_root_parameters = sum(p.numel() for p in root_model.parameters() if p.requires_grad)
    print('number of Root Transformer parameters:', num_root_parameters)
    root_param_dicts = [
        {"params": [p for n, p in root_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in root_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    root_optimizer = torch.optim.AdamW(root_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    root_lr_scheduler = torch.optim.lr_scheduler.StepLR(root_optimizer, args.lr_drop)
    if args.load_saved_root == True:
        """
        root_optimizer.load_state_dict(root_checkpoint["root_optimizer"])
        root_lr_scheduler.load_state_dict(root_checkpoint["root_lr_scheduler"])
        """
        args.root_start_epoch = root_checkpoint["root_epoch"] + 1

    # Dataset Sampler
    # For Leaf Transformer
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
    # For preprocessing
    if args.preprocess == True:
        preprocess_sampler_train = torch.utils.data.RandomSampler(dataset_train)
        preprocess_sampler_val = torch.utils.data.RandomSampler(dataset_val)
        preprocess_sampler_test = torch.utils.data.RandomSampler(dataset_test)

    output_dir = Path(args.output_dir)
    # dataset loader
    # For Leaf Transformer
    if not args.test and not args.dev:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.leaf_batch_size, drop_last=True)
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
    # For Preprocessing
    if args.preprocess == True:
        batch_preprocess_sampler_train = torch.utils.data.BatchSampler(preprocess_sampler_train, args.leaf_batch_size, drop_last=False)
        batch_preprocess_sampler_val = torch.utils.data.BatchSampler(preprocess_sampler_val, args.leaf_batch_size, drop_last=False)
        batch_preprocess_sampler_test = torch.utils.data.BatchSampler(preprocess_sampler_test, args.leaf_batch_size, drop_last=False)
        
        preprocess_data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_train)
        preprocess_data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_val)
        preprocess_data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_test)


    # use saved model for evaluation (using dev set or test set)
    if args.dev or args.test:
        leaf_checkpoint = torch.load(args.leaf_saved_model, map_location='cpu')
        leaf_model.load_state_dict(leaf_checkpoint["leaf_model"])
        root_checkpoint = torch.load(args.root_saved_model, map_location='cpu')
        root_model.load_state_dict(root_checkpoint["root_model"])
        # build dataset
        if args.dev:
            with open("__storage__/valDict.json") as val_json:
                val_dict = json.load(val_json)
            processed_dataset_val = build_processed_dataset(image_set='val', args=args, neighbors_dict=val_dict)
            if args.distributed:
                sampler_val = DistributedSampler(processed_dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(processed_dataset_val)
            data_loader = DataLoader(processed_dataset_val, num_workers=args.num_workers,
                                     drop_last=False, collate_fn=processed_collater,
                                     sampler=sampler_val)
        elif args.test:
            with open("__storage__/testDict.json") as test_json:
                test_dict = json.load(test_json)
            processed_dataset_test = build_processed_dataset(image_set='test', args=args, neighbors_dict=test_dict)
            if args.distributed:
                sampler_test = DistributedSampler(processed_dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(processed_dataset_test)
            data_loader = DataLoader(processed_dataset_test, num_workers=args.num_workers,
                                     drop_last=False, collate_fn=processed_collater,
                                     sampler=sampler_test)
            

        test_stats = evaluate_swig(leaf_model, root_model, root_criterion,
                                   data_loader, device, args.output_dir)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return None
    
    if not os.path.exists("__storage__"):
        os.mkdir("__storage__")
    # train HiFormer Leaf Transformer
    print("Start training HiFormer Leaf Transformer at epoch ",args.leaf_start_epoch)
    start_time = time.time()
    max_test_verb_acc_top1 = 4
    for epoch in range(args.leaf_start_epoch, args.leaf_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = leaf_train_one_epoch(leaf_model, leaf_criterion, data_loader_train, leaf_optimizer, 
                                      device, epoch, args.clip_max_norm)
        leaf_lr_scheduler.step()

        # evaluate
        test_stats = leaf_evaluate_swig(leaf_model, leaf_criterion, data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'leaf_epoch': epoch,
                     'num_leaf_parameters': num_leaf_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'leaf_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'leaf_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'leaf_model': leaf_model_without_ddp.state_dict(),
                                      'leaf_optimizer': leaf_optimizer.state_dict(),
                                      'leaf_lr_scheduler': leaf_lr_scheduler.state_dict(),
                                      'leaf_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Preprocess
    torch.cuda.empty_cache()
    if args.preprocess == True:
        preprocess_neighbors(leaf_model, preprocess_data_loader_train,
                             "train", device, args.images_per_segment)
        preprocess_neighbors(leaf_model, preprocess_data_loader_val,
                             "val", device, args.images_per_eval_segment)
        preprocess_neighbors(leaf_model, preprocess_data_loader_test,
                             "test", device, args.images_per_eval_segment)
    if args.root_start_epoch >= args.root_epochs:
        return None
    
    # build Root Transformer dataset
    with open("__storage__/trainDict.json") as train_json:
        train_dict = json.load(train_json)
    with open("__storage__/valDict.json") as val_json:
        val_dict = json.load(val_json)
    processed_dataset_train = build_processed_dataset("train", args, neighbors_dict=train_dict)
    processed_dataset_val = build_processed_dataset("val", args, neighbors_dict=val_dict)
    # build Root Transformer dataset sampler
    if args.distributed:
        sampler_train = DistributedSampler(processed_dataset_train)
        sampler_val = DistributedSampler(processed_dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(processed_dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(processed_dataset_val)
    
    # build Root Transformer dataset loader
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.root_batch_size, drop_last=True)
    data_loader_train = DataLoader(processed_dataset_train, num_workers=args.num_workers,
                                   collate_fn=processed_collater, batch_sampler=batch_sampler_train)
    data_loader_val = DataLoader(processed_dataset_val, num_workers=args.num_workers,
                                 drop_last=False, collate_fn=processed_collater, sampler=sampler_val)

    # Train HiFormer Root Transformer
    print("Start training HiFormer Root Transformer at epoch ",args.root_start_epoch)
    start_time = time.time()
    max_test_verb_acc_top1 = 43
    for epoch in range(args.root_start_epoch, args.root_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = root_train_one_epoch(root_model, root_criterion, leaf_model,
                                              data_loader_train, root_optimizer, 
                                              device, epoch, args.images_per_segment,
                                              args.clip_max_norm)
        root_lr_scheduler.step()

        # evaluate
        test_stats = evaluate_swig(leaf_model, root_model, root_criterion,
                                           data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'root_epoch': epoch,
                     'num_root_parameters': num_root_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'root_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'root_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'root_model': root_model_without_ddp.state_dict(),
                                      'root_optimizer': root_optimizer.state_dict(),
                                      'root_lr_scheduler': root_lr_scheduler.state_dict(),
                                      'root_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('HiFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
