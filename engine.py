# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import util.misc as utils
from typing import Iterable

def Encoder_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Encoder Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]
        
        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()

        # print unused parameters
        # print("------------------------------------")
        # print("Unused parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print("------------------------------------")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def preprocess_image_features(model: torch.nn.Module, data_loader: Iterable, device: torch.device, dataset):
    cur = 0
    tot = len(dataset)
    all_features, all_role_tokens = torch.zeros(0, device=device), torch.zeros(0, device=device)
    for batch in data_loader:
        samples, targets = batch
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]
        cur += len(targets)
        print(f"Preprocess [{cur}/{tot}]")
        # model output & calculate loss
        with torch.no_grad():
            outputs = model(samples, targets)
        all_features = torch.cat((all_features, outputs["features"]), dim=0)
        all_role_tokens = torch.cat((all_role_tokens, outputs["role_tokens"]), dim=0)
        # print("all_ft: ",all_features.shape)
        # print("all_RL: ",all_role_tokens.shape)
        
        # print unused parameters
        # print("------------------------------------")
        # print("Unused parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print("------------------------------------")

    return all_features, all_role_tokens


def Decoder_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, preprocessor: torch.nn.Module,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Decoder Epoch: [{}]'.format(epoch)
    print_freq = 10
    current_index = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]
        
        bs = len(targets)
        indices = torch.arange(start=current_index, end=current_index+bs)
        current_index += bs

        # model output & calculate loss
        with torch.no_grad():
            outs = preprocessor(samples, targets)
        features = outs["features"]
        role_tokens = outs["role_tokens"]
        outputs = model(features, role_tokens, idx=indices)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()

        # print unused parameters
        # print("------------------------------------")
        # print("Unused parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print("------------------------------------")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def encoder_evaluate_swig(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
@torch.no_grad()
def decoder_evaluate_swig(model, criterion, preprocessor, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    # current_index = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # bs = len(targets)
        # indices = torch.arange(start=current_index, end=current_index+bs)
        # print("cur idx",current_index)
        # print("batch size",bs)
        # current_index += bs

        # model output & calculate loss
        outs = preprocessor(samples, targets)
        features = outs["features"]
        role_tokens = outs["role_tokens"]
        outputs = model(features, role_tokens, inference=True)
        loss_dict = criterion(outputs, targets, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
