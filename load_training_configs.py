"""
This script provides functions for loading the optimizer and learning rate scheduler given a config file.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import logging

from optimization.lr_schedulers import CosineLR

from optimization.sam import *
from optimization.topk_cram import *

import pdb



def build_optimizer_from_config(model, optimizer_config):
    optimizer_class = optimizer_config['class']
    restricted_keys = ['class']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k not in restricted_keys}
    optimizer_args['params'] = model.parameters()
    optimizer = globals()[optimizer_class](**optimizer_args)
    return optimizer


def build_lr_scheduler_from_config(optimizer, lr_scheduler_config):
    lr_scheduler_class = lr_scheduler_config['class']
    lr_scheduler_args = {k: v for k, v in lr_scheduler_config.items() if k != 'class'}
    lr_scheduler_args['optimizer'] = optimizer
    lr_scheduler = globals()[lr_scheduler_class](**lr_scheduler_args)
    return lr_scheduler


def build_sam_optimizer_from_config(model, optimizer_config, sam_config, sam_rho=None):
    optimizer_class = optimizer_config['class']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k not in ['class']} 
    sam_class = sam_config['class']
    sam_optim_args = {k: v for k, v in sam_config.items() if k not in ['class']}
    if sam_rho is not None:
        # overwrite the SAM rho value from the command line
        sam_optim_args['rho'] = sam_rho
    sam_optim_args['params'] = model.parameters()
    sam_optim_args['base_optimizer'] = globals()[optimizer_class]
    sam_optimizer = globals()[sam_class](**sam_optim_args, **optimizer_args)
    return sam_optimizer


def build_training_schedule_from_config(model, training_config_dict, sam_rho=None):
    trainer_dict = training_config_dict['training']
    use_sam = False
    if 'SAM' in trainer_dict.keys():
        optimizer = build_sam_optimizer_from_config(model, trainer_dict['optimizer'], 
                                                   trainer_dict['SAM'], sam_rho=sam_rho)
        use_sam = True
        lr_scheduler = build_lr_scheduler_from_config(optimizer.base_optimizer, trainer_dict['lr_scheduler'])
    else:
        optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'])
        lr_scheduler = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
    return optimizer, lr_scheduler, use_sam 



