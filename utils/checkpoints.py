import os
import errno
import torch
import shutil
import logging
import inspect

from models import get_model

import pdb

__all__ = ['save_eval_checkpoint', 'save_checkpoint', 
           'load_eval_checkpoint', 'load_checkpoint', 'normalize_module_name']


def normalize_module_name(layer_name):
    """
    Normalize a each module's name in nn.Model
    in case of model was wrapped with DataParallel
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)



def save_eval_checkpoint(model_config: str, model: 'nn.Module', checkpoint_path: str):
    """
    Save the model state dict with all layer unwrapped and 
    pruning masks applied.
    
    Arguments:
        model_config {dict} -- {'arch': arch, 'dataset': dataset}
        path {str} -- path to save wrapped model (e.g.: exps_root/sample_run/run_id)
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    if isinstance(model, torch.nn.DataParallel):
        logging.debug('Was using data parallel')
        model = model.module
    model_state_dict = model.state_dict()
    state_dict = dict()
    state_dict['model_config'] = model_config
    state_dict['model_state_dict'] = model_state_dict
    torch.save(state_dict, os.path.join(checkpoint_path, 'eval_ready_chkpt.pth'))


def load_eval_checkpoint(checkpoint_path: str) -> 'nn.Module':
    """
    Load the evaluation ready model given the chepoint path.
    """
    try:
        state_dict = torch.load(os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))
    except:
        raise IOError(errno.ENOENT, 'Evaluation checkpoint does not exist at', os.path.abspath(checkpoint_path))
    model_config = state_dict['model_config']
    model = get_model(model_config['arch'], model_config['dataset'])
    model.load_state_dict(state_dict['model_state_dict'])
    return model


def save_checkpoint(epoch, model_config, model, optimizer, lr_scheduler,
                    checkpoint_path: str, is_best=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states. 
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
                                     'state_dict': optimizer.state_dict()
                                    }
    checkpoint_dict['lr_scheduler'] = {
                                        'state_dict': lr_scheduler.state_dict()
                                        }

    eval_chkpt_dict = dict()
    eval_chkpt_dict['model_config'] = model_config
    eval_chkpt_dict['model_state_dict'] = model.state_dict()

    path_last = os.path.join(checkpoint_path, f'last_checkpoint.ckpt')
    path_last_eval = os.path.join(checkpoint_path, f'last_checkpoint.pth')
    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}.ckpt')
    path_best = os.path.join(checkpoint_path, 'best_checkpoint.ckpt')
    path_best_eval = os.path.join(checkpoint_path, 'best_checkpoint.pth')
    torch.save(checkpoint_dict, path_last)
    torch.save(eval_chkpt_dict, path_last_eval)
    if is_best:
        print("util - saving best model")
        shutil.copyfile(path_last, path_best)
        shutil.copyfile(path_last_eval, path_best_eval)
    if is_scheduled_checkpoint:
        print("util - saving on schedule")
        shutil.copyfile(path_last, path_regular)


def load_checkpoint(full_checkpoint_path: str, only_model=False):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_dict = torch.load(full_checkpoint_path, map_location='cpu')
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))
    
    try:
        model_config = checkpoint_dict['model_config']
        model = get_model(*model_config.values())
        updated_state_dict = model.state_dict()
        if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
            checkpoint_dict['model_state_dict'] = {normalize_module_name(k): v for k, v in checkpoint_dict['model_state_dict'].items()}
        updated_state_dict.update(checkpoint_dict['model_state_dict'])
        model.load_state_dict(updated_state_dict)
        if only_model:
            return model
        optimizer_dict = checkpoint_dict['optimizer']['state_dict']
        lr_scheduler_dict = checkpoint_dict['lr_scheduler']['state_dict']
        epoch = checkpoint_dict['epoch']
        print('epoch:', epoch)
    except Exception as e:
        raise TypeError(f'Checkpoint file is not valid. {e}')
    
    return epoch + 1, model, optimizer_dict, lr_scheduler_dict
