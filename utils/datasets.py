"""
Dataset loading utilities
"""

import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, Subset, TensorDataset


import pdb

DATASETS_NAMES = ['imagenet', 'cifar10', 'cifar100']

__all__ = ["get_datasets", "imagenet_calibration_datasets"]


# see post: https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/2
# this is to avoid wasting memory by loading the same dataset twice, when using train/val splits, with different data aug applied
class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)


def get_train_val_split(full_train_dataset, train_transform, test_transform, train_ratio):
    #np.random.seed(42)
    # not necessary to set the random seed, as it is already set in the main script
    lazy_train_dset = MyLazyDataset(full_train_dataset, transform=train_transform)
    lazy_val_dset = MyLazyDataset(full_train_dataset, transform=test_transform)
        
    total_train_size = len(full_train_dataset)
    random_indices = np.random.permutation(total_train_size)
    train_size = int(np.floor(train_ratio * total_train_size))
    train_indices, val_indices = random_indices[:train_size], random_indices[train_size:]

    train_dataset = Subset(lazy_train_dset, train_indices)
    val_dataset = Subset(lazy_val_dset, val_indices)
    return train_dataset, val_dataset


def __dataset_factory(dataset):
    return globals()[f'{dataset}_get_datasets']


def get_datasets(dataset, dataset_dir, **kwargs):
    datasets_fn = __dataset_factory(dataset)
    return datasets_fn(dataset_dir, **kwargs)


def cifar10_get_datasets(data_dir, use_val_split=True):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                              (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    if use_val_split:
        # set aside 10% of the train set for validation purposes
        full_train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                              download=True)
        train_dataset, val_dataset = get_train_val_split(full_train_dataset, train_transform,
                                                         test_transform, train_ratio=0.9)
        return train_dataset, test_dataset, val_dataset    

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)
    return train_dataset, test_dataset, None



def cifar100_get_datasets(data_dir, use_val_split=True):
    means = (0.5071, 0.4867, 0.4408)
    stds = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])
    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_transform)
    if use_val_split:
        # set aside 10% of the train set for validation purposes
        full_train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                               download=True)
        train_dataset, val_dataset = get_train_val_split(full_train_dataset, train_transform,
                                                         test_transform, train_ratio=0.9)
        return train_dataset, test_dataset, val_dataset    
    
    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    return train_dataset, test_dataset, None



def imagenet_get_datasets(data_dir, use_val_split=True):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          ])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                         ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    if use_val_split:
        # set aside 10% of the train set for validation purposes
        full_train_dataset = datasets.ImageFolder(train_dir)
        train_dataset, val_dataset = get_train_val_split(full_train_dataset, train_transform,
                                                         test_transform, train_ratio=0.9)
 
        return train_dataset, test_dataset, val_dataset    

    return train_dataset, test_dataset, None



def imagenet_calibration_datasets(data_dir_calib, data_dir_val, calib_size=1, use_data_aug=True):
    calib_data = f'calib_{calib_size}'
    train_dir = os.path.join(data_dir_calib, calib_data)
    test_dir = os.path.join(data_dir_val, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)


    return train_dataset, test_dataset
