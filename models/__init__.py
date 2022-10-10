import torch

from models.resnet_cifar10 import *
from models.big_resnet_cifar10 import *
from models.vgg_cifar10 import *
from models.mobilenet import *
from models.wide_resnet_cifar import *

from torchvision.models import resnet18, resnet34, resnet50 


CIFAR10_MODELS = ['resnet20', 'resnet32', 'resnet18_cifar_big', 'resnet34_cifar_big', 'resnet50_cifar_big', 'VGG16']
CIFAR100_MODELS = ['wideresnet', 'resnet20', 'resnet32']
IMAGENET_MODELS = ['resnet18', 'resnet34', 'resnet50', 'mobilenet']


def get_model(name, dataset, pretrained=False):
    if (name.startswith('resnet') or name.startswith('VGG')) and dataset=='cifar10':
        return globals()[name]()
    if (name.startswith('resnet') or name.startswith('VGG')) and dataset=='cifar100':
        return globals()[name](num_classes=100)
    if name=='wideresnet' and dataset=='cifar100':
        model = Wide_ResNet(28, 10, 0.3, 100)
        return model
    if 'mobilenet' in name:
        return globals()[name]()
    return globals()[name](pretrained=pretrained)




     


