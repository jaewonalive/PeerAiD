import os

import numpy as np
import torch
from torchvision import datasets, transforms
from TinyImageNetDataset import *



def get_transform(dataset):
    if dataset == 'tinyimagenet':
        image_size = 64
    else:
        image_size = 32

    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataset(dataset, data_path, download=True):
    train_transform, test_transform = get_transform(dataset)

    if dataset == 'cifar10':
        image_size = (3, 32, 32)
        n_classes = 10
        train_set = datasets.CIFAR10(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (3, 32, 32)
        n_classes = 100
        train_set = datasets.CIFAR100(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(data_path, train=False, download=download, transform=test_transform)

    elif dataset == 'tinyimagenet':
        image_size = (3, 64, 64)
        n_classes = 200

        data_dir = os.path.join(data_path, 'tiny-imagenet-200')
        # train_set = datasets.ImageFolder(train_dir, transform=train_transform)

        train_set = TinyImageNetDataset(data_dir,
                                    preload=False,
                                    mode='train',
                                    transform=train_transform)


        # test_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
        # test_set = datasets.ImageFolder(test_dir, transform=test_transform)


        test_set = TinyImageNetDataset(data_dir,
                                    preload=False,
                                    mode='val',
                                    transform=test_transform)


    else:
        raise NotImplementedError()

    return train_set, test_set, image_size, n_classes