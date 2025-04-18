"""Implements MNIST DataModule."""

import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from .scale_mnist import ScaleMNIST
from .custom_sampler import CustomRandomSampler
from sklearn.model_selection import train_test_split
from scale_eq.utils.core_utils import AddUniformNoise, AddCircularShift


def None_func():
    return None


augment_map = {"add_noise": AddUniformNoise,
               "circular_shift": AddCircularShift, "None": None_func}


def get_augmentation(name):
    global augment_map
    return augment_map[name]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.num_classes = 10

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.mnist_train = MNIST(
                self.data_dir, train=True, transform=self.transform)
            self.mnist_val = MNIST(self.data_dir, train=False,
                                   transform=self.transform)

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False,
                                    transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)


class ScaleMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, training_size=None, test_size=50000, train_scales=[1], test_scales=[1],
                 data_resize_mode='None', data_downscale_mode='ideal', batch_size=32, num_workers=4, train_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.training_size = training_size
        self.test_size = test_size
        self.train_scales = train_scales
        self.test_scales = test_scales
        self.data_resize_mode = data_resize_mode
        self.data_downscale_mode = data_downscale_mode
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.train_transform = train_transform
        self.num_classes = 10

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:

            dataset_train = MNIST(self.data_dir, train=True,
                                  download=True, transform=transforms.ToTensor())
            dataset_test = MNIST(self.data_dir, train=False,
                                 download=True, transform=transforms.ToTensor())
            concat_dataset = ConcatDataset([dataset_train, dataset_test])

            total = len(dataset_train) + len(dataset_test)
            self.val_size = total - self.training_size - self.test_size
            train_val_size = self.training_size + self.val_size

            labels = [el[1] for el in concat_dataset]

            train_val, test = train_test_split(concat_dataset, train_size=train_val_size,
                                               test_size=self.test_size, stratify=labels)

            labels = [el[1] for el in train_val]
            train, val = train_test_split(train_val, train_size=self.training_size,
                                          test_size=self.val_size, stratify=labels)

            self.mnist_train = ScaleMNIST(
                train, self.train_scales, batch_size=self.batch_size, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=self.train_transform)

            self.mnist_val = ScaleMNIST(
                val, self.train_scales, batch_size=self.batch_size, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=None)

            self.mnist_test = ScaleMNIST(
                test, self.test_scales, batch_size=self.batch_size, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=None)

    def train_dataloader(self):
        sampler = CustomRandomSampler(self.mnist_train, self.batch_size, True)
        return DataLoader(self.mnist_train, batch_size=self.batch_size,  sampler=sampler, shuffle=False, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        sampler = CustomRandomSampler(self.mnist_val, self.batch_size, True)
        return DataLoader(self.mnist_val, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        sampler = CustomRandomSampler(self.mnist_test, self.batch_size, True)
        return DataLoader(self.mnist_test, batch_size=self.batch_size,  sampler=sampler, shuffle=False, num_workers=self.num_workers)
