"""Implements MNIST DataModule."""

import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, STL10
from .scale_stl import ScaleSTL

from .custom_sampler import CustomRandomSampler
from sklearn.model_selection import train_test_split
from scale_eq.utils.core_utils import AddUniformNoise, AddCircularShift, Cutout


def None_func():
    return None


augment_map = {"add_noise": AddUniformNoise,
               "circular_shift": AddCircularShift, "None": None_func}


def get_augmentation(name):
    global augment_map
    return augment_map[name]


class ScaleSTLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, training_size=None, test_size=5000, train_scales=None, test_scales=None,
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
        STL10(root=self.data_dir, split='train', download=True)
        STL10(root=self.data_dir, split='test', download=True)

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            dataset_train = STL10(
                root=self.data_dir, split='train', download=True, transform=transforms.ToTensor())
            dataset_test = STL10(root=self.data_dir, split='test',
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
            self.stl_train = ScaleSTL(
                train, self.train_scales, batch_size=self.batch_size, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=self.train_transform)

            self.stl_val = ScaleSTL(
                val, self.train_scales, batch_size=10, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=None)

            self.stl_test = ScaleSTL(
                test, self.test_scales, batch_size=self.batch_size, data_resize_mode=self.data_resize_mode,
                data_downscale_mode=self.data_downscale_mode, transform=None)

    def train_dataloader(self):
        sampler = CustomRandomSampler(self.stl_train, self.batch_size, True)
        return DataLoader(self.stl_train, batch_size=self.batch_size,  sampler=sampler, shuffle=False, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        sampler = CustomRandomSampler(self.stl_val, 10, True)
        return DataLoader(self.stl_val, batch_size=10, sampler=sampler, shuffle=False, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        sampler = CustomRandomSampler(self.stl_test, self.batch_size, True)
        return DataLoader(self.stl_test, batch_size=self.batch_size,  sampler=sampler, shuffle=False, num_workers=self.num_workers)
