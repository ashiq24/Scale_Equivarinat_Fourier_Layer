from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from utils.core_utils import resample
from torch.nn.functional import pad
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import math


class ScaleSTL(Dataset):
    def __init__(self, root_data, scales, batch_size, data_resize_mode, data_downscale_mode, transform=None):
        super(ScaleSTL, self).__init__()
        self.root_data = root_data

        size = len(self.root_data)
        batch_per_sacle = size//(len(scales)*batch_size)
        self.size = batch_per_sacle*(len(scales)*batch_size)

        length = len(self.root_data)
        self.scale_range = scales
        suffel_indices = [i for i in range(length)]
        np.random.shuffle(suffel_indices)

        k = length//len(self.scale_range)
        self.cluster_indices = [
            suffel_indices[i*k:(i+1)*k] for i in range(len(self.scale_range))]
        np.random.shuffle(self.cluster_indices)

        self.data_resize_mode = data_resize_mode
        self.data_downscale_mode = data_downscale_mode

        self.normalize = transforms.Normalize(
            (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        self.transform = transform
        self.target_transform = None
        self.max_size = max(scales)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        img, target = self.root_data.__getitem__(index)

        if self.transform is not None:
            img = self.transform(img)

        i = None
        for k in range(len(self.cluster_indices)):
            if index in self.cluster_indices[k]:
                i = k
                break
        if i is None:
            raise Exception("Index not in scale Group")

        if self.data_downscale_mode == "ideal":
            img = resample(img[None, :, :, :], (self.scale_range[i],
                           self.scale_range[i]), complex=False, skip_nyq=True)
        if self.data_downscale_mode == "bicubic":
            img = F.interpolate(img[None, :, :, :], size=(
                self.scale_range[i], self.scale_range[i]), mode='bicubic', align_corners=True,  antialias=True)

        if self.data_resize_mode == "pad":
            l = self.max_size - img.shape[-2]
            t = self.max_size - img.shape[-1]
            img = pad(img, pad=[l//2, l - l//2, t//2, t - t//2])
        elif self.data_resize_mode == "reflect":
            l = self.max_size - img.shape[-2]
            t = self.max_size - img.shape[-1]
            img = pad(img, pad=[l//2, l - l//2, t //
                      2, t - t//2], mode="reflect")
        elif self.data_resize_mode == "replicate":
            l = self.max_size - img.shape[-2]
            t = self.max_size - img.shape[-1]
            img = pad(img, pad=[l//2, l - l//2, t //
                      2, t - t//2], mode="replicate")
        elif self.data_resize_mode == "circular":
            l = self.max_size - img.shape[-2]
            t = self.max_size - img.shape[-1]
            img = pad(img, pad=[l//2, l - l//2, t //
                      2, t - t//2], mode='circular')
        elif self.data_resize_mode == 'resize':
            img = resample(img, (self.max_size, self.max_size), complex=False)
        elif self.data_resize_mode == 'None':
            pass
        else:
            raise Exception("Not a valid resizing mode")

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = self.normalize(img[0])

        return img, target
