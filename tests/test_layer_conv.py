"""Test cases for Conv. layers."""

import unittest
import torch
from layers.spectral_conv import SpectralConv2dLocalized
from .test_equievarience import TestEquivarinecError

class TestSpectralConv(unittest.TestCase):
  def test_equivarinace(self):
    batch_size = 7
    in_channels = 10
    out_channels = 20
    global_modes = 32
    local_modes = 8
    lowest_scale =8
    data = torch.randn(200, in_channels,global_modes, global_modes)
    layer = SpectralConv2dLocalized(in_channels,out_channels,global_modes,local_modes)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, data),batch_size=batch_size)
    tester = TestEquivarinecError(layer)
    scale_range = [i for i in range(lowest_scale,global_modes+1)]
    error = tester.get_equivarience_error(data_loader,scale_range)
    print("Error ", error)
    assert error < 10e-6

