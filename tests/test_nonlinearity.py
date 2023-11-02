"""Test cases for Non linearity."""

import unittest
import torch
from layers.scalq_eq_nonlin import scaleEqNonlin
from .test_equievarience import TestEquivarinecError

class TestNonLin(unittest.TestCase):
  def test_equivarinace(self):
    batch_size = 7
    in_channels = 10
    global_modes = 32
    lowest_scale =8
    data = torch.randn(200, in_channels,global_modes, global_modes)
    layer = scaleEqNonlin(torch.relu, lowest_scale, max_res=global_modes, increment = 1)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, data),batch_size=batch_size)
    tester = TestEquivarinecError(layer)
    scale_range = [i for i in range(lowest_scale,global_modes+1)]
    error = tester.get_equivarience_error(data_loader,scale_range)
    print("Error ", error)
    assert error < 10e-6