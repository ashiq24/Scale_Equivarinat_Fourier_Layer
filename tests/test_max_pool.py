"""Test cases for Pooling Layer"""

import unittest
import torch
from layers.scalq_eq_nonlin import scaleEqNonlinMaxp
from .test_equievarience import TestEquivarinecError


class Testpool(unittest.TestCase):
    def test_equivarinace(self):
        batch_size = 7
        in_channels = 10
        global_modes = 32
        lowest_scale = 8
        pool_size = 2
        data = torch.randn(200, in_channels, global_modes, global_modes)
        layer = scaleEqNonlinMaxp(
            torch.relu, lowest_scale, max_res=global_modes, pool_window=pool_size, increment=1)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, data), batch_size=batch_size)
        tester = TestEquivarinecError(layer)
        scale_range = [i for i in range(
            lowest_scale, global_modes+1, pool_size)]
        error = tester.get_equivarience_error(data_loader, scale_range)
        assert error < 10e-6
