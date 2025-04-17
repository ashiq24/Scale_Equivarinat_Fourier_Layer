"""Implements model getter support."""

from .fourier_seq_mnist import FourierSeqMist
from .fourier_seq_stl import FourierSeqStl
from .fourier_seq_1d import FourierSeq1d
from .fourier_seq_mixer import FourierScaleShifteqMixer, FourierSeqFrequencyMixer
_available_classifiers = {"fourier_mnist": FourierSeqMist, 'fourier_stl': FourierSeqStl,
                          "fourier_1d": FourierSeq1d, 'fourier_mixer': FourierSeqFrequencyMixer,
                          'fourier_mixer_shift_invarinat': FourierScaleShifteqMixer}


def get_available_models():
    return list(_available_classifiers.keys())


def get_model(name):
    return _available_classifiers[name]
