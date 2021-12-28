"""
Custom class defining the MDCT transform to be used
by PyTorch when loading the Dataset.
"""

import math
import torch
import torch.nn as nn
from mdct import mdct
from .window import window_vorbis

class ZeroPadTransform(nn.Module):
    def __init__(self, fft_size):
        self.fft_size = fft_size
    def __call__(self, snd):
        snd_size = snd.shape[0]
        padded_snd = torch.zeros((2**math.ceil(math.log2(snd_size)), snd.shape[1]))
        padded_snd[:snd_size, :] = snd
        return padded_snd

class MdctTransform(nn.Module):
    def __init__(self, nfft=256, window=None):
        if window is None:
            self.window = window_vorbis(nfft)
        else:
            self.window = window
        self.fft_size = nfft

    def __call__(self, snd):
        return mdct(snd, framelength=self.fft_size,
                    window=self.window, centered=True)[:, :-1, :]

class TransposeTransform(nn.Module):
    def __call__(self, snd):
        return torch.t(snd)

class ToTensorTransform(nn.Module):
    def __call__(self, snd):
        return torch.from_numpy(snd)
