"""
Custom class defining the MDCT transform to be used
by PyTorch when loading the Dataset.
"""

import math
import torch
from torch import nn
from mdct import mdct
from .window import window_vorbis
# from .psychoacoustic_filter import PsychoacousticModel


class ZeroPadTransform(nn.Module):
    """Transformation that pads the input with zeros"""
    def __init__(self, fft_size):
        self.fft_size = fft_size

    def __call__(self, snd):
        snd_size = snd.shape[0]
        padded_snd = torch.zeros((2**math.ceil(math.log2(snd_size)),
                                  snd.shape[1]))
        padded_snd[:snd_size, :] = snd
        return padded_snd

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class MdctTransform(nn.Module):
    """MDCT transformation used to train the discriminator"""
    def __init__(self, nfft=256, window=None, mono=True):
        self.mono = mono
        if window is None:
            self.window = window_vorbis(nfft)
        else:
            self.window = window
        self.fft_size = nfft

    def __call__(self, snd):
        out = mdct(snd, framelength=self.fft_size,
                   window=self.window, centered=True)
        if self.mono:
            out = out[:, :, None]
        out = out[:, :-1, :]
        return out

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class TransposeTransform(nn.Module):
    """Transposition Transformation"""
    def __call__(self, snd):
        return torch.t(snd)

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class ToTensorTransform(nn.Module):
    """Tensor transformation"""
    def __call__(self, snd):
        tensor = torch.tensor(torch.from_numpy(snd), dtype=torch.float32)
        return tensor.transpose(0, 2)

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class PsychoAcousticTransform(nn.Module):
    """Transformation with a psychoacoustic filter"""
    def __init__(self, psycho_model):
        self.psycho_model = psycho_model

    def __call__(self, snd):
        return self.psycho_model.apply_psycho_single(snd)

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class ReshapeTransform(nn.Module):
    """Transformation that reshapes"""
    def __call__(self, snd):
        snd = snd.transpose(0, 2)
        snd = snd.transpose(1, 2)
        return snd

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError


class ToDeviceTransform(nn.Module):
    """Move to a device"""
    def __init__(self, device):
        self.device = device

    def __call__(self, snd):
        return snd.to(self.device)

    def forward(self, inputs):
        """forward computation"""
        raise NotImplementedError
