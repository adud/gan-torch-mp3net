"""
Custom class defining the MDCT transform to be used
by PyTorch when loading the Dataset.
"""

from mdct import mdct
from .window import window_vorbis

class MdctTransform():
    def __init__(self, nfft=256, window=None):
        if window is None:
            self.window = window_vorbis(nfft)
        else:
            self.window = window
        self.fft_size = nfft

    def __call__(self, snd):
        return mdct(snd, framelength=self.fft_size, window=self.window)
