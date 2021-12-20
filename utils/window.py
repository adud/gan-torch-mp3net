"""
Window function used by the Vorbis project,
as defined Eq (6) in MP3net paper.
"""

from math import pi
import numpy as np


def window_vorbis(window_size):
    """
    Returns np.array of size window_size containing the samples
    of the window function used by the Vorbis project.
    A factor 2 is omitted since in the reference formula the
    window has a size of 2*N.
    """
    window = np.zeros(window_size)
    samples = np.linspace(0, window_size, window_size, endpoint=False)
    window = np.sin(0.5*pi * np.power(
                       np.sin((pi/(window_size)) * (samples + 0.5)), 2))
    return window
