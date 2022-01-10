"""
Default values for all variables to be used in main.py
"""

import torch
import torchaudio

# Signal processing

FFT_SIZE = 256
SAMPLE_RATE = 22050


# Model parameters

LATENT_DIM = 100  # dimensions of the latent space (also Generator input size)

# Adam optimizer parameters (see p6)

LR = 1e-3  # Learning rate
B1 = 0.5   # beta1
B2 = 0.9   # beta2


# Training parameters

BATCH_SIZE = 16
FADE_IN = 64  # in samples
FADE_OUT = 64  # in samples
TRAIN_PATH = './dataset/'

EX_GEN = 10  # number of examples to generate for log
EX_PATH = 'examples'
