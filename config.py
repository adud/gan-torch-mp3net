"""
Default values for all variables to be used in main.py
"""

# Signal processing

MDCT_SIZE = 256
FILTER_BANDS = 128
SAMPLE_RATE = 22050
BARK_BANDS = 64
NB_CHANNELS = 1

# Model parameters

LATENT_DIM = 512  # dimensions of the latent space (also Generator input size)
DEVICE = 'cuda'

# Adam optimizer parameters (see p6)

LR = 0.002  # Learning rate
B1 = 0.5   # beta1
B2 = 0.9   # beta2


# Training parameters

GEN_BONUS = 0
BATCH_SIZE = 32
EPOCH = 2
FADE_IN = 64  # in samples
FADE_OUT = 64  # in samples
TRAIN_PATH = './dataset/'

EX_GEN = 10  # number of examples to generate for log
EX_PATH = 'training_output'

TLOG = 30

CARBONTRACKER = True
