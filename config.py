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

LATENT_DIM = 100  # dimensions of the latent space (also Generator input size)
DEVICE = 'cuda'

# Adam optimizer parameters (see p6)

LR = 1e-4  # Learning rate
B1 = 0.5   # beta1
B2 = 0.9   # beta2


# Training parameters

DIS_BONUS = 1
BATCH_SIZE = 16
EPOCH = 10
FADE_IN = 64  # in samples
FADE_OUT = 64  # in samples
TRAIN_PATH = './dataset/'

EX_GEN = 10  # number of examples to generate for log
EX_PATH = 'examples'

TLOG = 30
