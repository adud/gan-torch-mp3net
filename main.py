from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--fft-size", type=int, dest="FFT_SIZE",
                    default=256,
                    help="size of the fft of the psychoacoustic filter")
parser.add_argument("--sample-rate", type=int, dest="SAMPLE_RATE",
                    default=22050,
                    help="sample rate of the generated audio")
parser.add_argument("--latent-dim", type=int, dest="LATENT_DIM",
                    default=100,
                    help="number of dimensions of the latent space")
parser.add_argument("--learning-rate", type=float, dest="LR",
                    default=1e-4,
                    help="learning rate of the Adam optimizer")
parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE",
                    default=16,
                    help="size of a batch training")
parser.add_argument("--fade-in", type=int, dest="FADE_IN",
                    default=64,
                    help="size of the fade-in in the extracts (in samples)")
parser.add_argument("--fade-out", type=int, dest="FADE_OUT",
                    default=64,
                    help="size of the fade-out in the extracts (in samples)")
parser.add_argument("--train-path", type=str, dest="TRAIN_PATH",
                    default="./dataset/",
                    help="path to store/get training data")
parser.add_argument("--ex-gen", type=int, dest="EX_GEN",
                    default=10,
                    help="number of examples to generate when logging")
parser.add_argument(
    "--ex-path", type=str, dest="EX_PATH",
    default="examples",
    help="path to store examples (will never overwrite, fail instead)"
)

parser.add_argument("--tlog", type=int, dest="TLOG",
                    default=100,
                    help="time between two logs (in number of batchs)")
