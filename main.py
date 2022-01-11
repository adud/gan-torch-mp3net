import torch
import torchaudio
from argparse import ArgumentParser
from model.discriminator import Discriminator
from model.generator import Generator
from train import train
from torch.utils.data import DataLoader
from utils.psychoacoustic_filter import PsychoacousticModel
import utils.dataset
import utils.transform
import config as c
from carbontracker.tracker import CarbonTracker


parser = ArgumentParser()

parser.add_argument("--sample-rate", type=int, dest="SAMPLE_RATE",
                    default=c.SAMPLE_RATE,
                    help="sample rate of the generated audio")
parser.add_argument("--latent-dim", type=int, dest="LATENT_DIM",
                    default=c.LATENT_DIM,
                    help="number of dimensions of the latent space")
parser.add_argument("--epoch", type=int, dest="EPOCH",
                    default=c.EPOCH,
                    help="number of epochs for training")
parser.add_argument("--learning-rate", type=float, dest="LR",
                    default=c.LR,
                    help="learning rate of the Adam optimizer")
parser.add_argument("--gen-bonus", type=int, dest="GEN_BONUS",
                    default=c.GEN_BONUS,
                    help="number of rounds of generator training for one round of discriminator training")
parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE",
                    default=c.BATCH_SIZE,
                    help="size of a batch training")
parser.add_argument("--fade-in", type=int, dest="FADE_IN",
                    default=c.FADE_IN,
                    help="size of the fade-in in the extracts (in samples)")
parser.add_argument("--fade-out", type=int, dest="FADE_OUT",
                    default=c.FADE_OUT,
                    help="size of the fade-out in the extracts (in samples)")
parser.add_argument("--train-path", type=str, dest="TRAIN_PATH",
                    default=c.TRAIN_PATH,
                    help="path to store/get training data")
parser.add_argument("--ex-gen", type=int, dest="EX_GEN",
                    default=c.EX_GEN,
                    help="number of examples to generate when logging")
parser.add_argument("--carbon-tracker", type=bool, dest="CARBONTRACKER",
                    default=c.CARBONTRACKER,
                    help="Activate carbon tracker or not")
parser.add_argument(
    "--ex-path", type=str, dest="EX_PATH",
    default=c.EX_PATH,
    help="path to store examples (will never overwrite, fail instead)"
)
parser.add_argument("--tlog", type=int, dest="TLOG",
                    default=c.TLOG,
                    help="time between two logs (in number of batchs)")


transforms = torch.nn.Sequential(
                        torchaudio.transforms.Fade(c.FADE_IN, c.FADE_OUT),
                        utils.transform.TransposeTransform(),
                        utils.transform.ZeroPadTransform(c.MDCT_SIZE),
                        utils.transform.MdctTransform(c.MDCT_SIZE, mono=True),
                        utils.transform.ToTensorTransform(),
                        utils.transform.PsychoAcousticTransform(PsychoacousticModel(c.SAMPLE_RATE, c.FILTER_BANDS, c.BARK_BANDS)),
                        utils.transform.ReshapeTransform()
                        )

training_data = utils.dataset.MP3NetDataset(c.TRAIN_PATH, num_channels=1, transform=transforms)

train_dataloader = DataLoader(training_data, c.BATCH_SIZE, shuffle=True)


device = torch.device(c.DEVICE)
gen = Generator(out_channels=c.NB_CHANNELS)
dis = Discriminator(in_channels=c.NB_CHANNELS)


print("")
loss = torch.nn.BCELoss()

print("Begin training...")

if c.CARBONTRACKER:
    c.CARBONTRACKER = CarbonTracker(epochs=c.EPOCH)

train((gen, dis), loss=loss, epoch=c.EPOCH, gen_bonus=c.GEN_BONUS,
      train_dataloader=train_dataloader, device=device, tlog=c.TLOG)

print("End of training")
