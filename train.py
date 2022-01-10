import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import optim
import config as c
import utils.dataset
import utils.transform
import os

transforms = torch.nn.Sequential(
                        torchaudio.transforms.Fade(c.FADE_IN, c.FADE_OUT),
                        utils.transform.TransposeTransform(),
                        utils.transform.ZeroPadTransform(c.FFT_SIZE),
                        utils.transform.MdctTransform(c.FFT_SIZE),
                        utils.transform.ToTensorTransform()
                        )

training_data = utils.dataset.MP3NetDataset(c.TRAIN_PATH, transform=transforms)

train_dataloader = DataLoader(training_data, c.BATCH_SIZE, shuffle=True)


def train(arch, loss, epoch, dis_bonus, train_dataloader, device, tlog=c.TLOG):
    """trains the generator and the discriminator
    arch: a pair (generator, discriminator) of a GAN architecture
    loss: a loss function
    optim: an optimizer class
    epoch: the number of training iterations on the dataset
    dis_bonus: the number of added iterations on the discriminator
    tlog: frequency of logs (number of batchs before printing)
          a negative value means no log at all
    device: device for computations (should be the same as arch)"""
    gen, dis = arch
    dis_opt = optim.Adam(dis.parameters(), lr=c.LR, betas=(c.B1, c.B2))
    gen_opt = optim.Adam(gen.parameters(), lr=c.LR, betas=(c.B1, c.B2))
    real_lab = torch.ones(c.BATCH_SIZE, 1, device=device)
    fake_lab = torch.zeros(c.BATCH_SIZE, 1, device=device)

    if tlog >= 0:
        fix_lat = torch.randn(c.EX_GEN, c.LATENT_DIM, device=device)
        if os.path.exists(c.EX_PATH):
            raise FileExistsError("training would erase logs: aborting")
        os.makedirs(c.EX_PATH)

    for ep in range(epoch):
        if tlog >= 0:
            print(f"## EPOCH {epoch} ##")
            os.makedirs(os.join(c.EX_PATH, str(ep)))
        for i, data in enumerate(train_dataloader):
            # train the discriminator
            for _ in range(dis_bonus):
                lat_vect = torch.randn(c.BATCH_SIZE, c.LATENT_DIM,
                                       device=device)
                with torch.no_grad():
                    fake = gen(lat_vect)

                real = data.to(device)

                dfake_loss = loss(dis(fake), fake_lab)
                dreal_loss = loss(dis(real), real_lab)
                dis_loss = (dfake_loss + dreal_loss) / 2

                dis.zero_grad()
                dis_loss.backward()
                dis_opt.step()

            # train the generator
            lat_vect = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, device=device)
            fake = gen(lat_vect)
            with torch.no_grad():
                guess = dis(fake)

            gen_loss = loss(guess, real_lab)
            gen.zero_grad()
            gen_loss.backward
            gen_opt.step()

            if not i % tlog:
                print(f"\tbatch {i}/{c.BATCH_SIZE}: {gen_loss}, {dis_loss}")
                with torch.no_grad():
                    out = gen(fix_lat)
                    with open(os.join(c.EX_PATH, str(ep), str(i))) as f:
                        torch.save(out, f)
