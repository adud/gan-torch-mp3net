import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import optim
import config as c
import utils.dataset
import utils.transform
from utils.psychoacoustic_filter import PsychoacousticModel
import os
from carbontracker.tracker import CarbonTracker


def train(arch, loss, epoch, gen_bonus, train_dataloader, device, tlog=c.TLOG):
    """trains the generator and the discriminator
    arch: a pair (generator, discriminator) of a GAN architecture
    loss: a loss function
    optim: an optimizer class
    epoch: the number of training iterations on the dataset
    dis_bonus: the number of added iterations on the discriminator
    tlog: frequency of logs (number of batchs before printing)
          a negative value means no log at all
    device: device for computations (should be the same as arch)"""
    
    if c.CARBONTRACKER:
        c.CARBONTRACKER.epoch_start()
    
    #Psychoacoustic model creation to add noise to generated mdct
    model_psych = PsychoacousticModel(c.SAMPLE_RATE, c.FILTER_BANDS, c.BARK_BANDS)
    
    #Open file to save losses on the fly
    log_loss = open(os.join(c.EX_PATH, "losses"),mode = 'w')
    log_loss.write('Epoch, batch, Loss Generator, Loss Discriminator \n')
    
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
            
            for _ in range(gen_bonus):
                # train the generator
                lat_vect = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, device=device)
                fake = model_psych.apply_psycho(gen(lat_vect))
                with torch.no_grad():
                    guess = dis(fake)

                gen_loss = loss(guess, real_lab)
                gen.zero_grad()
                gen_loss.backward
                gen_opt.step()
                
            # train the discriminator    
            lat_vect = torch.randn(c.BATCH_SIZE, c.LATENT_DIM,
                                   device=device)
            with torch.no_grad():
                fake = model_psych.apply_psycho(gen(lat_vect))
            real = data.to(device)
            dfake_loss = loss(dis(fake), fake_lab)
            dreal_loss = loss(dis(real), real_lab)
            dis_loss = (dfake_loss + dreal_loss) / 2
            dis.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            if not i % tlog:
                print(f"\tbatch {i}/{c.BATCH_SIZE}: {gen_loss}, {dis_loss}")
                log_loss.write(f"{ep}, {i}, {gen_loss}, {dis_loss}")
                with torch.no_grad():
                    out = gen(fix_lat)
                    with open(os.join(c.EX_PATH, str(ep), str(i))) as f:
                        torch.save(out, f)
        
        #Save the model parameters every 10 epochs
        if ep%10 == 0:
            with open(os.join(c.EX_PATH, "model_parameters",f"generator_{ep}")) as f:
                torch.save(gen.state_dict(),f)
            with open(os.join(c.EX_PATH, "model_parameters",f"discriminator_{ep}")) as f:
                torch.save(dis.state_dict(),f)
            print("Model parameters saved")
            
    log_loss.close()
    
    if c.CARBONTRACKER:
        c.CARBONTRACKER.epoch_end()
            
    
        
                        
                        
