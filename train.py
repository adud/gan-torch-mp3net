import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import optim
import utils.dataset
import utils.transform
from utils.psychoacoustic_filter import PsychoacousticModel
import os
from carbontracker.tracker import CarbonTracker


def train(arch, loss, epoch, gen_bonus, train_dataloader, device, c):
    """trains the generator and the discriminator
    arch: a pair (generator, discriminator) of a GAN architecture
    loss: a loss function
    optim: an optimizer class
    epoch: the number of training iterations on the dataset
    dis_bonus: the number of added iterations on the discriminator
    tlog: frequency of logs (number of batchs before printing)
          a negative value means no log at all
    device: device for computations (should be the same as arch)"""
    
    tlog = c.TLOG
    if c.CARBONTRACKER:
        c.CARBONTRACKER.epoch_start()
    
    #Psychoacoustic model creation to add noise to generated mdct
    model_psych = PsychoacousticModel(c.SAMPLE_RATE, device, c.FILTER_BANDS, c.BARK_BANDS)

    
    if tlog >= 0:
        fix_lat = torch.randn(c.EX_GEN, c.LATENT_DIM, c.NB_CHANNELS, 4, device=device)
        if os.path.exists(c.EX_PATH):
            raise FileExistsError("training would erase logs: aborting")
        os.makedirs(c.EX_PATH)

    #Create file to save losses
    log_loss = open(os.path.join(c.EX_PATH, "losses"),mode = 'w')
    log_loss.write('Epoch, batch, Loss Generator, Loss Discriminator \n')
    log_loss.close()

    gen, dis = arch
    dis_opt = optim.Adam(dis.parameters(), lr=c.LR, betas=(c.B1, c.B2))
    gen_opt = optim.Adam(gen.parameters(), lr=c.LR, betas=(c.B1, c.B2))
    real_lab = torch.ones(c.BATCH_SIZE, 1, device=device)
    fake_lab = torch.zeros(c.BATCH_SIZE, 1, device=device)

    os.makedirs(os.path.join(c.EX_PATH, 'model_parameters'))


    for ep in range(epoch):
        if tlog >= 0:
            print(f"## EPOCH {ep} ##")
            os.makedirs(os.path.join(c.EX_PATH, str(ep)))

        for i, data in enumerate(train_dataloader):
            
            #Update discriminator network
            dis.zero_grad()

            #Output real samples
            real = data.to(device)
            dreal = dis(real)
            dreal_loss = loss(dreal, real_lab)
            dreal_loss.backward()

            #Train with fake
            noise = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, c.NB_CHANNELS, 4,
                                   device=device)
            
            fake = model_psych.apply_psycho_batch(gen(noise))
            dfake = dis(fake)
            
            dfake_loss = loss(dfake, fake_lab)
            dfake_loss.backward(retain_graph = True)
            errD = dreal_loss + dfake_loss
            dis_opt.step()


            #Update Generator network
            gen.zero_grad()
            dfake = dis(fake)
            dfake_loss = loss(dfake, real_lab)

            dfake_loss.backward()
            gen_opt.step()

            """
            dis_opt.zero_grad()
            #with torch.no_grad():
            #    dfake = dis(fake)
            dfake = dis(fake)
            dfake_loss = loss(dfake, fake_lab)
            dreal_loss = loss(dis(real), real_lab)
            dis_loss = (dfake_loss + dreal_loss) / 2
            #print(dis(fake.detach()))
            #print(dis(real))
            #print(dis_loss)

            dis_loss.backward(retain_graph = True)
            dis_opt.step()
            
            
            gen_opt.zero_grad()
            #lat_vect2 = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, c.NB_CHANNELS, 4,
            #                       device=device)
            
            #fake2 = model_psych.apply_psycho_batch(gen(lat_vect2))
            gen_loss = loss(dfake,real_lab)
            #print(gen_loss)
            
            gen_loss.backward()
            gen_opt.step()
            


            for _ in range(gen_bonus):
                # train the generator
                lat_vect = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, c.NB_CHANNELS, 4, device=device)
                fake = gen(lat_vect)
                fake = model_psych.apply_psycho_batch(fake)
                #with torch.no_grad():
                guess = dis(fake)

                gen_loss = loss(guess, real_lab)
                print(gen_loss)
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()
            """
                


            if not i % tlog:
                print(f"\tbatch {i}: {dfake_loss}, {errD}")
                with open(os.path.join(c.EX_PATH, "losses"),mode = 'a') as f:
                    f.write(f"{ep}, {i}, {dfake_loss}, {errD}\n")
                with torch.no_grad():
                    out = model_psych.apply_psycho_batch(gen(fix_lat))
                    torch.save(out, os.path.join(c.EX_PATH, str(ep), str(i)))
            
        
        #Save the model parameters every 1 epochs
        if ep%1 == 0:
            torch.save(gen.state_dict(), os.path.join(c.EX_PATH, "model_parameters",f"generator_{ep}"))
            torch.save(dis.state_dict(),os.path.join(c.EX_PATH, "model_parameters",f"discriminator_{ep}"))
            print("Model parameters saved")
    
    if c.CARBONTRACKER:
        c.CARBONTRACKER.epoch_end()
            
    
        
                        
                        
