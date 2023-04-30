import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import UNet
import os
from sklearn.linear_model import LinearRegression

fit_model = LinearRegression()
torch.set_default_tensor_type(torch.DoubleTensor)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    def sample_promoters (self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, 4, 50)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def dataTogene(data, bz,name):
    data = data.reshape(bz,50,4)
    data = np.eye(data.shape[2])[data.argmax(2)]

    list = []
    for i in data:
        for l in i:
           l1 = np.array([1., 0., 0., 0.])
           l2 = np.array([0., 1., 0., 0.])
           l3 = np.array([0., 0., 1., 0.])
           l4 = np.array([0., 0., 0., 1.])
           if str((l == l1).all()) == "True":
               list.append('A')
           if str((l == l2).all()) == "True":
               list.append('C')
           if str((l == l3).all()) == "True":
               list.append('G')
           if str((l == l4).all()) == "True":
               list.append('T')
    f = open ('output_gene%s.txt'%name,'w')

    for i in range (bz):
        seq =(list[(i*50):((i+1)*(50))])
        seq = str(seq).replace(',','').replace("'",'').replace(' ','').replace('[','').replace(']','')
        f.write('>%s'%i)
        f.write('\n')
        f.write(seq)
        f.write('\n')
    f.close()

def get_six_mer(path,six_mer_list=[],generated=False):
    f = open (r'%s'%path)
    a = f.readlines()
    f.close()
    c = 0
    for i in a:
        c += 1
        i = i.strip('\n')
        for k in range (0,len(i)-6+1):
            Six_mer = i[k:(k+6)]
            six_mer_list.append(Six_mer)
        if generated:
            if c % 2 == 0:
                for k in range (0,len(i)-6+1):
                    Six_mer = i[k:(k+6)]
                    six_mer_list.append(Six_mer)
    return six_mer_list

def get_sixmer_score(path1='sequence_data_noreduct.txt',path2 = 'output_geneepoch66.txt',epoch=0, num_nat = 14098,num_gen = 512):
    gene = ['A','T','C','G']
    six_mer = []
    for a in gene:
        for b in gene:
            for c in gene:
                for d in gene:
                    for e in gene:
                        for f in gene:
                            combine = a+b+c+d+e+f
                            six_mer.append(combine)

    nat_Sixmer = []
    get_six_mer(path = path1, six_mer_list=nat_Sixmer)

    gen_Sixmer = []
    get_six_mer(path = path2, six_mer_list=gen_Sixmer,generated=True)

    f = open('kmer_epoch%s.txt'%epoch,'w')
    f.close()
    f = open('kmer_epoch%s.txt'%epoch,'a+')
    stat_nat = []
    stat_gen = []
    for i in six_mer:
        count_nat = nat_Sixmer.count(i)
        nat_freq = count_nat/num_nat
        stat_nat.append(nat_freq)
        count_gen = gen_Sixmer.count(i)
        gen_freq = count_gen/num_gen
        stat_gen.append(gen_freq)
        wr = i + ' ' + str(nat_freq) + ' ' + str(gen_freq) + '\n'
        f.write(wr)
    f.close()
    x = np.array(stat_nat)
    x = x.reshape(-1,1)
    y = np.array(stat_gen)
    y = y.reshape(-1,1)
    fit_model.fit(x,y)
    score = fit_model.score(x,y)
    f = open ('fit_score%s.txt'%epoch,'w')
    f.write(str(score))
    f.close()
    print(epoch,score)
    return score

def encode(seq):
    encoded_seq = np.zeros(len(seq)*4,int)
    for j in range(len(seq)):
        if seq[j] == 'A' or seq[j] == 'a':
            encoded_seq[j*4] = 1
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'C' or seq[j] == 'c':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 1
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'G' or seq[j] == 'g':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 1
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'T' or seq[j] == 't':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 1

        else:
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0
    encoded_seq = encoded_seq.reshape(len(seq),4)
    return encoded_seq

def get_input(seqs, seq_lenth):
    data = np.zeros((1,seq_lenth,4))
    count = 0
    for i in seqs:
        count += 1
        if count == 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = data + single_seq
            data = np.expand_dims(data, axis=0)
        if count != 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = np.concatenate ((data,single_seq),axis=0)
    return data

epochs = 2000

def train():

    ##create folder and loss text
    f = open('epoch_loss.txt','w')
    f.close()
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    ##load data
    file = 'sequence_data_noreduct.txt'
    seq_file = open(file).read().splitlines()
    data = get_input(seqs=seq_file,seq_lenth=50)
    real_bz = 128
    loader = DataLoader(dataset=data,batch_size=real_bz,shuffle=True,drop_last= True)

    ##model initiation
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = UNet(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(device=device)

    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for i, images in enumerate(loader):
            count += 1
            images = images.to(device)
            images = images.reshape(128,1,4,50)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            epoch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f = open('epoch_loss.txt','a+')
        loss_ave = 'epoch_loss %s\n'%(epoch_loss/count)
        f.write(loss_ave)
        f.close()
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), r"saved_models/" + 'epoch_%s.pt'%(str(epoch)))
        print(epoch)
        with torch.no_grad():
            if epoch > 0 and (epoch+1) % 100 == 0:
                load_model = torch.load("saved_models/epoch_%s.pt"%epoch)
                model.load_state_dict(load_model)
                bz = 1024
                data = diffusion.sample_promoters(model, bz).cpu().detach().numpy()
                dataTogene(data=data,bz=bz,name='epoch%s'%epoch)
                get_sixmer_score(path1='sequence_data_noreduct.txt',path2 = 'output_geneepoch%s.txt'%epoch,epoch=epoch,num_gen=1024)
        
if __name__ == '__main__':
    train()