import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Diff_modules import UNet
import os
import argparse
import torch
import PS_modules
import PA_modules
import PR_modules
import random
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

def dataTogene(data, bz):
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
    f = open ('output_gene.txt','w')

    for i in range (bz):
        seq =(list[(i*50):((i+1)*(50))])
        seq = str(seq).replace(',','').replace("'",'').replace(' ','').replace('[','').replace(']','')
        f.write('>%s'%i)
        f.write('\n')
        f.write(seq)
        f.write('\n')
    f.close()

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


def PromoDiff(create_num=5000, device='cpu'):
    ##model initiation
    device = torch.device('%s'%device)
    model = UNet(device=device).to(device)
    diffusion = Diffusion(device=device)
    load_model = torch.load("Diff_model.pt")
    model.load_state_dict(load_model)
    bz = create_num
    data = diffusion.sample_promoters(model, bz).cpu().detach().numpy()
    dataTogene(data=data,bz=bz)

def predict_in(input_file='output_gene.txt', device='cpu'):
    ##generate data
    f = open(r'%s' %input_file)
    all_seqs = f.readlines()
    f.close()
    seqs = []
    seq_lenth = 50
    num = 0
    for i in all_seqs:
        num += 1
        if num % 2 == 0:
            seqs.append(i.strip('\n'))

    ##load model
    model_classify = 'PS_best_model.pth'
    model_strength = 'PA_best_model.pth'
    model_R = 'PR_best.pth'
    device = torch.device('%s'%device)
    promoS_cp = torch.load(model_classify, map_location=device)
    promoA_cp = torch.load(model_strength, map_location=device)
    promoR_cp = torch.load(model_R, map_location=device)
    promoS = PS_modules.predictor().to(device)
    promoA = PA_modules.predictor().to(device)
    promoR = PR_modules.predictor().to(device)
    promoS.load_state_dict(promoS_cp['pred'])
    promoA.load_state_dict(promoA_cp['pred'])
    promoR.load_state_dict(promoR_cp['pred'])

    ##predict
    f = open('result_%s'%input_file, 'w')
    test_seq = get_input(seqs=seqs, seq_lenth=seq_lenth)
    c = 0
    for i in test_seq:
        c += 1
        i = i.reshape(1, 1, 4, 50)
        i = torch.from_numpy(i)
        i = i.to(device)
        predict_promoS = promoS(i, 1)
        predict_promoA = promoA(i, 1)
        predict_promoR = promoR(i, 1)
        predict_promoS = predict_promoS.cpu().detach().numpy()
        predict_promoA = predict_promoA.cpu().detach().numpy()
        predict_promoR = predict_promoR.cpu().detach().numpy()
        predict_promoS = predict_promoS.reshape(1, )
        predict_promoA = predict_promoA.reshape(1, )
        predict_promoR = predict_promoR.reshape(1, )

        for j, k, l in zip(predict_promoR, predict_promoS, predict_promoA):
            f.write('%s %s %s %s\n'%(str(np.round(j)), str(np.round(k)), str(l), all_seqs[c*2 - 1].strip('\n')))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict', dest='predict', type=str)
    parser.add_argument('-input_file', dest='input_file', type=str)
    parser.add_argument('-device', dest='device', type=str)
    parser.add_argument('-number_of_created_promoter', dest='create_num', type=int)
    args = parser.parse_args()
    PromoDiff(create_num=int("%s"%args.create_num), device='%s'%args.device)
    predict = '%s'%args.predict
    if predict == 'yes':
        predict_in(device="%s" % args.device)
