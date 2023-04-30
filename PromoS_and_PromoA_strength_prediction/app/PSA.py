import torch
import PS_modules
import PA_modules
import PR_modules
import numpy as np
import random
import argparse


class denovo_seq():
    def __init__(self, seq):
        self.seq = seq

    def denovo(self):
        input_seq = self.seq
        code_dic = ['A', 'T', 'C', 'G']
        padding_site = random.randint(1, 9)
        padding = random.choice(code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(
            code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(code_dic) + random.choice(
            code_dic) + random.choice(code_dic)
        input_seq = padding[0:padding_site] + input_seq + padding[padding_site:9]

        for i in range(random.randint(1, 10)):
            mut_site = random.randint(1, 50)
            input_seq = input_seq[0:(mut_site - 1)] + random.choice(code_dic) + input_seq[mut_site:50]
        return input_seq


## using promoter J23119 as input, which contains 41 bp, the promoter is extended to 50 bp using random padding.
def create_seq(create_num):
    rand_seq = denovo_seq('AATTCTTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCA')
    all_seqs = []
    for i in range(create_num):
        all_seqs.append(rand_seq.denovo())
    all_seqs = np.array(all_seqs)
    all_seqs = all_seqs.reshape([len(all_seqs), ])
    np.save('denovo_seq%s.npy' % create_num, all_seqs)


def encode(seq):
    encoded_seq = np.zeros(len(seq) * 4, int)
    for j in range(len(seq)):
        if seq[j] == 'A' or seq[j] == 'a':
            encoded_seq[j * 4] = 1
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'C' or seq[j] == 'c':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 1
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'G' or seq[j] == 'g':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 1
            encoded_seq[j * 4 + 3] = 0

        elif seq[j] == 'T' or seq[j] == 't':
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 1

        else:
            encoded_seq[j * 4] = 0
            encoded_seq[j * 4 + 1] = 0
            encoded_seq[j * 4 + 2] = 0
            encoded_seq[j * 4 + 3] = 0
    encoded_seq = encoded_seq.reshape(len(seq), 4)
    return encoded_seq


def get_input(seqs, seq_lenth):
    data = np.zeros((1, seq_lenth, 4))
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
            data = np.concatenate((data, single_seq), axis=0)
    return data

##for generate sample
def predict(num, device='cpu'):
    ##generate data
    create_seq(create_num=int("%s"%num))
    all_seqs = np.load('denovo_seq%s.npy'%num)
    seqs = []
    seq_lenth = 50
    for i in range(len(all_seqs)):
        seqs.append(all_seqs[i])
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
    f = open('result_generated.txt', 'w')
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
        predict_promoR = predict_promoA.cpu().detach().numpy()
        predict_promoS = predict_promoS.reshape(1, )
        predict_promoA = predict_promoA.reshape(1, )
        predict_promoR = predict_promoR.reshape(1, )

        for j, k, l in zip(predict_promoR, predict_promoS, predict_promoA):
            f.write('%s %s %s %s\n'%(str(np.round(j)), str(np.round(k)), str(l), all_seqs[c - 1]))
    f.close()

##for input_sample
def predict_in(input_file='sample.txt', device='cpu'):
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
    parser.add_argument('-sample_type', dest='sample_type', type=str)
    parser.add_argument('-input_file', dest='input_file', type=str)
    parser.add_argument('-device', dest='device', type=str)
    parser.add_argument('-number_of_created_promoter', dest='create_num', type=int)
    args = parser.parse_args()
    sample_type = "%s"%args.sample_type
    if sample_type == "input_sample":
        predict_in(input_file="%s"%args.input_file, device="%s"%args.device)
    if sample_type == "generate_sample":
        predict(num=int("%s"%args.create_num))