import torch
import PR_modules
import numpy as np
import argparse

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


def predict(load_file='PR_validation.txt', select_device='cpu'):
    ##generate data
    f = open(r'%s' % load_file)
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
    model_R = 'PR_best.pth'
    device = torch.device(select_device)
    promoR_cp = torch.load(model_R, map_location=device)
    promoR = PR_modules.predictor().to(device)
    promoR.load_state_dict(promoR_cp['pred'])

    ##predict
    f = open('result_%s' % load_file, 'w')
    test_seq = get_input(seqs=seqs, seq_lenth=seq_lenth)
    count = 0
    for i in test_seq:

        i = i.reshape(1, 1, 4, 50)
        i = torch.from_numpy(i)
        i = i.to(device)
        predict_promoR = promoR(i, 1)
        predict_promoR = predict_promoR.cpu().detach().numpy().reshape(-1, 1)
        x = int(str(np.round(predict_promoR)).replace('[[', '').replace('.]]', ''))
        f.write('%s %s\n' % (seqs[count], x))
        count += 1
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_file', dest='load_file', type=str)
    parser.add_argument('-select_device', dest='select_device', type=str)
    args = parser.parse_args()
    predict(load_file='%s'%args.load_file, select_device='%s'%args.select_device)