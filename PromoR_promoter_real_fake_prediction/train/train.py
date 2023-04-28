import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import modules
import os

f = open (r'Combined_train_set.txt')
seqs_activity = f.readlines()
f.close()
seqs = []
activity = []
seq_lenth = 50
for i in range(len(seqs_activity)):
    all_term = seqs_activity[i].split()
    seqs.append (all_term[0])
    activity.append (float(all_term[1]))
dataset = list (zip(seqs,activity))

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

def get_input(seqs):
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

def get_strength(activity):
    data = np.zeros((1,1))
    count = 0
    for i in activity:
        count += 1
        if count == 1:
            data = data + np.array([[[float("%s"%i)]]])
            data = np.expand_dims(data, axis=0)
        if count != 1:
            data = np.concatenate((data,np.array([[[[float("%s"%i)]]]])),axis=0)
    return data

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
dataset = shuffle_dataset(dataset, 4297)

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2
dataset_train, dataset_test = split_dataset(dataset, 0.8)
trainset_seq = []
trainset_strength = []
for i in dataset_train:
    trainset_seq.append(i[0])
    trainset_strength.append(i[1])
testset_seq = []
testset_strength = []
for i in dataset_test:
    testset_seq.append(i[0])
    testset_strength.append(i[1])
trainseq = get_input(seqs = trainset_seq)
testseq = get_input(seqs = testset_seq)
train_strength = get_strength(activity = trainset_strength)
test_strength = get_strength(activity = testset_strength)    

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

dataset = MyDataset(x=trainseq, y=train_strength)
testset = MyDataset(x=testseq, y=test_strength)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last= True)
testloader = DataLoader(dataset=testset, batch_size=6107, drop_last= True)

def train():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    p = modules.predictor().to(device)
    print(device)
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    f = open('epoch_loss.txt', 'w')
    f.close()
    f = open('accuracy.txt', 'w')
    f.close()

    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    for epoch in range (200):
        epoch_loss = 0
    
        count = 0
        for i, data in enumerate (dataloader):
            count += 1
            x,y = data
            x = x.reshape(32,1,4,seq_lenth)
    
            x, y = x.to(device), y.to(device)
    
            output = p(x,32)
    
            y = y.reshape(32,1)
    
            p_loss = loss_fn(output,y)
            optimizer.zero_grad()
    
            epoch_loss += float(p_loss)
            p_loss.backward()
            optimizer.step()
        f = open('epoch_loss.txt','a+')
        loss_ave = 'epoch_loss %s\n'%(epoch_loss/count)
        f.write(loss_ave)
        f.close()
        print(epoch)
    
        state_dict = {"pred": p.state_dict(), "optim": optimizer.state_dict(),"epoch": epoch}
        torch.save(state_dict, r'saved_models/epoch_%s.pth'%(str(epoch)))
    
        with torch.no_grad():
            if epoch > 0:
                for i, data in enumerate (testloader):
                    x, y = data
                    x = x.reshape(6107,1,4,seq_lenth)
                    x = x.to(device)
                    output = p(x,6107)
                    output = output.cpu().detach().numpy()
                    a = output.reshape(-1,1)
                    a = np.round(a)
                    y = y.cpu().detach().numpy()
                    b = y.reshape(-1,1)
                    c = 0
                    for a1, b1 in zip(a,b):
                        if a1 == b1:
                            c += 1
                    c = c / 6107
                    f = open('accuracy.txt','a+')
                    f.write('epoch %s %s\n'%(epoch,c))
                    f.close()
    
if __name__ == '__main__':
    train()