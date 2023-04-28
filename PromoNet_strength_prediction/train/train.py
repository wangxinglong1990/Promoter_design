import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import modules
from sklearn.linear_model import LinearRegression
import math
import os
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)
#torch.manual_seed(36)

f = open (r'constraint_based_dataset.txt')
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
    list = []
    for i in activity:
        list.append(math.log2(float(i)))
    count = 0
    for i in list:
        count += 1
        #a = float(i)
        #a = (a - float(min(list))) / (float((max(list))) - float(min(list))) ##max 2755441.06 min 9.91 9000-10000 99257.59 11.76
        if count == 1:
            data = data + np.array([[[float(i)]]])
            data = np.expand_dims(data, axis=0)
        if count != 1:
            data = np.concatenate((data,np.array([[[[float(i)]]]])),axis=0)

    return data #(11884, 1)

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
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, drop_last= True)
testloader = DataLoader(dataset=testset, batch_size=40, drop_last= True)

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

for i, data in enumerate (dataloader):
    a,b = data
    data = a.cpu().detach().numpy()
    dataTogene(data=data,bz=8,name='%s'%i)

for i, data in enumerate (testloader):
    a,b = data
    data = a.cpu().detach().numpy()
    dataTogene(data=data,bz=40,name='testloader')

def train(epochs=10000, lr=1e-3, device='cpu'):
    device = torch.device('%s'%device)
    p = modules.predictor().to(device)
    print(device)
    optimizer = torch.optim.Adam(p.parameters(),lr=lr,weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    f = open('epoch_loss.txt','w')
    f.close()
    f = open('accuracy.txt','w')
    f.close()
    f = open('predictValue.txt','w')
    f.close()

    model = LinearRegression()
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    for epoch in range (epochs):
        epoch_loss = 0

        count = 0
        f = open('visualize.txt','w')
        for i, data in enumerate (dataloader):
            count += 1
            x,y = data
            x = x.reshape(8,1,4,50)
            x, y = x.to(device), y.to(device)
            output = p(x,8)
            y = y.reshape(8,1)

            p_loss = loss_fn(output,y)
            a = str(output.cpu().detach().numpy().reshape(-1,1)).replace('[','').replace(']','')
            b = str(y.cpu().detach().numpy().reshape(-1,1)).replace('[','').replace(']','')
            f.write('epoch %s \n%s\n%s\n'%(epoch,a,b))
            optimizer.zero_grad()

            epoch_loss += float(p_loss)
            p_loss.backward()
            optimizer.step()
        f.close()
        f = open('epoch_loss.txt','a+')
        loss_ave = 'epoch_loss %s\n'%(epoch_loss/count)
        f.write(loss_ave)
        f.close()
        print(epoch)

        state_dict = {"pred": p.state_dict(), "optim": optimizer.state_dict(),"epoch": epoch}
        torch.save(state_dict, r'saved_models/epoch_%s.pth'%(str(epoch)))

        with torch.no_grad():
            cof = 0
            for i, data in enumerate (testloader):
                x, y = data
                x = x.reshape(40,1,4,50)
                x = x.to(device)
                output = p(x,40)
                output = output.cpu().detach().numpy()
                a = output.reshape(-1,1)
                y = y.cpu().detach().numpy()
                b = y.reshape(-1,1)
                model.fit(a,b)
                score = model.score(a,b)
                cof += float(score)

                cof = math.sqrt(cof)
                print(epoch,score)
                f = open('accuracy.txt','a+')
                f.write('epoch %s %s\n'%(epoch,cof))
                f.close()
                if epoch > 50:
                    f = open('predictValue.txt','a+')
                    f.write('epoch %s \n%s\n%s\n'%(epoch,str(a).replace('[','').replace(']',''),str(b).replace('[','').replace(']','')))
                    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', dest='epochs', type=int)
    parser.add_argument('-lr', dest='lr', type=int)
    parser.add_argument('-device', dest='device', type=str)
    args = parser.parse_args()
    train(epochs='%s'%args.epochs, lr='%s'%args.lr, device='%s'%args.device)