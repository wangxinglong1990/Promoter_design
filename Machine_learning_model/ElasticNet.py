import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import math
import os
seq_lenth=50
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
seqs = np.load('promoter.npy')
activity = np.load('gene_expression.npy')
dataset = list (zip(seqs,activity))


def get_input(seqs):
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


def get_strength(activity):
    data = np.zeros((1, 1))
    list = []
    for i in activity:
        list.append(math.log2(float(i)))
    count = 0
    for i in list:
        count += 1
        if count == 1:
            data = data + np.array([[[float(i)]]])
            data = np.expand_dims(data, axis=0)
        if count != 1:
            data = np.concatenate((data, np.array([[[[float(i)]]]])), axis=0)

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


dataset_train, dataset_test = split_dataset(dataset, 0.9)
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
trainseq = get_input(seqs=trainset_seq)
testseq = get_input(seqs=testset_seq)
train_strength = get_strength(activity=trainset_strength)
test_strength = get_strength(activity=testset_strength)


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
trainseq = trainseq.reshape(np.shape(trainseq)[0],np.shape(trainseq)[1]*np.shape(trainseq)[2]*np.shape(trainseq)[3])
train_strength = train_strength.reshape(np.shape(train_strength)[0]*np.shape(train_strength)[1]*np.shape(train_strength)[2]*np.shape(train_strength)[3],1)
y_train = train_strength.ravel()

testseq = testseq.reshape(np.shape(testseq)[0],np.shape(testseq)[1]*np.shape(testseq)[2]*np.shape(testseq)[3])
test_strength = test_strength.reshape(np.shape(test_strength)[0]*np.shape(test_strength)[1]*np.shape(test_strength)[2]*np.shape(test_strength)[3],1)

elastic= linear_model.ElasticNet(alpha=0.1,max_iter=10000,l1_ratio=0.5)
elastic.fit(trainseq,train_strength)

print(elastic.score(trainseq, train_strength))
y_pred = elastic.predict(testseq)
y_test = test_strength
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
a = y_pred.reshape(-1,1)
b = y_test.reshape(-1,1)
model = LinearRegression()
model.fit(a,b)
print(model.score(a,b))
