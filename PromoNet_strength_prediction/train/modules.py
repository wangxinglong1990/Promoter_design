import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)
        
class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.l1 = nn.Sequential((nn.Conv2d(1, 32, kernel_size = (3,3), stride = (3,3))), nn.BatchNorm2d(32), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = (1,3), stride = (1,3)), nn.BatchNorm2d(64),nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = (1,3), stride = (1,3)), nn.BatchNorm2d(128), nn.ReLU())
        self.dense = nn.Linear(128,1)
        #self.dense1 = nn.Linear(200,256)
        #self.dense2 = nn.Linear(256,128)
        #self.dense3 = nn.Linear(128,1)

    def forward(self,x,bz):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = x3.reshape(bz,128)
        out = self.dense(x3)
        #x3 = self.dense1(x2)
        #x4 = self.dense2(x3)
        #out = self.dense3(x4)
        return out
