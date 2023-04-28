import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)

class ResBlock(nn.Module):
    def __init__(self,chanels):
        super(ResBlock, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(chanels, chanels, kernel_size =(1,3), padding=(0,1)),nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(chanels, chanels, kernel_size =(1,3), padding=(0,1)),nn.ReLU())
    def forward(self,x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        out = x + x2
        return out
        
class Self_Attn(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= (1,1) )
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= (1,1) )
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= (1,1) )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out 
        
class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.l1 = nn.Sequential((nn.Conv2d(1, 32, kernel_size = (1,6), stride = 1, padding=(0,5))), nn.ReLU())
        self.res1 = ResBlock(32)
        self.res1_1 = ResBlock(32)
        self.res1_2 = ResBlock(32)
        self.res1_3 = ResBlock(32)
        self.res1_4 = ResBlock(32)
        self.attn1 = Self_Attn(32, 'relu')
        self.l2 = nn.Sequential((nn.Conv2d(32, 64, kernel_size = (1,2), stride = (1,2), padding=0)), nn.ReLU())
        self.res2 = ResBlock(64)
        self.res2_1 = ResBlock(64)
        self.res2_2 = ResBlock(64)
        self.res2_3 = ResBlock(64)
        self.res2_4 = ResBlock(64)
        self.attn2 = Self_Attn(64, 'relu')
        self.l3 = nn.Sequential((nn.Conv2d(64, 128, kernel_size = (1,2), stride = (1,2), padding=0)), nn.ReLU())
        self.res3 = ResBlock(128)
        self.res3_1 = ResBlock(128)
        self.res3_2 = ResBlock(128)
        self.res3_3 = ResBlock(128)
        self.res3_4 = ResBlock(128)
        self.attn3 = Self_Attn(128, 'relu')
        self.l4 = nn.Sequential((nn.Conv2d(128, 256, kernel_size = (1,2), stride = (1,2), padding=0)), nn.ReLU())
        self.res4 = ResBlock(256)
        self.res4_1 = ResBlock(256)
        self.res4_2 = ResBlock(256)
        self.res4_3 = ResBlock(256)
        self.res4_4 = ResBlock(256)
        self.attn4 = Self_Attn(256, 'relu')
        self.dense = nn.Linear(6144,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, bz):
        x1 = self.l1(x)
        x2 = self.res1(x1)
        x2 = self.res1_1(x2)
        x2 = self.res1_2(x2)
        x2 = self.res1_3(x2)
        x2 = self.res1_4(x2)
        SA1 = self.attn1(x2)
        x3 = self.l2(SA1)
        x4 = self.res2(x3)
        x4 = self.res2_1(x4)
        x4 = self.res2_1(x4)
        x4 = self.res2_2(x4)
        x4 = self.res2_3(x4)
        x4 = self.res2_4(x4)
        SA2 = self.attn2(x4)
        x5 = self.l3(SA2)
        x6 = self.res3(x5)
        x6 = self.res3_1(x6)
        x6 = self.res3_2(x6)
        x6 = self.res3_3(x6)
        x6 = self.res3_4(x6)
        SA3 = self.attn3(x6)
        x7 = self.l4(SA3)
        x8 = self.res4(x7)
        x8 = self.res4_1(x8)
        x8 = self.res4_2(x8)
        x8 = self.res4_3(x8)
        x8 = self.res4_4(x8)
        SA4 = self.attn4(x8)
        x9 = SA4.reshape(bz,6144)
        x10 = self.dense(x9)
        out = self.sigmoid(x10)
        return out
