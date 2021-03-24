import torch
import torch.nn as nn
import torch.nn.functional as F

class OurNeuralNetwork_flat(nn.Module):
    def __init__(self, input_size):
        super(OurNeuralNetwork_flat, self).__init__()
        self.linear1 = nn.Linear(input_size, 256) # input_size 1137
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 6)

    def forward(self, x):
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        return out

class OurNeuralNetwork_conv(nn.Module):
    def __init__(self):
        super(OurNeuralNetwork_conv, self).__init__()

        self.conv1 = nn.Conv2d(7, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv_dropout = nn.Dropout2d()

        self.linear1 = nn.Linear(64*2*2, 64)
        self.linear2 = nn.Linear(64, 6)

    def forward(self, x):
        out = self.conv1(x) # bs x 32 x 15 x 15
        #print(out.size()) # 15 is not divisible by 2 --> problem
        out = F.max_pool2d(out, 2) # bs x 32 x 7 x 7
        #print(out.size())
        out = F.relu(out) # bs x 32 x 7 x 7
        #print(out.size())

        out = self.conv2(out) # bs x 64 x 5 x 5
        #print(out.size())
        #out = self.conv_dropout(out) # bs x 64 x 5 x 5
        #print(out.size())
        out = F.max_pool2d(out, 2) # bs x 64 x 2 x 2
        #print(out.size())
        out = F.relu(out) # bs x 64 x 2 x 2
        #print(out.size())
        
        out = out.view(-1, 64*2*2) # bs x 256
        #print(out.size())
        out = self.linear1(out) # bs x 64
        #print(out.size())
        out = F.relu(out) # bs x 64
        #print(out.size())
        out = self.linear2(out) # bs x 6
        #print(out.size())
        
        return out

class OurNeuralNetwork_conv_2(nn.Module):
    def __init__(self):
        super(OurNeuralNetwork_conv, self).__init__()

        self.conv1 = nn.Conv2d(7, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv_dropout = nn.Dropout2d()

        self.linear1 = nn.Linear(64*2*2, 64)
        self.linear2 = nn.Linear(64, 6)

    def forward(self, x):
        out = self.conv1(x) # bs x 32 x 15 x 15
        #print(out.size()) # 15 is not divisible by 2 --> problem
        out = F.max_pool2d(out, 2) # bs x 32 x 7 x 7
        #print(out.size())
        out = F.relu(out) # bs x 32 x 7 x 7
        #print(out.size())

        out = self.conv2(out) # bs x 64 x 5 x 5
        #print(out.size())
        #out = self.conv_dropout(out) # bs x 64 x 5 x 5
        #print(out.size())
        out = F.max_pool2d(out, 2) # bs x 64 x 2 x 2
        #print(out.size())
        out = F.relu(out) # bs x 64 x 2 x 2
        #print(out.size())
        
        out = out.view(-1, 64*2*2) # bs x 256
        #print(out.size())
        out = self.linear1(out) # bs x 64
        #print(out.size())
        out = F.relu(out) # bs x 64
        #print(out.size())
        out = self.linear2(out) # bs x 6
        #print(out.size())
        
        return out
'''
import numpy as np
model = OurNeuralNetwork_conv()
load = np.load("neural_network_pretraining/test_data/coins_.npz")

x = load['features']
x = torch.tensor(x[0:3])
x = x.type(torch.float)

model(x)
'''