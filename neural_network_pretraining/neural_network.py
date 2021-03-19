import torch
import torch.nn as nn
import torch.nn.functional as F

class OurNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(OurNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 64) # input_size 257
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 6)

    def forward(self, x):
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        return out