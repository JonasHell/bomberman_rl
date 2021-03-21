import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

model = torch.load("neural_network_pretraining/15x15_ep100_bs32_lr0.01.pt", map_location=torch.device('cpu'))
print(model)
torch.save(model.state_dict(), "neural_network_pretraining/15x15_ep100_bs32_lr0.01_statedict.pt")