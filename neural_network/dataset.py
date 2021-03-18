import os
import numpy as np

import torch
from torch.utils.data import Dataset

class BomberManDataSet(Dataset):
    def __init__(self, directory, start_of_filename):

        # read csv files and append them to list
        inputs = []
        for filename in os.listdir(directory):
          print(filename)
          if filename.startswith(start_of_filename):
            print(filename)
            content = np.genfromtxt(filename, delimiter=",")
            inputs.append(content)
      
        # combine inputs to one large array
        data = np.concatenate(inputs, axis=0)
        print(data.shape)
        print(data)
        
        # divide data in features and label (=last column)
        # and convert to torch tensors
        self.x = torch.tensor(data[:, :-1], dtype=torch.float)
        self.y = torch.tensor(data[:, -1], dtype=torch.int)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]