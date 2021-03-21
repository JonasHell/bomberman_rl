import os
import numpy as np

import torch
from torch.utils.data import Dataset

class BomberManDataSet(Dataset):
    def __init__(self, directory, start_of_filename):
        # read csv files and append them to list
        features_input_list = []
        labels_input_list = []
        for filename in os.listdir(directory):
            if filename.startswith(start_of_filename):
                loaded = np.load(directory + filename)
                features_input_list.append(loaded["features"])
                labels_input_list.append(loaded["labels"])
                #content = np.genfromtxt(directory + filename, delimiter=",")
                #inputs.append(content)
      
        # combine input_lists to large arrays
        features = np.concatenate(features_input_list, axis=0)
        labels = np.concatenate(labels_input_list)
        #data = np.concatenate(inputs, axis=0)
        #print(data.shape)
        #print(data)
        
        # divide data in features and label (=last column)
        # and convert to torch tensors
        self.x = torch.tensor(features[0:4], dtype=torch.float)
        self.y = torch.tensor(labels[0:4], dtype=torch.int).long()
        #self.x = torch.tensor(data[:, :-1], dtype=torch.float)
        #self.y = torch.tensor(data[:, -1], dtype=torch.int).long()
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]