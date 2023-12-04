#%%
import torch
import torch.utils.data as data
import pandas as pd 
import numpy as np

class ECG(data.Dataset):
    def __init__(self, data):
        self.data = pd.read_csv(data, sep=',', header=None)
        self.samples = self.data.iloc[:, :187]
        self.targets = self.data[187].to_numpy()


    def __getitem__(self, index):
        x = self.samples.iloc[index, :]
        x = torch.from_numpy(x.values).float()
        x = torch.unsqueeze(x, 0)
        y = self.targets[index].astype(np.int64)
        return x, y

    def __len__(self):
        return len(self.data)

