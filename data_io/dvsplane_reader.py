#### dataset from https://github.com/PIX2NVS/NVS2Graph/blob/master/code/inputsdata.py
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import *
import h5py
from .events_timeslices import *
import os
import time
import matplotlib.pyplot as plt



class DVSPlaneDataset(Dataset):
    def __init__(self,
                 filename,
                 size=[240, 304],
                 num_classes=4,
                 ds=4,
                 target_transform=None,
                 chunk_size=500,
                 empty_size=None,
                 dt=1000,
                 ):
        super(DVSPlaneDataset, self).__init__()

        self.n = 0
        self.chunk_size = chunk_size
        self.window_size = chunk_size if empty_size is None else (empty_size + chunk_size)
        self.dt = dt
        self.target_transform = Compose(
            [toOneHot(num_classes)])
        self.size = size
        self.ds = ds
        self.output_size = [2, size[0] // ds[0], size[1] // ds[1]]

        self.f = h5py.File(filename, 'r', swmr=True, libver="latest")
        self.n = len(self.f)


    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        # Important to open and close in getitem to enable num_workers>0
        dset = self.f[str(idx)]
        
        time = dset['time'][()]
        time -= time[0]
        data = dset['data'][()]
        label = dset['labels'][()]
        # print(time[-1]/10000)
        data = get_tmad_slice(time, data, 0, self.chunk_size * self.dt)
        data = data[:, [0, 3, 1, 2]] # t, p, x, y

        
        data = my_chunk_evs_pol_dvs(data=data,
                                    dt=self.dt,
                                    T=self.chunk_size,
                                    size=self.output_size,
                                    ds=self.ds)
 
        data = np.concatenate((data,np.zeros(data.shape[:-1]+ (self.window_size - self.chunk_size,))), axis=-1)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        return data, label
  
