import os
import gzip, shutil

import numpy as np
import tables
from torch.utils.data import Dataset
from .transforms import *

 
# # The remote directory with the data files
# base_url = "https://zenkelab.org/datasets"
class SHDSSCDataset(Dataset):
    def __init__(self,
                 filename,
                 T=500,
                 empty_size=None,
                 dt=1,
                 num_classes=20,
                 ):
        super(SHDSSCDataset, self).__init__()
        self.chunk_size = T
        self.window_size = T if empty_size is None else (T+empty_size)
        self.dt = dt
        self.target_transform = Compose([toOneHot(num_classes)])

        
        fileh = tables.open_file(filename, mode='r')
        self.units = np.array(fileh.root.spikes.units, dtype=object)
        self.times = np.array(fileh.root.spikes.times, dtype=object)
        self.labels = np.array(fileh.root.labels)
        self.n = len(self.labels)
        fileh.close()
        
 
    def __len__(self):
        return self.n

    def spikes_to_tensor(self, idx):
        data = np.zeros((700, self.window_size), dtype=np.float32)
        time_bins = np.floor(self.times[idx] * 1000 / self.dt).astype(np.int16)
        used_ind = (time_bins < self.chunk_size)
        time_bins = time_bins[used_ind]
        units_bins = self.units[idx][used_ind]
        # print(max(time_bins), max(units_bins))
        # if len(units_bins) > 0:
        np.add.at(data, (units_bins, time_bins), 1) #######???
        
        
        # data = normalise(data)
        
        return data
        
    def __getitem__(self, idx):

        cur_data = self.spikes_to_tensor(idx)
        label = self.labels[idx]
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return cur_data, label


def get_and_gunzip(gz_file_path):
    hdf5_file_path=gz_file_path[:-3]
    if os.path.isfile(gz_file_path) and not os.path.exists(hdf5_file_path): # or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path
 

