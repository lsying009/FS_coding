## codes are modified from https://github.com/Barchid/SNN-PyTorch/blob/18caf3f5e4061bbfac6f7e7740198e86570d43cc/torchneuromorphic/ntidigits/ntidigits_dataloaders.py
import numpy as np
from torch.utils.data import Dataset
from .transforms import *
import h5py
from .events_timeslices import *
import os


mapping = { 0 :'0',
            1 :'1',
            2 :'2',
            3 :'3',
            4 :'4',
            5 :'5',
            6 :'6',
            7 :'7',
            8 :'8',
            9 :'9',
            10: '10'}


class NTIdigitsDataset(Dataset):
    # resources_url = [['https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1',None, 'n-tidigits.hdf5']]
    # directory = 'data/tidigits/'
    # resources_local = [directory+'/n-tidigits.hdf5']


    def __init__(
            self, 
            filename,
            train=True,
            dt=3000, #1000*20,
            chunk_size=90,
            empty_size=0,
            size=64,
            num_classes=11,
            ):

        super(NTIdigitsDataset, self).__init__()
        self.n = 0
        self.filename = filename
        self.train = train 
        self.size = size
        self.chunk_size = chunk_size
        self.window_size = chunk_size if empty_size is None else (empty_size + chunk_size)
        self.dt = dt
        self.target_transform = Compose(
            [toOneHot(num_classes)])
        
        self.f = h5py.File(self.filename, 'r', swmr=True, libver="latest")
        
        # with h5py.File(self.filename, 'r', swmr=True, libver="latest") as f:
        if train:
            self.n = self.f['extra'].attrs['Ntrain']
            self.keys = self.f['extra']['train_keys']
        else:
            self.n = self.f['extra'].attrs['Ntest']
            self.keys = self.f['extra']['test_keys']


    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        # with h5py.File(self.filename, 'r', swmr=True, libver="latest") as f:
        if not self.train:
            key = key + self.f['extra'].attrs['Ntrain']
        data, target = sample(
                self.f,
                key,
                T = self.chunk_size,
                dt = self.dt)
        
        data = my_chunk_evs_audio(data, dt=self.dt, T=self.chunk_size, size=self.size)
        data = np.concatenate((data,np.zeros(data.shape[:-1]+ (self.window_size - self.chunk_size,))), axis=-1)
        
        #--------
        # data = normalise(data)
        
        target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key, T, dt):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    start_time = 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*dt)
    tmad[:,0]-=tmad[0,0]
    return tmad, label


def create_events_hdf5(directory, hdf5_filename):
    train_evs, train_labels_isolated = load_tidigit_hdf5(directory+'/n-tidigits.hdf5', train=True)
    test_evs, test_labels_isolated = load_tidigit_hdf5(directory+'/n-tidigits.hdf5', train=False)
    border = len(train_labels_isolated)

    tmad = train_evs + test_evs
    labels = train_labels_isolated + test_labels_isolated 
    test_keys = []
    train_keys = []

    with h5py.File(os.path.join(directory, hdf5_filename), 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        channels = 0
        end_times = []
        for i,data in enumerate(tmad):
            times = data[:,0]
            addrs = data[:,1:]
            label = labels[i]
            out = []
            istrain = i<border
            if istrain: 
                train_keys.append(key)
            else:
                test_keys.append(key)
            metas.append({'key':str(key), 'training sample':istrain}) 
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype=np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            assert label in mapping
            key += 1
            channels = addrs.max()
            end_times.append(times[-1]/1e6)
            
            # print(addrs.max(), times[-1]/1e6)
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
        print(len(train_keys), len(test_keys))
        end_times = np.sort(np.array(end_times))
        print(channels, end_times.min(), end_times.max(), end_times.mean(), end_times[:50], end_times[-50:])

def load_tidigit_hdf5(filename, train=True):
    with h5py.File(filename, 'r', swmr=True, libver="latest") as f:
        if train:
            train_evs = []
            train_labels_isolated = []
            for tl in f['train_labels']:
                label_ = tl.decode()
                label_s = label_.split('-')
                if len(label_s[-1])==1:
                    digit = label_s[-1]
                    if digit == 'o':
                        digit=10
                    if digit == 'z':
                        digit=0
                    else:
                        digit=int(digit)
                    train_labels_isolated.append(digit)
                    tm = np.int32(f['train_timestamps'][label_][:]*1e6)
                    ad = np.int32(f['train_addresses'][label_][:])
                    train_evs.append(np.column_stack([tm,ad]))
            return train_evs, train_labels_isolated

        else:
            test_evs = []
            test_labels_isolated  = []
            for tl in f['test_labels']:
                label_ = tl.decode()
                label_s = label_.split('-')
                if len(label_s[-1])==1:
                    digit = label_s[-1]
                    if digit == 'o':
                        digit=10
                    if digit == 'z':
                        digit=0
                    else:
                        digit=int(digit)
                    test_labels_isolated.append(digit)
                    tm = np.int32(f['test_timestamps'][label_][:]*1e6)
                    ad = np.int32(f['test_addresses'][label_][:])
                    test_evs.append(np.column_stack([tm,ad]))
            
            return test_evs, test_labels_isolated


