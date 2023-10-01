#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------

import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import *
import h5py
from .events_timeslices import *
import os
import time
import matplotlib.pyplot as plt

# dcll_folder = os.path.dirname(__file__)
data_folder = '/home/sl220/Documents/data/DVSGesture/'
mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}


class DVSGestureDataset(Dataset):
    def __init__(self,
                 filename,
                 group='train',
                 transform=None,
                 target_transform=None,
                 chunk_size=500,
                 empty_size=None,
                 clip=10,
                 dt=1000,
                 size=[2, 32, 32],
                 ds=4,
                 ):
        super(DVSGestureDataset, self).__init__()

        self.n = 0
        self.chunk_size = chunk_size
        self.window_size = chunk_size if empty_size is None else (empty_size + chunk_size)
        self.clip = clip
        self.dt = dt
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.ds = ds
        self.group = group

        f = h5py.File(filename, 'r', swmr=True, libver="latest")
        # self.stats = f['stats']
        self.grp = f[group]
        self.n = len(self.grp)


    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        # Important to open and close in getitem to enable num_workers>0

        # if self.group == 'train':
            # assert idx < 1175
        dset = self.grp[str(idx)]

        data, labels = sample_train(
            dset, T=self.chunk_size, dt=self.dt) #, is_train_Enhanced=self.is_train_Enhanced, 
        
        # plot_event_tensor(data)
        # plot_event_voxel_grid(data, 20)
        
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        
        data = my_chunk_evs_pol_dvs(data=data,
                                    dt=self.dt,
                                    T=self.chunk_size,
                                    size=self.size,
                                    ds=self.ds)
        
        
        data = np.concatenate((data,np.zeros(data.shape[:-1]+ (self.window_size - self.chunk_size,))), axis=-1)
        
        if self.transform is not None:
            data = self.transform(data)

        #----------------#
        # data = normalise(data)

        return data, labels


def sample_train(hdf5_file,
                 T=60,
                 dt=1000,
                #  is_train_Enhanced=False
                 ):
    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['time'][0]
    tend = np.maximum(0, hdf5_file['time'][-1] - T * dt)

    # start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0
    start_time = 0

    tmad = get_tmad_slice(hdf5_file['time'][()],
                          hdf5_file['data'][()],
                          start_time,
                          T * dt)
    tmad[:, 0] -= tmad[0, 0]
    return tmad[:, [0, 3, 1, 2]], label


def sample_test(hdf5_file,
                T=60,
                clip=10,
                dt=1000
                ):

    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['time'][0]
    tend = np.maximum(0, hdf5_file['time'][-1])

    tmad = get_tmad_slice(hdf5_file['time'][()],
                          hdf5_file['data'][()],
                          tbegin,
                          tend - tbegin)
    # 初试从1开始
    tmad[:, 0] = tmad[:, 0] - tmad[0, 0] +1

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clip * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clip * T * dt - (end_time - start_time)) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clip * T * dt) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg
        temp.append(tmad[idx_beg:idx_end][:, [0, 3, 1, 2]])

    return temp, label


def create_datasets(filename,
                    group='train',
                    data_size = [128,128], #y,x
                    chunk_size=60,
                    empty_size=None,
                    ds=[4,4],
                    dt=1000,
                    num_classes=10,
                    transform=None,
                    target_transform=None,
                    add_noise=False,
                    ):

    if isinstance(ds, int):
        ds = [ds, ds]

    size = [2, data_size[0] // ds[0], data_size[1] // ds[1]]

    # if n_events_attention is None:
    #     def default_transform(): return Compose([
    #         ToTensor()
    #     ])
    # else:
    def default_transform(): return Compose([
        ToTensor()
    ])

    if transform is None:
        transform = default_transform()

    if target_transform is not None:
        target_transform = Compose(
            [toOneHot(num_classes)]) #Repeat(chunk_size),  


    dataset = DVSGestureDataset(filename,
                                group=group,
                                transform=transform,
                                target_transform=target_transform,
                                chunk_size=chunk_size,
                                empty_size=empty_size,
                                # is_train_Enhanced=is_train_Enhanced,
                                dt=dt,
                                size=size,
                                ds=ds,
                                add_noise=add_noise)
    return dataset



# OUT
def plot_gestures_imshow(images, labels, nim=11, avg=50, do1h = True, transpose=False):
    import pylab as plt
    plt.figure(figsize = [nim+2,16])
    import matplotlib.gridspec as gridspec
    if not transpose:
        # print(images.shape)
        gs = gridspec.GridSpec(images.shape[2]//avg, nim)
    else:
        gs = gridspec.GridSpec(nim, images.shape[2]//avg)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=.0, hspace=.04)
    if do1h:
        categories = labels.argmax(axis=1)
    else:
        categories = labels
    s=[]
    for j in range(nim):
         for i in range(images.shape[2]//avg):
             if not transpose:
                 ax = plt.subplot(gs[i, j])
             else:
                 ax = plt.subplot(gs[j, i])
             plt.imshow(images.cpu().data[j,0, i*avg:(i*avg+avg),:,:].sum(axis=-1).T \
                 - images.cpu().data[j,1, i*avg:(i*avg+avg),:,:].sum(axis=-1).T)
             
             plt.xticks([])
             if i==0:  plt.title(mapping[labels[j,:].cpu().numpy().argmax()], fontsize=10)
             plt.yticks([])
             plt.gray()
         s.append(images[j].sum())
    print(s)
    plt.show()




if __name__ == '__main__':
    # path = os.path.dirname(
    #     os.path.dirname(
    #         os.path.dirname(
    #             os.path.abspath(__file__)))) + os.sep + 'dataset' + os.sep + 'DVS_Gesture'

    T = 60
    batch_size = 36
    train_dataset = create_datasets(filename='/home/sl220/Documents/DVS  Gesture dataset/DVS-Gesture-train.hdf5',
                                    group='train',
                                    # is_train_Enhanced=False,
                                    ds=[4,4],
                                    dt=1000*20,
                                    chunk_size=T,
                                    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0,
                                               pin_memory=True)

    test_dataset = create_datasets(filename='/home/sl220/Documents/DVS  Gesture dataset/DVS-Gesture-test.hdf5',
                                   group='test',
                                   ds=[4,4],
                                   dt=1000 * 20,
                                   chunk_size=T,
                                   clip=10
                                   )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              pin_memory=True)

    start = time.time()

    # i = 1
    # for idx, (input, labels) in enumerate(train_loader):
    #     print(i)
    #     i += 1
    #     print(input.size(), labels.size())

    # print('train:', time.time() - start)

    start = time.time()

    i = 1
    for idx, (input, labels) in enumerate(test_loader):
        print(i)
        i += 1
        print(input.size(), labels.size())

    print('test:',time.time() - start)



    # prefetcher = data_prefetcher(test_loader)
    # i =1
    # for i in range(len(test_loader)):
    #     data = prefetcher.next()
    #     print(i)
    #     i+=1


