import os
import torch

from .dvs_gesture_reader import create_datasets
from .shd_ssc_reader import SHDSSCDataset
from .dvsplane_reader import DVSPlaneDataset
from .ntidigits_reader import NTIdigitsDataset
from .transforms import *


def load_dataset(args, kwargs, sim_params, mode='test'):
    num_classes = sim_params['dataset']['num_classes']
    batch_size =  args.batch_size if mode == 'train' else args.test_batch_size
    shuffle = True if mode == 'train' else False
    dataset_type = sim_params['dataset']['name'].casefold()
    path_to_dataset = sim_params['dataset']['path_to_dataset']

    if dataset_type in ['dvsgesture']:
        dataset_name = 'DVS-Gesture-train10.hdf5' if mode == 'train' else 'DVS-Gesture-test10.hdf5'
        group = 'train' if mode == 'train' else 'test'
        dataset = create_datasets(filename=os.path.join(path_to_dataset, dataset_name), # test10
                                group=group,
                                ds=[4,4],
                                dt=sim_params['simulation']['dt'] * 1000, #1000*20,
                                chunk_size=sim_params['simulation']['T'],
                                empty_size=sim_params['simulation']['T_empty'],
                                num_classes=num_classes,
                                target_transform = 'onehot',
                                )
        
    if dataset_type in ['shd']:
        dataset_name = 'shd_train.h5' if mode == 'train' else 'shd_test.h5'
        dataset = SHDSSCDataset(filename= os.path.join(path_to_dataset, dataset_name),
                                    T=sim_params['simulation']['T'],
                                    empty_size=sim_params['simulation']['T_empty'],
                                    dt=sim_params['simulation']['dt'],
                                    num_classes=num_classes)
        
    if dataset_type in ['dvsplane']:
        dataset_name = 'DVSPlane-train.hdf5' if mode == 'train' else 'DVSPlane-test.hdf5'
        dataset = DVSPlaneDataset(filename=os.path.join(path_to_dataset, dataset_name), # test10
                                    size=[240, 304],
                                    ds=[4,4],
                                    dt=sim_params['simulation']['dt']*1000, #1000*20,
                                    chunk_size=sim_params['simulation']['T'],
                                    empty_size=sim_params['simulation']['T_empty'],
                                    num_classes=num_classes,
                                    target_transform = 'onehot'
                                    )
    if dataset_type == 'ntidigits':
        dataset_name = 'n-tidigits-single.hdf5'
        is_train = True if mode == 'train' else False
        dataset = NTIdigitsDataset(filename=os.path.join(path_to_dataset, dataset_name), # test10
                                    train=is_train,
                                    size=64,
                                    dt=sim_params['simulation']['dt']*1000, #1000*20,
                                    chunk_size=sim_params['simulation']['T'],
                                    empty_size=sim_params['simulation']['T_empty'],
                                    num_classes=num_classes,
                                    )
    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               drop_last=False,
                                               **kwargs)

    return dataset_loader

