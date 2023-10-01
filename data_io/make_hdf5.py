# import tarfile
import os
import h5py
import numpy as np
import struct
import scipy.io as sio
import glob
from events_timeslices import *

data_folder = '/home/sl220/Documents/DVS  Gesture dataset/'

alphabet_label = {}
alphabet_label['a'] = 0
alphabet_label['b'] = 1
alphabet_label['c'] = 2
alphabet_label['d'] = 3
alphabet_label['e'] = 4
alphabet_label['f'] = 5
alphabet_label['g'] = 6
alphabet_label['h'] = 7
alphabet_label['i'] = 8
alphabet_label['k'] = 9
alphabet_label['l'] = 10
alphabet_label['m'] = 11
alphabet_label['n'] = 12
alphabet_label['o'] = 13
alphabet_label['p'] = 14
alphabet_label['q'] = 15
alphabet_label['r'] = 16
alphabet_label['s'] = 17
alphabet_label['t'] = 18
alphabet_label['u'] = 19
alphabet_label['v'] = 20
alphabet_label['w'] = 21
alphabet_label['x'] = 22
alphabet_label['y'] = 23

plane_label = {}
plane_label['F117'] = 0
plane_label['Mig31'] = 1
plane_label['Su24'] = 2
plane_label['Su35'] = 3


# def untar(fname, dirs):
    
#     t = tarfile.open(fname)
#     t.extractall(path=dirs)

# read aedat file names
def gather_aedat(directory, start_id, end_id, filename_prefix = 'user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))
    import glob
    fns = []
    for i in range(start_id,end_id):
        search_mask = directory+'/'+filename_prefix+"{0:02d}".format(i)+'*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out)>0:
            fns+=glob_out
    return fns

# read aedat file data
def aedat_to_events(filename):
    # read classes
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename,
                        skiprows=1,
                        delimiter=',',
                        dtype='uint32')

    events = []
    with open(filename, 'rb') as f:

        # skip other info
        for i in range(5):
            _ = f.readline()

        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize),
                                            'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)

    events = np.column_stack(events)
    events = events.astype('uint32')

    clipped_events = np.zeros([4, 0], 'uint32')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events,
                                          events[:, start:end]])

    return clipped_events.T, labels


# XX
def gather_gestures_stats(hdf5_grp):
    from collections import Counter
    labels = []
    for d in hdf5_grp:
        labels += hdf5_grp[d]['labels'][:,0].tolist()
    count = Counter(labels)
    stats = np.array(list(count.values()))
    stats = stats/ stats.sum()
    return stats
   

   
 
# create hdf5 file, train/test, not slice
def create_events_hdf5(path_save):
    fns_train = gather_aedat(os.path.join(data_folder, 'DvsGesture'),1,24)
    fns_test = gather_aedat(os.path.join(data_folder, 'DvsGesture'),24,30)

    with h5py.File(os.path.join(path_save,'DVS-Gesture-train.hdf5'), 'w') as f:
        f.clear()

        print("processing training data...")
        key = 0
        train_grp = f.create_group('train')
        for file_d in fns_train:
            print(key)
            events, labels = aedat_to_events(file_d)
            print(events, labels)
            subgrp = train_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1
        stats =  gather_gestures_stats(train_grp)
        f.create_dataset('stats',stats.shape, dtype = stats.dtype)
        f['stats'][:] = stats

    with h5py.File(os.path.join(path_save,'DVS-Gesture-test.hdf5'), 'w') as f:
        f.clear()
        print("processing testing data...")
        key = 0
        test_grp = f.create_group('test')
        for file_d in fns_test:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = test_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1
        stats =  gather_gestures_stats(test_grp)
        f.create_dataset('stats',stats.shape, dtype = stats.dtype)
        f['stats'][:] = stats
        

# create hdf5 file, slice data in single file
def create_hdf5(save_path):
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'train')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)

    fns_train = gather_aedat(os.path.join(data_folder, 'DvsGesture'),1,24)
    fns_test = gather_aedat(os.path.join(data_folder, 'DvsGesture'),24,30)

    with h5py.File(save_path_train + os.sep + 'DVS-Gesture-train.hdf5', 'w') as f:
        train_grp = f.create_group('train')
        key = 0
        for i in range(len(fns_train)):
            print('processing' + str(i + 1) + 'th train data')
            data, labels_starttime = aedat_to_events(fns_train[i])
            tms = data[:, 0]
            ads = data[:, 1:]
            lbls = labels_starttime[:, 0]
            start_tms = labels_starttime[:, 1]
            end_tms = labels_starttime[:, 2]

            for lbls_idx in range(len(lbls)):
                print(lbls_idx, key)
                subgrp = train_grp.create_group(str(key))
                s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
                times = s_[0]
                addrs = s_[1]
                # str(i * 12 + lbls_idx + 1)
                # with h5py.File(save_path_train + os.sep + 'DVS-Gesture-train' + str(i * 12 + lbls_idx + 1) + '.hdf5',
                #             'w') as f:
                tm_dset = subgrp.create_dataset('time', data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset('data', data=addrs, dtype=np.uint8)
                lbl_dset = subgrp.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)
                key += 1

        print('training dataset ... done!')


    print('processing test data...')
    save_path_test = os.path.join(save_path, 'test')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    with h5py.File(save_path_train + os.sep + 'DVS-Gesture-test.hdf5', 'w') as f:
        test_grp = f.create_group('test')
        key = 0
        for i in range(len(fns_test)):
            print('processing' + str(i + 1) + 'th test data')
            data, labels_starttime = aedat_to_events(fns_test[i])
            tms = data[:, 0]
            ads = data[:, 1:]
            lbls = labels_starttime[:, 0]
            start_tms = labels_starttime[:, 1]
            end_tms = labels_starttime[:, 2]

            for lbls_idx in range(len(lbls)):
                subgrp = test_grp.create_group(str(key))
                print(lbls_idx, key)
                s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
                times = s_[0]
                addrs = s_[1]
                # with h5py.File(save_path_test + os.sep + 'DVS-Gesture-test' + str(i * 12 + lbls_idx + 1) + '.hdf5', 'w') as f:
                tm_dset = subgrp.create_dataset('time', data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset('data', data=addrs, dtype=np.uint8)
                lbl_dset = subgrp.create_dataset('labels', data=lbls[lbls_idx] - 1, dtype=np.uint8)
                key += 1

    print('test dataset ... done!')


def rearrange_files(file_list):
    result_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return result_list

def gather_asldvs_filenames(src_path, start_id, end_id):
    selected_files = []
    for dir in sorted(os.listdir(src_path)):
        filenames = glob.glob(os.path.join(src_path, dir, '*.mat'))
        filenames = sorted(filenames)
        
        for file in filenames:
            id = int(file.split('/')[-1].split('.')[0].split('_')[-1])
            if id > start_id and id <= end_id:
                selected_files.append(file)
                
    ## rearrange file names
    arranged_files =  rearrange_files(selected_files)

    return arranged_files

def gather_dvsplane_filenames(src_path, start_id, end_id):
    selected_files = []
    
    filenames = glob.glob(os.path.join(src_path, '*.mat'))
    filenames = sorted(filenames)
    
    for file in filenames:
        id = int(file.split('/')[-1].split('.')[0].split('_')[-1])
        if id > start_id and id <= end_id:
            selected_files.append(file)
    arranged_files =  rearrange_files(selected_files)
    return arranged_files
        

def mat_to_events(filename):
    # ASL-DVS, 240 x 180
    # DVS-plane 304 x 240
    # Read data from `raw_path`.
    content = sio.loadmat(filename)
    pol = np.array(content['pol'], np.int8)
    ts = np.array(content['ts'], np.float32)
    x = np.array(content['x'], np.uint16)
    y = np.array(content['y'], np.uint16)
    
    events = np.column_stack([ts, x, y, pol])
    alphabet = filename.split('/')[-2]
    label = alphabet_label[alphabet]
    return events, label


def mat_to_events_plane(filename):
    # DVS-plane 304 x 240, p=1/-1, x/y start from 1 to 304/240
    # Read data from `raw_path`.
    content = sio.loadmat(filename, matlab_compatible=True)
    content = content['TD']

    pol = np.array(content['p'][0,0], np.int8) #np.array(np.squeeze(content['p']), np.int8)
    ts = np.array(content['ts'][0,0], np.float32)
    x =  np.array(content['x'][0,0], np.int16)
    y =  np.array(content['y'][0,0], np.int16)
    
    x[x > 0] -= 1
    y[y > 0] -= 1
    pol[pol < 0] = 0
    
    events = np.column_stack([ts, x, y, pol])
    plane = filename.split('/')[-1].split('_')[0]
    label = plane_label[plane]
    return events, label


def clip_events(events):
    ts = events[:, 0] / 1000
    win = 1000
    
    iterators = [ts[i+win]-ts[i] for i in range(len(ts)-win)]
    slide_slope = np.stack(iterators) / win
    m_val = 10*slide_slope.min()
    id = np.where(slide_slope < m_val)[0]
    
    clipped_events = events[id[0]:id[-1],:]
    time_diff = (clipped_events[-1,0]-clipped_events[0,0])/1000
    
    if time_diff > 300:
        m_val /= 2
        id = np.where(slide_slope < m_val)[0]
        clipped_events = events[id[0]:id[-1],:]
        id_diff = id[1:] - id[:-1]
        id_gap = np.where(id_diff>2)[0]
        if len(id_gap)>0 and (id_gap > 2000).any():
            id = id[:id_gap[id_gap > 2000][0]]
            clipped_events = events[id[0]:id[-1],:]
            
    if (clipped_events[-1,0]-clipped_events[0,0])/1000 > 300 or (clipped_events[-1,0]-clipped_events[0,0])/1000 < 180:
        print(ts.shape)
        print(clipped_events.shape, (clipped_events[-1,0]-clipped_events[0,0])/1000)
    return clipped_events

# create hdf5 file, slice data in single file
def create_hdf5_asldvs(src_path, save_path, save_name):
    # print('processing train data...')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fns_train = gather_asldvs_filenames(src_path, start_id=0, end_id=3360)
    fns_test = gather_asldvs_filenames(src_path, start_id=3360, end_id=4200)
    fns_valid = gather_asldvs_filenames(src_path, start_id=0, end_id=500)
    fns_small_test = gather_asldvs_filenames(src_path, start_id=3360, end_id=3760)
    # #ASL-DVS-train
    # with h5py.File(save_path + os.sep + '{}-train.hdf5'.format(save_name), 'w') as f:
        
    #     for key in range(len(fns_train)):
    #         subgrp = f.create_group(str(key))
    #         print('processing' + str(key + 1) + 'th train data')
    #         events, label = mat_to_events(fns_train[key])
    #         time = events[:, 0]
    #         data = events[:, 1:]
            
    #         subgrp.create_dataset('time', data=time, dtype=np.uint32)
    #         subgrp.create_dataset('data', data=data, dtype=np.uint8)
    #         subgrp.create_dataset('labels', data=label, dtype=np.uint8)

    #     print('training dataset ... done!')


    # print('processing test data...')

    with h5py.File(save_path + os.sep + '{}-test.hdf5'.format(save_name), 'w') as f:
        # test_grp = f.create_group('test')
        key = 0
        for key in range(len(fns_test)):
            subgrp = f.create_group(str(key))
            print('processing' + str(key + 1) + 'th test data')
            events, label = mat_to_events(fns_test[key])
            time = events[:, 0]
            data = events[:, 1:]
            
            subgrp.create_dataset('time', data=time, dtype=np.uint32)
            subgrp.create_dataset('data', data=data, dtype=np.uint8)
            subgrp.create_dataset('labels', data=label, dtype=np.uint8)

    print('test dataset ... done!')
    
    # with h5py.File(save_path + os.sep + '{}-valid.hdf5'.format(save_name), 'w') as f:
    #     # test_grp = f.create_group('test')
    #     key = 0
    #     for key in range(len(fns_valid)):
    #         subgrp = f.create_group(str(key))
    #         print('processing' + str(key + 1) + 'th validation data')
    #         events, label = mat_to_events(fns_valid[key])
    #         time = events[:, 0]
    #         data = events[:, 1:]
            
    #         subgrp.create_dataset('time', data=time, dtype=np.uint32)
    #         subgrp.create_dataset('data', data=data, dtype=np.uint8)
    #         subgrp.create_dataset('labels', data=label, dtype=np.uint8)

    # print('Validation dataset ... done!')

    # with h5py.File(save_path + os.sep + '{}-small-test.hdf5'.format(save_name), 'w') as f:
    #     # test_grp = f.create_group('test')
    #     key = 0
    #     for key in range(len(fns_small_test)):
    #         subgrp = f.create_group(str(key))
    #         print('processing' + str(key + 1) + 'th small test data')
    #         events, label = mat_to_events(fns_small_test[key])
    #         time = events[:, 0]
    #         data = events[:, 1:]
            
    #         subgrp.create_dataset('time', data=time, dtype=np.uint32)
    #         subgrp.create_dataset('data', data=data, dtype=np.uint8)
    #         subgrp.create_dataset('labels', data=label, dtype=np.uint8)

    # print('Small test dataset ... done!')
    

# create hdf5 file, slice data in single file
def create_hdf5_dvsplane(src_path, save_path, save_name):
    print('processing train data...')
    # save_path_train = os.path.join(save_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fns_train = gather_dvsplane_filenames(src_path, start_id=0, end_id=160)
    fns_test = gather_dvsplane_filenames(src_path, start_id=160, end_id=200)

    #ASL-DVS-train
    with h5py.File(save_path + os.sep + '{}-train.hdf5'.format(save_name), 'w') as f:
        
        for key in range(len(fns_train)):
            subgrp = f.create_group(str(key))
            print('processing ' + str(key + 1) + 'th train data')
            # print(fns_train[key])
            events, label = mat_to_events_plane(fns_train[key])
            events = clip_events(events)
            time = events[:, 0]
            data = events[:, 1:]
            
            subgrp.create_dataset('time', data=time, dtype=np.uint32)
            subgrp.create_dataset('data', data=data, dtype=np.uint16)
            subgrp.create_dataset('labels', data=label, dtype=np.uint8)

        print('training dataset ... done!')


    print('processing test data...')

    with h5py.File(save_path + os.sep + '{}-test.hdf5'.format(save_name), 'w') as f:
        # test_grp = f.create_group('test')
        for key in range(len(fns_test)):
            subgrp = f.create_group(str(key))
            print('processing ' + str(key + 1) + 'th test data')
            events, label = mat_to_events_plane(fns_test[key])
            events = clip_events(events)
            time = events[:, 0]
            data = events[:, 1:]
            
            subgrp.create_dataset('time', data=time, dtype=np.uint32)
            subgrp.create_dataset('data', data=data, dtype=np.uint16)
            subgrp.create_dataset('labels', data=label, dtype=np.uint8)

    print('test dataset ... done!')



if __name__ == '__main__':
    # path = os.path.dirname(
    #     os.path.dirname(
    #         os.path.dirname(
    #             os.path.abspath(__file__)))) + os.sep + 'dataset' + os.sep + 'DVS_Gesture'

    # create_events_hdf5(data_folder)
    # create_hdf5(data_folder)
    
    
    asldvs_dir = '/home/sl220/Documents/data/ASL-DVS-src/'
    save_path = '/home/sl220/Documents/data/ASL-DVS/'
    # fns_train = gather_asldvs_filenames(asldvs_dir, start_id=0, end_id=3360)
    create_hdf5_asldvs(asldvs_dir, save_path, save_name='ASL-DVS')
    
    # asldvs_dir = '/home/sl220/Documents/data/DVSPlane/dvsplane/'
    # save_path = '/home/sl220/Documents/data/DVSPlane/'
    # create_hdf5_dvsplane(asldvs_dir, save_path, save_name='DVSPlane')