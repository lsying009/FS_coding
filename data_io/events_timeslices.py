"""
Codes are modified from
https://github.com/Barchid/SNN-PyTorch/blob/18caf3f5e4061bbfac6f7e7740198e86570d43cc/torchneuromorphic/events_timeslices.py
"""
from __future__ import print_function

import numpy as np



def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y

def one_hot(mbt, num_classes):
    out = np.zeros([mbt.shape[0], num_classes])
    out[np.arange(mbt.shape[0], dtype='int'),mbt.astype('int')] = 1
    return out

import bisect
def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)



def my_chunk_evs_pol_dvs(data, dt=1000, T=500, size=[2, 240, 304], ds=[4,4]):
    t_start = data[0][0]
    ts = np.arange(t_start, t_start + T * dt, dt)
    chunks = np.zeros(size+[len(ts)], dtype='int8')
    # print(data[:,3].min(), data[:,3].max())
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:,0], t+dt)
        if idx_end > idx_start:
            ee = data[idx_start:idx_end,1:]
            pol, x, y = ee[:, 0], (ee[:, 1] // ds[1]).astype(np.int32), (ee[:, 2] // ds[0]).astype(np.int32)
            np.add.at(chunks, (pol, y, x, i), 1)
        idx_start = idx_end
    return chunks


def my_chunk_evs_audio(data, dt=1000, T=500, size=64):
    t_start = data[0][0]
    ts = np.arange(t_start, t_start + T * dt, dt)
    chunks =  np.zeros([size,]+[len(ts)], dtype='float32')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:,0], t+dt)
        if idx_end > idx_start:
            channel = data[idx_start:idx_end,1]
            np.add.at(chunks, (channel, i), 1)
        idx_start = idx_end
    return chunks



def my_chunk_evs_pol_time(data, dt=1000, T=500, size=[2, 240, 304], ds=[4,4]):
    t_start = data[0][0]
    ts = np.arange(t_start, t_start + T * dt, dt)
    chunks_sum = np.zeros(size+[len(ts)], dtype='int8')
    chunks_time =  np.zeros(size+[len(ts)], dtype='float32')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:,0], t+dt)
        # print(idx_end - idx_start)
        if idx_end > idx_start:
            ee = data[idx_start:idx_end]
            tm, pol, x, y = ee[:, 0], ee[:, 1], (ee[:, 2] // ds[0]).astype(np.int), (ee[:, 3] // ds[1]).astype(np.int)
            np.add.at(chunks_time, (pol, y, x, i), tm/1e6)
            np.add.at(chunks_sum, (pol, y, x, i), 1)
        idx_start = idx_end
    # max_id = np.argmax(chunks_sum.flatten())
    # print(np.sort(chunks_sum[chunks_sum > 0].flatten())[-500:])
    # print('max_sum', len(chunks_time[chunks_sum > 0]), chunks_sum.max())
    chunks_time[chunks_sum > 0] /= chunks_sum[chunks_sum>0]
    # print((chunks_time[chunks_time > 0].flatten()))
    chunks_spike = chunks_sum
    chunks_spike[chunks_spike>0] = 1
    return chunks_time, chunks_spike




def get_slice(times, addrs, start_time, end_time):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], end_time)+idx_beg
        return times[idx_beg:idx_end]-times[idx_beg], addrs[idx_beg:idx_end]
    except IndexError:
        raise IndexError("Empty batch found")
    

def get_tmad_slice(times, addrs, start_time, T):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time+T)+idx_beg
        return np.column_stack([times[idx_beg:idx_end], addrs[idx_beg:idx_end]])
    except IndexError:
        raise IndexError("Empty batch found")

