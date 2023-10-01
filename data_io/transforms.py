#!/bin/python
# -----------------------------------------------------------------------------
# File Name :
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 16:26:06 2019
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch, bisect
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integers):
        # print(integers.shape)
        y_onehot = torch.FloatTensor(self.num_classes) #, integers.shape[-1])
        y_onehot.zero_()
        # print(integers, torch.LongTensor([integers]))
        return y_onehot.scatter_(0, torch.LongTensor([integers]), 1)


class Downsample(object):
    """Resize the address event Tensor to the given size.
    Args:
        factor: : Desired resize factor. Applied to all dimensions including time
    """

    def __init__(self, factor):
        assert isinstance(factor, int) or hasattr(factor, '__iter__')
        self.factor = factor

    def __call__(self, tmad):
        return tmad // self.factor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Crop(object):
    def __init__(self, low_crop, high_crop):
        '''
        Crop all dimensions
        '''
        self.low = low_crop
        self.high = high_crop

    def __call__(self, tmad):
        idx = np.where(np.any(tmad > self.high, axis=1))
        tmad = np.delete(tmad, idx, 0)
        idx = np.where(np.any(tmad < self.high, axis=1))
        tmad = np.delete(tmad, idx, 0)
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropDims(object):
    def __init__(self, low_crop, high_crop, dims):
        self.low_crop = low_crop
        self.high_crop = high_crop
        self.dims = dims

    def __call__(self, tmad):
        for i, d in enumerate(self.dims):
            idx = np.where(tmad[:, d] >= self.high_crop[i])
            tmad = np.delete(tmad, idx, 0)
            idx = np.where(tmad[:, d] < self.low_crop[i])
            tmad = np.delete(tmad, idx, 0)
            # Normalize
            tmad[:, d] = tmad[:, d] - self.low_crop[i]
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Attention(object):
    def __init__(self, n_attention_events, size):
        '''
        Crop around the median event in the last n_events.
        '''
        self.att_shape = np.array(size[1:], dtype=np.int64)
        self.n_att_events = n_attention_events

    def __call__(self, tmad):
        df = pd.DataFrame(tmad, columns=['t', 'p', 'x', 'y'])
        # compute centroid in x and y
        centroids = df.loc[:, ['x', 'y']].rolling(window=self.n_att_events,
                                                  min_periods=1).median().astype(int)
        # re-address (translate) events with respect to centroid corner
        df.loc[:, ['x', 'y']] -= centroids - self.att_shape // 2
        # remove out of range events
        df = df.loc[(df.x >= 0) & (df.x < self.att_shape[1]) & (df.y >= 0) & (df.y < self.att_shape[0])]
        return df.to_numpy()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToChannelHeightWidth(object):
    def __call__(self, tmad):
        n = tmad.shape[1]
        if n == 2:
            o = np.zeros(tmad.shape[0], dtype=tmad.dtype)
            return np.column_stack([tmad, o, o])

        elif n == 4:
            return tmad

        else:
            raise TypeError('Wrong number of dimensions. Found {0}, expected 1 or 3'.format(n - 1))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToCountFrame(object):
    """Convert Address Events to Binary tensor.
    Converts a numpy.ndarray (C x H x W x T??) to a torch.FloatTensor of shape (C x H x W x T) in the range [0., 1., ...]
    """

    def __init__(self, T=500, size=[2, 32, 32]):
        # pol, y, x
        self.T = T
        self.size = size

    def __call__(self, tmad):
        times = tmad[:, 0]
        t_start = times[0]
        t_end = times[-1]
        addrs = tmad[:, 1:]

        ts = range(0, self.T)
        chunks = np.zeros(self.size + [len(ts)], dtype='int8')
        idx_start = 0
        idx_end = 0
        for i, t in enumerate(ts):
            idx_end += find_first(times[idx_end:], t)
            if idx_end > idx_start:
                ee = addrs[idx_start:idx_end]
                pol_y_x_i = (ee[:, 0], ee[:, 2], ee[:, 1],  i)
                np.add.at(chunks, pol_y_x_i, 1)
            idx_start = idx_end
        return chunks

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Repeat(object):
    '''
    Replicate np.array (C) as (n_repeat X C). This is useful to transform sample labels into sequences
    '''

    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def __call__(self, target):
        return np.tile(np.expand_dims(target, -1), [1, self.n_repeat])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T X H x W x C)
    """

    def __call__(self, frame):
        """
        Args:
            frame (numpy.ndarray): numpy array of frames
        Returns:
            Tensor: Converted data.
        """
        return torch.FloatTensor(frame)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AddEmptySeq(object):
    """Add empty sequence to a numpy.ndarray (C x H x W x T) to (C x H x W x (T+TE))
    """
    
    def __init__(self, T_empty):
        self.T_empty = T_empty

    def __call__(self, frame):
        frame = np.concatenate((frame,np.zeros(frame.shape[:-1] +(self.T_empty,))), axis=-1)
        return frame
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ExChannel(object):
    """Exchange channel from (T x C x H x W) to  (C x H x W x T)
    """
    
    def __call__(self, frame):
        if frame.shape[1] == 2 :
            frame = frame.transpose(1,2,3,0)
        return frame
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class SpatialDownsample(object):
    """Resize the address event Tensor to the given size.
    Args:
        factor: : Desired resize factor. Applied to X, Y, input C x H x W x T
    """

    def __init__(self, factor=[4,4]):
        assert len(factor) == 2
        self.factor = factor

    def __call__(self, frame):
        return frame[:,::self.factor[0],::self.factor[1],:]

    def __repr__(self):
        return self.__class__.__name__ + '()'


def normalise(data):
    # min-max normalisation
    return (data- data.min())/(data.max()- data.min()+1e-8)