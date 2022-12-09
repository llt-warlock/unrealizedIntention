import os
import pickle
from math import floor
from typing import Any, Callable, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from pytorchvideo.data.video import VideoPathHandler




class AccelExtractor():
    def __init__(self, accel_path, strict=False):
        self.accel = pickle.load(open(accel_path, 'rb'))
        self.strict = strict

        self.num_channels = 3
        self.fs = 20

    def __call__(self, pid, start, end):
        if self.strict and pid not in self.accel:
            raise ValueError(f'pid {pid} not in self.accel')
        
        if pid not in self.accel:
            return np.zeros((self.num_channels, round(self.fs * (end-start))), dtype=np.float32)

        # accel_ini = video_seconds_to_accel_sample(start)
        # accel_fin = video_seconds_to_accel_sample(end)

        accel_ini = start
        accel_fin = end

        my_subj_accel = self.accel[pid]
        ini_idx = np.argmax(my_subj_accel[:,0] > accel_ini)
        fin_idx = ini_idx + round(self.fs * (end-start))

        # if ini_idx == 0:
        #     print('out of bounds. pid={:d}, accel_ini={:.2f}'.format(ex['person'], accel_ini))

        accel = my_subj_accel[ini_idx: fin_idx, 1:]

        if len(accel) < round(self.fs * (end-start)):
            accel = np.pad(accel, 
                ((0, round(self.fs * (end-start))-len(accel)), (0, 0)),
                mode='constant',
                constant_values= 0)

        return accel.transpose().astype(np.float32)

    def extract_multiple(self, keys):
        return np.stack([self(*k) for k in keys])
