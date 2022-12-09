import csv
import os
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.interpolate import interp1d

from constants import (
    raw_videos_path,
    balloon_pop_1_video_frame,
    balloon_pop_1_accel_frame,
    balloon_pop_3_video_frame,
    balloon_pop_3_accel_frame
)

csv_path = "../preprocess/audio/vad/"


def reset_examples_ids(examples):
    for i, ex in enumerate(examples):
        ex['id'] = i


'''
Generate information for example.pkl file
'''


class Maker():
    def __init__(self, accel_path=None, vad_path=None):

        self.accel = {}
        if accel_path is not None:
            self.load_accel(accel_path)

        self.vad = {}
        if vad_path is not None:
            self.load_vad(vad_path)

        self.examples = None

    def load_accel(self, accel_path):
        # self.accel = pickle.load(open(accel_path, 'rb'))
        self.accel = pickle.load(open('../data/subj_accel_interp.pkl', 'rb'))

    def load_vad(self, vad_path):
        self.vad = {}
        for i in range(1, 45):
            fpath = os.path.join(vad_path, f'{i}.vad')
            if os.path.exists(fpath) and os.path.isfile(fpath):
                self.vad[i] = pd.read_csv(fpath, header=None).to_numpy()
                print("i : ", self.vad[i], "  length : ", len(self.vad[i]))

        if len(self.vad) == 0:
            print('load_vad called but nothing loaded.')
        print("type : ", type(self.vad))

    # set time window
    def _get_vad(self, pid, ini_time, end_time, vad_fs=100):
        # note audio (vad) and video start at the same time
        if pid not in self.vad:
            return None

        # ini = round(ini_time * vad_fs)
        # width = round((end_time - ini_time) * vad_fs)
        # end = ini + width

        return self.vad[pid][ini_time:end_time].flatten()

    # def _interp_vad(self, vad, in_fs, out_fs):
    #     t = np.arange(0, len(vad) / in_fs, 1 / in_fs)
    #     f = interp1d(t, vad, kind='nearest')
    #     tnew = np.arange(0, len(vad) / in_fs, 1 / out_fs)
    #     return f(tnew)

    def make_examples(self, window_len=60):
        examples = list()
        example_id = 0

        #for _, key, value in enumerate(self.vad):

        for key, value in self.vad.items():
            # 打开对应的pid csv 和提取对应的list tuple
            time_window_list = []
            with open(csv_path + str(key) + ".csv") as infile:
                reader = csv.reader(infile)

                for line in reader:
                    if line:
                        print("line : ", line)
                        time_window_list.append(tuple([int(line[0]), int(line[1])]))

            for i in range(0, len(time_window_list)):
                ini_time = time_window_list[i][0]
                end_time = time_window_list[i][1]

                # pid, 开始时间，结束时间
                temp_vad = self._get_vad(key, ini_time, end_time)

                # interp_vad 是干嘛的
                # interp_vad = self._interp_vad(temp_vad, 100, 20)

                examples.append({
                    'id': example_id,
                    'pid': key,
                    'int_time': ini_time,
                    'end_time': end_time,
                    # data
                    'vad': temp_vad,

                })
                example_id += 1

        self.examples = examples
        return examples

    def filter_examples_by_movement_threshold(self, ts=20):
        new_examples = list()

        for ex in self.examples:
            track = ex['poses']
            std_x = np.std(track[:, 3])
            std_y = np.std(track[:, 4])

            if std_x > ts or std_y > ts:
                continue

            new_examples.append(ex)


