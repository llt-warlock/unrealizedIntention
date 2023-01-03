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

# csv_path = "../preprocess/audio/vad/"
csv_path = "../preprocess/audio/po_ne_csv/"
test_csv_path = "../preprocess/audio/test_data_po_ne_csv/"
test_csv_path_unsuccessful = "../preprocess/audio/unsuccessful_intention_sample/"
txt_path = "../preprocess/audio/output_file/"
def reset_examples_ids(examples):
    for i, ex in enumerate(examples):
        ex['id'] = i


'''
Generate information for example.pkl file
'''


class Maker():
    def __init__(self, accel_path=None, vad_path=None, unsuccessful_vad_path=None):

        self.accel = {}
        if accel_path is not None:
            self.load_accel(accel_path)

        self.vad = {}
        if vad_path is not None:
            self.load_vad(vad_path)

        self.unsuccessful_vad = {}
        if unsuccessful_vad_path is not None:
            self.load_unsuccessful_vad(unsuccessful_vad_path)

        self.examples = None
        self.test_examples = None
        self.unsuccessful_examples = None

    def load_accel(self, accel_path):
        # self.accel = pickle.load(open(accel_path, 'rb'))
        self.accel = pickle.load(open('../data/subj_accel_interp.pkl', 'rb'))

    def load_vad(self, vad_path):
        print(" in load vad ")
        # load csv directly
        self.vad = {}
        pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        for i in pid_list:
            fpath = os.path.join(vad_path, f'{i}.csv')

            self.vad[i] = pd.read_csv(fpath, header=None).to_numpy()

        if len(self.vad) == 0:
            print('load_vad called but nothing loaded.')

    def load_unsuccessful_vad(self, unsuccessful_vad):
        start_pid = [2,3,4,7,10,11,17,22,23, 34]
        continue_pid = []
        maybe_pid = []
        for i in start_pid:
            fpath = os.path.join(unsuccessful_vad, f'{i}.csv')

            self.unsuccessful_vad[i] = pd.read_csv(fpath, header=None).to_numpy()

        if len(self.vad) == 0:
            print('load_vad called but nothing loaded.')

    # set time window
    def _get_vad(self, pid, ini_time, end_time, vad_fs=100):
        # note audio (vad) and video start at the same time
        if pid not in self.vad:
            return None

        #if len(self.vad[pid][ini_time * vad_fs: end_time * vad_fs]) == 0:
            # print(self.vad[pid][ini_time * vad_fs: end_time * vad_fs])
        unique, counts = np.unique(self.vad[pid][ini_time*100: end_time*100], return_counts=True)
        #print(" valid time : ",  len(self.vad[pid][ini_time*100: end_time*100]), "ini : ", ini_time, " end : ", end_time,  "! : ", dict(zip(unique, counts)))
        return self.vad[pid][ini_time*100: end_time*100].flatten()

    def _get_unsuccessful_vad(self, pid, ini_time, end_time, vad_fs=100):
        # note audio (vad) and video start at the same time
        if pid not in self.unsuccessful_vad:
            return None
        #if len(self.vad[pid][ini_time * vad_fs: end_time * vad_fs]) == 0:
            # print(self.vad[pid][ini_time * vad_fs: end_time * vad_fs])
        unique, counts = np.unique(self.vad[pid][ini_time*100: end_time*100], return_counts=True)
        #print(" valid time : ",  len(self.vad[pid][ini_time*100: end_time*100]), "ini : ", ini_time, " end : ", end_time,  "! : ", dict(zip(unique, counts)))
        return self.unsuccessful_vad[pid][ini_time*100: end_time*100].flatten()

    def make_examples(self, unsuccessful_pid, window_len=60):
        examples = list()
        example_id = 0
        test_examples = list()
        test_example_id = 0
        label_1 = 0
        label_0 = 0
        unsuccessful_examples = list()
        unsuccessful_example_id = 0


        # valid_list = [2, 3, 4, 5, 7, 10, 11, 12, 14, 15, 17, 18, 19, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35]
        valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]
        for i in valid_list:

            print("pid : ", i)

            time_window_list = []
            # read train participant csv
            with open(csv_path + str(i) + ".csv") as infile:
                reader = csv.reader(infile)

                for line in reader:
                    if line:
                        time_window_list.append(tuple([int(line[0]), int(line[1])]))


            # generate test dataset time window list:
            test_time_window_list = []
            with open(test_csv_path + str(i) + ".csv") as infile:
                test_reader = csv.reader(infile)

                for test_line in test_reader:
                    if test_line:
                        test_time_window_list.append(tuple([int(test_line[0]), int(test_line[1])]))

            if unsuccessful_pid.__contains__(i):
                unsuccessful_test_time_window_list = []
                with open(test_csv_path_unsuccessful + str(i) + ".csv") as infile:
                    unsuccessful_reader = csv.reader(infile)

                    for unsuccessful_line in unsuccessful_reader:
                        if unsuccessful_line:
                            unsuccessful_test_time_window_list.append(tuple([int(unsuccessful_line[0]), int(unsuccessful_line[1])]))

            # add into example
            for j in range(0, len(time_window_list)):
                print("in train list : ", j, "  , ", len(time_window_list))
                ini_time = time_window_list[j][0]
                end_time = time_window_list[j][1]

                temp_vad = self._get_vad(i, ini_time, end_time)
                #print("temp vad: ", temp_vad)

                examples.append({
                    'id': example_id,
                    'pid': i,
                    'ini_time': ini_time,
                    'end_time': end_time,
                    # data
                    'vad': temp_vad
                })
                example_id += 1


            # adding test dataset
            for j in range(0, len(test_time_window_list)):
                print("in test list : ", j, "  , ", len(test_time_window_list))
                test_ini_time = test_time_window_list[j][0]
                test_end_time = test_time_window_list[j][1]

                test_temp_vad = self._get_vad(i, test_ini_time, test_end_time)
                #print("temp vad: ", temp_vad)

                test_examples.append({
                    'id': test_example_id,
                    'pid': i,
                    'ini_time': test_ini_time,
                    'end_time': test_end_time,
                    # data
                    'vad': test_temp_vad
                })
                test_example_id += 1

            if unsuccessful_pid.__contains__(i):

                with open(txt_path + str(i) + ".txt", 'w') as f:

                    for j in range(0, len(unsuccessful_test_time_window_list)):
                        print("in test list : ", j, "  , ", len(unsuccessful_test_time_window_list))
                        unsuccessful_ini_time = unsuccessful_test_time_window_list[j][0]
                        unsuccessful_end_time = unsuccessful_test_time_window_list[j][1]

                        unsuccessful_temp_vad = self._get_unsuccessful_vad(i, unsuccessful_ini_time, unsuccessful_end_time)
                        # print("temp vad: ", temp_vad)

                        unsuccessful_examples.append({
                            'id': unsuccessful_example_id,
                            'pid': i,
                            'ini_time': unsuccessful_ini_time,
                            'end_time': unsuccessful_end_time,
                            # data
                            'vad': unsuccessful_temp_vad
                        })

                        temp = "indice : " + str(unsuccessful_example_id) + "  pid : " + str(i) + "  start time : " + str((unsuccessful_ini_time-3600)*1000) + "  end time :  " + str((unsuccessful_end_time-3600)*1000)
                        f.write(temp + '  \n')

                        unsuccessful_example_id += 1

                        # write txt
                        # with open(txt_path + str(i) + ".txt", 'w') as f:


                f.close()



        print("train len : ", len(examples), " test len : ", len(test_examples))
        self.examples = examples
        self.test_examples = test_examples
        self.unsuccessful_examples = unsuccessful_examples
        print("unsuccessful_example_id : ", unsuccessful_example_id)

        return examples, test_examples, unsuccessful_examples

    def filter_examples_by_movement_threshold(self, ts=20):
        new_examples = list()

        for ex in self.examples:
            track = ex['poses']
            std_x = np.std(track[:, 3])
            std_y = np.std(track[:, 4])

            if std_x > ts or std_y > ts:
                continue

            new_examples.append(ex)
