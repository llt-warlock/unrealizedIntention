import csv
import os
from random import random

import numpy as np
import pandas as pd
from scipy.io.wavfile import write as wavwrite, read as wavread
from constants import (processed_audio_path)

diarizations_path = './rttmFile'
vad_path = './vad'


def load_diarization(fpath):
    return pd.read_csv(fpath,
                       header=None,
                       names=['x', 'y', 'z', 'ini', 'dur', 'n1', 'n2', 'speaker', 'n3', 'n4'],
                       usecols=['ini', 'dur', 'speaker'],
                       delim_whitespace=True,
                       index_col=False)


# d = load_diarization(os.path.join(diarizations_path, '7.rttm'))
# row = d.iloc[7, :]

main_speakers = {
    1: [0],
    2: [0],
    3: [0],
    4: [0],
    5: [0],
    7: [1, 3],
    9: [0],
    10: [1],
    11: [0],
    12: [0, 1],
    13: [1],
    14: [1],
    15: [0],
    16: [0],
    17: [0],
    18: [0],  # fail: two women with same voice
    19: [0],
    20: [0],
    21: [1],
    22: [0],
    23: [1],
    24: [0],
    25: [1],
    26: [0],
    27: [2],
    29: [1, 3],  # check
    30: [0, 3],  # check
    31: [0],
    32: [0],
    33: [0],
    34: [0],
    35: [1],
    45: [0]
}

# produce VAD

import numpy as np
from scipy.io.wavfile import write as wavwrite


'''
Generate positive example from vad files
'''
def generate_positive_sample(vad, size=9900, fs=100):
    valid = True
    intention_time_window = []
    intention_label = np.zeros((size * fs))
    change = False
    for i in range(0, len(vad)):
        if vad[i] == 1 and i - 200 >= 0 and not change:
            change = True
            # check valid time window
            for j in range(i, i - 200, -1):
                if vad[j] == 1:
                    valid = False
                    break

            # intention 2 seconds window (2 * 100)
            if valid:
                time_ini = i - 200
                time_end = i - 1
                intention_time_window.append(tuple([time_ini, time_end]))
                intention_label[time_ini:time_end] = 1

        if vad[i] == 0 and change:
            change = False

    return intention_time_window, intention_label


'''
Generate intention time window, and vad fiels from ".rttm" files. 
'''
def make_vad(df: pd.DataFrame, pid, size=9900, fs=100):
    ''' len is in seconds
    '''
    time_window = []

    vad = np.zeros((size * fs))
    '''
    loop the rttm files 
    
    '''
    for idx, row in df.iterrows():
        spk = int(row['speaker'].split('_')[1])
        if pid in main_speakers and spk not in main_speakers[pid]:
            continue

        print("ini is :  ", row['ini'])
        ini = round(row['ini'] * fs)
        end = round((row['ini'] + row['dur']) * fs)

        # 2 seconds
        time_ini = round((row['ini'] - 2) * fs)
        time_end = round(row['ini'] * fs)
        time_window.append(tuple([time_ini, time_end]))
        vad[ini:end] = 1

    return vad, time_window


def store_vad(df: pd.DataFrame, pid, fname, size=9900, fs=100):
    vad, time_list = make_vad(df, pid, size=size, fs=fs)

    # write csv
    with open(fname + '.csv', "w") as f_f:
        csv_writer = csv.writer(f_f)
        for mytuple in time_list:
            csv_writer.writerow(mytuple)

    # write .vad file
    np.savetxt(fname + '.vad', vad, fmt='%d')
    # write .wav file
    wavwrite(fname + '.wav', fs, vad)


def load_vad(vad_path_0, pid_list):
    vad = {}
    for i in range(0, len(pid_list)):
        fpath = os.path.join(vad_path_0, f'{pid_list[i]}.vad')
        print("fpath : ", fpath)
        if os.path.exists(fpath) and os.path.isfile(fpath):
            print("valid pid file : ", fpath)
            vad[pid_list[i]] = pd.read_csv(fpath, header=None).to_numpy()

    if len(vad) == 0:
        print('load_vad called but nothing loaded.')
    print("type : ", type(vad))
    return vad


# make vad files
from pathlib import Path

if __name__ == '__main__':
    for f in Path(diarizations_path).glob('*.rttm'):
        df = load_diarization(f)
        pid = int(f.stem)
        # load corresponding vad file based on rttm files
        out_path = os.path.join(vad_path, f.stem)
        print("out_path : ", out_path)
        '''
        df : diarization file
        out_path : output path
        '''

        store_vad(df, pid, out_path)

# if __name__ == '__main__':
#     pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 18, 22, 23, 27, 34, 35]  # len = 14
#     vad_dict = load_vad(vad_path, pid_list)
#     print("keys: ", vad_dict.keys())
#     for i in range(0, len(pid_list)):
#         print("i : ", i, " pid = ", pid_list[i])
#         pos_example_time_list, label_pos = generate_positive_sample(vad_dict[pid_list[i]])
#         neg_example_time_list, label_neg = generate_negative_sample(vad_dict[pid_list[i]])
#         time_window = pos_example_time_list + neg_example_time_list
#         # write csv
#         with open('./po_ne_csv/' + str(pid_list[i]) + '.csv', "w") as f:
#             csv_writer_new = csv.writer(f)
#             for time_tuple in time_window:
#                 csv_writer_new.writerow(time_tuple)
#
#
#         with open('./target_label/' + str(pid_list[i]) + '.csv', "w") as f:
#             csv_writer_new = csv.writer(f)
#             for time_tuple in time_window:
#                 csv_writer_new.writerow(time_tuple)