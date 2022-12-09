import csv
import os

import numpy as np
import pandas as pd
from scipy.io.wavfile import write as wavwrite, read as wavread
from constants import (processed_audio_path)

diarizations_path = './rttmFile'
vad_path =  './vad'


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
            print("不会吧")
            continue

        ini = round(row['ini'] * fs)
        end = round((row['ini'] + row['dur']) * fs)

        # 1 秒 用多少表示?
        time_ini = round((row['ini'] - 60) * fs)
        time_end = round(row['ini'] * fs)
        time_window.append(tuple([time_ini, time_end]))
        print("我在这里")
        vad[ini:end] = 1

    return vad, time_window


def store_vad(df: pd.DataFrame, pid, fname, size=9900, fs=100):
    print("进来没")
    vad, time_list = make_vad(df, pid, size=size, fs=fs)


    # write csv
    with open(fname + '.csv', "w") as f_f:
        csv_writer = csv.writer(f_f)
        for mytuple in time_list:
            csv_writer.writerow(mytuple)

    # write .vad file
    print("我也在这里")
    np.savetxt(fname + '.vad', vad, fmt='%d')
    # write .wav file
    wavwrite(fname + '.wav', fs, vad)


from pathlib import Path


if __name__ == '__main__':
    for f in Path(diarizations_path).glob('*.rttm'):
        print("不是把")
        df = load_diarization(f)
        pid = int(f.stem)
        # load corresponding vad file based on rttm files
        out_path = os.path.join(vad_path, f.stem)
        print("out_path : ", out_path)
        '''
        df : diarization file
        pid : 编号
        out_path : output path
        生成一个人的vad
        '''
        print("这也没有？")
        store_vad(df, pid, out_path)

    print("什么J8")
