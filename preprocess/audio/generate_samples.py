import csv
import os
import random as rand

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite, read as wavread
from constants import (processed_audio_path)

diarizations_path = './rttmFile'
vad_path = 'filter_vad/'


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


unsuccessful_intention = [(2, 386160, 388520, 'INTS_start'), (3, 203033, 211840, 'INTS_start'), (3, 371673, 372980, 'INTS_continue'), (4, 42000, 46320, 'INTS_continue'),
 (4, 111633, 115986, 'INTS_start'), (4, 127933, 129653, 'INTS_start'), (4, 175846, 178086, 'INTS_continue'), (4, 250906, 254353, 'INTS_continue'),
 (4, 263833, 267353, 'INTS_continue'), (4, 284286, 286060, 'INTS_continue'), (4, 295220, 298173, 'INTS_continue'), (4, 310566, 311986, 'INTS_continue'), (4, 428740, 429906, 'INTS_start'),
 (4, 454213, 456400, 'INTS_start'), (4, 461800, 464920, 'INTS_start'), (7, 260780, 261773, 'INTS_maybe'), (7, 573726, 576053, 'INTS_start'), (10, 298766, 302260, 'INTS_continue'),
 (10, 327113, 327926, 'INTS_start'), (11, 200620, 202100, 'INTS_start'), (11, 265993, 268886, 'INTS_continue'), (11, 453580, 455440, 'INTS_continue'),
 (11, 504746, 506253, 'INTS_start'), (11, 569646, 572020, 'INTS_start'), (11, 584906, 586813, 'INTS_continue'), (18, 231240, 232246, 'INTS_maybe'), (18, 326506, 329080, 'INTS_start'),
 (18, 530173, 532306, 'INTS_continue'), (17, 126940, 128820, 'INTS_start'), (17, 419506, 421513, 'INTS_maybe'), (17, 447546, 451213, 'INTS_continue'), (22, 270433, 272220, 'INTS_continue'),
 (22, 320566, 321906, 'INTS_continue'), (22, 356313, 358780, 'INTS_start'), (22, 532260, 534773, 'INTS_continue'), (22, 555600, 556740, 'INTS_start'), (23, 371340, 374440, 'INTS_start'),
 (27, 169253, 170973, 'INTS_continue'), (27, 260046, 262333, 'INTS_maybe'), (34, 423106, 424286, 'INTS_continue'), (34, 424980, 433200, 'INTS_start'), (34, 439420, 440633, 'INTS_start'),
 (34, 477426, 479106, 'INTS_start'), (34, 513746, 515126, 'INTS_continue'), (35, 378093, 379920, 'INTS_continue'), (5, 412920, 414420, 'INTS_maybe')]


# produce VAD

import numpy as np
from scipy.io.wavfile import write as wavwrite



# exclue 3600 - 4200 seconds (1:00 h - 1:10 h)
def generate_negative_sample(train_intention_time_window, test_intention_time_window, intention_label, size=9900, fs=100, generate="Train"):
    # negative_intention_label = np.zeros((size * fs))
    train_negative_time_sample_size = len(train_intention_time_window) * 10
    test_negative_time_sample_size = len(test_intention_time_window) * 10
    negative_intention_time_window_list_train = []
    negative_intention_time_window_list_test = []
    # 3600 - 4200 seconds (1:00 h - 1:10 h)
    false_negative_check_time_window = []
    collect_false_negative = 3

    while len(negative_intention_time_window_list_train) < train_negative_time_sample_size and len(negative_intention_time_window_list_test) < test_negative_time_sample_size:
        random_point = (rand.randint(0, 9900))

        # check left side of random point
        left_point = random_point - 2
        if left_point >= 0:

            if not intention_label[left_point * fs:random_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200  for training dataset
                if left_point > 4200 or random_point < 3600:
                    negative_intention_time_window_list_train.append(tuple([left_point, random_point]))

                # during time interval of 3600 - 4200 for test dataset
                if left_point >= 3600 and random_point <= 4200:
                    negative_intention_time_window_list_test.append(tuple([left_point, random_point]))


        # check right side of random point
        right_point = random_point + 2

        if right_point <= 9900:

            if not intention_label[random_point * fs:right_point * fs].__contains__(1):

                # exclude time interval of 3600 - 4200
                if random_point > 4200 or right_point < 3600:
                    negative_intention_time_window_list_train.append(tuple([random_point, right_point]))

                # during time interval of 3600 - 4200
                #print("in test nagetive sample")
                if random_point >= 3600 and right_point <= 4200:
                    negative_intention_time_window_list_test.append(tuple([random_point, right_point]))

    return negative_intention_time_window_list_train, negative_intention_time_window_list_test


# exclue 3600 - 4200 seconds (1:00 h - 1:10 h)
def generate_positive_sample(vad, size=9900, fs=100):
    valid = True
    train_intention_time_window = []
    test_intention_time_window = []
    intention_label = np.zeros((size * fs))
    unique, counts = np.unique(intention_label, return_counts=True)
    #print(dict(zip(unique, counts)))
    previous_is_zero = True
    for i in range(0, size):
        if i - 2 >= 0:

            if (vad[i * fs] == 1) and previous_is_zero:
                previous_is_zero = False
                # check valid time window
                for j in range((i - 1) * fs, (i - 2 - 1) * fs, -1):
                    if vad[j] == 1:
                        valid = False
                        break

                # intention 2 seconds window (2 * 100)
                if valid:
                    time_ini = i - 2
                    time_end = i

                    # exclude time interval of 3600 - 4200
                    if time_ini > 4200 or time_end < 3600:
                        train_intention_time_window.append(tuple([time_ini, time_end]))

                    # during time interval of 3600 - 4200
                    if time_ini >= 3600 and time_end <= 4200:
                        #print("in test positive sample")
                        test_intention_time_window.append(tuple([time_ini, time_end]))

                    intention_label[time_ini * fs:time_end * fs] = 1

            if vad[i * fs] == 0 and not previous_is_zero:
                previous_is_zero = True

            valid = True

    print("count 1: ", len(train_intention_time_window), " count 1 test : ", len(test_intention_time_window))

    return train_intention_time_window, test_intention_time_window, intention_label


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


def load_filter_vad(vad_path_filter, pid_list_filter):
    vad = {}
    for i in range(0, len(pid_list_filter)):
        temp = wavfile.read(vad_path_filter + str(pid_list_filter[i]) + ".wav")
        vad[pid_list_filter[i]] = temp[1]
    if len(vad) == 0:
        print('load_vad called but nothing loaded.')
    return vad


# make vad files
from pathlib import Path

if __name__ == '__main__':
    # for f in Path(diarizations_path).glob('*.rttm'):
    #     df = load_diarization(f)
    #     pid = int(f.stem)
    #     # load corresponding vad file based on rttm files
    #     out_path = os.path.join(vad_path, f.stem)
    #     print("out_path : ", out_path)
    #     '''
    #     df : diarization file
    #     out_path : output path
    #     '''
    #
    #     store_vad(df, pid, out_path)

    # temp = wavfile.read('filter_vad/3.wav')
    # print(type(temp))
    # a, b = generate_positive_sample(temp[1])
    # print(len(a), " : ", len(b))
    #
    # c, d = generate_positive_sample_0(temp[1])
    # print(len(c), " : ", len(d))
    #
    # e = generate_negative_sample(a, b)
    # print(len(e))

    pid_list = [2, 3, 4, 5, 7, 10, 11, 17, 18, 22, 23, 27, 34, 35]  # len = 14
    vad_dict = load_filter_vad(vad_path, pid_list)
    for k in range(0, len(pid_list)):
        # Generate training and validation dataset
        # unique, counts = np.unique(label_pos, return_counts=True)
        # print(" len of label : ", len(label_pos))
        # print(dict(zip(unique, counts)))

        pos_example_train_time_list, pos_example_test_time_list, label_pos = generate_positive_sample(vad_dict[pid_list[k]])
        neg_example_train_time_list, neg_example_test_time_list = generate_negative_sample(pos_example_train_time_list, pos_example_test_time_list, label_pos)
        train_time_window = pos_example_train_time_list + neg_example_train_time_list
        np.random.shuffle(train_time_window)

        print("train time sample : ", len(train_time_window))

        test_time_window = pos_example_test_time_list + neg_example_test_time_list
        np.random.shuffle(test_time_window)

        print("test time sample : ", len(test_time_window))

        # write csv for train dataset
        with open('./po_ne_csv/' + str(pid_list[k]) + '.csv', "w") as f:
            csv_writer_new = csv.writer(f)
            for time_tuple in train_time_window:
                csv_writer_new.writerow(time_tuple)


        # write csv for test dataset
        with open('./test_data_po_ne_csv/' + str(pid_list[k]) + '.csv', "w") as ff:
            test_csv_writer_new = csv.writer(ff)
            for test_time_tuple in test_time_window:
                test_csv_writer_new.writerow(test_time_tuple)

        outfile = open('./target_label/' + str(pid_list[k]) + '.csv', "w", newline='')
        out = csv.writer(outfile)
        out.writerows(map(lambda x: [x], label_pos))
        outfile.close()

