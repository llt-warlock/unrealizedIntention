
import pickle

import utils

accel_path = "../data/subj_accel_interp.pkl"

def make_all_examples(Num, windowSize, label_length_fs, numberOfExperiment=None, category=None):
    accel_path = "../data/subj_accel_interp.pkl"

    # successful train ground truth label
    realized_intention_label_path = "../preprocess/audio/successful_train_ground_truth/"


    # unsuccessful intention start/continue case ground truth. (which one depends on selection)
    unrealized_intention_path = "../preprocess/audio/unsuccessful_intention_test_label/"

    # Both start and continue unsuccessful intention ground truth label.
    all_intention_label_path = "../preprocess/audio/all_label/"

    start_pid = [2, 3, 4, 7, 10, 11, 17, 22, 23, 34]


    if Num == 1:
        temp = all_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, all_sample_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 1 -- ", index)
            all_test_samples = maker.make_all_examples(index, windowSize, label_length_fs)
            pickle.dump(all_test_samples, open('../data/all_test_pkl/' +  str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))

    elif Num == 2:
        temp = realized_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, vad_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 2 -- ", index)
            successful_test_samples = maker.make_test_examples(index, windowSize, label_length_fs)
            pickle.dump(successful_test_samples, open('../data/successful_test_pkl/' + str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))

    elif Num == 0:
        print("Generate training pkl")
        temp = realized_intention_label_path + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, vad_path=temp)
        train_samples = maker.make_train_examples(windowSize, label_length_fs)
        pickle.dump(train_samples, open('../data/train_pkl/' + str(windowSize) + "s/" + '_INTS_train.pkl', 'wb'))

    elif Num == 3 or Num == 4 or Num == 5:
        temp = unrealized_intention_path + str(category) + "/" + str(windowSize) + "s/"
        maker = utils.Maker(accel_path=accel_path, unsuccessful_vad_path=temp)
        for index in range(0, numberOfExperiment):
            print("num : 3-4-5 -- ", index, "  ", str(category))
            all_test_samples = maker.make_unsuccessful_examples(start_pid, index, windowSize, label_length_fs, category)
            pickle.dump(all_test_samples, open('../data/unsuccessful_test_pkl/' + str(category) + "/" + str(windowSize) + "s/" + str(index) + '_INTS_test.pkl', 'wb'))


def main(Num, numberOfExperiment, category=None):
    if Num == 1:
        make_all_examples(Num, numberOfExperiment)
    elif Num == 2:
        make_all_examples(Num, numberOfExperiment)
    elif Num == 3 or  numberOfExperiment == 4 or  numberOfExperiment == 5:
        make_all_examples(Num, numberOfExperiment, category)
    elif Num == 0:
        make_all_examples(0, 1)


if __name__ == '__main__':

    for window_size in range(1,5):
        print("window size : ", window_size)
        # experiment 0
        make_all_examples(0, window_size, label_length_fs=20)

        # experiment 1
        make_all_examples(1, window_size, label_length_fs=20, numberOfExperiment=100)

        # experiment 2  done
        make_all_examples(2, window_size, label_length_fs=20, numberOfExperiment=100)

        # experiment 3
        make_all_examples(3, window_size, label_length_fs=20, numberOfExperiment=100, category='all_unsuccessful')

        # experiment 4
        make_all_examples(4, window_size, label_length_fs=20, numberOfExperiment=100, category='start')

        # experiment 5
        make_all_examples(5, window_size, label_length_fs=20, numberOfExperiment=100, category='continue')





