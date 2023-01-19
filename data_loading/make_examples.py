import os
import pickle

import utils

from utils import (
    Maker,
    reset_examples_ids)


def make_all_examples():
    processed_accel_path = "../data/subj_accel_interp.pkl"
    #vad_path = "../filter_vad/"
    #examples = []
    #test_examples = []
    # for cam in [2, 3]:
    #     tracks_path = os.path.join(processed_pose_path, 'tracks', f'cam{cam}_final.pkl')
    #
    #     accel_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
    #     maker = utils.Maker(tracks_path, accel_path, vad_path)
    #     examples += maker.make_examples(cam=cam)

    # accel_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
    accel_path = "../data/subj_accel_interp.pkl"
    #vad_path = "../data/"
    vad_path = "../preprocess/audio/target_label/"
    unsuccessful_vad_path = "../preprocess/audio/unsuccessful_intention_label/"
    #maker = utils.Maker(accel_path, vad_path)
    #examples += maker.make_examples()
    start_pid = [2,3,4,7,10,11,17,22,23, 34]
    maker = utils.Maker(accel_path, vad_path, unsuccessful_vad_path)
    train_examples, test_example, unsuccessful_example = maker.make_examples(start_pid)

    return train_examples, test_example, unsuccessful_example


# def write_examples(examples):
#     #out_path = os.path.join(processed_videos_path, 'examples')
#     out_path = './examples'
#     reset_examples_ids(examples)
#     write_all_example_videos(examples, out_path)
#

if __name__ == '__main__':
    # unsuccessful_examples
    examples, test_examples, unsuccessful_examples = make_all_examples()

    len(examples)

    pickle.dump(examples, open('../data/INTS_examples_12_24.pkl', 'wb'))

    pickle.dump(test_examples, open('../data/INTS_examples_test_24.pkl', 'wb'))

    pickle.dump(unsuccessful_examples, open('../data/INTS_unsuccessful_test_29.pkl', 'wb'))
