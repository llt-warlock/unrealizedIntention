import os
import pickle

import utils

from utils import (
    Maker,
    reset_examples_ids)


def make_all_examples():
    processed_accel_path = "../data/subj_accel_interp.pkl"
    vad_path = "../data/"
    examples = []
    # for cam in [2, 3]:
    #     tracks_path = os.path.join(processed_pose_path, 'tracks', f'cam{cam}_final.pkl')
    #
    #     accel_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
    #     maker = utils.Maker(tracks_path, accel_path, vad_path)
    #     examples += maker.make_examples(cam=cam)

    # accel_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
    accel_path = "../data/subj_accel_interp.pkl"
    vad_path = "../data/"
    maker = utils.Maker(accel_path, vad_path)
    examples += maker.make_examples()

    print("in function 'make_all_example'")

    return examples


# def write_examples(examples):
#     #out_path = os.path.join(processed_videos_path, 'examples')
#     out_path = './examples'
#     reset_examples_ids(examples)
#     write_all_example_videos(examples, out_path)
#

if __name__ == '__main__':

    examples = make_all_examples()

    len(examples)

    with open('INTS_examples.pkl', 'wb') as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
