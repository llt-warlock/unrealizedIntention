import os
import sys
import logging
import pickle
import traceback

import torch
import lightning_lite
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from IPython.display import clear_output
import pytorch_lightning as pl

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# set the cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from data_loading.dataset import FatherDataset, FatherDatasetSubset
from data_loading.extractors import AccelExtractor
from constants import (
    processed_data_path,
    processed_accel_path,
    processed_videos_path,
    examples_path, dataset_path)
from train import System, train, test

import multiprocessing
from multiprocessing import Process
import threading

'''
do_train : 
'''


def do_cross_validation(do_train, ds, input_modalities, seed, prefix=None, deterministic=False):
    # cv_splits : [array, array, array, ...]
    # split data into 3 sets
    cv_splits = list(GroupKFold(n_splits=3).split(range(len(ds)), groups=ds.get_groups()))
    print("csv_splits: ", cv_splits)
    for i in cv_splits:
        print(" length : ", len(i))
    all_results = []

    for f, (train_idx, test_idx) in enumerate(cv_splits):
        print("f :", f, "   train_idx : ", len(train_idx), "  test_idx : ", len(test_idx), "\n")
        if f == 0 or f == 1:
            continue
        # load feature caches for fold f
        #
        # ###########################  make weight :
        train_all_label = ds.get_all_labels()
        temp = np.take(train_all_label,train_idx)
        temp_tensor = torch.from_numpy(temp)
        class_sample_count = torch.tensor(
            [(temp_tensor == t).sum() for t in torch.unique(temp_tensor, sorted=True)])
        weight = 1. / class_sample_count.float()
        train_samples_weight = torch.tensor([weight[int(t)] for t in temp_tensor])
        print("train : ", weight, train_samples_weight)


        ########  val weight
        train_all_label = ds.get_all_labels()
        temp_val = np.take(train_all_label,test_idx)
        temp_val_tensor = torch.from_numpy(temp_val)
        class_val_sample_count = torch.tensor(
            [(temp_val_tensor == t).sum() for t in torch.unique(temp_val_tensor, sorted=True)])
        val_weight = 1. / class_val_sample_count.float()
        val_samples_weight = torch.tensor([val_weight[int(t)] for t in temp_val_tensor])
        print("val : ", val_weight, val_samples_weight)

        ################################################

        train_ds = FatherDatasetSubset(ds, train_idx, eval=False)
        test_ds = FatherDatasetSubset(ds, test_idx, eval=True)

        weights_path = os.path.join(
            'weights',
            f'I{"-".join(input_modalities)}_fold{f}.ckpt'
        )

        pl.utilities.seed.seed_everything(seed + f + 734890573)
        if do_train:
            trainer = train(train_samples_weight, val_samples_weight, f, train_ds, test_ds, input_modalities,
                            prefix=prefix + f'_fold{f}' if prefix else None,
                            eval_every_epoch=True,
                            deterministic=deterministic,
                            weights_path=weights_path)
            model = trainer.model
        else:
            model = System.load_from_checkpoint(checkpoint_path=weights_path)

        # ensures that the testing is reproducible regardless of training
        pl.utilities.seed.seed_everything(seed + f + 2980374334)
        fold_outputs = test(f, model, test_ds, prefix=prefix + f'_fold{f}' if prefix else None, )
        all_results.append(fold_outputs)
        clear_output(wait=False)

    outputs = [r['proba'].numpy() for r in all_results]
    indices = [r['index'].numpy() for r in all_results]
    metrics = [r['metric'] for r in all_results]
    precision = [r['precision'] for r in all_results]
    recall = [r['recall'] for r in all_results]
    return metrics, outputs, indices, precision, recall


def do_run(examples, input_modalities,
           do_train=True, deterministic=True, prefix=''):
    ''' train = True will train the models, and requires
            model_label_modality = test_label_modality
        train = False will load weights to test the models and does not require
            model_label_modality = test_label_modality
    '''
    print(f'Using {len(examples)} examples')

    # create the feature datasets
    extractors = {}

    if 'accel' in input_modalities:
        # accel_ds_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
        # get accel data
        accel_ds_path = '../data/subj_accel_interp.pkl'
        extractors['accel'] = AccelExtractor(accel_ds_path)

    # extract data based on features selected
    ds = FatherDataset(examples, extractors)

    seed = 22
    metrics, probas, indices, precision, recall  = do_cross_validation(
        do_train,
        ds,
        input_modalities=input_modalities,
        deterministic=deterministic,
        seed=seed,
        prefix=f'{prefix}I{"-".join(input_modalities)}')

    torch.cuda.empty_cache()

    return {
        'metrics': metrics,
        'probas': probas,
        'indices': indices,
        'seed': seed,
        'precision':precision,
        'recall':recall
    }


def get_table(do_train=True, deterministic=True):
    # examples = pickle.load(open(examples_path, 'rb'))
    # data set
    examples = pickle.load(open("../data/INTS_examples_12_16.pkl", 'rb'))

    all_input_modalities = [
        # ('video',),
        # ('pose',),
        ('accel',),
    ]

    res = {}
    '''
    examples: 输入的数据
    '''
    for input_modalities in all_input_modalities:
        run_results = do_run(
            examples,
            input_modalities,
            do_train=do_train,
            deterministic=deterministic)

        res['-'.join(input_modalities)] = run_results
    return res


if __name__ == '__main__':

    try:
        res = get_table(do_train=True, deterministic=False)
        print(res)
    except Exception:
        print(traceback.format_exc())
