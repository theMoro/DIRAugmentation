import os
import pathlib
import random

import torch
from ba3l.ingredients.datasets import Dataset
import pandas as pd
from sacred.config import DynamicIngredient, CMD
from scipy.signal import convolve
from scipy.fft import irfft, rfft
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset, DistributedSampler, WeightedRandomSampler
import numpy as np
from datasets.audiodatasets import FilesCachedDataset, SimpleSelectionDataset, \
    PreprocessDataset, ObjectCacher

from audio_processor import get_processor_default

AUDIO_PATH = "/share/rk6/shared/dcase22/TAU-urban-acoustic-scenes-2022-mobile-development/"
DIR_PATH = "/share/rk6/shared/dcase22/ir_filters/external_devices/"
TRAIN_FILES_CSV = "/share/rk6/shared/dcase22/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold{}_train.csv"
META_CSV = "/share/rk6/shared/dcase22/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv"
TEST_FILES_CSV = "/share/rk6/shared/dcase22/TAU-urban-acoustic-scenes-2022-mobile-development/evaluation_setup/fold{}_evaluate.csv"
CACHE_ROOT_PATH = "/share/rk6/shared/kofta_cached_datasets/d22t1_tobiasm/"


class BasicDCASE22Dataset(TorchDataset):
    """
    Basic DCASE22 Dataset
    """
    def __init__(self):
        df = pd.read_csv(META_CSV, sep="\t")
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(df[['scene_label']].values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)
        self.labels = labels
        self.df = df

    def __getitem__(self, index):
        return self.files[index], self.labels[index],

    def __len__(self):
        return len(self.files)


class SpectrogramDataset(TorchDataset):
    """
    gets the spectrogram from files using audioprocessor

    """

    def __init__(self):
        self.ds = BasicDCASE22Dataset()
        self.process_func = get_processor_default()
        self.audio_path = AUDIO_PATH

    def __getitem__(self, index):
        file, label = self.ds[index]
        x = self.process_func(self.audio_path + file)
        return x, file, label

    def __len__(self):
        return len(self.ds)


def get_file_cached_dataset(name='d22t1'):
    audio_processor = dict(
        identifier='resample22050',
        n_fft=2048,
        sr=22050,
        mono=True,
        log_spec=False,
        n_mels=256,
        hop_length=512
    )

    print("get_file_cached_dataset::", name, audio_processor['identifier'], "sr=", audio_processor['sr'],
          CACHE_ROOT_PATH)
    ds = FilesCachedDataset(SpectrogramDataset, name, audio_processor['identifier'], CACHE_ROOT_PATH)
    return ds


def get_base_training_set(fold=1, a_train_only=False, use_random_labels=False,
                          use_all_meta_to_train=False):
    if use_all_meta_to_train:
        if a_train_only: raise RuntimeError("Not Implemented")
        return get_file_cached_dataset()
    ds = BasicDCASE22Dataset()
    training_files = pd.read_csv(TRAIN_FILES_CSV.format(fold), sep='\t')['filename'].values.reshape(-1)
    if a_train_only:
        print("\nWarning: using only device a for training \n")
        training_files = [d for d in training_files if d.rsplit("-", 1)[1][:-4] == 'a']
    print(f"\nWarning: using all device a for training {len(set(training_files))}\n")

    devices = set([d.rsplit("-", 1)[1][:-4] for d in training_files])
    print("Training Devices: ", devices)

    training_files_set = set(training_files)
    train_indices = [i for i, f in enumerate(ds.files) if f in training_files_set]
    ds = SimpleSelectionDataset(get_file_cached_dataset(), train_indices)

    if use_random_labels:
        ds = RandomFixedLabelDataset(ds)
    return ds


def get_base_test_set(fold=1):
    ds = BasicDCASE22Dataset()
    training_files = pd.read_csv(TEST_FILES_CSV.format(fold), sep='\t')['filename'].values.reshape(-1)
    training_files_set = set(training_files)
    train_indeces = [i for i, f in enumerate(ds.files) if f in training_files_set]
    devices = set([d.rsplit("-", 1)[1][:-4] for d in training_files])
    print("Validation Devices: ", devices)
    return SimpleSelectionDataset(get_file_cached_dataset(), train_indeces)


def get_roll_func(axis=2, shift=None, shift_range=4000):
    print("rolling...")

    def roll_func(b):
        x, i, y = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        global FirstTime

        axis = 2
        if x.dim() == 2:
            axis = 1
        return x.roll(sf, axis), i, y

    return roll_func


def load_dirs(dirs_path, cut_dirs_offset=None):
    all_paths = [path for path in pathlib.Path(os.path.expanduser(dirs_path)).rglob('*.wav')]
    all_paths = sorted(all_paths)

    if cut_dirs_offset is not None:
        all_paths = all_paths[cut_dirs_offset:cut_dirs_offset + 10]
    all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]

    print("will use these Device IRs:")
    for i in range(len(all_paths_name)):
        print(i, ": ", all_paths_name[i])

    audio_processor = get_processor_default()

    return [audio_processor(p) for p in all_paths]


class DIRAugmentDataset(TorchDataset):
    """
   Augments Audio with a DIR (Device Impulse Response)

    """

    def __init__(self, ds, dirs, prob):
        self.ds = ds
        self.dirs = dirs
        self.prob = prob
        self.audio_path = AUDIO_PATH

    def __getitem__(self, index):
        x, file, label = self.ds[index]
        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]

        dir_idx = -1
        dir = None

        if device == 'a':
            if torch.rand(1) < self.prob:
                dir_idx = int(np.random.randint(0, len(self.dirs)))
                dir = self.dirs[dir_idx]

                x_convolved = convolve(x, dir, 'full')[:, :x.shape[1]]
                return x_convolved, file, label

        return x, file, label

    def __len__(self):
        return len(self.ds)


def add_dir_augment_ds(ds, apply=False, prob=0.4):
    if not apply:
        return ds
    return DIRAugmentDataset(ds, load_dirs(DIR_PATH), prob)


def get_training_set(normalize=False, roll=True, apply_dir=False, prob_dir=0.4):
    ds = get_base_training_set()
    if normalize:
        print("normalized train!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)
    ds = add_dir_augment_ds(ds, apply=apply_dir, prob=prob_dir)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    return ds

def get_test_set(normalize=False, roll=True):
    ds = get_base_test_set()
    if normalize:
        print("normalized test!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)

    return ds
