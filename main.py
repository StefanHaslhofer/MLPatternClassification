import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd

# TODO set to True to show graphs
show_sample_graphs = False


def plot_feature(feat, feat_name, snip_meta):
    """
    plot a single feature of an audio snippet

    :param feat: feature values
    :param feat_name: name of feature
    :param snip_meta: snippet metadata (including label)
    """
    plt.rcParams.update({'font.size': 14})
    plt.title(snip_meta[3] + ' - ' + feat_name)  # use recording label as title
    plt.xlabel('t')
    plt.ylabel('value')
    plt.plot(np.arange(0, len(feat)), feat)
    plt.show()


def plot_spectrogram(data, feat_name, snip_meta):
    plt.title(snip_meta[3] + ' - ' + feat_name)  # use recording label as title
    plt.xlabel('t')
    plt.ylabel('frequency bin')
    plt.pcolormesh(data, shading='auto')
    plt.show()


def plot_features(std, speaker_id):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(range(len(std)), std),  # tick_label=feature_names)
    plt.xlabel('feature id')
    plt.ylabel('standard deviation')
    plt.title('standard deviation of speaker ' + str(speaker_id))
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent overlapping labels
    plt.show()


# load from data files provided on moodle
label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
idx_to_feature = np.loadtxt('metadata/idx_to_feature_name.csv', delimiter=',', usecols=1, skiprows=1, dtype='U')
data = np.load('development_numpy/development.npy')


def get_data_for_speakers(speaker_ids):
    """
    get all recording data by speaker id

    :param speaker_ids: fold of speakers
    :return: recordings id
    """
    metadata = list(filter(lambda el: np.isin(el[2], speaker_ids), label_metadata))
    # metadata elements at index 0 = indices of recordings
    recording_ids = [el[0] for el in metadata]
    # get recording data by list of indices
    return data[recording_ids]


"""
TASK 3: Classification
"""
# 1. split data into n sub-sets
n = 10
# get unique speaker ids from metadata
speaker_ids = np.unique([el[2] for el in label_metadata])
# random shuffle
random.shuffle(speaker_ids)
# split speaker ids into 10 folds
speaker_splits_ids = np.array_split(speaker_ids, n)
# get the recording of the speaker ids of each of the 10 folds
recording_folds = []
for fold_idx, fold in enumerate(speaker_splits_ids):
    print("get speaker data for fold {}/{}".format(fold_idx, n))
    recording_folds.append(get_data_for_speakers(fold))


# plot every feature of a single audio snippet
# (just for showcasing purposes, so you can get an idea what the data looks like)
sample_idx = 0
for feat_idx, feat in enumerate(data[sample_idx]):
    if show_sample_graphs:
        plot_feature(feat, idx_to_feature[feat_idx], label_metadata[sample_idx])

# plot mel-spectrogram of first audio snippet
# mel-spectrogram data is from idx 12 to 75
plot_spectrogram(data[sample_idx][12:75], 'mel-spectrogram', label_metadata[sample_idx])
plot_spectrogram(data[sample_idx][76:107], 'mfcc-spectrogram', label_metadata[sample_idx])

# compare energy of a silent and a loud recording
plot_feature(data[26606][9], idx_to_feature[9], label_metadata[26606])
plot_feature(data[26606][172], idx_to_feature[172], label_metadata[26606])

None
