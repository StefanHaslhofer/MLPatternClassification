import numpy as np
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

# 1a) check if there are inconsistencies in the audio by comparing file path and assigned label
false_count = 0
for l in label_metadata:
    value = str(l[1]).split('/')[1]
    label = l[3]
    if value != label:
        false_count = false_count + 1

# compare energy of a silent and a loud recording
plot_feature(data[26606][9], idx_to_feature[9], label_metadata[26606])
plot_feature(data[26606][172], idx_to_feature[172], label_metadata[26606])
# standard deviation for feature with index 9
std = np.std(data[26606][9])

plot_feature(data[3000][9], idx_to_feature[9], label_metadata[3000])
plot_feature(data[3000][172], idx_to_feature[172], label_metadata[3000])

# quiet
speaker_1_idx = [i[0] for i in list(filter(lambda x: x[2] == 3, label_metadata))]
# loud
speaker_2_idx = [i[0] for i in list(filter(lambda x: x[2] == 14, label_metadata))]

speaker_1 = data[speaker_1_idx]
speaker_2 = data[speaker_2_idx]

downsampled_data_1 = np.mean(speaker_1, axis=2)
downsampled_data_2 = np.mean(speaker_2, axis=2)

std_1 = np.std(downsampled_data_1, axis=0)
std_2 = np.std(downsampled_data_2, axis=0)

plot_features(std_1, 3)
plot_features(std_2, 14)

# 10 = German speaker


None