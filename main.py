import numpy as np
from matplotlib import pyplot as plt

# TODO set to True to show graphs
show_sample_graphs = False


def plot_feature(feat, feat_name, snip_meta):
    """
    plot a single feature of an audio snippet

    :param feat: feature values
    :param feat_name: name of feature
    :param snip_meta: snippet metadata (including label)
    """
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

# 1a) check if there are inconsistencies in the audio by comparing file path and assigned label
false_count = 0
for l in label_metadata:
    value = str(l[1]).split('/')[1]
    label = l[3]
    if value != label:
        false_count = false_count + 1

# compare energy of a silent and a loud recording
plot_feature(data[26606][9], idx_to_feature[9], label_metadata[26606])
plot_feature(data[28049][9], idx_to_feature[9], label_metadata[28049])
