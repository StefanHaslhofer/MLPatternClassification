import numpy as np
from matplotlib import pyplot as plt

# TODO set to True to show graphs
show_graphs = False


def plot_feature(feat, feat_name, snip_meta):
    """
    plot a single feature of an audio snippet

    :param feat: feature values
    :param feat_name: name of feature
    :param snip_meta: snippet metadata (including label)
    """
    plt.title(snip_meta[3] + ' - ' + feat_name)  # use recording label as title
    plt.xlabel("t")
    plt.ylabel("value")
    plt.plot(np.arange(0, len(feat)), feat)
    plt.show()


# load from data files provided on moodle
label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
idx_to_feature = np.loadtxt('metadata/idx_to_feature_name.csv', delimiter=',', usecols=1, skiprows=1, dtype='U')
data = np.load('development_numpy/development.npy')

# plot features of a single audio snippet
# (just for showcasing purposes, so you can get an idea what the data looks like)
sample_idx = 0
for feat_idx, feat in enumerate(data[sample_idx]):
    if show_graphs:
        plot_feature(feat, idx_to_feature[feat_idx], label_metadata[sample_idx])
