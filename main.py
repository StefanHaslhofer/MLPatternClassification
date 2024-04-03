import numpy as np
from matplotlib import pyplot as plt


def plotFeature(snippet, feature):
    plt.title("Matplotlib demo")
    plt.xlabel("t")
    plt.ylabel("val")
    plt.plot(np.arange(0, len(feature)), feature)
    plt.show()


data = np.load('development_numpy/development.npy')

data = data[0:1]
for snippet in data:
    for feature in snippet:
        plotFeature(snippet, feature)

