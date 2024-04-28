import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

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
try:
    label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
    idx_to_feature = np.loadtxt('metadata/idx_to_feature_name.csv', delimiter=',', usecols=1, skiprows=1, dtype='U')
    data = np.load('development_numpy/development.npy')
except Exception as e:
    print(f"Failed to load data: {e}")
    raise

'''
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
'''


# Second version of get_data_for_speakers, should be more efficient
def get_data_for_speakers(speaker_ids, label_metadata, data):
    """
    Retrieve all recording data and labels for specified array of speaker IDs, and flatten the data.
    """
    condition = np.isin(label_metadata['speaker_id'], speaker_ids)
    indices = np.where(condition)[0]
    # Flatten each sample's feature data within the selected indices
    selected_data = data[indices]
    flattened_data = np.array([sample.flatten() for sample in selected_data])
    return flattened_data, label_metadata['word'][indices]


# get map of encoder for labels and their numerical representation as dictionary
def get_label_map(le: LabelEncoder):
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_map


"""
TASK 3: Classification
"""

'''
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
    print("get speaker data for fold {}/{}".format(fold_idx + 1, n))
    recording_folds.append(get_data_for_speakers(fold))
'''
# Initialize the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder with the labels from the training data
le.fit(label_metadata['word'])

# Transform the labels into numerical labels
label_metadata['word'] = le.transform(label_metadata['word'])
print(label_metadata['word'])

# Second version of splitting data into n sub-sets
n = 10
speaker_ids = np.unique(label_metadata['speaker_id'])
np.random.shuffle(speaker_ids)
speaker_splits_ids = np.array_split(speaker_ids, n)

recording_folds = []
for fold_idx, fold in enumerate(speaker_splits_ids):
    print(f"Retrieving data for fold {fold_idx + 1}/{n}")
    fold_data, fold_labels = get_data_for_speakers(fold, label_metadata, data)
    recording_folds.append((fold_data, fold_labels))
    print(f"Retrieved {len(fold_data)} samples with labels.")

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

"""
Set up SVM
"""


# Call this function before appending the labels to the data
def normalize_feature(data, feature_index):
    """
    Normalize a feature of the data to a value between 0 and 1 using the scikit Standard Scaler.
    """
    # Extract the feature values from the data
    feature_values = np.array([sample[feature_index] for sample in data])
    # flatten to 1d array
    feature_values = feature_values.flatten().reshape(-1, 1)
    # Create a StandardScaler object
    scaler = MinMaxScaler()
    # Fit the scaler to the feature values
    scaler.fit(feature_values)
    # Transform the feature values using the scaler
    normalized_feature = scaler.transform(feature_values)
    # transform again to 2d array, take the number of samples per array
    normalized_feature = normalized_feature.reshape(-1, len(data))
    return normalized_feature


def normalize_data(data):
    """
    Normalize all features of the data to a value between 0 and 1 using the scikit Standard Scaler.
    """
    normalized_data = []
    for i in range(len(data[0])):
        normalized_data.append(normalize_feature(data, i))

    # flatten to 2d array
    #normalized_data = np.array(normalized_data).reshape(-1, 44 * 175)
    return np.array(normalized_data)


# Append the labels to the data, assume that the labels are in the same order as the data
def append_labels(data, labels):
    """
    Append the labels to the data using NumPy's column_stack function.
    """
    # Flatten the data to 2D if it's not already
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    # Convert labels to a 2D array with shape (n, 1)
    labels_2d = labels.reshape(-1, 1)
    # Check if the number of rows in data and labels_2d match
    if data.shape[0] != labels_2d.shape[0]:
        raise ValueError(f"Number of rows in data ({data.shape[0]}) and labels ({labels_2d.shape[0]}) do not match.")
    # Use np.column_stack to append labels to the data
    appended_data = np.column_stack((data, labels_2d))
    return appended_data

#normalized_data = normalize_data(data)
#print(normalized_data.shape)

# Set up the SVM classifier with the normalized data
clf = make_pipeline(StandardScaler(), SVC())

# Initialize the accuracy list
accuracies = []

# Iterate over the recording folds
for fold_idx, (fold_data, fold_labels) in enumerate(recording_folds):
    print(f"Training SVM for fold {fold_idx + 1}/{n}")
    # Normalize the data
    normalized_fold_data = normalize_data(fold_data)
    # Append the labels to the data
    appended_fold_data = append_labels(normalized_fold_data, fold_labels)
    # Shuffle the data
    np.random.shuffle(appended_fold_data)
    # Split the data into features and labels
    X = appended_fold_data[:, :-1]
    y = appended_fold_data[:, -1]
    # Fit the SVM classifier
    clf.fit(X, y)
    # Predict the labels
    y_pred = clf.predict(X)
    # Calculate the accuracy
    accuracy = accuracy_score(y, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for fold {fold_idx + 1}/{n}: {accuracy}")

# Calculate the mean accuracy
mean_accuracy = np.mean(accuracies)
print(f"Mean accuracy: {mean_accuracy}")
