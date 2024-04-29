import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import neighbors
import sklearn

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

# plot every feature of a single audio snippet
# (just for showcasing purposes, so you can get an idea what the data looks like)
sample_idx = 0
for feat_idx, feat in enumerate(data[sample_idx]):
    if show_sample_graphs:
        plot_feature(feat, idx_to_feature[feat_idx], label_metadata[sample_idx])

        # compare energy of a silent and a loud recording
        plot_feature(data[26606][9], idx_to_feature[9], label_metadata[26606])
        plot_feature(data[26606][172], idx_to_feature[172], label_metadata[26606])

        # plot mel-spectrogram of first audio snippet
        # mel-spectrogram data is from idx 12 to 75
        plot_spectrogram(data[3000][76:107], 'mfcc-spectrogram', label_metadata[3000])
        plot_spectrogram(data[6000][76:107], 'mfcc-spectrogram', label_metadata[6000])


# Second version of get_data_for_speakers, should be more efficient
def get_data_for_speakers(speaker_ids, label_metadata, data):
    """
    Retrieve all recording data and labels for specified array of speaker IDs, and flatten the data.
    """
    condition = np.isin(label_metadata['speaker_id'], speaker_ids)
    indices = np.where(condition)[0]
    selected_data = data[indices]
    return selected_data, label_metadata['word'][indices]


# get map of encoder for labels and their numerical representation as dictionary
def get_label_map(le: LabelEncoder):
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_map


def flatten_data_by_mean(data):
    """
    Flatten 3D data to 2D by calculating the mean for each feature and frame.

    :param data: 3D array of shape (n_recordings, n_features, n_frames)
    :return: 2D array of shape (n_recordings, n_features)
    """
    return np.mean(data, axis=2)


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

# Second version of splitting data into n sub-sets
n = 10
speaker_ids = np.unique(label_metadata['speaker_id'])
np.random.shuffle(speaker_ids)
speaker_splits_ids = np.array_split(speaker_ids, n)

recording_folds = []
for fold_idx, fold in enumerate(speaker_splits_ids):
    print(f"Retrieving data for fold {fold_idx + 1}/{n}")
    fold_data, fold_labels = get_data_for_speakers(fold, label_metadata, data_flat)
    recording_folds.append((fold_data, fold_labels))
    print(f"Retrieved {len(fold_data)} samples with labels.")

None
"""
Set up SVM
"""


def normalize_feature(data, feature_index):
    """
    Normalize a feature of the data to a value between 0 and 1 using the scikit MinMax Scaler.
    """
    # Extract the feature values from the data
    feature_values = data[:, feature_index].reshape(-1, 1)
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    # Fit the scaler to the feature values
    scaler.fit(feature_values)
    # Transform the feature values using the scaler
    normalized_feature = scaler.transform(feature_values)
    return normalized_feature


def normalize_data(data):
    """
    Normalize all features of the data to a value between 0 and 1 using the scikit MinMax Scaler.
    """
    normalized_data = []
    for i in range(data.shape[1]):
        normalized_data.append(normalize_feature(data, i))

    # Stack the normalized features along the last axis
    normalized_data = np.column_stack(normalized_data)
    return normalized_data


# Append the labels to the data, assume that the labels are in the same order as the data
def append_labels(data, labels):
    """
    Append the labels to the data using NumPy's column_stack function.
    """
    # Convert labels to a 2D array with shape (n, 1)
    labels_2d = labels.reshape(-1, 1)
    # Check if the number of rows in data and labels_2d match
    if data.shape[0] != labels_2d.shape[0]:
        raise ValueError(f"Number of rows in data ({data.shape[0]}) and labels ({labels_2d.shape[0]}) do not match.")
    # Use np.column_stack to append labels to the data
    appended_data = np.column_stack((data, labels_2d))
    return appended_data


print("orginal data shape: ", data.shape)
flattened_data = flatten_data_by_mean(data)
print("flattened data shape: ", flattened_data.shape)
normalized_data = normalize_data(flattened_data)
print("data normalized: ", normalized_data.shape)
appended_data = append_labels(normalized_data, label_metadata['word'])
print("label appended: ", appended_data.shape)

"""
# Initialize the SVM model
svm = SVC(kernel='linear')

# List to store the accuracy for each fold
accuracies = []

# Create a progress bar
with tqdm(total=n, desc="Training Progress", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    # Iterate over the 10 folds
    for i in range(n):
        # Use the i-th fold as the test set
        test_data, test_labels = recording_folds[i]

        # Use the remaining folds as the training set
        train_data = np.concatenate([recording_folds[j][0] for j in range(n) if j != i])
        train_labels = np.concatenate([recording_folds[j][1] for j in range(n) if j != i])

        # Normalize the training and test data
        normalized_train_data = normalize_data(train_data)
        normalized_test_data = normalize_data(test_data)

        # Train the SVM model with the training data
        svm.fit(normalized_train_data, train_labels)

        # Make predictions on the test data
        predictions = svm.predict(normalized_test_data)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(test_labels, predictions)
        accuracies.append(accuracy)

        # Update the progress bar
        pbar.update(1)

# Print the average accuracy of the model
print("Average Accuracy:", np.mean(accuracies))
"""

"""
Set up kNN
"""
None


def train_kNN(X_train, train_label, k_train):
    """
    Function that fits a kNN to given data
    @param X_train, np array, training data
    @param train_label, np array, training labels
    @param k_train, integer, k for the kNN
    @output classifier, kNN instance, classifier that was fitted to training data
    """
    # your code goes here ↓↓↓
    classifier = neighbors.KNeighborsClassifier(n_neighbors=k_train)
    classifier.fit(X_train, train_label)

    return classifier


def eval_kNN(classifier, X_eval):
    """
    Function that returns predictions for some input data
    @param classifier, kNN instance, trained kNN classifier
    @param X_eval, np array, data that you want to predict the labels for
    @output predicitons, np array, predicted labels
    """
    # your code goes here ↓↓↓
    predictions = classifier.predict(X_eval)

    return predictions


def mean_zero_one_loss(true_label, pred_label):
    """
    Function that calculates the mean zero-one loss for given true and predicted la
    @param true_label, np array, true labels
    @param pred_label, np array, predicted labels
    @output loss, float, mean zero-one loss
    """
    # your code goes here ↓↓↓
    loss = sklearn.metrics.zero_one_loss(true_label, pred_label)
    return loss


def run_kNN(X, labels, nf, k):
    """
    Function that combines all functions using CV
    @param X, np array, training data
    @param labels, np array, training labels
    @param nf, integer, number of folds for CV
    @param k, integer, k for kNN
    @output mean_error, float, mean error over all folds
    """
    # your code goes here ↓↓↓
    error = 0
    # split lists into nf distinct sets
    sub_sets_X = np.array_split(X, nf)
    sub_sets_y = np.array_split(labels, nf)
    for i in range(nf):
        # use set at index i as evaluation set
        classifier = train_kNN(
            np.concatenate(sub_sets_X[:i] + sub_sets_X[i + 1:]),  # leave out test sub
            np.concatenate(sub_sets_y[:i] + sub_sets_y[i + 1:]),
            k)
        predictions = eval_kNN(classifier, sub_sets_X[i])
        error += mean_zero_one_loss(sub_sets_y[i], predictions)
    return error / nf


error_holder = []
for k in range(1, m + 1, 2):  # range with 179 included and step of 2
    error_holder.append(run_kNN(X, y, nf, k))
