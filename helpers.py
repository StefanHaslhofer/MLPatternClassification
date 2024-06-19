import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib


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


def scatter_features(d, labels):
    # your code goes here ↓↓↓
    plt.scatter(d[:20, 9], d[:20, 11], c=labels[:20].astype(int))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.show()
    None


def calc_deltas(data, axis=2):
    """
    calculate the change (delta) of the frequency energy over time
    """
    deltas = np.diff(data, axis=axis)
    # trailing_zeros = np.zeros(shape=(deltas.shape[0] ,deltas.shape[1], 1))
    # deltas = np.concatenate((deltas, trailing_zeros), axis=2)
    # return np.append(data, deltas, axis=1)
    return deltas


# Second version of get_data_for_speakers, should be more efficient
def get_data_for_speakers(speaker_ids, label_metadata, data):
    """
    Retrieve all recording data and labels for specified array of speaker IDs, and flatten the data.
    """
    condition = np.isin(label_metadata['speaker_id'], speaker_ids)
    indices = np.where(condition)[0]
    selected_data = data[indices]
    return selected_data, label_metadata['word'][indices]


def filter_by_label(label_filter, label_metadata, data):
    """
    filter datapoints by label
    """
    condition = np.isin(label_metadata['word'], label_filter)
    indices = np.where(condition)[0]
    selected_data = data[indices]
    return selected_data, label_metadata[indices]

def get_data(label_metadata, data):
    """
    retrieve all recording data alongside labels
    """
    return data, label_metadata['word']


# get map of encoder for labels and their numerical representation as dictionary
def get_label_map(le: LabelEncoder):
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_map


def flatten_data_by_mean(data, axis):
    """
    Flatten 3D data to 2D by calculating the mean along axis

    :param data: 3D array of shape (n_recordings, n_features, n_frames)
    :return: 2D array of shape (n_recordings, n_frames)
    """
    return np.mean(data, axis=axis)


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
    norm_data = []
    for i in range(data.shape[1]):
        norm_data.append(normalize_feature(data, i))

    # Stack the normalized features along the last axis
    norm_data = np.column_stack(norm_data)
    return norm_data


# Initialize the LabelEncoder
le = LabelEncoder()
label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
# Fit the LabelEncoder with the labels from the training data
le.fit(label_metadata['word'])

# Transform the labels into numerical labels
label_metadata['word'] = le.transform(label_metadata['word'])
label_map = get_label_map(le)
label_map_reverted = {v: k for k, v in label_map.items()}
print(label_map_reverted)

def convert_int_to_label(label_index):
    return label_map_reverted[label_index]


## FEATURE IMPORTANCE SECTION DO NOT TOUCH!!!!!!!!

#load model
rfc = joblib.load('random_forest_model.pkl')
feature_importances = rfc.feature_importances_

#load csv file with feature names and their index
featurescsv = 'metadata/idx_to_feature_name.csv'

#load first column of csv file into list
feature_names = []
with open(featurescsv) as f:
    for line in f:
        feature_names.append(line.split(',')[1].strip())
#remove first entry as it is the header
feature_names.pop(0)

def get_feature_name_from_index(index):
    return feature_names[index]


# save feature importance into map with feature name as key and importance as value
feature_importances_map = {}
for i in range(len(feature_importances)):
    name = get_feature_name_from_index(i)
    feature_importances_map[name] = feature_importances[i]

# sort feature importance map by value
sorted_feature_importances_map = dict(sorted(feature_importances_map.items(), key=lambda item: item[1], reverse=True))

def get_most_important_features_names(n):
    """
    Get the n most important features from a sorted feature importance map

    :param sorted_feature_importances_map: dictionary with feature names as keys and importance as values
    :param n: number of most important features to return
    :return: list of n most important features
    """
    return list(sorted_feature_importances_map.keys())[:n]

def get_feature_index_from_name(name):
    return feature_names.index(name)