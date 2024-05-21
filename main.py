import joblib
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import sklearn

# TODO set to True to show graphs
show_sample_graphs = False
show_scatter_plot = False
train_knn = False
tain_random_forest = False
validate_knn = True
validate_random_forest = False

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

def calc_deltas(data):
    """
    calculate the change (delta) of the frequency energy over time
    """
    deltas = np.diff(data, axis=2)
    # trailing_zeros = np.zeros(shape=(deltas.shape[0] ,deltas.shape[1], 1))
    # deltas = np.concatenate((deltas, trailing_zeros), axis=2)
    # return np.append(data, deltas, axis=1)
    return deltas


def setup_knn(data, isTraining):
    # only use mfcc (also cut away mfcc bin 0 and upper bins)
    reduced_feature_data = data[:, 13:60]
    print("reduced feature data shape: ", reduced_feature_data.shape)
    # delta_features = append_deltas(reduced_feature_data)
    # calculate mean for each frame (along axis 1)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    print("flattened data shape: ", flattened_data.shape)
    # don't normalize spectrograms
    # normalized_data = normalize_data(data)
    # print("data normalized: ", normalized_data.shape)

    if isTraining:
        recording_folds = []
        for fold_idx, fold in enumerate(speaker_splits_ids):
            print(f"Retrieving data for fold {fold_idx + 1}/{n}")
            fold_data, fold_labels = get_data_for_speakers(fold, label_metadata, flattened_data)
            recording_folds.append({'data': fold_data, 'labels': fold_labels})
            print(f"Retrieved {len(fold_data)} samples with labels.")

        return recording_folds

    else:
        return flattened_data


if(train_knn):
    recording_folds = setup_knn(data, train_knn)

"""
Set up SVM
"""


def setup_svm(data):
    # Flatten the data, by taking the mean of each feature, for each audio snippet.
    # Effectively we will have 1 row with 175 features for each audio snippet

    flatten_data = []
    for i in range(len(data)):
        flatten_data.append(np.mean(data[i], axis=1))

    flatten_data = np.array(flatten_data)
    print(f"Data flattened ${flatten_data.shape}")  # Should be (45k, 175)

    # Now we will normalize the data accross the features, so that the max value is 1 and the min value is 0
    normalized_data = normalize_data(flatten_data)
    print("Data normalized")

    svm_recording_folds = []
    for fold_idx, fold in enumerate(speaker_splits_ids):
        print(f"Retrieving data for fold {fold_idx + 1}/{n}")
        fold_data, fold_labels = get_data_for_speakers(fold, label_metadata, normalized_data)
        svm_recording_folds.append({'data': fold_data, 'labels': fold_labels})
        print(f"Retrieved {len(fold_data)} samples with labels.")
        # print("recording folds: ", recording_folds)

    return svm_recording_folds

def train_svm(recording_folds):
    """
    Train and evaluate the SVM classifier using k-fold cross-validation.
    """
    param_dist = {
        'svc__C': [0.1, 1, 10, 100, 1000],
        'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'svc__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }

    accuracies = []
    for i in range(n):
        print(f"Fold {i + 1}/{n}")
        test_data = recording_folds[i]['data']
        test_labels = recording_folds[i]['labels']
        training_data = np.concatenate([recording_folds[j]['data'] for j in range(n) if j != i])
        training_labels = np.concatenate([recording_folds[j]['labels'] for j in range(n) if j != i])

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ])

        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, refit=True, verbose=3)
        random_search.fit(training_data, training_labels)

        print(f"Best parameters for fold {i + 1}: {random_search.best_params_}")
        y_pred = random_search.predict(test_data)
        accuracy = accuracy_score(test_labels, y_pred)
        accuracies.append(accuracy)
        print(f"Accuracy for fold {i + 1}: {accuracy}")

    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy across all folds: {mean_accuracy}")
    return mean_accuracy

'''print("SVM Classifier")
svm_recording_folds = setup_svm(data)
mean_accuracy = train_svm(svm_recording_folds)'''


"""
Set up kNN
"""


def train_kNN(X_train, train_label, k_train):
    """
    Function that fits a kNN to given data
    @param X_train, np array, training data
    @param train_label, np array, training labels
    @param k_train, integer, k for the kNN
    @output classifier, kNN instance, classifier that was fitted to training data
    """
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
    predictions = classifier.predict(X_eval)

    return predictions


def mean_zero_one_loss(true_label, pred_label):
    """
    Function that calculates the mean zero-one loss for given true and predicted la
    @param true_label, np array, true labels
    @param pred_label, np array, predicted labels
    @output loss, float, mean zero-one loss
    """
    loss = sklearn.metrics.zero_one_loss(true_label, pred_label)
    return loss


def run_kNN(d, nf, k):
    """
    Function that combines all functions using CV
    @param d, np array, training data with labels
    @param nf, integer, number of folds for CV
    @param k, integer, k for kNN
    @output mean_error, float, mean error over all folds
    """
    error = 0
    print(f"Training kNN (folds={nf}, k={k})...")
    # split lists into nf distinct sets
    X_train = [e.get('data') for e in d]
    label_train = [e.get('labels') for e in d]
    for i in range(nf):
        if show_scatter_plot:
            scatter_features(X_train[i], label_train[i])
        # use set at index i as evaluation set
        classifier = train_kNN(
            np.concatenate(X_train[:i] + X_train[i + 1:]),  # leave out test sub
            np.concatenate(label_train[:i] + label_train[i + 1:]),
            k)
        predictions = eval_kNN(classifier, X_train[i])
        error += mean_zero_one_loss(label_train[i], predictions)

        # We found out that the model performs best with k=5, so we will save the model with k=5
        if(i == 5):
            joblib.dump(classifier, "knn_model.pkl")

    return error / nf


if train_knn:
    max_k = 179
    error_holder = []
    for k in range(1, max_k + 1, 2):  # range with 179 included and step of 2
        error_holder.append(run_kNN(recording_folds, n, k))
        None

"""
Set up random forests
"""
RSEED = 10


def setup_random_forest(data, isTraining):
    # only use mfcc (also cut away mfcc bin 0 and upper bins)
    reduced_feature_data = data[:, 13:60]
    print("reduced feature data shape: ", reduced_feature_data.shape)
    # calculate mean for each frame (along axis 1)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    print("flattened data shape: ", flattened_data.shape)
    # normalized_data = normalize_data(data)
    # print("data normalized: ", normalized_data.shape)

    if(isTraining):
        # 80% - 20% | train - test split
        speaker_train = speaker_ids[:int(speaker_ids.size * 0.8)]
        speaker_test = speaker_ids[int(speaker_ids.size * 0.8):]

        # get training data
        x_train, y_train = get_data_for_speakers(speaker_train, label_metadata, flattened_data)
        # get test data
        x_test, y_test = get_data_for_speakers(speaker_test, label_metadata, flattened_data)
        return x_train, y_train, x_test, y_test

    else:
        return flattened_data

if(tain_random_forest):
    x_train, y_train, x_test, y_test = setup_random_forest(data, tain_random_forest)


def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        rseed: int
) -> tuple[RandomForestClassifier, np.ndarray]:
    print(f"Train random forest classifier with {x_train.shape[0]} samples.")
    clf = RandomForestClassifier(random_state=rseed)
    model = clf.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("Finished random forest prediction.")
    return model, prediction


if(tain_random_forest):
    model, predictions = fit_predict(x_train, y_train, x_test, y_test, RSEED)
    # error_train = mean_zero_one_loss(y_train, predictions)
    error_test = mean_zero_one_loss(y_test, predictions)
    print(f"random forest error: {error_test}")

    # save model
    joblib.dump(model, "random_forest_model.pkl")


# validate the models with new data in the validation folder, which contains 5 separate .npy files
# load from data files provided on moodle
try:
    filename = '3_Verena_Staubsauger_an_Alarm_an.npy' # change filename to test different files
    data_val = np.load(f'validation_data/{filename}')
    plot_spectrogram(data_val[0][13:75], 'mfcc-spectrogram', ['x','y','z','Verena_Staubsauger_an_Alarm_an'])
except Exception as e:
    print(f"Failed to load data: {e}")
    raise

print("Loaded validation data, with shape: ", data_val.shape)
rfc = joblib.load("random_forest_model.pkl")
knn = joblib.load("knn_model.pkl")

label_map = get_label_map(le)
print("Label map: ", label_map)
# revert key value pairs in label map
label_map_reverted = {v: k for k, v in label_map.items()}
print("Label map reverted: ", label_map_reverted)

if validate_random_forest:
    # setup data for validation for random forest
    x_val_rfc = setup_random_forest(data_val, False)
    print("x_val_rfc shape: ", x_val_rfc.shape)

    # run random forest on validation data, but only 44 samples at a time
    for i in range(0, x_val_rfc.shape[1], 1):
        predictions_rfc = rfc.predict(x_val_rfc[:, i:i + 44])
        if predictions_rfc != ['18']:
            print(f"Predicting for samples {i} to {i + 44}")
            print("Predictions: ", label_map_reverted[int(predictions_rfc[0])])

if validate_knn:
    # setup data for validation for kNN
    x_val_knn = setup_knn(data_val, False)
    print("x_val_knn shape: ", x_val_knn.shape)

    # run kNN on validation data
    for i in range(0, x_val_knn.shape[1], 1):
        predictions_knn = knn.predict(x_val_knn[:, i:i + 44])
        if predictions_knn != ['18']:
            print(f"Predicting for samples {i} to {i + 44}")
            print("Predictions: ", label_map_reverted[int(predictions_knn[0])])