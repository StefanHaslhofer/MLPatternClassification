import csv
import joblib
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

from helpers import plot_spectrogram, calc_deltas, flatten_data_by_mean, normalize_data, get_data_for_speakers, get_label_map

# TODO set to True to show graphs
show_sample_graphs = False
show_scatter_plot = False
train_dummy = False
train_random_forest = False
validate_random_forest = True

# load from data files provided on moodle
try:
    label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
    idx_to_feature = np.loadtxt('metadata/idx_to_feature_name.csv', delimiter=',', usecols=1, skiprows=1, dtype='U')
    data = np.load('development_numpy/development.npy')
except Exception as e:
    print(f"Failed to load data: {e}")
    raise

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
    labels_2d = labels.reshape(-1, 1)
    if data.shape[0] != labels_2d.shape[0]:
        raise ValueError(f"Number of rows in data ({data.shape[0]}) and labels ({labels_2d.shape[0]}) do not match.")
    appended_data = np.column_stack((data, labels_2d))
    return appended_data


def mean_zero_one_loss(true_label, pred_label):
    loss = sklearn.metrics.zero_one_loss(true_label, pred_label)
    return loss


def plot_error_vs_k(error_holder, m):
    plt.bar(range(1, m + 1, 2), error_holder, color='maroon')
    plt.xlabel("k")
    plt.ylabel("error (%)")
    plt.title("Mean error for k-nearest-neighbors")
    plt.show()


if show_sample_graphs:
    None


def dummy_cls(data):
    reduced_feature_data = data[:, 13:60]
    print("reduced feature data shape: ", reduced_feature_data.shape)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    print("flattened data shape: ", flattened_data.shape)
    normalized_data = normalize_data(flattened_data)

    X_train, X_test, y_train, y_test = train_test_split(normalized_data, label_metadata['word'], test_size=0.2,
                                                        random_state=42)
    dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    error = mean_zero_one_loss(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Baseline Classifier Error:", error)
    print("Classification Report:")
    print(report)

    return error


if train_dummy:
    dummy_cls(data)


RSEED = 10


def setup_random_forest(data, isTraining):
    reduced_feature_data = data[:, 13:60]
    print("reduced feature data shape: ", reduced_feature_data.shape)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    print("flattened data shape: ", flattened_data.shape)

    if isTraining:
        speaker_train = speaker_ids[:int(speaker_ids.size * 0.8)]
        speaker_test = speaker_ids[int(speaker_ids.size * 0.8):]
        x_train, y_train = get_data_for_speakers(speaker_train, label_metadata, flattened_data)
        x_test, y_test = get_data_for_speakers(speaker_test, label_metadata, flattened_data)
        return x_train, y_train, x_test, y_test

    else:
        return flattened_data


def fit_predict(x_train, y_train, x_test, y_test, rseed, params) -> tuple[RandomForestClassifier, np.ndarray]:
    print(f"Train random forest classifier with {x_train.shape[0]} samples.")
    clf = RandomForestClassifier(random_state=rseed, **params)
    model = clf.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("Finished random forest prediction.")
    return model, prediction


def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


if train_random_forest:
    x_train, y_train, x_test, y_test = setup_random_forest(data, train_random_forest)

    # Define hyperparameter grid
    hyperparams_grid = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 20},
        {'n_estimators': 300, 'max_depth': 30},
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 200, 'max_depth': None},
    ]

    best_accuracy = 0
    best_params = None

    for params in hyperparams_grid:
        print(f"Testing parameters: {params}")
        model, predictions = fit_predict(x_train, y_train, x_test, y_test, RSEED, params)
        accuracy, report = evaluate_model(y_test, predictions)
        print(f"Accuracy: {accuracy}\n{report}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Best accuracy: {best_accuracy} with params: {best_params}")

    # Save the best model
    model, predictions = fit_predict(x_train, y_train, x_test, y_test, RSEED, best_params)
    joblib.dump(model, "random_forest_model_best.pkl")

def validate_and_export_predictions():
    development_scenes_path = 'development_scenes_npy/development_scenes'
    file_names = os.listdir(development_scenes_path)

    results = []

    for file_name in file_names:
        try:
            data_val = np.load(f'{development_scenes_path}/{file_name}')
            np.transpose(data_val)
            data_val = np.expand_dims(data_val, axis=0)
            if show_sample_graphs:
                plot_spectrogram(data_val[0][13:75], 'mfcc-spectrogram', ['x', 'y', 'z', 'Verena_Staubsauger_an_Alarm_an'])
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise

        print("Loaded validation data, with shape: ", data_val.shape)
        rfc = joblib.load("random_forest_model_best.pkl")

        label_map = get_label_map(le)
        label_map_reverted = {v: k for k, v in label_map.items()}

        if validate_random_forest:
            x_val_rfc = setup_random_forest(data_val, False)
            print("x_val_rfc shape: ", x_val_rfc.shape)

            SKIP_SAMPLES = 44
            word_predicted = False
            i = 0
            while i < x_val_rfc.shape[1]:
                prediction_data = x_val_rfc[:, i:i + 44]
                if np.shape(prediction_data)[1] == 44:
                    predictions_rfc = rfc.predict(prediction_data)
                    if predictions_rfc != ['18']:
                        print(f"Predicting for samples {i} to {i + 44}")
                        print("Predictions: ", label_map_reverted[int(predictions_rfc[0])])
                        word_predicted = True
                        base_filename = os.path.splitext(file_name)[0]
                        timestamp = i * 0.025
                        results.append((base_filename, label_map_reverted[int(predictions_rfc[0])], round(timestamp, 3)))
                if word_predicted and i + SKIP_SAMPLES <= x_val_rfc.shape[1]:
                    i += SKIP_SAMPLES
                    word_predicted = False
                i += 1

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "command", "timestamp"])
        for result in results:
            writer.writerow(result)

    print("Results written to results.csv")
