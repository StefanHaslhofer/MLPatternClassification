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


from helpers import plot_spectrogram, calc_deltas, flatten_data_by_mean, normalize_data, get_data_for_speakers, \
    get_label_map, filter_by_label

import helpers

# TODO set to True to show graphs
show_sample_graphs = False
show_scatter_plot = False
train_dummy = False
train_random_forest = False
train_command_random_forest = False
validate_random_forest = True
train_with_augmented_data = True

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
print(f"speaker ids : {speaker_ids.size} ")
np.random.shuffle(speaker_ids)
speaker_splits_ids = np.array_split(speaker_ids, n)


def calc_deltas(data):
    """
    calculate the change (delta) of the frequency energy over time
    """
    deltas = np.diff(data, axis=2)
    # trailing_zeros = np.zeros(shape=(deltas.shape[0] ,deltas.shape[1], 1))
    # deltas = np.concatenate((deltas, trailing_zeros), axis=2)
    # return np.append(data, deltas, axis=1)
    return deltas


def setup_knn(data):
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

    recording_folds = []
    for fold_idx, fold in enumerate(speaker_splits_ids):
        print(f"Retrieving data for fold {fold_idx + 1}/{n}")
        fold_data, fold_labels = get_data_for_speakers(fold, label_metadata, flattened_data)
        recording_folds.append({'data': fold_data, 'labels': fold_labels})
        print(f"Retrieved {len(fold_data)} samples with labels.")

    return recording_folds



#recording_folds = setup_knn(data)

def add_white_noise(wave, noise_factor: float):
   """
   :param wave: our scenes
   :param noise_factor: how strong the noise shouldbe
   :return:a white noise augmented image
   """
   noise = np.random.normal(0, wave.std(), wave.shape)
   augmented_wave = wave + noise * noise_factor
   return augmented_wave


def add_time_stretch(wave,rate:float):
    """
    Makes the recording slower or faster. I think it will be more beneficial in our task if the recordings
    are bit faster
    :param wave: your npy file wave
    :param rate: the speed in which you want to stretch the audio
    :return: augmented npy files
    """

    return librosa.effects.time_stretch(wave,rate=rate)



def pitch_shifting(wave,sr,n_steps):

    return librosa.effects.pitch_shift(wave,sr=sr,n_steps=n_steps)

def apply_augmentation_nparray(data,  noise_factor: float, n_steps, rate):
    data

def apply_augmentation_folder(inp_dir,out_dir,noise_factor:float,n_steps,rate):
    """
    Gets the data and loads it ,then apply three different augmentations
    with an option of plotting them
    :param inp_dir:
    :param out_dir:
    :param noise_factor: controlling factor of the noise
    :param n_steps: n_steps for the stretch rate
    :param rate: rate of the pitching scale
    :return: augmented npy files in the out_dir
    """
    #make the out_dir if it doesnt exist
    os.makedirs(out_dir,exist_ok=True)

    files = os.listdir(inp_dir)

    n_of_files = 0
    n_noise = 0
    n_time_stretch = 0
    n_pitch_scaled = 0
    augmented_data = []

    def plot_signal(signal, augmented_signal):
        fig, ax = plt.subplots(nrows=2)
        librosa.display.waveshow(signal, sr=sr, ax=ax[0])
        ax[0].set(title=f"Original{file}")
        librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
        ax[1].set(title="Augmented")
        plt.show()

    for file in files :
        if file.endswith(".npy"):

           file_path = os.path.join(inp_dir, file)
           wave = np.load(file_path)
           sr = 160000
           try:
               if n_of_files <= 300:
                   augmented_wave = add_white_noise(wave=wave, noise_factor=noise_factor)
                   augmentation_name = "noise"
                   n_noise += 1



               elif 300 < n_of_files <= 550:
                   augmented_wave = add_time_stretch(wave, rate=5)
                   augmentation_name = "time_stretched"
                   # n_of_files += 1
                   n_time_stretch += 1





               else:
                   augmented_wave = pitch_shifting(wave=wave, sr=sr, n_steps=0.2)
                   augmentation_name = "pitch_scaled"
                   n_pitch_scaled += 1

                   augmented_wave = add_white_noise(wave,0.5)

                   clean_file_name = file.replace(" ", "_").replace(",", "_").replace(";", "_")
                   out_file = os.path.join(out_dir, "augmented_" + clean_file_name)

                   np.save(out_file, augmented_wave)
                   print("File succesfully augmented")

                   if np.array_equal(wave, augmented_wave):
                        print("Warning: Augmented npy file is identical to original,check if augmentation worked!")

                   augmented_data.append(augmented_wave)
           except:
               print("An error occured!")




           return np.array(augmented_data)




        print(n_of_files)
        print(f"num of noise augmented:{n_noise}")
        print(f"num of pitch scaled augmented:{n_pitch_scaled}")
        print(f"num of time stretched augmented:{n_time_stretch}")









"""
Set up SVM
"""


# Append the labels to the data, assume that the labels are in the same order as the data
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
    # print("reduced feature data shape: ", reduced_feature_data.shape)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    # print("flattened data shape: ", flattened_data.shape)
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


def setup_random_forest(data, label_metadata, isTraining):
    reduced_feature_data = data[:, 13:60]
    if isTraining:
        print("reduced feature data shape: ", reduced_feature_data.shape)
    flattened_data = flatten_data_by_mean(reduced_feature_data, 1)
    if isTraining:
        print("flattened data shape: ", flattened_data.shape)

    if isTraining:
        speaker_train = speaker_ids[:int(speaker_ids.size * 0.8)]
        speaker_test = speaker_ids[int(speaker_ids.size * 0.8):]

        x_train, y_train = get_data_for_speakers(speaker_train, label_metadata, flattened_data)
        x_test, y_test = get_data_for_speakers(speaker_test, label_metadata, flattened_data)


            #Get speakers for augmented data
        if (train_with_augmented_data):
            # Get speakers for augmented data
            speaker_train_wn = speaker_ids[0:25]
            speaker_train_pitch = speaker_ids[25:45]
            # speaker_train_stretch = speaker_ids[31:45]

            x_train_wn, y_train_wn = get_data_for_speakers(speaker_train_wn, label_metadata, flattened_data)
            x_train_wn = add_white_noise(x_train_wn, 0.5)

            x_train_pitch, y_train_pitch = get_data_for_speakers(speaker_train_pitch, label_metadata, flattened_data)
            x_train_pitch = pitch_shifting(x_train_pitch, 16000, -4)

            # x_train_stretch, y_train_stretch = get_data_for_speakers(speaker_train_stretch, label_metadata, flattened_data)
            # x_train_stretch = add_time_stretch(x_train_stretch, 2)


            print(x_train_wn.shape)

            print(y_train_wn.shape)

            # y= pd.DataFrame(x_train_pitch)
            print(x_train_pitch.shape)

            print(f"Speaker ids training size : {speaker_train.size}")
            print(f"Speaker ids augmentation  size: {speaker_train_wn.size + speaker_train_pitch.size}")
            print(f"x white noise train shape : {x_train_wn.shape}")
            print(f"y white noise train label shape: {y_train_wn.shape}")
            print(f"x pitch noise train shape : {x_train_pitch.shape}")
            print(f"y pitch noise train label shape: {y_train_pitch.shape}")
            print(f"y pitch noise train label : {y_train_pitch}")

            print(x_train.shape)
            x_train = np.vstack((x_train, x_train_wn, x_train_pitch))
            y_train = np.hstack((y_train, y_train_pitch, y_train_wn))
            print(x_train.shape)

            return x_train, y_train, x_test, y_test
        else:
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
    x_train, y_train, x_test, y_test = setup_random_forest(data, label_metadata, train_random_forest)

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

    # validate the models with new data in the validation folder, which contains 5 separate .npy files
    # load from data files provided on moodle
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

if (train_command_random_forest):
    command_data = filter_by_label(['13', '14', '18'], label_metadata, data)
    x_train, y_train, x_test, y_test = setup_random_forest(command_data[0], command_data[1],
                                                           train_command_random_forest)
    model, predictions = fit_predict(x_train, y_train, x_test, y_test, RSEED)
    # error_train = mean_zero_one_loss(y_train, predictions)
    error_test = mean_zero_one_loss(y_test, predictions)
    print(f"command random forest error: {error_test}")

    # save model
    joblib.dump(model, "command_random_forest_model.pkl")

def validate_and_export_predictions():
    # Define the directory path
    development_scenes_path = 'development_scenes_npy/development_scenes'
    file_names = os.listdir(development_scenes_path)

    results = []

    for file_name in file_names:
        print(f"Filename: {file_name}")
        try:
            data_val = np.load(f'{development_scenes_path}/{file_name}')
            np.transpose(data_val)
            data_val = np.expand_dims(data_val, axis=0)
            if show_sample_graphs:
                plot_spectrogram(data_val[0][13:75], 'mfcc-spectrogram', ['x', 'y', 'z', file_name])
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise

        #print("Loaded validation data, with shape: ", data_val.shape)
        rfc = joblib.load("random_forest_model_best.pkl")
        crfc = joblib.load("command_random_forest_model.pkl")

        label_map = get_label_map(le)
        label_map_reverted = {v: k for k, v in label_map.items()}

        if validate_random_forest:
            x_val_rfc = setup_random_forest(data_val, label_metadata,False)
            print("x_val_rfc shape: ", x_val_rfc.shape)

            # heuristic: if the classifier predicts a keyword we assume that no other keyword follows for at least 1s
            SKIP_SAMPLES = 44 * 2
            word_predicted = False
            i = 0
            # run random forest on validation data, but only 44 samples at a time
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

                # if word was predicted, advance loop without prediction
                if word_predicted and i + SKIP_SAMPLES + 44 < x_val_rfc.shape[1]:
                    # heuristic: if a word is recognized a command must follow in the next 1.1 seconds
                    for j in range(i, i + SKIP_SAMPLES):
                        predictions_crfc = crfc.predict(x_val_rfc[:, j:j + 44])
                        if predictions_crfc != ['18']:
                            print(f"Predicting for samples {j} to {j + 44}")
                            print("Predictions: ", label_map_reverted[int(predictions_crfc[0])])
                            break

                if word_predicted:
                    i += SKIP_SAMPLES
                    word_predicted = False

                i += 1

    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "command", "timestamp"])
        for result in results:
            writer.writerow(result)

    print("Results written to results.csv")


validate_and_export_predictions()