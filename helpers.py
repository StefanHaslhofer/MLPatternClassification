import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def get_label_map(le: LabelEncoder):
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return label_map

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