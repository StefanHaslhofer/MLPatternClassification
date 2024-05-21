from sklearn.model_selection import train_test_split

import main
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

# In this file we will implement a Support Vector Machine (SVM) classifier to classify the speaker of an audio snippet.
# It is the same data as in main.py and we we will reuse some functions

# First step is to load the data
label_metadata = main.label_metadata
idx_to_feature = main.idx_to_feature
data = main.data

print("Data loaded")

# Flatten the data, by taking the mean of each feature, for each audio snippet. Effectively we will have 1 row with 175 features for each audio snippet
flatten_data = []
for i in range(len(data)):
    flatten_data.append(np.mean(data[i], axis=0))

flatten_data = np.array(flatten_data)
print(f"Data flattened ${flatten_data.shape}") #Should be (45k, 175)

# Now we will normalize the data accross the features, so that the max value is 1 and the min value is 0
normalize_data = main.normalize_data(flatten_data)
print("Data normalized")

# Append labels to the data
data = normalize_data
labels = label_metadata['word']
# print the shapes
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(normalize_data, labels, test_size=0.2, random_state=42)

# Create the SVM model
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# Train the model
model.fit(X_train, y_train)

# Predict the labels
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)