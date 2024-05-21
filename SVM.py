import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import helpers

# In this file we will implement a Support Vector Machine (SVM) classifier to classify the speaker of an audio snippet.
# It is the same data as in main.py and we  will reuse some functions

# First step is to load the data
label_metadata = np.genfromtxt('metadata/development.csv', dtype=None, delimiter=',', names=True, encoding='utf-8')
idx_to_feature = np.loadtxt('metadata/idx_to_feature_name.csv', delimiter=',', usecols=1, skiprows=1, dtype='U')
data = np.load('development_numpy/development.npy')

print("SVM Classifier file")
print(f"Data loaded: {data.shape}")

def preprocess_data(data):
    """
    Preprocess the data by flattening it, normalizing it and appending the labels to it.
    """
    # Flatten the data, by taking the mean of each feature, for each audio snippet. Effectively we will have 1 row with 175 features for each audio snippet
    flatten_data = []
    for i in range(len(data)):
        flatten_data.append(np.mean(data[i], axis=1))

    flatten_data = np.array(flatten_data)
    print(f"Data flattened ${flatten_data.shape}") #Should be (45k, 175)

    # Now we will normalize the data accross the features, so that the max value is 1 and the min value is 0
    normalize_data = helpers.normalize_data(flatten_data)
    print("Data normalized")

    # Append labels to the data
    data = normalize_data
    labels = label_metadata['word']
    # print the shapes
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    return data, labels

# Preprocess the data
normalize_data, labels = preprocess_data(data)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(normalize_data, labels, test_size=0.2, random_state=42)

#print the shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Create the SVM model
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# Train the model
model.fit(X_train, y_train)

# Predict the labels
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Calculate the f1 score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1 Score: {f1}")

# Save the model using joblib
print("Saving the model")
joblib.dump(model, 'svm_model.joblib')


'''# Define the parameter values that should be searched
param_grid = {'svc__C': [0.1, 1, 10, 100, 1000],
              'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'svc__kernel': ['rbf', 'linear', 'poly']}

# Create a pipeline
pipeline = make_pipeline(StandardScaler(), SVC())

# Instantiate the grid
grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=3)

# Fit the grid with data
grid.fit(X_train, y_train)

# Print the best parameters
print("Grid best params: ",grid.best_params_)

# Predict the labels using the best model
y_pred = grid.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the f1 score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1 Score: {f1}")

# Save the model using joblib
joblib.dump(grid.best_estimator_, 'svm_model.joblib') '''