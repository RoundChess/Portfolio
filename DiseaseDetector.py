#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 17:57:07 2024 by following a geeksforgeeks.com tutorial:
https://www.geeksforgeeks.org/disease-prediction-using-machine-learning/
"""

# Import main libraries

import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Read 'train.csv' file and drop last column because it is empty
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Determining if dataset is balanced
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values,
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Transforming labels into numerical values
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Splitting the data into training and testing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Defining model accuracy
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing machine learning (ML) Models
models = {
    "SVC": SVC(),
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(random_state=18)
}

# Scoring the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)

    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean score {np.mean(scores)}")

# Training and testing SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"SVM classifier's accuracy on train data: {accuracy_score(y_train, svm_model.predict(X_train)) * 100}")
print(f"SVM classifier's accuracy on test data: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion matrix for SVM classifier on test data")
plt.show()

# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)

print(f"Naive Bayes classifier's accuracy on train data: {accuracy_score(y_train, nb_model.predict(X_train)) * 100}")
print(f"Naive Bayes classifier's accuracy on test data: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes classifier on test data")
plt.show()


# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)

print(f"Random Forest Classifier's accuracy on train data: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
print(f"Random Forest Classifier's accuracy on test data: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes classifier on test data")
plt.show()

# Training the models on the entire dataset
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("Testing.csv")
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Make final prediction based on mode of the classifier's predictions
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

from scipy import stats

final_preds = [stats.mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
print(f"Combined model's accuracy on test data: {accuracy_score(test_Y, final_preds) * 100}")

# Plot a confusion matrix for the combined model on test dataset
cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))

sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion matrix for combined model on test dataset")
plt.show()

# Converts symptoms into numerical form
symptoms = X.columns.values

symptom_index = {}

for index, value, in enumerate(symptoms):
    symptom = "".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_,
}

# Disease predictor function
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # The input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    
    # Transforming the input data into a more appropriate format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating the outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Making final prediction based on the mode of the predictions
    import statistics
    
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
    }

    return predictions

# Testing the function

print(predictDisease("Itching,Skin_Rash,"))
