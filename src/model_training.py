# use the data given by the user, split it into train and test data, train an sklearn pipeline to classify URLs from the column 'url' into the given labels in the column 'label', use SGDclassifier, and tfidf vectorizer with word-level of (3,6) n-grams, evaluate the model using accuracy, precision, recall, f1 and f2 scores, and finally save the trained pipeline to 'url-classification/models/SGD_pipeline.pkl'

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix

# Get the path of the dataset from the user, check if it is correct, then load it
data_path = input('Enter the path to the dataset (put it in the data directory, and paste its name here): ')
data_path = os.path.join(os.path.dirname(__file__), f'../data/{data_path}')
if not os.path.exists(data_path):
    raise FileNotFoundError(f'File "{data_path}" not found')

print(f'Loading data from {data_path}')
data = pd.read_csv(data_path)

# Check if the dataset has the columns 'url' and 'label'
if 'url' not in data.columns or 'label' not in data.columns:
    raise ValueError('Dataset must have columns "url" and/or "label"')

# Split the data into train and test sets
X = data['url']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weight={label: 1/len(y_train[y_train == label]) for label in np.unique(y_train)}

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(3,6))),
    ('clf', SGDClassifier(loss='modified_huber', class_weight=class_weight, random_state=42))
])

print('Training the model...')
# Train the model
pipeline.fit(X_train, y_train)

print('Evaluating the model...')
# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
f2 = fbeta_score(y_test, y_pred, average='weighted', beta=2)
conf_matrix = confusion_matrix(y_test, y_pred)

# Get the evaluation metrics for each class
precision_per_class = precision_score(y_test, y_pred, average=None, labels=np.unique(y_pred))
recall_per_class = recall_score(y_test, y_pred, average=None, labels=np.unique(y_pred))
f1_per_class = f1_score(y_test, y_pred, average=None, labels=np.unique(y_pred))


print('Saving the model...')
# Save the model
model_path = os.path.join(os.path.dirname(__file__), '../models/SGD_pipeline.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(pipeline, file)

# In url-classification/model_results save the evaluation metrics in  model_results.csv (if it does not exist, create it) with the columns 'model_name' 'date' 'accuracy', 'precision', 'recall', 'f1', 'f2' and per class metrics in the columns 'precision_per_class', 'recall_per_class', 'f1_per_class'
model_results_path = os.path.join(os.path.dirname(__file__), '../model_results/model_results.csv')
class_metrics = pd.DataFrame({
    'label': np.unique(y_pred),
    'precision_per_class': precision_per_class,
    'recall_per_class': recall_per_class,
    'f1_per_class': f1_per_class
})

model_results = pd.DataFrame({
    'model_name': ['SGDClassifier'],
    'date': [pd.Timestamp.now()],
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f1': [f1],
    'f2': [f2],
})
for label in np.unique(y_pred):
    model_results[f'precision_{label}'] = class_metrics[class_metrics['label'] == label]['precision_per_class'].values
    model_results[f'recall_{label}'] = class_metrics[class_metrics['label'] == label]['recall_per_class'].values
    model_results[f'f1_{label}'] = class_metrics[class_metrics['label'] == label]['f1_per_class'].values

if os.path.exists(model_results_path):
    model_results.to_csv(model_results_path, mode='a', header=False, index=False)
else:
    model_results.to_csv(model_results_path, index=False)


# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'F2: {f2}')

# print the confusion matrix in a readable format with the labels given, and values are normlaized
print('Confusion matrix:')
print(pd.DataFrame(conf_matrix, columns=np.unique(y_pred), index=np.unique(y_pred)))


print('Per class metrics:')
for label, precision, recall, f1 in zip(np.unique(y_pred), precision_per_class, recall_per_class, f1_per_class):
    print(f'Label: {label}, Precision: {precision}, Recall: {recall}, F1: {f1}')

