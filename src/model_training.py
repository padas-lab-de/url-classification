from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import pickle
import os

def load_data(data_path):
    full_path = os.path.join(os.path.dirname(__file__), f'../data/{data_path}')
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File "{full_path}" not found')
    print(f'Loading data from {full_path}')
    return pd.read_csv(full_path)

def split_data(data):
    if 'url' not in data.columns or 'label' not in data.columns:
        raise ValueError('Dataset must have columns "url" and "label"')
    X = data['url']
    y = data['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline(model_choice, class_weight=None):
    vectorizer = TfidfVectorizer(ngram_range=(3, 6))
    if model_choice == '1':
        classifier = SGDClassifier(loss='modified_huber', class_weight=class_weight, random_state=42)
    elif model_choice == '2':
        classifier = LinearSVC(class_weight=class_weight, random_state=42)
    else:
        raise ValueError("Invalid model choice")
    return Pipeline([('tfidf', vectorizer), ('clf', classifier)])

def evaluate_model(model, X_test, y_test, model_name, model_results_path):
    y_pred = model.predict(X_test)
    # Calculate general metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'f2': fbeta_score(y_test, y_pred, beta=2, average='weighted'),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Calculate per-class metrics
    unique_labels = ['Adult', 'Benign', 'Malicious']
    precision_per_class = precision_score(y_test, y_pred, average=None, labels=unique_labels)
    recall_per_class = recall_score(y_test, y_pred, average=None, labels=unique_labels)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=unique_labels)
    
    # Create DataFrame for general metrics
    results_df = pd.DataFrame({
        'model_name': [model_name],
        'date': [pd.Timestamp.now()],
        'accuracy': [metrics['accuracy']],
        'precision_weighted': [metrics['precision']],
        'recall_weighted': [metrics['recall']],
        'f1_weighted': [metrics['f1']],
        'f2_weighted': [metrics['f2']]
    })
    
    # Adding per-class metrics to DataFrame
    for label, prec, rec, f1 in zip(unique_labels, precision_per_class, recall_per_class, f1_per_class):
        results_df[f'precision_{label}'] = prec
        results_df[f'recall_{label}'] = rec
        results_df[f'f1_{label}'] = f1
    
    # Append results to the CSV file
    if os.path.exists(model_results_path):
        results_df.to_csv(model_results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(model_results_path, index=False)
    
    return y_pred, metrics

def save_model(model, model_name):
    model_path = os.path.join(os.path.dirname(__file__), f'../models/{model_name}.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Main execution flow
if __name__ == "__main__":
    model_choices = { '1': 'SGDClassifier', '2': 'SVC' }
    print("Supported models:")
    for key, value in model_choices.items():
        print(f"{key}: {value}")
    choice = input("Enter the number of the model you want to use: ")
    
    data_path = input('Enter the path to the dataset (put it in the data directory, and paste its name here): ')
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data)
    
    class_weight = {label: 1/len(y_train[y_train == label]) for label in np.unique(y_train)}
    
    pipeline = create_pipeline(choice, class_weight=class_weight)
    print('Training the model...')
    pipeline.fit(X_train, y_train)
    
    print('Evaluating the model...')
    y_pred, metrics = evaluate_model(pipeline, X_test, y_test, model_choices[choice], os.path.join(os.path.dirname(__file__), '../model_results/model_results.csv'))
    
    print('Saving the model...')
    save_model(pipeline, model_choices[choice])
    
    print(f"Model performance metrics:")
    for key, value in metrics.items():
        if key != 'conf_matrix':
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}:")
            print(pd.DataFrame(value, columns=np.unique(y_pred), index=np.unique(y_pred)))
