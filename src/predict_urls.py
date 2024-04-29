# load a list of URLs either in a .csv file (the column should be named 'url') or a list of URLs in a .txt file, and predict the labels for the URLs using the trained pipeline saved in 'url-classification/models/SGD_pipeline.pkl'

import os
import pickle
import pandas as pd
import numpy as np

def predict_urls(url_list):
    # Load the trained pipeline
    model_path = os.path.join(os.path.dirname(__file__), '../models/SGD_pipeline.pkl')
    with open(model_path, 'rb') as file:
        pipeline = pickle.load(file)

    # Load the URLs to predict
    if url_list.endswith('.csv'):
        data = pd.read_csv(url_list)
        X = data['url']
    elif url_list.endswith('.txt'):
        with open(url_list, 'r') as file:
            X = file.readlines()
    else:
        raise ValueError('URL list must be either a .csv file or a .txt file')

    # Predict the labels
    y_pred = pipeline.predict(X)
    return X, y_pred

# Get input from the user, use predict_urls to get the prediction and save them in a .csv file with the URLs and their predicted labels
url_list = input('Enter the path to the file containing the list of URLs: ')
X, y_pred = predict_urls(url_list)
output = pd.DataFrame({'url': X, 'predicted_label': y_pred})
output.to_csv('predictions.csv', index=False)

print(f'Predictions saved to predictions.csv')

