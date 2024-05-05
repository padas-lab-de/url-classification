import os
import pickle
import pandas as pd

def list_models(model_directory):
    return [f for f in os.listdir(model_directory) if f.endswith('.pkl')]

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def load_urls(url_list):
    if url_list.endswith('.csv'):
        data = pd.read_csv(url_list)
        return data['url']
    elif url_list.endswith('.txt'):
        with open(url_list, 'r') as file:
            return [line.strip() for line in file]
    else:
        raise ValueError('URL list must be either a .csv file or a .txt file')

def predict_urls(model, urls):
    return model.predict(urls)

def save_predictions(urls, predictions, output_path):
    pd.DataFrame({'url': urls, 'predicted_label': predictions}).to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')

# Main execution flow
if __name__ == "__main__":
    model_directory = os.path.join(os.path.dirname(__file__), '../models/')
    models = list_models(model_directory)
    if not models:
        raise FileNotFoundError("No model files found in the directory.")

    print("Available models:")
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    model_index = int(input("Enter the number of the model you want to use: ")) - 1
    model_path = os.path.join(model_directory, models[model_index])

    model = load_model(model_path)
    
    url_list_path = input('Enter the path to the file containing the list of URLs: ')
    urls = load_urls(url_list_path)
    
    predictions = predict_urls(model, urls)
    
    save_predictions(urls, predictions, 'predictions.csv')
