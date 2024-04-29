# Webpage URL Classification

This repository contains code for URL Classification , designed to classify URLs into predefined categories using machine learning techniques. The project utilizes a leanier classifier with Stochastic Gradient Descent (SGD) optimizer, alongside a TfidfVectorizer for feature extraction.


### Key Components

- **data/**: Contains the dataset used for training and evaluating the model.
- **model_results/**: Stores the evaluation metrics and results from model training.
- **models/**: Contains the serialized form of the trained model.
- **src/**: Source scripts including model training and URL prediction functionalities.

## Setup

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/padas-lab-de/url-classification.git
   cd url-classification
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows:**
     ```cmd
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, navigate to the `src/` directory and run the `model_training.py` script. You will be prompted to enter the path to the dataset file:

```bash
cd src
python model_training.py
```

Follow the prompts to enter the dataset name (e.g., `OWS_URL_DS.csv`). The script will train the model and save it along with the evaluation metrics.

### Predicting URL Labels

To classify new URLs, use the `predict_urls.py` script. You will need to provide a path to a file containing URLs in either `.csv` or `.txt` format:

```bash
python predict_urls.py
```

The predictions will be saved to `predictions.csv` in the root directory.

## Contributing

Contributions to this project are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
