import argparse
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import yaml

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return ' '.join(words)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def predict_text(config, text):
    vectorizer = TfidfVectorizer(**config['vectorizer_params'])
    encoder = LabelEncoder()

    model = SVC(kernel='linear')
    model.set_params(**config['model_params'])

    # Load the pre-trained model and encoder
    model.load(config['model_path'])
    encoder.load(config['encoder_path'])

    # Preprocess the text
    text = preprocess_text(text)

    # Transform the preprocessed text into a feature vector
    text_vec = vectorizer.transform([text])

    # Make a prediction
    predicted_label = encoder.inverse_transform(model.predict(text_vec))[0]
    lst = predicted_label.split("_")
    result = {"Action": lst[0], "Object": lst[1], "Location": lst[2]}

    return result

def main():
    parser = argparse.ArgumentParser(description='CPU Inferencing script for text classification')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--text', type=str, help='Text to classify')
    args = parser.parse_args()

    config_path = args.config
    text = args.text

    config = load_config(config_path)

    result = predict_text(config, text)
    print(result)

if __name__ == '__main__':
    main()
