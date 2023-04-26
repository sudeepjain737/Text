import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import joblib
import yaml

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return ' '.join(words)

def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load data
    data = pd.read_csv(config['train_data_path'])
    data['transcription'] = data['transcription'].apply(preprocess_text)
    data['label'] = data['action'] + '_' + data['object'] + '_' + data['location']

    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(data['transcription'], data['label'], test_size=0.2, random_state=42)

    # create vectorizer and encoder
    vectorizer = TfidfVectorizer(**config['vectorizer_config'])
    X_train = vectorizer.fit_transform(X_train)
    X_valid = vectorizer.transform(X_valid)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_valid = encoder.transform(y_valid)

    # train model
    clf = SVC(**config['clf_config'])
    clf.fit(X_train, y_train)

    # evaluate model
    y_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='macro')
    recall = recall_score(y_valid, y_pred, average='macro')
    f1 = f1_score(y_valid, y_pred, average='macro')
    print('Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(acc, precision, recall, f1))

    # save model and config
    joblib.dump(clf, config['model_save_path'])
    config['label_encoder_classes'] = encoder.classes_
    joblib.dump(config, config['config_save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
