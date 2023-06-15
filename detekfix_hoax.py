import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import tensorflow as tf
import xgboost as xgb
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

with open('tokenizer_fix.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
key_norm = pd.read_csv('https://raw.githubusercontent.com/kinanti18/ofa-2022/main/key_norm.csv')

rf_class = joblib.load("rf_fix.joblib")

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
    text = str.lower(text)
    
    return text

def stemming(text):
    text = stemmer.stem(text)
    
    return text

def casefolding(text):
    text = text.lower()                               
    text = re.sub(r'https?://\S+|www\.\S+', '', text)           
    text = re.sub(r'[^\w\s]','', text)                
    text = text.strip()
    
    return text


def predict_model_rf(text):
    maxlen = 32
    X = stemming(text_normalize(casefolding(text)))
    X_tok = tokenizer.texts_to_sequences([X])
    X_pad = pad_sequences(X_tok, maxlen=maxlen)
    
    output = rf_class.predict_proba(X_pad)
    
    return output