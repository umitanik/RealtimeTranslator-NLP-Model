import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

def load_data():
    data_1 = pd.read_csv('data/EN_TR_711767.csv', sep=";", header=0, names=["English", "Turkish"])
    data_2 = pd.read_csv('data/EN_TR_1048575.csv', sep=",", header=0, names=["English", "Turkish"])
    data = pd.concat([data_1, data_2])
    return data

def load_tokenizer(filepath='tokenization/tokenizer.pkl'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Tokenizer dosyası bulunamadı: {filepath}")
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✅ Tokenizer yüklendi: {filepath}")
    return tokenizer

def load_trained_model(filepath='models/model.keras'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {filepath}")
    model = load_model(filepath)
    print(f"✅ Model yüklendi: {filepath}")
    return model