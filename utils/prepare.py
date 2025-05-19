import os
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def prepare_data(data, random_state=42):
    data['English'] = data['English'].str.lower().str.replace(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ ]+", "", regex=True)
    data['Turkish'] = data['Turkish'].str.lower().str.replace(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ ]+", "", regex=True)

    def clean_and_tag(text):
        return 'bos ' + text.strip() + ' eos'

    data['English'] = data['English'].astype(str).apply(clean_and_tag)
    data['Turkish'] = data['Turkish'].astype(str).apply(clean_and_tag)


    df_reverse = data.rename(columns={"English": "Turkish", "Turkish": "English"})
    data = pd.concat([data, df_reverse], ignore_index=True).drop_duplicates().reset_index(drop=True)

    data.rename(columns={"English": "Input", "Turkish": "Target"}, inplace=True)

    data = shuffle(data, random_state=random_state).reset_index(drop=True)

    data.to_csv("data/bidirectional_EN_TR_TR_EN.csv", index=False, encoding="utf-8-sig")
    print("✅ Çift yönlü veri seti 'data/bidirectional_EN_TR_TR_EN.csv'")
    return data

def tokenize_data(input_texts, target_texts):
    tokenizer = Tokenizer(filters='', lower=False, oov_token='<unk>')
    tokenizer.fit_on_texts(input_texts + target_texts)
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    max_seq_length = max(
        max(len(seq) for seq in input_sequences),
        max(len(seq) for seq in target_sequences)
    )

    encoder_input_data = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
    decoder_input_data = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

    decoder_target_data = np.zeros((len(target_texts), max_seq_length), dtype='int32')
    for i, seqs in enumerate(decoder_input_data):
        decoder_target_data[i, 0:max_seq_length - 1] = seqs[1:max_seq_length]

    num_tokens = len(tokenizer.word_index) + 1

    def save_tokenizer(tokenization, filepath='tokenization/tokenizer.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(tokenization, f)
        print(f"✅ Tokenizer kaydedildi: {filepath}")

    save_tokenizer(tokenizer)

    os.makedirs('tokenization', exist_ok=True)
    with open('tokenization/max_seq_length.pkl', 'wb') as f:
        pickle.dump(max_seq_length, f)
    print(f"✅ Max sequence length kaydedildi: tokenization/max_seq_length.pkl")

    return tokenizer, max_seq_length, num_tokens, encoder_input_data, decoder_input_data, decoder_target_data