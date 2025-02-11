import pickle
import warnings
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer




def set_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            print("🚀 GPU kullanımına zorlandı:", gpus)
        except RuntimeError as e:
            print("Hata:", e)
    else:
        print("🚨 GPU algılanmadı! TensorFlow ve CUDA uyumluluğunu tekrar kontrol et.")

def load_and_prepare_data():
    df1 = pd.read_csv("Data/try_data.csv", sep=";", header=0, names=["English", "Turkish"])
    df2 = pd.read_csv("Data/try2_data.csv", sep=";", header=0, names=["English", "Turkish"])
    df3 = pd.read_csv("Data/new_data.csv", sep=";", header=0, names=["English", "Turkish"])
    df4 = pd.read_csv("Data/newData555.csv", sep=";", header=0, names=["English", "Turkish"])
    df5 = pd.read_csv("Data/newData68.csv", sep=";", header=0, names=["English", "Turkish"])

    data = pd.concat([df1, df2, df3, df4, df5], ignore_index=True).drop_duplicates().reset_index(drop=True)

    data['English'] = data['English'].str.lower().str.replace(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ ]+", "", regex=True)
    data['Turkish'] = data['Turkish'].str.lower().str.replace(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ ]+", "", regex=True)

    data['English'] = data['English'].apply(lambda x: 'bos ' + x.strip() + ' eos')
    data['Turkish'] = data['Turkish'].apply(lambda x: 'bos ' + x.strip() + ' eos')

    df_reverse = data.rename(columns={"English": "Turkish", "Turkish": "English"})
    data = pd.concat([data, df_reverse], ignore_index=True).drop_duplicates().reset_index(drop=True)

    data.rename(columns={"English": "Input", "Turkish": "Target"}, inplace=True)

    data.to_csv("Data/MergedDataBoth_EN_TR.csv", index=False, encoding="utf-8-sig")
    print("✅ Veri kaydedildi: 'MergedDataBoth_EN_TR.csv'")
    return data

def split_data(data):
    return train_test_split(data['Input'].values, data['Target'].values, test_size=0.2, random_state=42)

def tokenize_data(input_texts, target_texts):
    tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(np.concatenate((input_texts, target_texts)))

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

    return tokenizer, max_seq_length, num_tokens, encoder_input_data, decoder_input_data, decoder_target_data

def build_model(num_tokens, latent_dim):
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(num_tokens, latent_dim, name='encoder_embedding')(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(
        latent_dim,
        return_state=True,
        name='encoder_lstm'
    )(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(num_tokens, latent_dim, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(
        num_tokens,
        activation='softmax',
        name='decoder_dense'
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('model2.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping, model_checkpoint]
    )
    print("✅ Model eğitildi ve kaydedildi!")

    return history

def save_tokenizer(tokenizer):
    with open('tokenizer2.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    with open('tokenizer2.pkl', 'rb') as f:
        return pickle.load(f)

def load_trained_model():
    return load_model("model2.keras")

def load_inference_models(model, latent_dim):
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + [state_h_dec, state_c_dec]
    )

    return encoder_model, decoder_model

def preprocess_sentence(sentence):
    sentence = sentence.strip()
    sentence = 'bos ' + sentence + ' eos'
    return sentence

def translate_sentence(sentence, tokenizer, encoder_model, decoder_model, max_seq_length):
    sentence = preprocess_sentence(sentence)
    input_seq = tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1), dtype='int32')
    target_seq[0, 0] = tokenizer.word_index.get('bos')

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'eos' or len(decoded_sentence.split()) > max_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

            target_seq = np.zeros((1,1), dtype='int32')
            target_seq[0, 0] = sampled_token_index

            states_value = [h, c]

    return decoded_sentence.strip()

if __name__ == "__main__":
    data = load_and_prepare_data()
    tokenizer = load_tokenizer()
    model = load_trained_model()
    encoder_model, decoder_model = load_inference_models(model, 64)

    with open('max_seq_length.pkl', 'rb') as f:
        max_seq_length = pickle.load(f)

    while True:
        sentence = input("Çeviri için cümle girin (çıkmak için 'q'): ")
        if sentence.lower() == 'q':
            break
        translation = translate_sentence(sentence, tokenizer, encoder_model, decoder_model, max_seq_length)
        print(f"📥 {sentence} → 📤 {translation}")
  #data = pd.read_csv("Data/MergedDataBoth_EN_TR.csv", header=0, names=["Input", "Target"]).head(100000)
    #input_train, input_test, target_train, target_test = split_data(data)
#
    #tokenizer, max_seq_length, num_tokens, encoder_input_data, decoder_input_data, decoder_target_data = tokenize_data(
    #   input_train, target_train
    #)
#
    #save_tokenizer(tokenizer)
#
    #with open('max_seq_length.pkl', 'wb') as f:
    #   pickle.dump(max_seq_length, f)
#
    #print("⏳ Model oluşturuluyor ve eğitiliyor...")
    #model = build_model(num_tokens, latent_dim=64)
    #history = train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=50)
#
    ## Eğitim ve doğrulama kaybını görselleştirme
    #plt.plot(history.history['loss'], label='Eğitim Kaybı')
    #plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    #plt.title('Eğitim ve Doğrulama Kaybı')
    #plt.xlabel('Epoch')
    #plt.ylabel('Kayıp')
    #plt.legend()
    #plt.show()
#
    #print("✅ Model eğitildi ve kaydedildi!")

    # Modeli yükleme ve çeviri