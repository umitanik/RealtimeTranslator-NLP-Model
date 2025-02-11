import pickle
import threading

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


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

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + [state_h_dec, state_c_dec])

    return encoder_model, decoder_model


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    sentence = 'bos ' + sentence + ' eos'
    return sentence


model_lock = threading.Lock()


def translate_sentence(sentence, tokenizer, encoder_model, decoder_model, max_seq_length):
    sentence = preprocess_sentence(sentence)
    input_seq = tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    with model_lock:
        states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = tokenizer.word_index.get('bos')

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        with model_lock:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'eos' or len(decoded_sentence.split()) > max_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.zeros((1, 1), dtype='int32')
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return decoded_sentence.strip()


try:
    tokenizer = load_tokenizer()
    trained_model = load_trained_model()
    latent_dim = 64
    encoder_model, decoder_model = load_inference_models(trained_model, latent_dim)

    with open('max_seq_length.pkl', 'rb') as f:
        max_seq_length = pickle.load(f)

    print("Model, tokenizer ve inference modelleri başarıyla yüklendi.")
except Exception as e:
    print("Model veya tokenizer yüklenirken hata oluştu:", e)


@app.route('/')
def home():
    return "Translation API sunucusu çalışıyor!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        sentence = data.get('text')
        if sentence is None:
            return jsonify({'error': 'Gönderilen veride "text" parametresi eksik!'}), 400

        translation = translate_sentence(sentence, tokenizer, encoder_model, decoder_model, max_seq_length)
        print(translation)
        print(type(translation))
        return jsonify({'translation': translation})
    except Exception as e:
        print("Hata oluştu:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)
