import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    decoder_dense = model.get_layer('output_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs, state_h_dec, state_c_dec]
    )

    return encoder_model, decoder_model


def preprocess_sentence(sentence):
    return 'bos ' + sentence.strip() + ' eos'


def translate_sentence(sentence, tokenizer, encoder_model, decoder_model, max_seq_length, beam_width=1):
    tf.config.run_functions_eagerly(False)

    sentence = preprocess_sentence(sentence)
    input_seq = tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = tokenizer.word_index.get('bos')

    if beam_width > 1:
        translations = [(target_seq, states_value, 0.0)]
        completed_translations = []
    else:
        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
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

    max_iterations = max_seq_length * 2
    for _ in range(max_iterations):
        candidates = []

        for seq, states, score in translations:
            if seq[0, -1] == tokenizer.word_index.get('eos'):
                completed_translations.append((seq, states, score))
                continue

            output_tokens, h, c = decoder_model.predict([seq] + states, verbose=0)

            top_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]

            for idx in top_indices:
                word_probability = output_tokens[0, -1, idx]
                new_seq = np.zeros((1, seq.shape[1] + 1), dtype='int32')
                new_seq[0, :-1] = seq[0]
                new_seq[0, -1] = idx
                new_score = score - np.log(word_probability + 1e-10)
                candidates.append((new_seq, [h, c], new_score))

        translations = sorted(candidates, key=lambda x: x[2])[:beam_width]

        if all(seq[0, -1] == tokenizer.word_index.get('eos') for seq, _, _ in translations):
            break

    best_seq = min(translations + completed_translations, key=lambda x: x[2])[0]
    words = [tokenizer.index_word.get(idx, '') for idx in best_seq[0, 1:]]

    if words and words[-1] == 'eos':
        words = words[:-1]

    return ' '.join(words)
