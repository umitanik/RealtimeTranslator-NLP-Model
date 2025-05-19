import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model


def build_seq2seq_model(num_tokens, latent_dim=256):
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(input_dim=num_tokens, output_dim=latent_dim, name="encoder_embedding")(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = Embedding(input_dim=num_tokens, output_dim=latent_dim, name="decoder_embedding")(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(num_tokens, activation="softmax", name="output_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_seq2seq_model(model, encoder_input_data, decoder_input_data, decoder_target_data,
                        batch_size=256, epochs=2, model_path="models/saved_model.keras"):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    print(f"✅ Eğitim tamamlandı. En iyi model '{model_path}' olarak kaydedildi.")
    return history
