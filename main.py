from utils.config import *
from utils.io import *
from utils.prepare import *
from models.seq2seq_model import *
from utils.interactive import *
from utils.visualization import *

set_environment()


if __name__ == "__main__":
    data = load_data()
    data = data.head(1000)
    data = prepare_data(data)
    tokenizer, max_seq_length, vocab_size, enc_in, dec_in, dec_out = tokenize_data(data["Input"], data["Target"])

    print("⏳ Model oluşturuluyor ve eğitiliyor...")
    model = build_seq2seq_model(num_tokens=vocab_size, latent_dim=64)
    history = train_seq2seq_model(model, enc_in, dec_in, dec_out, batch_size=64, epochs=5)

    plot_training_history(
        history,
        metrics=['loss', 'accuracy'],
        save_path='visualizations/training_history.png',
        show=True
        )

    interactive_translation()




