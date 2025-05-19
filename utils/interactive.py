import os
import pickle

from models.inference import load_inference_models, translate_sentence
from utils.io import load_trained_model, load_tokenizer


def interactive_translation(model_path=None, tokenizer_path=None, max_seq_path=None):
    """
    Interactive translation function that takes user input and translates it.
    
    Args:
        model_path: Path to the trained model
        tokenizer_path: Path to the tokenizer
        max_seq_path: Path to the max sequence length
    """
    # Get the absolute path of the current workspace
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set default paths using absolute paths
    if model_path is None:
        model_path = os.path.join(workspace_path, "models", "saved_model.keras")
    if tokenizer_path is None:
        tokenizer_path = os.path.join(workspace_path, "tokenization", "tokenizer.pkl")
    if max_seq_path is None:
        max_seq_path = os.path.join(workspace_path, "tokenization", "max_seq_length.pkl")

    print("⏳ Model yükleniyor...")
    print(f"Model yolu: {model_path}")
    print(f"Tokenizer yolu: {tokenizer_path}")
    print(f"Max seq length yolu: {max_seq_path}")

    try:
        # Check if files exist
        for path, name in [(model_path, "Model"), (tokenizer_path, "Tokenizer"), (max_seq_path, "Max sequence length")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ {name} dosyası bulunamadı: {path}")

        # Load model and tokenizer
        model = load_trained_model(model_path)
        tokenizer = load_tokenizer(tokenizer_path)

        # Load max sequence length
        with open(max_seq_path, 'rb') as f:
            max_seq_length = pickle.load(f)

        # Create inference models
        latent_dim = 64  # Eğitimde kullanılan değer
        encoder_model, decoder_model = load_inference_models(model, latent_dim)

        print("\n🔍 İnteraktif Çeviri Modu")
        print("=" * 50)
        print("Çıkmak için 'exit', 'quit' veya 'q' yazın")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n📝 Çevrilecek metin: ")

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Çevirici kapatılıyor...")
                    break

                if not user_input.strip():
                    print("Lütfen çevirmek için bir metin girin.")
                    continue

                translation = translate_sentence(user_input, tokenizer, encoder_model, decoder_model, max_seq_length)
                print(f"🔤 Çeviri: {translation}")

            except KeyboardInterrupt:
                print("\nÇevirici kapatılıyor...")
                break
            except Exception as e:
                print(f"❌ Çeviri sırasında hata: {str(e)}")
                continue

    except FileNotFoundError as e:
        print(e)
        print("Model, tokenizer veya max sequence length dosyalarının mevcut olduğundan emin olun.")
        print("Önce modeli eğitin veya doğru dosya konumlarını sağlayın.")
    except Exception as e:
        print(f"❌ Model yükleme sırasında hata: {str(e)}")
