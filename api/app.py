import logging
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

from models.inference import load_inference_models, translate_sentence
from utils.io import load_trained_model, load_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models and tokenizer
try:
    # Load the tokenizer
    tokenizer = load_tokenizer(filepath='tokenization/tokenizer.pkl')

    # Load the trained model
    trained_model = load_trained_model(filepath='models/saved_model.keras')

    # Load the max sequence length
    with open('tokenization/max_seq_length.pkl', 'rb') as f:
        max_seq_length = pickle.load(f)

    # Create inference models
    latent_dim = 64  # Must match the value used during training
    encoder_model, decoder_model = load_inference_models(trained_model, latent_dim)

    logger.info("✅ Model, tokenizer, and inference models successfully loaded.")
except Exception as e:
    logger.error(f"❌ Error loading model or tokenizer: {e}")
    raise


@app.route('/')
def home():
    return jsonify({"status": "Translation API server is running!"})


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json(force=True)

        if 'text' not in data:
            return jsonify({'error': 'Missing "text" parameter in request!'}), 400

        text = data['text']
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'auto')

        # Log the request
        logger.info(f"Translation request: {len(text)} chars from {source_lang} to {target_lang}")

        # Simple language detection if auto is specified
        if source_lang == 'auto' or target_lang == 'auto':
            # This is a very simplistic approach - you would want a proper language detector
            turkish_chars = set('ğüşöçıİĞÜŞÖÇ')
            has_turkish = any(c in turkish_chars for c in text.lower())

            if source_lang == 'auto':
                source_lang = 'tr' if has_turkish else 'en'

            if target_lang == 'auto':
                target_lang = 'en' if source_lang == 'tr' else 'tr'

        # Perform translation
        translation = translate_sentence(text, tokenizer, encoder_model, decoder_model, max_seq_length)

        return jsonify({
            'original': text,
            'translation': translation,
            'source_lang': source_lang,
            'target_lang': target_lang
        })

    except Exception as error:
        logger.error(f"Error during translation API call: {str(error)}")
        return jsonify({'error': str(error)}), 500


if __name__ == '__main__':
    # Use Waitress or Gunicorn for production
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
