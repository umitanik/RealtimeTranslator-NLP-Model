import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
import tensorflow as tf
import warnings


def set_environment():
    absl.logging.set_verbosity(absl.logging.ERROR)
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            print(f"✅ GPU bulundu: {gpus}")
        except RuntimeError as e:
            print(f"❌ Hata: {e}")
    else:
        print("⚠️ GPU bulunamadı! TensorFlow ve CUDA'yı kontrol et.")
