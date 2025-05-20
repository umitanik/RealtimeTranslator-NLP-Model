# Deep Learning Translation Application

Bu proje, İngilizce ve Türkçe dilleri arasında çift yönlü çeviri yapabilen derin öğrenme tabanlı bir NLP uygulamasıdır.
Sequence-to-Sequence (Seq2Seq) modeli kullanarak metinleri bir dilden diğerine çevirir.

## 🌟 Özellikler

- 🔄 İngilizce ↔️ Türkçe çift yönlü çeviri
- 🧠 Derin öğrenme tabanlı çeviri motoru (LSTM Seq2Seq model)
- 🌐 REST API ile entegrasyon imkanı
- 💻 İnteraktif komut satırı arayüzü
- 📊 Eğitim performansı görselleştirme
- 🔍 Otomatik dil algılama

## 📋 Gereksinimler

Projeyi çalıştırmak için aşağıdaki gereksinimlere ihtiyacınız var:

```
tensorflow>=2.6.0
numpy>=1.19.2
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
flask>=2.0.1
flask-cors>=3.0.10
waitress>=2.0.0
absl-py~=2.2.2
```

## 🔧 Kurulum

1. Repoyu klonlayın:
   ```
   git clone [repo-url]
   cd TranslationAppwith_DL
   ```

2. Gerekli paketleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

3. Veri setini `data/` klasörüne yerleştirin:
    - EN_TR_711767.csv
    - EN_TR_1048575.csv

## 🚀 Kullanım

### Model Eğitimi

Modeli eğitmek için aşağıdaki komutu çalıştırın:

```
python main.py
```

Bu komut şunları yapacaktır:

- Veri setini yükler ve hazırlar
- Verileri tokenize eder
- Seq2Seq modelini oluşturur ve eğitir
- Eğitim performansını görselleştirir
- İnteraktif çeviri modunu başlatır

### İnteraktif Çeviri

Model eğitildikten sonra, interaktif çeviri modunu kullanabilirsiniz:

```python
from utils.interactive import interactive_translation

interactive_translation()
```

Bu mod, komut satırı üzerinden metin girmenize ve anında çeviri almanıza olanak tanır.


## 📁 Proje Yapısı

```
TranslationAppwith_DL/
│
├── api/                    # API ile ilgili dosyalar
│   └── app.py              # Flask API uygulaması
│
├── data/                   # Veri setleri
│   ├── EN_TR_711767.csv    # İngilizce-Türkçe çeviri veri seti
│   ├── EN_TR_1048575.csv   # İngilizce-Türkçe çeviri veri seti
│   └── bidirectional_EN_TR_TR_EN.csv  # İşlenmiş çift yönlü veri seti
│
├── models/                 # Model tanımlamaları ve kaydedilen modeller
│   ├── inference.py        # Çıkarım (inference) ile ilgili fonksiyonlar
│   ├── seq2seq_model.py    # Sequence-to-Sequence model tanımlamaları
│   └── saved_model.keras   # Eğitilmiş model dosyası (eğitimden sonra oluşur)
│
├── tokenization/           # Tokenizer ve ilgili dosyalar
│   ├── tokenizer.pkl       # Kaydedilen tokenizer (eğitimden sonra oluşur)
│   └── max_seq_length.pkl  # Maximum sequence uzunluğu (eğitimden sonra oluşur)
│
├── utils/                  # Yardımcı fonksiyonlar
│   ├── config.py           # Konfigürasyon ayarları
│   ├── interactive.py      # İnteraktif çeviri fonksiyonları
│   ├── io.py               # Giriş/çıkış işlemleri
│   ├── prepare.py          # Veri hazırlama fonksiyonları
│   ├── split.py            # Veri bölme fonksiyonları
│   └── visualization.py    # Görselleştirme fonksiyonları
│
├── visualizations/         # Eğitim görselleştirmeleri (eğitimden sonra oluşur)
│
├── main.py                 # Ana uygulama dosyası
└── requirements.txt        # Proje bağımlılıkları
```

## 🔍 Teknik Detaylar

Bu projede kullanılan Seq2Seq modeli şu bileşenlerden oluşmaktadır:

- Embedding katmanı
- LSTM tabanlı encoder
- LSTM tabanlı decoder
- Dense output katmanı

Modelin eğitimi sırasında early stopping ve model checkpoint kullanılarak en iyi performans elde edilmeye çalışılır.

## 📝 Notlar

- Model eğitiminin tamamlanması için veri boyutuna bağlı olarak zaman gerekebilir.
- GPU kullanımı, eğitim süresini önemli ölçüde hızlandırabilir.
- `main.py` dosyasında 1000 örnek kullanılarak hızlı bir eğitim yapılmaktadır. Tam veri seti ile eğitim için bu
  kısıtlamayı kaldırın.

## 🤝 Katkı

Katkılarınızı bekliyoruz! Lütfen Pull Request göndermeden önce değişikliklerinizin proje standartlarına uygun olduğundan
emin olun.