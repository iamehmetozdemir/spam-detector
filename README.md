# Bilingual Spam Detector / İki Dilli Spam Dedektörü

[🇬🇧 English](#english) | [🇹🇷 Türkçe](#türkçe)

---

## English

A Turkish + English spam classifier. TF-IDF features and a Multinomial Naive Bayes model are trained on a merged English e-mail and Turkish SMS dataset, served through a FastAPI endpoint and a Streamlit UI.

### Project structure

| File | Purpose |
|------|---------|
| `main.py` | Loads both datasets, trains the model, prints metrics for several confidence thresholds, saves `spam_model_bilingual.pkl` and `tfidf_vectorizer_bilingual.pkl` |
| `api.py` | FastAPI app exposing `POST /predict` |
| `frontend.py` | Streamlit UI that calls the API at `http://127.0.0.1:8000/predict` |
| `TurkishSMSCollection.csv` | Turkish SMS dataset |
| `emails.csv` | English e-mail dataset (not in the repo, see below) |

### Installation

```bash
git clone https://github.com/iamehmetozdemir/spam-detector.git
cd spam-detector
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Download `emails.csv` (see [Datasets](#datasets)) and place it in the project root.

### Training

```bash
python main.py
```

This downloads the NLTK stop-word lists (English + Turkish), trains the model on an 80/20 split (`random_state=42`) and writes the two `*_bilingual.pkl` files. The trained `*_bilingual.pkl` files are committed, so you can skip training and run the API directly.

### Running the API

```bash
uvicorn api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Example `/predict` request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free prize. Click here to claim now!"}'
```

Response (actual output):

```json
{
  "prediction": "Spam",
  "email_text": "Congratulations! You won a free prize. Click here to claim now!",
  "confidence": 53.58
}
```

Turkish example (actual output):

```json
{
  "prediction": "Spam",
  "email_text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin",
  "confidence": 99.52
}
```

`confidence` is the probability (in %) of the predicted class.

### Streamlit UI

With the API running, in a second terminal:

```bash
streamlit run frontend.py
```

### Test results

Output of `python main.py` on the 20 % hold-out split (2,096 messages: 1,336 not spam, 760 spam). Messages are classified as spam when `P(spam) >= threshold`.

| Threshold | Accuracy | Spam precision | Spam recall | Spam F1 |
|-----------|----------|----------------|-------------|---------|
| 0.30 | 0.92 | 0.90 | 0.89 | 0.89 |
| 0.50 | 0.93 | 0.98 | 0.83 | 0.90 |
| 0.70 | 0.92 | 0.99 | 0.78 | 0.87 |
| 0.85 | 0.90 | 1.00 | 0.73 | 0.85 |

Higher thresholds reduce false positives (better precision) at the cost of missing more spam (lower recall). Note that `api.py` uses the model's default decision (equivalent to threshold 0.5).

### Datasets

- **English:** [Spam email classification dataset `emails.csv`](https://www.kaggle.com/datasets) from Kaggle (5,728 e-mails, columns `text`, `spam`; 1,368 spam). It is about 9 MB and therefore excluded via `.gitignore`.
- **Turkish:** `TurkishSMSCollection.csv` – Turkish SMS Collection (4,751 messages, `;`-separated, columns `Message`, `Group`, `GroupText`; 2,536 spam / 2,215 normal).

Please check each dataset's original license before redistributing.

---

## Türkçe

Türkçe + İngilizce çalışan bir spam sınıflandırıcı. İngilizce e-posta ve Türkçe SMS veri setleri birleştirilerek TF-IDF + Multinomial Naive Bayes modeli eğitilir; model bir FastAPI uç noktası ve Streamlit arayüzü ile sunulur.

### Proje yapısı

| Dosya | Açıklama |
|-------|----------|
| `main.py` | İki veri setini yükler, modeli eğitir, farklı güven eşikleri için metrikleri yazdırır, `spam_model_bilingual.pkl` ve `tfidf_vectorizer_bilingual.pkl` dosyalarını kaydeder |
| `api.py` | `POST /predict` uç noktasını sunan FastAPI uygulaması |
| `frontend.py` | `http://127.0.0.1:8000/predict` adresini çağıran Streamlit arayüzü |
| `TurkishSMSCollection.csv` | Türkçe SMS veri seti |
| `emails.csv` | İngilizce e-posta veri seti (repoda yok, aşağıya bakın) |

### Kurulum

```bash
git clone https://github.com/iamehmetozdemir/spam-detector.git
cd spam-detector
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`emails.csv` dosyasını indirip (bkz. [Veri setleri](#veri-setleri)) proje köküne koyun.

### Eğitim

```bash
python main.py
```

NLTK stop-word listelerini (İngilizce + Türkçe) indirir, modeli %80/%20 bölmeyle (`random_state=42`) eğitir ve iki `*_bilingual.pkl` dosyasını yazar. Eğitilmiş `*_bilingual.pkl` dosyaları repoda hazır olduğundan eğitimi atlayıp doğrudan API'yi çalıştırabilirsiniz.

### API'yi çalıştırma

```bash
uvicorn api:app --reload
```

Etkileşimli dokümantasyon: http://127.0.0.1:8000/docs

### Örnek `/predict` isteği

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin"}'
```

Yanıt (gerçek çıktı):

```json
{
  "prediction": "Spam",
  "email_text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin",
  "confidence": 99.52
}
```

`confidence`, tahmin edilen sınıfın olasılığıdır (%).

### Streamlit arayüzü

API çalışırken ikinci bir terminalde:

```bash
streamlit run frontend.py
```

### Test sonuçları

`python main.py` çıktısı, %20'lik test kümesi (2.096 mesaj: 1.336 spam değil, 760 spam). `P(spam) >= eşik` olduğunda mesaj spam sayılır.

| Eşik | Doğruluk (accuracy) | Spam precision | Spam recall | Spam F1 |
|------|---------------------|----------------|-------------|---------|
| 0.30 | 0.92 | 0.90 | 0.89 | 0.89 |
| 0.50 | 0.93 | 0.98 | 0.83 | 0.90 |
| 0.70 | 0.92 | 0.99 | 0.78 | 0.87 |
| 0.85 | 0.90 | 1.00 | 0.73 | 0.85 |

Eşik yükseldikçe yanlış alarm azalır (precision artar) ama kaçan spam artar (recall düşer). `api.py` modelin varsayılan kararını kullanır (0.5 eşiğine denk).

### Veri setleri

- **İngilizce:** Kaggle'daki [`emails.csv` spam e-posta veri seti](https://www.kaggle.com/datasets) (5.728 e-posta, `text` ve `spam` sütunları; 1.368 spam). Yaklaşık 9 MB olduğu için `.gitignore` ile repo dışında bırakılmıştır.
- **Türkçe:** `TurkishSMSCollection.csv` – Turkish SMS Collection (4.751 mesaj, `;` ayraçlı, `Message`, `Group`, `GroupText` sütunları; 2.536 spam / 2.215 normal).

Dağıtmadan önce her veri setinin özgün lisansını kontrol edin.
