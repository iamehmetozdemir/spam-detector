# 📧 Bilingual Spam Detector

[🇬🇧 English](#english) · [🇹🇷 Türkçe](#türkçe)

![Streamlit UI](docs/screenshot.png)

---

## English

A small spam classifier that works on both **Turkish** and **English** text. I merged an English e-mail dataset with a Turkish SMS dataset, trained a TF-IDF + Multinomial Naive Bayes model on it, and wrapped it in a FastAPI endpoint with a simple Streamlit page on top.

It started as a learning project for text classification, so the model is intentionally simple. The interesting part for me was seeing how differently it behaves on the two languages (see [Results](#results)).

### Files

| File | What it does |
|------|--------------|
| `main.py` | Loads both datasets, trains the model, prints metrics, saves the `.pkl` files |
| `api.py` | FastAPI app with `POST /predict` |
| `frontend.py` | Streamlit UI that talks to the API |
| `TurkishSMSCollection.csv` | Turkish SMS data (included) |
| `emails.csv` | English e-mail data (not included, ~9 MB, download it, see [Datasets](#datasets)) |
| `*_bilingual.pkl` | Trained model + vectorizer, so you can run the API without training |

### Setup

```bash
git clone https://github.com/iamehmetozdemir/spam-detector.git
cd spam-detector
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The included `.pkl` files were saved with scikit-learn 1.8.0. If you get a version warning, just retrain.

### Train

Put `emails.csv` in the project folder, then:

```bash
python main.py
```

It downloads the NLTK stop-word lists (English + Turkish), trains on an 80/20 split (`random_state=42`) and overwrites the two `*_bilingual.pkl` files.

### Run the API

```bash
uvicorn api:app --reload
```

Swagger docs are at http://127.0.0.1:8000/docs.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free prize. Click here to claim now!"}'
```

```json
{
  "prediction": "Spam",
  "email_text": "Congratulations! You won a free prize. Click here to claim now!",
  "confidence": 53.58
}
```

`confidence` is the model's probability (in %) for the predicted class. Note the 53 % here: the model leans spam but isn't sure, which ties into the English results below.

### Run the UI

With the API running, open a second terminal:

```bash
streamlit run frontend.py
```

### Results

Hold-out set: 2,096 messages (20 % of the merged data). A message counts as spam when `P(spam) >= threshold`.

**Overall, by threshold**

| Threshold | Accuracy | Spam precision | Spam recall | Spam F1 |
|-----------|----------|----------------|-------------|---------|
| 0.30 | 0.92 | 0.90 | 0.89 | 0.89 |
| 0.50 | 0.93 | 0.98 | 0.83 | 0.90 |
| 0.70 | 0.92 | 0.99 | 0.78 | 0.87 |
| 0.85 | 0.90 | 1.00 | 0.73 | 0.85 |

**By language (threshold 0.5, what the API uses by default)**

| Language | Test messages | Accuracy | Spam precision | Spam recall | Spam F1 |
|----------|---------------|----------|----------------|-------------|---------|
| Turkish (SMS) | 919 | 0.98 | 0.97 | 1.00 | 0.98 |
| English (e-mail) | 1,177 | 0.89 | 1.00 | 0.54 | 0.70 |

### Limitations

- **English spam recall is low (0.54).** The model almost never flags a good English e-mail as spam, but it misses almost half of the real spam. Part of the reason is class imbalance (only ~24 % of the English e-mails are spam) and that e-mails are long and varied. Lowering the threshold would help recall.
- **The two datasets are very different.** Short Turkish SMS are much easier than long English e-mails, so the headline accuracy flatters the model. The per-language table is the more honest number.
- **Domain mismatch.** An English *SMS* or a Turkish *e-mail* is outside what the model was trained on, so expect worse results there.
- Naive Bayes + TF-IDF ignores word order and context and the training data is static, so new scam styles won't be recognised.

### Ideas for later

- Pick the threshold per language, or train one model per language
- Try Logistic Regression / a linear SVM, or a multilingual transformer
- Add more recent data and Turkish e-mails

### Datasets

- **English:** [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) by jackksoncsie on Kaggle, `emails.csv` (5,728 e-mails, 1,368 spam).
- **Turkish:** [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) by onurkarasoy on Kaggle (4,751 messages: 2,536 spam / 2,215 normal).

Both belong to their authors, please check the licenses on the Kaggle pages. The code in this repo is MIT licensed (see [LICENSE](LICENSE)).

---

## Türkçe

Hem **Türkçe** hem **İngilizce** metinlerde çalışan küçük bir spam sınıflandırıcı. İngilizce e-posta veri setiyle Türkçe SMS veri setini birleştirip TF-IDF + Multinomial Naive Bayes modeli eğittim; üstüne bir FastAPI uç noktası ve basit bir Streamlit sayfası ekledim.

Metin sınıflandırmayı öğrenmek için yaptığım bir proje, o yüzden model bilerek basit tutuldu. Benim için en ilginç kısım, modelin iki dilde ne kadar farklı davrandığını görmekti ([Sonuçlar](#sonuçlar)).

### Dosyalar

| Dosya | Ne yapar |
|-------|----------|
| `main.py` | İki veri setini yükler, modeli eğitir, metrikleri yazar, `.pkl` dosyalarını kaydeder |
| `api.py` | `POST /predict` sunan FastAPI uygulaması |
| `frontend.py` | API ile konuşan Streamlit arayüzü |
| `TurkishSMSCollection.csv` | Türkçe SMS verisi (repoda var) |
| `emails.csv` | İngilizce e-posta verisi (repoda yok, ~9 MB, indirin, bkz. [Veri setleri](#veri-setleri)) |
| `*_bilingual.pkl` | Eğitilmiş model + vectorizer; eğitmeden API'yi çalıştırabilirsiniz |

### Kurulum

```bash
git clone https://github.com/iamehmetozdemir/spam-detector.git
cd spam-detector
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Repodaki `.pkl` dosyaları scikit-learn 1.8.0 ile kaydedildi. Sürüm uyarısı alırsanız modeli yeniden eğitin.

### Eğitim

`emails.csv` dosyasını proje klasörüne koyun, sonra:

```bash
python main.py
```

NLTK stop-word listelerini (İngilizce + Türkçe) indirir, %80/%20 bölmeyle (`random_state=42`) eğitir ve iki `*_bilingual.pkl` dosyasının üzerine yazar.

### API'yi çalıştırma

```bash
uvicorn api:app --reload
```

Swagger dokümantasyonu: http://127.0.0.1:8000/docs

Örnek istek:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin"}'
```

```json
{
  "prediction": "Spam",
  "email_text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin",
  "confidence": 99.52
}
```

`confidence`, modelin tahmin ettiği sınıf için olasılığıdır (%).

### Arayüz

API çalışırken ikinci bir terminalde:

```bash
streamlit run frontend.py
```

### Sonuçlar

Test kümesi: 2.096 mesaj (birleşik verinin %20'si). `P(spam) >= eşik` ise mesaj spam sayılır.

**Genel, eşiğe göre**

| Eşik | Doğruluk | Spam precision | Spam recall | Spam F1 |
|------|----------|----------------|-------------|---------|
| 0.30 | 0.92 | 0.90 | 0.89 | 0.89 |
| 0.50 | 0.93 | 0.98 | 0.83 | 0.90 |
| 0.70 | 0.92 | 0.99 | 0.78 | 0.87 |
| 0.85 | 0.90 | 1.00 | 0.73 | 0.85 |

**Dile göre (eşik 0.5, API'nin varsayılanı)**

| Dil | Test mesajı | Doğruluk | Spam precision | Spam recall | Spam F1 |
|-----|-------------|----------|----------------|-------------|---------|
| Türkçe (SMS) | 919 | 0.98 | 0.97 | 1.00 | 0.98 |
| İngilizce (e-posta) | 1.177 | 0.89 | 1.00 | 0.54 | 0.70 |

### Sınırlılıklar

- **İngilizce spam recall'ı düşük (0.54).** Model iyi İngilizce e-postaları neredeyse hiç spam işaretlemiyor ama gerçek spam'in yaklaşık yarısını kaçırıyor. Sebeplerden biri sınıf dengesizliği (İngilizce e-postaların sadece ~%24'ü spam) ve e-postaların uzun/çeşitli olması. Eşiği düşürmek recall'ı artırır.
- **İki veri seti çok farklı.** Kısa Türkçe SMS'ler uzun İngilizce e-postalardan çok daha kolay, bu yüzden genel doğruluk modeli olduğundan iyi gösteriyor. Dile göre tablo daha dürüst.
- **Alan uyuşmazlığı.** İngilizce bir *SMS* ya da Türkçe bir *e-posta* modelin eğitildiği şeyin dışında; orada daha kötü sonuç bekleyin.
- Naive Bayes + TF-IDF kelime sırasını ve bağlamı görmez ve eğitim verisi sabit olduğundan yeni dolandırıcılık tarzlarını tanımaz.

### İleride yapılabilecekler

- Eşiği dile göre seçmek veya dil başına ayrı model eğitmek
- Logistic Regression / lineer SVM ya da çok dilli bir transformer denemek
- Daha yeni veri ve Türkçe e-posta verisi eklemek

### Veri setleri

- **İngilizce:** Kaggle'da jackksoncsie'nin [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) veri seti, `emails.csv` (5.728 e-posta, 1.368 spam).
- **Türkçe:** Kaggle'da onurkarasoy'un [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) veri seti (4.751 mesaj: 2.536 spam / 2.215 normal).

İkisi de kendi sahiplerine aittir, lisansları için Kaggle sayfalarına bakın. Bu repodaki kod MIT lisanslıdır ([LICENSE](LICENSE)).
