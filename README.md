# 📧 Bilingual Spam Detector

[🇬🇧 English](#english) · [🇹🇷 Türkçe](#türkçe)

### 🔎 [Try it live in your browser](https://iamehmetozdemir.github.io/spam-detector/) · [Tarayıcıda canlı dene](https://iamehmetozdemir.github.io/spam-detector/)

![Streamlit UI](docs/screenshot.png)

---

## English

A small spam classifier that works on both **Turkish** and **English** text. I merged an English e-mail dataset with a Turkish SMS dataset, trained a TF-IDF + Logistic Regression model on it, and wrapped it in a FastAPI endpoint with a simple Streamlit page on top.

It started as a learning project for text classification, so the model is intentionally simple. I first used Naive Bayes, saw it do badly on English e-mails (54 % spam recall), and switched to a linear model with word + character n-grams, which fixed most of it (see [Results](#results)).

### Files

| File | What it does |
|------|--------------|
| `main.py` | Loads both datasets, trains the model, prints metrics, saves the `.pkl` files |
| `api.py` | FastAPI app with `POST /predict` |
| `frontend.py` | Streamlit UI that talks to the API |
| `docs/` | Live demo for GitHub Pages: the same model running in the browser (`index.html` + `model.json`) |
| `export_web_model.py` | Exports the trained model to `docs/model.json` for the demo |
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

The included `.pkl` files were saved with scikit-learn 1.8.0 (the vectorizer file is ~7 MB because of the character n-grams). If you get a version warning, just retrain.

### Train

Put `emails.csv` in the project folder, then:

```bash
python main.py
```

It downloads the NLTK stop-word lists (English + Turkish), removes duplicate texts, trains on an 80/20 split (`random_state=42`) and overwrites the two `*_bilingual.pkl` files. Training takes well under a minute on a normal laptop, no GPU needed.

The model is Logistic Regression (`C=10`, balanced class weights) on TF-IDF features: word 1–2-grams plus character 2–5-grams. The character n-grams help with Turkish word endings and misspelled spam words.

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
  "confidence": 81.34
}
```

`confidence` is the model's probability (in %) for the predicted class.

### Live demo (no server)

The page in `docs/` runs the model entirely in the browser, so it works on GitHub Pages. `export_web_model.py` writes the trained TF-IDF vocabulary and the logistic regression weights to `docs/model.json`, and `docs/index.html` re-implements the vectorizer and the classifier in JavaScript. I checked it against the Python model on 252 messages (including emoji, the Turkish İ and empty input): the largest difference in probability was about 2·10⁻⁷ and no message changed label. After retraining, run `python export_web_model.py` to refresh the demo.

### Run the UI

With the API running, open a second terminal:

```bash
streamlit run frontend.py
```

### Results

Hold-out set: 2,087 messages (20 % of the merged data after removing 44 duplicate texts; 791 spam). A message counts as spam when `P(spam) >= threshold`.

**Overall, by threshold**

| Threshold | Accuracy | Spam precision | Spam recall | Spam F1 |
|-----------|----------|----------------|-------------|---------|
| 0.30 | 0.997 | 0.993 | 1.000 | 0.996 |
| 0.50 | 0.998 | 0.999 | 0.996 | 0.998 |
| 0.70 | 0.994 | 1.000 | 0.984 | 0.992 |
| 0.85 | 0.985 | 1.000 | 0.961 | 0.980 |

**By language (threshold 0.5, what the API uses by default)**

| Language | Test messages | Accuracy | Spam precision | Spam recall | Spam F1 |
|----------|---------------|----------|----------------|-------------|---------|
| Turkish (SMS) | 947 | 0.999 | 0.998 | 1.000 | 0.999 |
| English (e-mail) | 1,140 | 0.997 | 1.000 | 0.989 | 0.995 |

**Naive Bayes vs. Logistic Regression** (my first model vs. the current one, same split idea, before/after deduplication)

| | Overall accuracy | English spam recall |
|---|---|---|
| TF-IDF + Naive Bayes (first version) | 0.93 | 0.54 |
| TF-IDF (word + char n-grams) + Logistic Regression | 0.998 | 0.989 |

### Limitations

- **The scores are probably too good to be true in practice.** ~99.8 % on a random split mostly says these datasets are easy and have recognisable patterns (for example the English e-mails come from one source and have a very particular style). Don't expect that number on messages from somewhere else.
- **Domain mismatch.** An English *SMS* or a Turkish *e-mail* is outside what the model was trained on, so expect worse results there. I haven't measured it.
- **Confidence is not always high.** The English example above only gets 81 %, so treat `confidence` as a rough signal.
- TF-IDF ignores word meaning and the training data is static, so new scam styles won't be recognised.

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

Hem **Türkçe** hem **İngilizce** metinlerde çalışan küçük bir spam sınıflandırıcı. İngilizce e-posta veri setiyle Türkçe SMS veri setini birleştirip TF-IDF + Logistic Regression modeli eğittim; üstüne bir FastAPI uç noktası ve basit bir Streamlit sayfası ekledim.

Metin sınıflandırmayı öğrenmek için yaptığım bir proje, o yüzden model bilerek basit tutuldu. İlk önce Naive Bayes kullandım, İngilizce e-postalarda kötü çıktı (%54 spam recall); kelime + karakter n-gram'lı lineer modele geçince sorunun çoğu çözüldü ([Sonuçlar](#sonuçlar)).

### Dosyalar

| Dosya | Ne yapar |
|-------|----------|
| `main.py` | İki veri setini yükler, modeli eğitir, metrikleri yazar, `.pkl` dosyalarını kaydeder |
| `api.py` | `POST /predict` sunan FastAPI uygulaması |
| `frontend.py` | API ile konuşan Streamlit arayüzü |
| `docs/` | GitHub Pages canlı demosu: aynı model tarayıcıda çalışır (`index.html` + `model.json`) |
| `export_web_model.py` | Eğitilmiş modeli demo için `docs/model.json` olarak dışa aktarır |
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

Repodaki `.pkl` dosyaları scikit-learn 1.8.0 ile kaydedildi (vectorizer dosyası karakter n-gram'ları yüzünden ~7 MB). Sürüm uyarısı alırsanız modeli yeniden eğitin.

### Eğitim

`emails.csv` dosyasını proje klasörüne koyun, sonra:

```bash
python main.py
```

NLTK stop-word listelerini (İngilizce + Türkçe) indirir, tekrar eden metinleri atar, %80/%20 bölmeyle (`random_state=42`) eğitir ve iki `*_bilingual.pkl` dosyasının üzerine yazar. Eğitim normal bir laptopta bir dakikadan kısa sürer, GPU gerekmez.

Model: Logistic Regression (`C=10`, dengeli sınıf ağırlıkları), TF-IDF özellikleriyle: kelime 1–2'li + karakter 2–5'li n-gram'lar. Karakter n-gram'ları Türkçe ekleri ve yazım hatalı spam kelimelerini yakalamaya yardım ediyor.

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
  "confidence": 98.49
}
```

`confidence`, modelin tahmin ettiği sınıf için olasılığıdır (%).

### Canlı demo (sunucusuz)

`docs/` içindeki sayfa modeli tamamen tarayıcıda çalıştırır, bu yüzden GitHub Pages'te çalışır. `export_web_model.py` eğitilmiş TF-IDF sözlüğünü ve logistic regression ağırlıklarını `docs/model.json`'a yazar; `docs/index.html` vektörleştiriciyi ve sınıflandırıcıyı JavaScript'te yeniden yazar. Python modeliyle 252 mesajda karşılaştırdım (emoji, Türkçe İ ve boş metin dahil): olasılıktaki en büyük fark yaklaşık 2·10⁻⁷, hiçbir mesajın etiketi değişmedi. Yeniden eğitimden sonra demoyu yenilemek için `python export_web_model.py` çalıştır.

### Arayüz

API çalışırken ikinci bir terminalde:

```bash
streamlit run frontend.py
```

### Sonuçlar

Test kümesi: 2.087 mesaj (44 tekrar eden metin atıldıktan sonra birleşik verinin %20'si; 791 spam). `P(spam) >= eşik` ise mesaj spam sayılır.

**Genel, eşiğe göre**

| Eşik | Doğruluk | Spam precision | Spam recall | Spam F1 |
|------|----------|----------------|-------------|---------|
| 0.30 | 0.997 | 0.993 | 1.000 | 0.996 |
| 0.50 | 0.998 | 0.999 | 0.996 | 0.998 |
| 0.70 | 0.994 | 1.000 | 0.984 | 0.992 |
| 0.85 | 0.985 | 1.000 | 0.961 | 0.980 |

**Dile göre (eşik 0.5, API'nin varsayılanı)**

| Dil | Test mesajı | Doğruluk | Spam precision | Spam recall | Spam F1 |
|-----|-------------|----------|----------------|-------------|---------|
| Türkçe (SMS) | 947 | 0.999 | 0.998 | 1.000 | 0.999 |
| İngilizce (e-posta) | 1.140 | 0.997 | 1.000 | 0.989 | 0.995 |

**Naive Bayes ve Logistic Regression** (ilk modelim ve şimdiki; tekrar temizliğinden önce/sonra)

| | Genel doğruluk | İngilizce spam recall |
|---|---|---|
| TF-IDF + Naive Bayes (ilk sürüm) | 0.93 | 0.54 |
| TF-IDF (kelime + karakter n-gram) + Logistic Regression | 0.998 | 0.989 |

### Sınırlılıklar

- **Skorlar pratikte muhtemelen bu kadar iyi olmaz.** Rastgele bölmede ~%99,8, bu veri setlerinin kolay ve kalıpları belli olduğunu gösteriyor (mesela İngilizce e-postalar tek bir kaynaktan ve çok belirgin bir tarzda). Başka kaynaktan gelen mesajlarda bu skoru beklemeyin.
- **Alan uyuşmazlığı.** İngilizce bir *SMS* ya da Türkçe bir *e-posta* modelin eğitildiği şeyin dışında; orada daha kötü sonuç bekleyin. Bunu ölçmedim.
- **Güven skoru her zaman yüksek değil.** Yukarıdaki İngilizce örnek sadece %81 alıyor, `confidence`'ı kaba bir sinyal olarak görün.
- TF-IDF kelimelerin anlamını görmez ve eğitim verisi sabit olduğundan yeni dolandırıcılık tarzlarını tanımaz.

### İleride yapılabilecekler

- Gerçekte ne kadar genelleştiğini görmek için başka kaynaktan veriyle test etmek
- Çok dilli bir transformer deneyip karşılaştırmak
- Daha yeni veri ve Türkçe e-posta verisi eklemek

### Veri setleri

- **İngilizce:** Kaggle'da jackksoncsie'nin [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) veri seti, `emails.csv` (5.728 e-posta, 1.368 spam).
- **Türkçe:** Kaggle'da onurkarasoy'un [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) veri seti (4.751 mesaj: 2.536 spam / 2.215 normal).

İkisi de kendi sahiplerine aittir, lisansları için Kaggle sayfalarına bakın. Bu repodaki kod MIT lisanslıdır ([LICENSE](LICENSE)).
