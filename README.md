# Spam Detector (TR + EN)

[English](#english) · [Türkçe](#türkçe)

**Live demo:** https://iamehmetozdemir.github.io/spam-detector/

![demo](docs/screenshot.png)

## English

A spam classifier that works on Turkish and English messages. I made it while learning machine learning. It turns text into TF-IDF features (words, word pairs and character n-grams) and classifies it with Logistic Regression. There's a FastAPI endpoint, a small Streamlit page, and a demo page that runs the same model directly in the browser.

### Run it

```bash
pip install -r requirements.txt
python main.py                  # train (needs emails.csv, see Datasets)
uvicorn api:app --reload        # API on http://127.0.0.1:8000
streamlit run frontend.py       # UI, needs the API running
```

The trained model files are already in the repo, so you can skip the first step and start the API.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin"}'
# {"prediction":"Spam","email_text":"...","confidence":98.49}
```

### How well does it work?

On a 20% hold-out split (2,087 messages) the accuracy is about 99.8%. Per language: Turkish 0.999, English 0.997 (it catches 98.9% of English spam).

My first model was Naive Bayes. It got 93% overall and missed almost half of the English spam, so I switched to Logistic Regression with character n-grams, which fixed that.

Take the numbers with a grain of salt:
- I compared models and picked settings using the same test split, so the real score is probably a bit lower.
- Both datasets are fairly easy, and an English SMS or a Turkish e-mail is something the model has never seen.

### The browser demo

GitHub Pages can't run Python, so `export_web_model.py` saves the vocabulary and weights to `docs/model.json` and `docs/index.html` redoes the vectorizer and the classifier in JavaScript. I checked it against the Python model on 252 messages and the probabilities match to about 0.0000002, with no label changes. Run `python export_web_model.py` after retraining to update it.

### Datasets

- English: [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) (`emails.csv`, ~9 MB, not in the repo, download it and put it next to `main.py`)
- Turkish: [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) (`TurkishSMSCollection.csv`, included)

Check their licenses on Kaggle before reusing. The code here is MIT.

## Türkçe

Türkçe ve İngilizce mesajlarda çalışan bir spam sınıflandırıcı. Makine öğrenmesini öğrenirken yaptım. Metni TF-IDF özelliklerine (kelimeler, kelime çiftleri, karakter n-gram'ları) çevirip Logistic Regression ile sınıflandırıyor. Bir FastAPI uç noktası, küçük bir Streamlit sayfası ve aynı modeli doğrudan tarayıcıda çalıştıran bir demo sayfası var.

### Çalıştırma

```bash
pip install -r requirements.txt
python main.py                  # eğitim (emails.csv lazım, aşağıya bak)
uvicorn api:app --reload        # API: http://127.0.0.1:8000
streamlit run frontend.py       # arayüz, API açık olmalı
```

Eğitilmiş model dosyaları repoda hazır, ilk adımı atlayıp direkt API'yi açabilirsin.

### Ne kadar iyi çalışıyor?

%20'lik test kümesinde (2.087 mesaj) doğruluk yaklaşık %99,8. Dile göre: Türkçe 0,999, İngilizce 0,997 (İngilizce spam'in %98,9'unu yakalıyor).

İlk modelim Naive Bayes'ti. Genelde %93 yaptı ve İngilizce spam'in neredeyse yarısını kaçırdı. Karakter n-gram'lı Logistic Regression'a geçince bu düzeldi.

Rakamlara biraz temkinli yaklaş:
- Modelleri ve ayarları aynı test kümesinde karşılaştırıp seçtim, gerçek skor muhtemelen biraz daha düşük.
- İki veri seti de oldukça kolay. İngilizce bir SMS ya da Türkçe bir e-posta modelin hiç görmediği bir şey.

### Tarayıcı demosu

GitHub Pages Python çalıştıramıyor. O yüzden `export_web_model.py` sözlüğü ve ağırlıkları `docs/model.json`'a kaydediyor, `docs/index.html` de vektörleştiriciyi ve sınıflandırıcıyı JavaScript'te yeniden yapıyor. Python modeliyle 252 mesajda karşılaştırdım, olasılıklar yaklaşık 0,0000002 farkla aynı çıktı, hiçbir etiket değişmedi. Yeniden eğittikten sonra güncellemek için `python export_web_model.py` çalıştır.

### Veri setleri

- İngilizce: [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) (`emails.csv`, ~9 MB, repoda yok, indirip `main.py`'nin yanına koy)
- Türkçe: [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) (`TurkishSMSCollection.csv`, repoda var)

Kullanmadan önce Kaggle'daki lisanslarına bak. Buradaki kod MIT lisanslı.
