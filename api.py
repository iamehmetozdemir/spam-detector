from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Spam Email Detection API")

BASE_DIR = Path(__file__).parent

# Model ve vectorizer'ı yüklüyoruz
try:
    model = joblib.load(BASE_DIR / 'spam_model_bilingual.pkl')
    vectorizer = joblib.load(BASE_DIR / 'tfidf_vectorizer_bilingual.pkl')
except FileNotFoundError as e:
    raise SystemExit(f"Model dosyası bulunamadı ({e.filename}). Önce `python main.py` ile modeli eğitin.")


class EmailRequest(BaseModel):
    # Boş metin anlamsız, çok uzun metin de gereksiz yük
    text: str = Field(min_length=1, max_length=20000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_spam(request: EmailRequest):
    text_tfidf = vectorizer.transform([request.text])

    spam_probability = float(model.predict_proba(text_tfidf)[0][1])
    is_spam = spam_probability >= 0.5

    return {
        "prediction": "Spam" if is_spam else "Not Spam",
        "email_text": request.text,
        # Tahmin edilen sınıfın olasılığı (%)
        "confidence": round(max(spam_probability, 1 - spam_probability) * 100, 2),
        "spam_probability": round(spam_probability, 4),
    }
