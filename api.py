from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Spam Email Detection API")

BASE_DIR = Path(__file__).parent

# Model ve vectorizer'ı yüklüyoruz
try:
    model = joblib.load(BASE_DIR / 'spam_model_bilingual.pkl')
    vectorizer = joblib.load(BASE_DIR / 'tfidf_vectorizer_bilingual.pkl')
except FileNotFoundError as e:
    raise SystemExit(f"Model dosyası bulunamadı ({e.filename}). Önce `python main.py` ile modeli eğitin.")


class EmailRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_spam(request: EmailRequest):
    text_tfidf = vectorizer.transform([request.text])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = max(probabilities) * 100

    label = "Spam" if prediction == 1 else "Not Spam"

    return {
        "prediction": label,
        "email_text": request.text,
        "confidence": round(confidence, 2)
    }
