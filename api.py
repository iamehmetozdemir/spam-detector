from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Email Detection API")

# Model ve Vectorizer'ı yüklüyoruz
model = joblib.load('spam_model_bilingual.pkl')
vectorizer = joblib.load('tfidf_vectorizer_bilingual.pkl')

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(request: EmailRequest):
    text_tfidf = vectorizer.transform([request.text])
    
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    confidence = max(probabilities) * 100
    
    # Etiketleri İngilizce yapıyoruz
    label = "Spam" if prediction == 1 else "Not Spam"
    
    return {
        "prediction": label,
        "email_text": request.text,
        "confidence": round(confidence, 2)
    }