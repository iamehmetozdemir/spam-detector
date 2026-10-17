from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def predict(text):
    return client.post("/predict", json={"text": text})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_turkish_spam_is_detected():
    body = predict("Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin").json()
    assert body["prediction"] == "Spam"
    assert body["spam_probability"] > 0.9


def test_normal_messages_are_not_spam():
    for text in ["Yarın akşam toplantı kaçta başlıyor?",
                 "Hi John, attached is the agenda for tomorrow's meeting. Let me know if the time works for you."]:
        body = predict(text).json()
        assert body["prediction"] == "Not Spam", text


def test_response_fields_are_consistent():
    body = predict("Congratulations! You won a free prize. Click here to claim now!").json()
    assert set(body) == {"prediction", "email_text", "confidence", "spam_probability"}
    assert 50 <= body["confidence"] <= 100
    assert 0 <= body["spam_probability"] <= 1
    expected = "Spam" if body["spam_probability"] >= 0.5 else "Not Spam"
    assert body["prediction"] == expected


def test_empty_or_missing_text_is_rejected():
    assert predict("").status_code == 422
    assert client.post("/predict", json={}).status_code == 422
