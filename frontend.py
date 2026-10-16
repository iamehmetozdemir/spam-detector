import streamlit as st
import requests

# Sayfa başlığı ve tasarımı
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Bilingual Spam Email Detector")
st.markdown("Analyze your emails or messages to detect whether they are Spam or Safe using Machine Learning.")

# Metin giriş alanı
email_text = st.text_area("Enter email or message content here:", placeholder="Type or paste your message here...")

# Analiz Butonu
if st.button("Analyze Message"):
    if email_text.strip():
        # FastAPI'ye istek atıyoruz
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json={"text": email_text}, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            prediction = data["prediction"]
            confidence = data["confidence"]
            
            st.markdown("---")
            st.subheader("Analysis Results")
            
            # Sonuca göre İngilizce şık uyarılar ve Confidence skoru
            if prediction == "Spam":
                st.error(f"🚨 **SPAM DETECTED!** (Confidence Score: %{confidence})")
            else:
                st.success(f"✅ **SAFE MESSAGE / NOT SPAM** (Confidence Score: %{confidence})")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to the API. Make sure FastAPI (Uvicorn) is running in the background.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ The API returned an error: {e}")
    else:
            st.warning("Please enter some text before analyzing.")