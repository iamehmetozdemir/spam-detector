import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.corpus import stopwords
import numpy as np

# 1. NLTK Stop words (İngilizce ve Türkçe birleşimi)
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
stop_words_tr = set(stopwords.words('turkish'))
combined_stop_words = list(stop_words_en.union(stop_words_tr))

# 2. İngilizce Dataset (Kaggle emails.csv)
df_en = pd.read_csv("emails.csv")
df_en = df_en[['text', 'spam']]

# 3. Türkçe Dataset (TurkishSMSCollection.csv)
df_tr = pd.read_csv("TurkishSMSCollection.csv", sep=";")
df_tr.rename(columns={'Message': 'text', 'GroupText': 'spam'}, inplace=True)
df_tr['spam'] = df_tr['spam'].map({'Spam': 1, 'Normal': 0})
df_tr = df_tr[['text', 'spam']]

# 4. Data Concatenation
df_combined = pd.concat([df_en, df_tr], ignore_index=True)
df_combined.dropna(inplace=True)

# 5. Feature Extraction & Train/Test Split
X = df_combined['text']
y = df_combined['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words=combined_stop_words)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Confidence Score (Predict Proba) ve Threshold Metrikleri
y_proba = model.predict_proba(X_test_tfidf)[:, 1]
thresholds = [0.3, 0.5, 0.7, 0.85]

print("Bilingual Model Başarısı:")
for t in thresholds:
    print(f"\n==========================================")
    print(f"       THRESHOLD (Confidence Sınırı): {t}")
    print(f"==========================================")
    y_custom_pred = (y_proba >= t).astype(int)
    print(classification_report(y_test, y_custom_pred))

# 8. Modeli Diske Kaydet
joblib.dump(model, 'spam_model_bilingual.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_bilingual.pkl')
print("\n✅ İki dilli model (Bilingual) ve Vectorizer başarıyla diske kaydedildi.")