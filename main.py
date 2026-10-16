from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent

# 1. NLTK stop words (İngilizce + Türkçe birleşimi)
nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
stop_words_tr = set(stopwords.words('turkish'))
combined_stop_words = list(stop_words_en.union(stop_words_tr))

# 2. İngilizce dataset (Kaggle emails.csv)
emails_path = BASE_DIR / "emails.csv"
if not emails_path.exists():
    raise SystemExit("emails.csv bulunamadı. README'deki Kaggle linkinden indirip proje köküne koyun.")
df_en = pd.read_csv(emails_path)[['text', 'spam']]
df_en['lang'] = 'EN'

# 3. Türkçe dataset (TurkishSMSCollection.csv)
df_tr = pd.read_csv(BASE_DIR / "TurkishSMSCollection.csv", sep=";")
df_tr.rename(columns={'Message': 'text', 'GroupText': 'spam'}, inplace=True)
df_tr['spam'] = df_tr['spam'].map({'Spam': 1, 'Normal': 0})
df_tr = df_tr[['text', 'spam']]
df_tr['lang'] = 'TR'

# 4. Birleştirme
df_combined = pd.concat([df_en, df_tr], ignore_index=True)
df_combined.dropna(inplace=True)
# Birebir aynı metinler train/test'e bölünürse sonuç şişer, o yüzden tekrarları atıyoruz
df_combined.drop_duplicates(subset='text', inplace=True)

# 5. Feature extraction & train/test split
X = df_combined['text']
y = df_combined['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kelime (1-2'li) + karakter (2-5'li) n-gram TF-IDF. Karakter n-gram'ları Türkçe gibi
# ek alan dillerde ve yazım hatalı spam metinlerinde (ör. "kazandiniz") işe yarıyor.
vectorizer = FeatureUnion([
    ('word', TfidfVectorizer(stop_words=combined_stop_words, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)),
])
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Model eğitimi
model = LogisticRegression(C=10, class_weight='balanced', max_iter=2000)
model.fit(X_train_tfidf, y_train)

# 7. Confidence score (predict_proba) ve threshold metrikleri
y_proba = model.predict_proba(X_test_tfidf)[:, 1]
thresholds = [0.3, 0.5, 0.7, 0.85]

print("Bilingual Model Başarısı:")
for t in thresholds:
    print(f"\n==========================================")
    print(f"       THRESHOLD (Confidence Sınırı): {t}")
    print(f"==========================================")
    y_custom_pred = (y_proba >= t).astype(int)
    print(classification_report(y_test, y_custom_pred))

# 8. Dile göre ayrı sonuçlar (threshold 0.5)
print("\n==========================================")
print("       DİLE GÖRE SONUÇLAR (threshold 0.5)")
print("==========================================")
y_pred = (y_proba >= 0.5).astype(int)
test_langs = df_combined.loc[X_test.index, 'lang'].values
for lang in ['EN', 'TR']:
    mask = test_langs == lang
    print(f"\n--- {lang} ({mask.sum()} test mesajı) ---")
    print(classification_report(y_test[mask], y_pred[mask]))

# 9. Modeli diske kaydet
joblib.dump(model, BASE_DIR / 'spam_model_bilingual.pkl')
joblib.dump(vectorizer, BASE_DIR / 'tfidf_vectorizer_bilingual.pkl')
print("\n✅ İki dilli model (Bilingual) ve Vectorizer başarıyla diske kaydedildi.")
