"""Özellik çıkarımı ve denenecek modeller."""
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline


def stop_words():
    """İngilizce + Türkçe NLTK stop word listesi."""
    nltk.download("stopwords", quiet=True)
    return sorted(set(stopwords.words("english")) | set(stopwords.words("turkish")))


def char_vectorizer():
    # Karakter n-gram'ları Türkçe ekler ve "kazandiniz" gibi yazım hatalı spam kelimeleri için işe yarıyor
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)


def make_pipeline(stops, memory=None):
    features = FeatureUnion([
        ("word", TfidfVectorizer(stop_words=stops, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", char_vectorizer()),
    ])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", clf)], memory=memory)


def param_grid():
    """Çapraz doğrulamada karşılaştırılan modeller: karakter n-gram'lı/sız, Naive Bayes ve Logistic Regression."""
    char_options = ["drop", char_vectorizer()]
    return [
        {
            "features__char": char_options,
            "clf": [MultinomialNB()],
            "clf__alpha": [0.01, 0.1, 1.0],
        },
        {
            "features__char": char_options,
            "clf": [LogisticRegression(class_weight="balanced", max_iter=3000)],
            "clf__C": [1, 10, 100],
        },
    ]


def describe(params):
    """Bir parametre kombinasyonunu tablo için kısa yazıya çevirir."""
    clf = params["clf"]
    name = "Naive Bayes" if isinstance(clf, MultinomialNB) else "Logistic Regression"
    setting = f"alpha={params['clf__alpha']}" if "clf__alpha" in params else f"C={params['clf__C']}"
    features = "words" if params["features__char"] == "drop" else "words + chars"
    return name, features, setting
