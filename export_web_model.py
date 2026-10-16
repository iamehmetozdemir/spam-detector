"""Exports the trained model to docs/model.json for the in-browser demo (docs/index.html).

The demo re-implements the TF-IDF vectorizer and the logistic regression in JavaScript,
so it runs on GitHub Pages without a server. Run `python main.py` first, then:

    python export_web_model.py
"""
import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).parent
OUT = BASE_DIR / "docs" / "model.json"


def export_vectorizer(vectorizer):
    # Terms ordered by their feature index, so they line up with the model coefficients.
    terms = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms[index] = term
    return {
        "terms": terms,
        "idf": [round(float(x), 6) for x in vectorizer.idf_],
    }


def main():
    union = joblib.load(BASE_DIR / "tfidf_vectorizer_bilingual.pkl")
    model = joblib.load(BASE_DIR / "spam_model_bilingual.pkl")
    parts = dict(union.transformer_list)
    word, char = parts["word"], parts["char"]

    n_word = len(word.vocabulary_)
    coef = [round(float(x), 6) for x in model.coef_[0]]

    data = {
        "stop_words": sorted(word.stop_words),
        "word": export_vectorizer(word),
        "char": export_vectorizer(char),
        # Coefficients: first the word features, then the char features (FeatureUnion order).
        "word_coef": coef[:n_word],
        "char_coef": coef[n_word:],
        "intercept": round(float(model.intercept_[0]), 6),
    }

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
