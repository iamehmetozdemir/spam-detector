"""Exports the trained model to docs/model.json for the in-browser demo (docs/index.html).

The demo re-implements the TF-IDF vectorizer and the linear classifier in JavaScript
(docs/spam-model.js), so it runs on GitHub Pages without a server. Run `python main.py` first, then:

    python export_web_model.py
"""
import json
from pathlib import Path

import joblib
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).parent
OUT = BASE_DIR / "docs" / "model.json"


def export_vectorizer(vectorizer):
    # Terms ordered by their feature index, so they line up with the model weights.
    terms = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms[index] = term
    return {
        "terms": terms,
        "idf": [round(float(x), 6) for x in vectorizer.idf_],
        "ngram_range": list(vectorizer.ngram_range),
    }


def linear_weights(clf):
    """Weights and bias so that P(spam) = sigmoid(bias + weights . x)."""
    if isinstance(clf, MultinomialNB):
        # Naive Bayes is linear in log space: log-odds = log prior ratio + x . (log p_spam - log p_ham)
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        bias = clf.class_log_prior_[1] - clf.class_log_prior_[0]
    else:
        weights = clf.coef_[0]
        bias = clf.intercept_[0]
    return [round(float(w), 6) for w in weights], round(float(bias), 6)


def build(union, clf):
    parts = dict(union.transformer_list)
    word, char = parts["word"], parts.get("char")
    has_char = char is not None and char != "drop"

    weights, bias = linear_weights(clf)
    n_word = len(word.vocabulary_)
    return {
        "stop_words": sorted(word.stop_words or []),
        "word": export_vectorizer(word),
        "char": export_vectorizer(char) if has_char else None,
        # Weights: first the word features, then the char features (FeatureUnion order).
        "word_coef": weights[:n_word],
        "char_coef": weights[n_word:] if has_char else [],
        "intercept": bias,
    }


def main(out=OUT):
    union = joblib.load(BASE_DIR / "tfidf_vectorizer_bilingual.pkl")
    clf = joblib.load(BASE_DIR / "spam_model_bilingual.pkl")
    data = build(union, clf)

    out = Path(out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {out} ({out.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
