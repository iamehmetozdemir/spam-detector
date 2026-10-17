"""Eğitilmiş modeli test kümesinde ayrıntılı inceler ve reports/ klasörüne yazar.

- Dile göre karışıklık matrisleri (confusion_matrix.png)
- Kalibrasyon: model "%90 spam" dediğinde gerçekten ~%90'ı spam mi? (calibration.png)
- Öğrenme eğrisi: daha fazla veri işe yarar mı? (learning_curve.png, sadece eğitim verisiyle)
- Yanlış sınıflandırılan mesajlar (errors.md)

Önce `python main.py` çalıştırılmalı.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, brier_score_loss
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

from data import BASE_DIR, SEED, load_data, split

REPORTS = BASE_DIR / "reports"


def load_model():
    features = joblib.load(BASE_DIR / "tfidf_vectorizer_bilingual.pkl")
    clf = joblib.load(BASE_DIR / "spam_model_bilingual.pkl")
    return Pipeline([("features", features), ("clf", clf)])


def confusion_matrices(test, pred):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, lang, title in zip(axes, ["EN", "TR"], ["English e-mails", "Turkish SMS"]):
        mask = (test["lang"] == lang).values
        ConfusionMatrixDisplay.from_predictions(
            test["spam"][mask], pred[mask], display_labels=["not spam", "spam"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{title} (n={mask.sum()})")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def calibration(test, proba):
    frac_pos, mean_pred = calibration_curve(test["spam"], proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("predicted spam probability")
    ax.set_ylabel("actual share of spam")
    ax.set_title("Calibration on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=120)
    plt.close(fig)
    return float(brier_score_loss(test["spam"], proba))


def learning(train, model):
    # Sadece eğitim verisi: her boyut için 5 katlı çapraz doğrulama
    sizes, train_scores, val_scores = learning_curve(
        clone(model), train["text"], train["spam"],
        train_sizes=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for scores, label in [(train_scores, "training"), (val_scores, "cross-validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", label=label)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("training messages")
    ax.set_ylabel("F1 (spam)")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "learning_curve.png", dpi=120)
    plt.close(fig)
    return {int(s): round(float(v), 4) for s, v in zip(sizes, val_scores.mean(axis=1))}


def errors(test, proba, pred, limit=40):
    wrong = test.assign(proba=proba, pred=pred)
    wrong = wrong[wrong["pred"] != wrong["spam"]].sort_values("proba")
    lines = ["# Misclassified test messages", "",
             f"{len(wrong)} of {len(test)} test messages are wrong. Texts are cut to 160 characters.", "",
             "| Language | Actual | Spam probability | Text |", "|---|---|---|---|"]
    for _, r in wrong.head(limit).iterrows():
        text = " ".join(str(r["text"]).split())[:160].replace("|", "/")
        actual = "spam" if r["spam"] == 1 else "not spam"
        lines.append(f"| {r['lang']} | {actual} | {r['proba']:.2f} | {text} |")
    (REPORTS / "errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(wrong)


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_data()
    train, test = split(df)
    model = load_model()

    proba = model.predict_proba(test["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)

    confusion_matrices(test, pred)
    brier = calibration(test, proba)
    curve = learning(train, model)
    n_wrong = errors(test, proba, pred)

    summary = {"brier_score": round(brier, 4), "misclassified": n_wrong, "learning_curve_cv_f1": curve}
    (REPORTS / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
