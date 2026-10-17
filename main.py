"""Modeli eğitir.

1. Veriyi %80 eğitim / %20 test olarak böler (dil ve etikete göre katmanlı).
2. Sadece eğitim kısmında 5 katlı çapraz doğrulama ile modelleri ve ayarlarını karşılaştırır.
3. En iyisini tüm eğitim verisiyle eğitip test kümesinde bir kez değerlendirir.
4. Modeli ve sonuçları kaydeder (reports/).
"""
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import BASE_DIR, SEED, load_data, split
from model import describe, make_pipeline, param_grid, stop_words

REPORTS = BASE_DIR / "reports"


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n": int(len(y_true)),
    }


def main():
    df = load_data()
    train, test = split(df)
    print(f"{len(df)} mesaj: {len(train)} eğitim, {len(test)} test")

    # 1) Model seçimi: test kümesine hiç bakmadan, eğitim kısmında çapraz doğrulama
    cache = tempfile.mkdtemp()  # vektörleştirici her katta bir kez fit edilsin diye
    try:
        grid = GridSearchCV(
            make_pipeline(stop_words(), memory=cache),
            param_grid(),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring={"f1": "f1", "precision": "precision", "recall": "recall", "accuracy": "accuracy"},
            refit="f1",
            n_jobs=-1,
        )
        grid.fit(train["text"], train["spam"])
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    rows = []
    for i, params in enumerate(grid.cv_results_["params"]):
        name, features, setting = describe(params)
        rows.append({
            "model": name, "features": features, "setting": setting,
            "cv_f1": grid.cv_results_["mean_test_f1"][i],
            "cv_f1_std": grid.cv_results_["std_test_f1"][i],
            "cv_precision": grid.cv_results_["mean_test_precision"][i],
            "cv_recall": grid.cv_results_["mean_test_recall"][i],
        })
    table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    print("\nÇapraz doğrulama (5 kat, sadece eğitim verisi):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name, best_features, best_setting = describe(grid.best_params_)
    print(f"\nSeçilen model: {best_name}, {best_features}, {best_setting}")

    # 2) Test kümesine sadece burada, bir kez bakıyoruz
    best = grid.best_estimator_
    proba = best.predict_proba(test["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\nTest kümesi (threshold 0.5):")
    print(classification_report(test["spam"], pred, digits=3))

    metrics = {
        "selected": {"model": best_name, "features": best_features, "setting": best_setting,
                     "cv_f1": round(float(grid.best_score_), 4)},
        "test": {"all": scores(test["spam"], pred)},
        "thresholds": {},
    }
    for lang in ["EN", "TR"]:
        mask = (test["lang"] == lang).values
        metrics["test"][lang] = scores(test["spam"][mask], pred[mask])
        print(f"{lang}: {metrics['test'][lang]}")
    for t in [0.3, 0.5, 0.7, 0.85]:
        metrics["thresholds"][str(t)] = scores(test["spam"], (proba >= t).astype(int))

    # 3) Kaydet
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(REPORTS / "model_selection.md", "w", encoding="utf-8") as f:
        f.write("| Model | Features | Setting | CV F1 | ± | CV precision | CV recall |\n|---|---|---|---|---|---|---|\n")
        for _, r in table.iterrows():
            f.write(f"| {r.model} | {r.features} | {r.setting} | {r.cv_f1:.4f} | {r.cv_f1_std:.4f} | {r.cv_precision:.4f} | {r.cv_recall:.4f} |\n")

    joblib.dump(best.named_steps["clf"], BASE_DIR / "spam_model_bilingual.pkl")
    joblib.dump(best.named_steps["features"], BASE_DIR / "tfidf_vectorizer_bilingual.pkl")
    print("\nModel, vectorizer ve reports/ kaydedildi.")


if __name__ == "__main__":
    main()
