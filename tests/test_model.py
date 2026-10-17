import json
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

import export_web_model

ROOT = Path(__file__).resolve().parent.parent

# Mixed sample: Turkish and English, spam and normal, plus awkward inputs for the tokenizer.
SAMPLES = [
    "Tebrikler! 1000 TL hediye kazandiniz, hemen tiklayin",
    "Yarın akşam toplantı kaçta başlıyor?",
    "Congratulations! You won a free prize. Click here to claim now!",
    "Subject: meeting notes attached, see you on monday",
    "İSTANBUL'DA İNDİRİM!!! ÇOK ÖZEL FIRSAT",
    "FREE $$$ 😀 click http://x.co now",
    "tab\tand\nnewline   multiple   spaces",
    "a",
    "ß ǅ Σ ΑΣ ı İ word_with_underscore 12345",
]


@pytest.fixture(scope="module")
def pipeline():
    features = joblib.load(ROOT / "tfidf_vectorizer_bilingual.pkl")
    clf = joblib.load(ROOT / "spam_model_bilingual.pkl")
    return Pipeline([("features", features), ("clf", clf)])


def test_probabilities_are_valid(pipeline):
    proba = pipeline.predict_proba(SAMPLES)
    assert proba.shape == (len(SAMPLES), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_exported_weights_reproduce_the_model(pipeline):
    """The linear form used by the browser demo must give the same probabilities as sklearn."""
    data = export_web_model.build(pipeline.named_steps["features"], pipeline.named_steps["clf"])
    x = pipeline.named_steps["features"].transform(SAMPLES)
    weights = np.array(data["word_coef"] + data["char_coef"])
    manual = 1 / (1 + np.exp(-(x @ weights + data["intercept"])))
    assert np.allclose(manual, pipeline.predict_proba(SAMPLES)[:, 1], atol=1e-5)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_port_matches_python(pipeline, tmp_path):
    """Runs docs/spam-model.js in Node and compares it with the Python model."""
    model_path = tmp_path / "model.json"
    export_web_model.main(model_path)
    cases = {"texts": SAMPLES, "probs": pipeline.predict_proba(SAMPLES)[:, 1].tolist()}
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    out = subprocess.run(["node", str(ROOT / "tests" / "js_parity.js"), str(model_path), str(cases_path)],
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["max_diff"] < 1e-5, result
    assert result["label_flips"] == 0, result
