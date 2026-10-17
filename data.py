"""Veri setlerini yükleme ve eğitim/test bölme (main.py ve evaluate.py ortak kullanır)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
SEED = 42


def load_data(base_dir=BASE_DIR):
    """İngilizce e-postalar + Türkçe SMS'ler: text, spam (0/1), lang (EN/TR) sütunları."""
    emails_path = Path(base_dir) / "emails.csv"
    if not emails_path.exists():
        raise FileNotFoundError("emails.csv bulunamadı. README'deki Kaggle linkinden indirip proje köküne koyun.")

    en = pd.read_csv(emails_path)[["text", "spam"]]
    en["lang"] = "EN"

    tr = pd.read_csv(Path(base_dir) / "TurkishSMSCollection.csv", sep=";")
    tr = pd.DataFrame({"text": tr["Message"], "spam": tr["GroupText"].map({"Spam": 1, "Normal": 0}), "lang": "TR"})

    df = pd.concat([en, tr], ignore_index=True).dropna()
    # Birebir aynı metin hem eğitimde hem testte olursa skor şişer, tekrarları atıyoruz
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["spam"] = df["spam"].astype(int)
    return df


def split(df, test_size=0.2, seed=SEED):
    """Dil + etikete göre katmanlı bölme: iki tarafta da dil ve spam oranları aynı kalsın."""
    strata = df["lang"] + "_" + df["spam"].astype(str)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
