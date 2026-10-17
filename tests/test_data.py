import pytest

from data import BASE_DIR, load_data, split

pytestmark = pytest.mark.skipif(not (BASE_DIR / "emails.csv").exists(),
                                reason="emails.csv is not in the repo (download it from Kaggle)")


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_columns_and_labels(df):
    assert list(df.columns) == ["text", "spam", "lang"]
    assert set(df["spam"].unique()) == {0, 1}
    assert set(df["lang"].unique()) == {"EN", "TR"}


def test_no_duplicate_texts(df):
    assert not df["text"].duplicated().any()


def test_split_keeps_language_and_spam_ratios(df):
    train, test = split(df)
    assert len(test) == pytest.approx(0.2 * len(df), abs=1)
    assert not set(train["text"]) & set(test["text"])
    for lang in ["EN", "TR"]:
        tr = train[train["lang"] == lang]["spam"].mean()
        te = test[test["lang"] == lang]["spam"].mean()
        assert abs(tr - te) < 0.01, lang
