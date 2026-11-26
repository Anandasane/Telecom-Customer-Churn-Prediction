import joblib
from src.config import MODEL_PATH
from src.data import load_data, split_xy
from src.features import FeatureEngineer


def test_data_loads_clean():
    df = load_data()
    assert df.shape == (3150, 14)
    assert df.isna().sum().sum() == 0
    assert set(df["churn"].unique()) == {0, 1}


def test_feature_engineering_is_finite_and_drops_age():
    X, _ = split_xy(load_data())
    out = FeatureEngineer().transform(X)
    assert "age" not in out.columns
    assert out.replace([float("inf"), float("-inf")], float("nan")).isna().sum().sum() == 0


def test_saved_model_predicts_probabilities():
    b = joblib.load(MODEL_PATH)
    X, _ = split_xy(load_data())
    p = b["model"].predict_proba(X[b["features"]].head(20))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
