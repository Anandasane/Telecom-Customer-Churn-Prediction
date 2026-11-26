import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CATEGORICAL = ["age_group", "tariff_plan", "status", "complains"]


def cat_cols(X):
    return [c for c in CATEGORICAL if c in X.columns]


def num_cols(X):
    return [c for c in X.columns if c not in CATEGORICAL]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise feature engineering. Stateless, so it cannot leak across the split."""

    DROP = ["age"]  # duplicates age_group (5 bins)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6
        X["avg_seconds_per_call"] = X["seconds_of_use"] / (X["frequency_of_use"] + eps)
        X["sms_to_call_ratio"] = X["frequency_of_sms"] / (X["frequency_of_use"] + 1)
        X["calls_per_month"] = X["frequency_of_use"] / (X["subscription_length"] + 1)
        X["failure_rate"] = X["call_failure"] / (X["frequency_of_use"] + 1)
        X["value_per_second"] = X["customer_value"] / (X["seconds_of_use"] + 1)
        X["is_inactive_usage"] = (X["seconds_of_use"] == 0).astype(int)
        for col in ["seconds_of_use", "frequency_of_sms", "customer_value"]:
            X[f"log_{col}"] = np.log1p(X[col])
        return X.drop(columns=[c for c in self.DROP if c in X.columns])
