"""Train + compare models with stratified CV, tune the best, evaluate once on the held-out test set.

Two scenarios are run:
  full        - all features
  early_warn  - drops `status` (active/inactive flag, which may be recorded close to/after churn),
                 giving a more conservative, leakage-aware estimate.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import MODEL_PATH, RANDOM_STATE, REPORT_DIR, TEST_SIZE
from .data import load_data, split_xy
from .features import CATEGORICAL, FeatureEngineer, cat_cols, num_cols

def make_preprocessor(columns, scale: bool):
    """Column selectors are callables so they resolve AFTER FeatureEngineer changes the columns."""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])


def build_models(columns, pos_weight):
    return {
        "Baseline (majority)": Pipeline([("fe", FeatureEngineer()), ("pre", make_preprocessor(columns, True)),
                                         ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([("fe", FeatureEngineer()), ("pre", make_preprocessor(columns, True)),
                                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("fe", FeatureEngineer()), ("pre", make_preprocessor(columns, False)),
                                   ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                                  min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE))]),
        "XGBoost": Pipeline([("fe", FeatureEngineer()), ("pre", make_preprocessor(columns, False)),
                             ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                                   colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                                   eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1))]),
    }


def best_f1_threshold(y_true, proba):
    ts = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, proba >= t) for t in ts]
    return float(ts[int(np.argmax(f1s))])


def run_scenario(name, drop_cols):
    df = load_data()
    X, y = split_xy(df)
    X = X.drop(columns=drop_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    models = build_models(X.columns.tolist(), pos_weight)
    for mname, pipe in models.items():
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                           scoring=["roc_auc", "average_precision", "f1", "recall", "precision"])
        rows.append({"model": mname, **{k.replace("test_", ""): float(v.mean()) for k, v in s.items() if k.startswith("test_")}})
    cv_df = pd.DataFrame(rows).set_index("model").round(3)

    # tune XGBoost on average precision (better than accuracy/ROC for a 16% positive class)
    search = RandomizedSearchCV(
        models["XGBoost"],
        {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [3, 4, 5, 6],
         "clf__learning_rate": [0.02, 0.05, 0.1], "clf__min_child_weight": [1, 3, 5],
         "clf__subsample": [0.7, 0.8, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0],
         "clf__reg_lambda": [1, 3, 10]},
        n_iter=25, scoring="average_precision", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    # choose the decision threshold on out-of-fold train predictions, never on the test set
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_tr, oof)

    best.fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    test = {
        "roc_auc": roc_auc_score(y_te, proba), "pr_auc": average_precision_score(y_te, proba),
        "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred), "threshold": thr,
        "test_size": int(len(y_te)), "test_churners": int(y_te.sum()),
    }
    return dict(name=name, cv=cv_df, best_params=search.best_params_, test=test,
                best=best, X_te=X_te, y_te=y_te, proba=proba, thr=thr, X_tr=X_tr, y_tr=y_tr)


def main():
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    results = {}
    for name, drop in [("full", []), ("early_warning_no_status", ["status"])]:
        r = run_scenario(name, drop)
        results[name] = r
        print(f"\n=== {name} ===\nCV (train folds):\n{r['cv']}\nBest params: {r['best_params']}")
        print("Held-out test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["test"].items()})

    # ship the leakage-aware model as the deployable one
    deploy = results["early_warning_no_status"]
    joblib.dump({"model": deploy["best"], "threshold": deploy["thr"],
                 "features": deploy["X_te"].columns.tolist()}, MODEL_PATH)
    joblib.dump(results, REPORT_DIR / "_results.joblib")
    summary = {k: {"cv": v["cv"].to_dict("index"), "test": v["test"], "best_params": v["best_params"]}
               for k, v in results.items()}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
