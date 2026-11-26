"""Plots + interpretation for the deployable (no-`status`) model, plus a leakage ablation."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, roc_auc_score)

from .config import FIG_DIR, REPORT_DIR

sns.set_theme(style="whitegrid")
COST_FN, COST_FP = 5.0, 1.0   # ASSUMPTION: losing a customer costs 5x a wasted retention offer. Change freely.


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = joblib.load(REPORT_DIR / "_results.joblib")["early_warning_no_status"]
    model, X_te, y_te, proba, thr = res["best"], res["X_te"], res["y_te"], res["proba"], res["thr"]
    pred = (proba >= thr).astype(int)
    notes = []

    # ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax[0]); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_title(f"ROC (AUC={roc_auc_score(y_te, proba):.3f})")
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax[1])
    ax[1].axhline(y_te.mean(), color="k", ls="--", lw=1)
    ax[1].set_title(f"Precision-Recall (AP={average_precision_score(y_te, proba):.3f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_roc_pr.png", dpi=150); plt.close(fig)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix @ threshold {thr:.2f}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=150); plt.close(fig)

    # calibration
    frac, mean = calibration_curve(y_te, proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(mean, frac, "o-", color="#4C78A8"); ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed churn rate"); ax.set_title("Calibration")
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_calibration.png", dpi=150); plt.close(fig)

    # business cost curve
    ts = np.linspace(0.02, 0.98, 97)
    costs = []
    for t in ts:
        p = proba >= t
        fn = ((~p) & (y_te == 1)).sum(); fp = (p & (y_te == 0)).sum()
        costs.append(COST_FN * fn + COST_FP * fp)
    best_t = float(ts[int(np.argmin(costs))])
    all_ignore = COST_FN * int(y_te.sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, costs, color="#E45756"); ax.axvline(best_t, ls="--", color="k")
    ax.axhline(all_ignore, ls=":", color="grey"); ax.text(0.5, all_ignore * 0.96, "do nothing", color="grey")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel(f"Total cost (FN={COST_FN:g}, FP={COST_FP:g})")
    ax.set_title("Business cost vs threshold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "08_cost_curve.png", dpi=150); plt.close(fig)
    p = pred.astype(bool)
    cost_at_thr = COST_FN * int(((~p) & (y_te == 1)).sum()) + COST_FP * int((p & (y_te == 0)).sum())
    saving = 1 - cost_at_thr / all_ignore
    notes.append(f"At the train-selected threshold ({thr:.2f}) the total cost is {saving:.0%} lower than doing nothing "
                 f"under the ASSUMED FN:FP cost ratio {COST_FN:g}:{COST_FP:g}. "
                 f"(Test-set cost minimum is at ~{best_t:.2f}; shown on the plot for illustration only, not used for the model.)")

    # SHAP
    fe_pre, clf = model[:-1], model.named_steps["clf"]
    Xt = fe_pre.transform(X_te)
    names = [n.split("__", 1)[1] for n in fe_pre.named_steps["pre"].get_feature_names_out()]
    Xt = pd.DataFrame(Xt, columns=names)
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIG_DIR / "09_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIG_DIR / "10_shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    notes.append("Top SHAP features:\n" + imp.head(8).round(3).to_string())

    # leakage ablation: how much do the two dominant flags matter?
    from .train import run_scenario  # noqa
    from .data import load_data, split_xy
    from sklearn.model_selection import train_test_split
    from .config import RANDOM_STATE, TEST_SIZE
    df = load_data(); X, y = split_xy(df)
    abl = []
    for label, drop in [("all features", []), ("minus status", ["status"]),
                        ("minus complains", ["complains"]), ("minus status & complains", ["status", "complains"])]:
        Xa = X.drop(columns=drop)
        a_tr, a_te, b_tr, b_te = train_test_split(Xa, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        m = clone(model).fit(a_tr, b_tr)
        pr = m.predict_proba(a_te)[:, 1]
        abl.append({"features": label, "roc_auc": roc_auc_score(b_te, pr), "pr_auc": average_precision_score(b_te, pr)})
    abl = pd.DataFrame(abl).round(3)
    notes.append("Ablation (same tuned XGBoost, refit):\n" + abl.to_string(index=False))
    abl.to_csv(REPORT_DIR / "ablation.csv", index=False)

    (REPORT_DIR / "evaluation_notes.txt").write_text("\n\n".join(notes))
    print("\n\n".join(notes))


if __name__ == "__main__":
    main()
