"""Exploratory analysis: writes figures + a short findings file to reports/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_DIR, REPORT_DIR
from .data import load_data

sns.set_theme(style="whitegrid")


def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    lines = []
    rate = df["churn"].mean()
    lines.append(f"Rows: {len(df)}, columns: {df.shape[1]}, missing values: {int(df.isna().sum().sum())}")
    lines.append(f"Overall churn rate: {rate:.1%} ({int(df['churn'].sum())} churned)")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    df["churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot.bar(ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Class balance"); ax.set_ylabel("Customers"); plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_class_balance.png", dpi=150); plt.close(fig)

    cats = ["complains", "tariff_plan", "status", "age_group", "charge_amount"]
    fig, axes = plt.subplots(1, len(cats), figsize=(18, 3.6), sharey=True)
    for ax, c in zip(axes, cats):
        g = df.groupby(c)["churn"].mean()
        g.plot.bar(ax=ax, color="#4C78A8"); ax.set_title(c); ax.set_ylabel("Churn rate"); ax.set_xlabel("")
        ax.axhline(rate, color="red", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_churn_rate_by_category.png", dpi=150); plt.close(fig)
    for c in ["complains", "tariff_plan", "status"]:
        g = df.groupby(c)["churn"].mean()
        lines.append(f"Churn rate by {c}: " + ", ".join(f"{k}={v:.1%}" for k, v in g.items()))

    nums = ["seconds_of_use", "frequency_of_use", "frequency_of_sms", "distinct_called_numbers",
            "customer_value", "subscription_length", "call_failure"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, c in zip(axes.ravel(), nums):
        sns.kdeplot(data=df, x=c, hue="churn", common_norm=False, fill=True, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_numeric_distributions.png", dpi=150); plt.close(fig)
    med = df.groupby("churn")[nums].median().T
    lines.append("Median by churn (0/1):\n" + med.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_correlation.png", dpi=150); plt.close(fig)
    corr = df.corr()["churn"].drop("churn").sort_values(key=abs, ascending=False)
    lines.append("Correlation with churn (top 6):\n" + corr.head(6).round(3).to_string())

    (REPORT_DIR / "eda_findings.txt").write_text("\n\n".join(lines))
    print("\n\n".join(lines))


if __name__ == "__main__":
    run()
