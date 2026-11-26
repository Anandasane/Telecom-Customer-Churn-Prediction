# 📉 Telecom Customer Churn Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project predicting telecom customer churn. The pipeline covers **EDA → stateless feature engineering → model comparison → leakage-aware evaluation → SHAP interpretation → business cost analysis**, culminating in an interactive **Streamlit dashboard**.

**Dataset:** 3,150 telecom customers | 13 features | Binary `Churn` target (15.7% churn rate, no missing values).

---

## 💡 Business Impact & Key Findings

* **Massive Cost Reduction:** Under an assumed 5:1 cost ratio (missed churner vs. wasted retention offer), deploying this model's predictions **cuts total retention costs by ~93%** compared to a "do nothing" baseline. *(Adjustable in `src/evaluate.py`)*.
* **High-Risk Profiles Identified:** Customers who file a **complaint churn at 83%**, compared to just 10% for those who don't. 
* **Usage Drop-offs:** Churners use the service far less: median **1,182 vs 3,530 seconds** of use and **25 vs 63 calls**.
* **Tariff Sensitivity:** Tariff plan 2 customers churn at just 2.4%, compared to 16.8% for plan 1.
* **Top SHAP Drivers:** Frequency of use, total seconds of use, complaints, subscription length, and customer value.

![SHAP Summary](reports/figures/09_shap_summary.png)

![SHAP Feature Importance](reports/figures/10_shap_bar.png)

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./run_all.sh                       # EDA, training, evaluation, tests (~1-2 min)
streamlit run app/streamlit_app.py # interactive predictor