from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Customer_Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
