import pandas as pd
from .config import DATA_PATH, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise column names (the raw file has double spaces)."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
    )
    return df


def split_xy(df: pd.DataFrame):
    target = TARGET.lower()
    return df.drop(columns=[target]), df[target]
