import logging
from pathlib import Path
import pandas as pd


def load_data(csv_filename: str = "manufacturing_dataset_1000_samples.csv") -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / csv_filename
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    num_rows, num_cols = df.shape
    missing_total = int(df.isna().sum().sum())

    logging.info(f"Loaded data from {data_path}")
    logging.info(f"Shape: {num_rows} rows x {num_cols} columns")
    logging.info(f"Total missing values: {missing_total}")

    return df
