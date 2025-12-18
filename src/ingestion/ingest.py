import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        if not self.input_path.exists():
            raise FileNotFoundError(f"{self.input_path} not found")

        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df

    def validate(self, df: pd.DataFrame) -> None:
        logger.info("Validating dataset")

        if df.empty:
            raise ValueError("Dataset is empty")

        required_columns = {"customerID", "Churn"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def save(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved processed data to {self.output_path}")

    def run(self) -> None:
        df = self.load()
        self.validate(df)
        self.save(df)

if __name__ == "__main__":
    ingestor = DataIngestor(
        input_path="data/raw/telco_churn.csv",
        output_path="data/processed/ingested.csv"
    )
    ingestor.run()
