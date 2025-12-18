import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    def __init__(self, input_path: str, output_path: str, pipeline_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pipeline_path = Path(pipeline_path)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading ingested data from {self.input_path}")
        if not self.input_path.exists():
            raise FileNotFoundError(f"{self.input_path} not found")
        return pd.read_csv(self.input_path)

    def build_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        logger.info("Building preprocessing pipeline")

        numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = df.select_dtypes(include=["object"]).columns

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        return preprocessor

    def run(self) -> None:
        df = self.load_data()

        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        pipeline = self.build_pipeline(X)

        logger.info("Fitting preprocessing pipeline")
        X_processed = pipeline.fit_transform(X)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((X_processed, y), self.output_path)

        self.pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.pipeline_path)

        logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    processor = Preprocessor(
        input_path="data/processed/ingested.csv",
        output_path="data/features/features.joblib",
        pipeline_path="data/features/preprocessing_pipeline.joblib",
    )
    processor.run()
