import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, feature_path: str, model_path: str):
        self.feature_path = Path(feature_path)
        self.model_path = Path(model_path)

    def load_features(self):
        logger.info(f"Loading features from {self.feature_path}")
        if not self.feature_path.exists():
            raise FileNotFoundError(f"{self.feature_path} not found")

        X, y = joblib.load(self.feature_path)
        return X, y

    def train(self):
        X, y = self.load_features()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=1000)

        logger.info("Starting MLflow run")
        with mlflow.start_run():
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", auc)

            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"ROC-AUC: {auc:.4f}")

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)

            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Training completed successfully")


if __name__ == "__main__":
    trainer = ModelTrainer(
        feature_path="data/features/features.joblib",
        model_path="models/churn_model.joblib",
    )
    trainer.train()
