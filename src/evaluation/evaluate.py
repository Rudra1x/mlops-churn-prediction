import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, model_path: str, feature_path: str):
        self.model_path = Path(model_path)
        self.feature_path = Path(feature_path)

    def load_artifacts(self):
        logger.info("Loading model and features")

        if not self.model_path.exists():
            raise FileNotFoundError(f"{self.model_path} not found")

        if not self.feature_path.exists():
            raise FileNotFoundError(f"{self.feature_path} not found")

        model = joblib.load(self.model_path)
        X, y = joblib.load(self.feature_path)

        return model, X, y

    def evaluate(self):
        model, X, y = self.load_artifacts()

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        logger.info(f"Evaluation Accuracy: {accuracy:.4f}")
        logger.info(f"Evaluation ROC-AUC: {auc:.4f}")

        return {
            "accuracy": accuracy,
            "roc_auc": auc
        }

    def promote(self, metrics: dict) -> bool:
        logger.info("Applying promotion rules")

        if metrics["roc_auc"] < 0.75:
            logger.warning("Model rejected: ROC-AUC below threshold")
            return False

        if metrics["accuracy"] < 0.70:
            logger.warning("Model rejected: Accuracy below threshold")
            return False

        logger.info("Model approved for deployment")
        return True


if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_path="models/churn_model.joblib",
        feature_path="data/features/features.joblib",
    )

    metrics = evaluator.evaluate()
    approved = evaluator.promote(metrics)

    if not approved:
        raise SystemExit("Model did not pass evaluation gates")
