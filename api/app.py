import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import ChurnRequest, ChurnResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade ML API for customer churn prediction",
    version="1.0.0",
)

# Load artifacts ONCE at startup
logger.info("Loading preprocessing pipeline and model")

preprocessor = joblib.load("data/features/preprocessing_pipeline.joblib")
model = joblib.load("models/churn_model.joblib")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    logger.info("Received prediction request")

    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])

    # Apply SAME preprocessing as training
    X_processed = preprocessor.transform(input_df)

    # Predict
    probability = model.predict_proba(X_processed)[0][1]
    prediction = int(probability >= 0.5)

    return ChurnResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
    )
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}
