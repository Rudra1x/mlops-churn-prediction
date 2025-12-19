import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time


from api.schemas import ChurnRequest, ChurnResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latency of prediction requests"
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of churn probabilities"
)


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
    start_time = time.time()
    REQUEST_COUNT.inc()

    input_df = pd.DataFrame([request.dict()])
    X_processed = preprocessor.transform(input_df)

    probability = model.predict_proba(X_processed)[0][1]
    prediction = int(probability >= 0.5)

    CHURN_PROBABILITY.observe(probability)
    REQUEST_LATENCY.observe(time.time() - start_time)

    return ChurnResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
    )

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
