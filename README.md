# End-to-End MLOps Pipeline for Customer Churn Prediction

This project implements a full production-grade MLOps pipeline for customer churn prediction.
It covers data ingestion, preprocessing, model training, evaluation, deployment, CI/CD,
Kubernetes orchestration, and production monitoring.

The system is designed to prevent training-serving skew, enforce automated model quality
gates, and support scalable, observable ML inference in production.

## Architecture Overview

Data → Training Pipeline → Model Evaluation Gates → FastAPI Inference API  
→ Docker → Kubernetes → Monitoring (Prometheus-style metrics)

The same preprocessing pipeline used during training is reused at inference to
guarantee consistency.

## Tech Stack

- Python 3.10
- scikit-learn (Logistic Regression baseline)
- FastAPI (model serving)
- Docker (containerization)
- Kubernetes (deployment & scaling)
- GitHub Actions (CI/CD)
- MLflow (experiment tracking)
- Prometheus-style metrics (monitoring hooks)

## MLOps Features

- Modular training pipeline with reproducible artifacts
- Persisted preprocessing to prevent training-serving skew
- Automated model evaluation gates (ROC-AUC & accuracy thresholds)
- REST API for real-time inference
- Dockerized deployment
- Kubernetes deployment with replicas and health checks
- CI/CD pipeline blocking low-quality models
- Prediction and latency monitoring hooks


## Run Locally

1. Train the model:
   python -m src.training.train

2. Run the API:
   uvicorn api.app:app --reload

3. Open:
   http://127.0.0.1:8000/docs

## Docker

docker build -f docker/Dockerfile.api -t churn-api .
docker run -p 8000:8000 churn-api

## Kubernetes Deployment

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

Access:
http://localhost:30007/docs

## Monitoring

The API exposes a /metrics endpoint for:
- Request counts
- Latency
- Prediction probability distribution

These metrics can be scraped by Prometheus and visualized in Grafana.

## Key Learnings

- Handling model serialization and dependency compatibility
- Preventing training-serving skew
- Automating ML quality gates in CI/CD
- Deploying scalable ML services with Kubernetes
- Monitoring ML systems beyond accuracy

