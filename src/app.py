# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import mlflow
import uuid
import json
import os

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("FinBERT_Inference")

# --- FastAPI Setup ---
app = FastAPI(
    title="Financial News Sentiment API",
    description="Returns sentiment scores for financial news using FinBERT, with MLflow logging",
    version="1.1"
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load FinBERT ---
logger.info("🔍 Loading FinBERT model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    logger.info("✅ FinBERT model loaded.")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise RuntimeError("Model loading failed")

# --- Pydantic Schema ---
class NewsRequest(BaseModel):
    headlines: List[str]

# --- Prediction Logic ---
def predict_sentiment(headlines: List[str]):
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_labels = torch.argmax(probs, dim=1).tolist()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    results = []
    for i, (headline, label) in enumerate(zip(headlines, pred_labels)):
        results.append({
            "headline": headline,
            "sentiment": label_map[label],
            "probabilities": {
                "negative": round(float(probs[i][0]), 4),
                "neutral": round(float(probs[i][1]), 4),
                "positive": round(float(probs[i][2]), 4),
            }
        })
    return results

# --- Routes ---
@app.get("/")
def health_check():
    return {"message": "✅ FinBERT Sentiment API is running."}

@app.post("/predict")
def get_sentiment(data: NewsRequest):
    if not data.headlines:
        raise HTTPException(status_code=400, detail="No headlines provided.")

    try:
        results = predict_sentiment(data.headlines)

        # --- MLflow Logging ---
        run_name = f"inference-{uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name):
            # Params (inputs)
            mlflow.log_param("num_headlines", len(data.headlines))
            mlflow.log_param("headlines_sample", data.headlines[:3])

            # Metrics
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for r in results:
                sentiment_counts[r["sentiment"]] += 1

            for sentiment, count in sentiment_counts.items():
                mlflow.log_metric(f"count_{sentiment}", count)

            # Save result as temp JSON
            os.makedirs("tmp", exist_ok=True)
            result_path = f"tmp/prediction_{run_name}.json"
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)

            # Log as artifact
            mlflow.log_artifact(result_path, artifact_path="predictions")

        return {"results": results}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
