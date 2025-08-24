from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model & tokenizer once at startup
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

app = FastAPI(title="FinBERT Sentiment API")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "FinBERT API is running"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    labels = ["negative", "neutral", "positive"]
    return {
        "text": request.text,
        "sentiment": labels[predicted_class_id]
    }
