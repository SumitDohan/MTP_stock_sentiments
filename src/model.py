# src/model.py

import os
import yfinance as yf
import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
from datetime import date

# --- Configuration ---
api_key = "c30d7a7f8b784290bf8106ae22ef4a2c"
ticker = "^NSEI"
query = "Nifty"
start_date = "2025-07-15"
end_date = date.today().isoformat()

# --- Get stock data ---
def get_stock_data(ticker, start_date=start_date, end_date=end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    data = data[["Close"]].copy()
    data.index.name = "Date"
    return data

# --- Get news data ---
def get_news(query, api_key, start_date=start_date, end_date=end_date):
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&from={start_date}&to={end_date}"
        f"&sortBy=publishedAt&language=en&pageSize=100&apiKey={api_key}"
    )
    res = requests.get(url)
    try:
        data = res.json()
    except Exception as e:
        print("❌ Could not parse JSON:", e)
        print("Response text:", res.text)
        return pd.DataFrame()

    if res.status_code != 200 or data.get("status") != "ok" or "articles" not in data:
        print("❌ Error in NewsAPI response:", data)
        return pd.DataFrame()

    articles = data["articles"]
    if not articles:
        print("⚠️ No articles returned by NewsAPI.")
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df["date"] = df["publishedAt"].dt.date
    return df[["title", "publishedAt", "date"]]

# --- Run sentiment analysis and log with MLflow ---
def run_sentiment_analysis_and_train(news_json, stock_df):
    print("📦 Converting news articles to DataFrame...")
    articles = news_json.get("articles", [])
    if not articles:
        print("⚠️ No articles found in JSON.")
        return

    news_df = pd.DataFrame(articles)
    if news_df.empty or "title" not in news_df.columns:
        print("⚠️ News DataFrame is empty or missing 'title'.")
        return

    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], errors="coerce")
    news_df["date"] = news_df["publishedAt"].dt.date

    print("🔍 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    print("✅ FinBERT model loaded.")

    def predict_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs[0][2].item() - probs[0][0].item()  # pos - neg

    print("🔬 Running sentiment analysis...")
    news_df["sentiment_score"] = news_df["title"].apply(predict_sentiment)

    avg_sentiment = news_df["sentiment_score"].mean()

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="Pipeline_Sentiment_Analysis"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("avg_sentiment", avg_sentiment)
        news_df.to_csv("pipeline_sentiment_output.csv", index=False)
        mlflow.log_artifact("pipeline_sentiment_output.csv", artifact_path="results")

    print("✅ Sentiment scores logged to MLflow.")
