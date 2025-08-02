import os
import yfinance as yf
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
from datetime import date

# --- Configuration ---
api_key = "c30d7a7f8b784290bf8106ae22ef4a2c"
ticker = "^NSEI"
query = "Nifty50"
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
        return {"articles": []}

    if res.status_code != 200 or data.get("status") != "ok" or "articles" not in data:
        print("❌ Error in NewsAPI response:", data)
        return {"articles": []}

    return data

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

    # Clean and prepare
    news_df = news_df[news_df["title"].notnull() & (news_df["title"].str.strip() != "")]
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], errors="coerce")
    news_df["date"] = news_df["publishedAt"].dt.date

    # Load FinBERT
    print("🔍 Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    model.eval()
    labels = ['negative', 'neutral', 'positive']
    print("✅ FinBERT model loaded.")

    # Predict label
    def predict_label(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return labels[torch.argmax(probs)]

    # Predict score
    def predict_score(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs[0][2].item() - probs[0][0].item()  # pos - neg

    print("🔬 Running sentiment analysis...")
    news_df["sentiment_label"] = news_df["title"].apply(predict_label)
    news_df["sentiment_score"] = news_df["title"].apply(predict_score)

    avg_sentiment = news_df["sentiment_score"].mean()
    label_counts = news_df["sentiment_label"].value_counts().to_dict()
    total_articles = sum(label_counts.values())

    percent_positive = (label_counts.get("positive", 0) / total_articles) * 100
    percent_neutral = (label_counts.get("neutral", 0) / total_articles) * 100
    percent_negative = (label_counts.get("negative", 0) / total_articles) * 100

    # Print sample outputs
    print("\n🧪 Sample Predictions:")
    for _, row in news_df.head(5).iterrows():
        print(f"📰 {row['title']}\n→ Label: {row['sentiment_label']}, Score: {row['sentiment_score']:.3f}\n")

    print("📊 Sentiment Label Distribution:", label_counts)
    print(f"📈 Positive: {percent_positive:.2f}%, Neutral: {percent_neutral:.2f}%, Negative: {percent_negative:.2f}%")

    # --- Investment Suggestion ---
    print("\n📉 Investment Advice:")
    if 33 <= percent_positive <= 40 and 25 <= percent_neutral <= 30:
        investment_advice = "GOOD TIME TO BUY STOCKS"
        print("✅ Sentiment indicates a GOOD TIME TO BUY STOCKS.")
    elif 45 <= percent_neutral <= 50:
        investment_advice = "NORMAL DAY to do transactions"
        print("ℹ️ Neutral sentiment is high — it's a NORMAL DAY to do transactions.")
    elif percent_negative > 40:
        investment_advice = "NOT GOOD TO INVEST"
        print("❌ Too many negative sentiments — NOT A GOOD TIME TO INVEST.")
    else:
        investment_advice = "MIXED SIGNALS — use caution"
        print("⚠️ Mixed signals — proceed with CAUTION.")

    # --- MLflow Logging ---
    mlflow.set_tracking_uri("file:D:/MTP/mlruns")
    mlflow.set_experiment("Financial_Sentiment_Pipeline")

    with mlflow.start_run(run_name="Pipeline_Sentiment_Analysis"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("query", query)
        mlflow.log_param("num_articles", len(news_df))
        mlflow.log_param("investment_advice", investment_advice)
        mlflow.log_metric("avg_sentiment", avg_sentiment)
        mlflow.log_metric("percent_positive", percent_positive)
        mlflow.log_metric("percent_neutral", percent_neutral)
        mlflow.log_metric("percent_negative", percent_negative)
        for label, count in label_counts.items():
            mlflow.log_metric(f"count_{label}", count)
        news_df.to_csv("pipeline_sentiment_output.csv", index=False)
        mlflow.log_artifact("pipeline_sentiment_output.csv", artifact_path="results")

    print("✅ Sentiment scores and investment advice logged to MLflow.")

# --- Main ---
if __name__ == "__main__":
    news_json = get_news(query, api_key)
    stock_df = get_stock_data(ticker)
    run_sentiment_analysis_and_train(news_json, stock_df)
