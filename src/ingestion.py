# src/ingestion.py
import os
import json
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from newsapi import NewsApiClient
import mlflow

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:/home/sweta/MTP/mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Configuration ---
PRIMARY_TICKER = "^NSEI"        # Nifty 50 index
FALLBACK_TICKER = "NSEI.NS"    # Alternate Yahoo Finance ticker
query = "Nifty"
start_date = "2025-08-25"
end_date = (date.today() - timedelta(days=1)).isoformat()

raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

stock_csv_path = os.path.join(raw_dir, "stock.csv")
news_json_path = os.path.join(raw_dir, "news.json")

# --- Fetch stock data ---
def fetch_stock_data():
    print(f"📥 Downloading stock data for {PRIMARY_TICKER} from {start_date} to {end_date}")
    df = yf.download(PRIMARY_TICKER, start=start_date, end=end_date)
    
    if df.empty:
        print(f"⚠️ No data returned for {PRIMARY_TICKER}. Trying fallback ticker {FALLBACK_TICKER}...")
        df = yf.download(FALLBACK_TICKER, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("❌ Still no stock data returned. Check ticker or date range.")
    
    df.to_csv(stock_csv_path)
    print(f"✅ Stock data saved to {stock_csv_path}")
    return stock_csv_path

# --- Fetch news data ---
def fetch_news_data():
    print(f"📰 Fetching news articles about '{query}' from NewsAPI")
    newsapi = NewsApiClient(api_key="c30d7a7f8b784290bf8106ae22ef4a2c")
    
    try:
        articles = newsapi.get_everything(
            q=query,
            language='en',
            from_param=start_date,
            to=end_date,
            page_size=100,
        )
        if articles.get("status") != "ok" or not articles.get("articles"):
            print("⚠️ No articles returned or API error. Logging empty list.")
            articles = {"status": "ok", "articles": []}
    except Exception as e:
        print(f"❌ Failed to fetch news: {e}")
        articles = {"status": "error", "articles": []}

    with open(news_json_path, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"✅ News data saved to {news_json_path}")
    return news_json_path

# --- MLflow logging ---
def log_with_mlflow(stock_path, news_path):
    with mlflow.start_run(run_name="data_ingestion"):
        # Log artifacts
        mlflow.log_artifact(stock_path, artifact_path="raw_data")
        mlflow.log_artifact(news_path, artifact_path="raw_data")

        # Log parameters
        mlflow.log_param("ticker", PRIMARY_TICKER)
        mlflow.log_param("query", query)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)

        # Log metrics
        df = pd.read_csv(stock_path)
        mlflow.log_metric("num_stock_records", len(df))

        with open(news_path) as f:
            news_data = json.load(f)
            articles = news_data.get("articles", [])
        mlflow.log_metric("num_news_articles", len(articles))

        # Log tags
        mlflow.set_tag("phase", "data_ingestion")
        mlflow.set_tag("data_source", "yfinance_and_newsapi")

        # Log summary JSON
        summary = {
            "ticker": PRIMARY_TICKER,
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "num_stock_records": len(df),
            "num_news_articles": len(articles)
        }
        summary_path = os.path.join(raw_dir, "ingestion_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="raw_data")

        # Optional: Log stock plot
        try:
            df["Close"].plot(title="Closing Prices")
            plot_path = os.path.join(raw_dir, "stock_plot.png")
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="visuals")
        except Exception as e:
            print(f"⚠️ Failed to plot closing prices: {e}")

        print("📦 Data artifacts, params, metrics, and summary logged to MLflow")

# --- Main ---
if __name__ == "__main__":
    try:
        stock_path = fetch_stock_data()
        news_path = fetch_news_data()
        log_with_mlflow(stock_path, news_path)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
