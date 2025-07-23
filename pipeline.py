# pipeline.py

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ingestion import fetch_stock_data, fetch_news_data
from preprocessing import load_stock_data, load_news_data, preprocess_stock_data
from model import run_sentiment_analysis_and_train

from prefect import flow

@flow(name="Stock-Sentiment-Pipeline")
def stock_sentiment_flow():
    print("🚀 Starting Stock Sentiment Pipeline")

    # Step 1: Ingestion
    stock_path = fetch_stock_data()
    news_path = fetch_news_data()

    # Step 2: Preprocessing
    stock_df = load_stock_data(stock_path)
    news_json = load_news_data(news_path)
    processed_stock_df = preprocess_stock_data(stock_df)

    # Step 3: Sentiment Analysis + Training
    run_sentiment_analysis_and_train(news_json, processed_stock_df)

    print("✅ Pipeline execution complete.")

if __name__ == "__main__":
    stock_sentiment_flow()
