# src/preprocessing.py
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import mlflow
from datetime import date

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:/home/sweta/MTP/mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Directory Setup ---
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
PLOTS_PATH = "data/plots"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# --- Load stock data ---
def load_stock_data(filepath):
    print("📂 Loading stock.csv from raw folder...")
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    df.index.name = "Date"
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.ffill().bfill()  # fill missing values
    return df

# --- Load news data ---
def load_news_data(filepath):
    print("📰 Loading news.json from raw folder...")
    with open(filepath, "r") as f:
        articles = json.load(f)
    return articles

# --- Preprocess stock data ---
def preprocess_stock_data(df):
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change().fillna(0)
    df["SMA_7"] = df["Close"].rolling(window=7).mean().fillna(method="bfill")
    df["SMA_21"] = df["Close"].rolling(window=21).mean().fillna(method="bfill")
    df["Volatility"] = df["Close"].rolling(window=7).std().fillna(0)
    return df

# --- Plotting function ---
def plot_stock(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=2)
    plt.plot(df.index, df["SMA_7"], label="7-day SMA", linestyle="--")
    plt.plot(df.index, df["SMA_21"], label="21-day SMA", linestyle="--")
    plt.title(f"{ticker} Stock Price with Indicators")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_file = os.path.join(PLOTS_PATH, f"{ticker}_plot.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    return plot_file

# --- Main script ---
if __name__ == "__main__":
    stock_file = os.path.join(RAW_PATH, "stock.csv")
    news_file = os.path.join(RAW_PATH, "news.json")
    ticker = "NSEI"

    with mlflow.start_run(run_name="preprocessing"):
        # --- Load data ---
        stock_df = load_stock_data(stock_file)
        news_data = load_news_data(news_file)

        # --- Preprocess ---
        processed_df = preprocess_stock_data(stock_df)
        processed_path = os.path.join(PROCESSED_PATH, f"{ticker}_processed.csv")
        processed_df.to_csv(processed_path)
        mlflow.log_artifact(processed_path, artifact_path="processed_data")

        # --- Plot ---
        plot_path = plot_stock(processed_df, ticker)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # --- Log raw news ---
        mlflow.log_artifact(news_file, artifact_path="news_raw")

        print("✅ Preprocessing complete and artifacts logged to MLflow.")
