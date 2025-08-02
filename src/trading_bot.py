import requests
import yfinance as yf
import mlflow
import json
from datetime import datetime

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"
TICKER = "^NSEI"
HEADLINES = [
    "Nifty surges to all-time high amid positive global cues",
    "Market uncertain ahead of interest rate decisions",
    "Investors show strong interest in blue-chip stocks"
]

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("FinBERT_TradingBot")

# --- Call FastAPI Sentiment API ---
try:
    res = requests.post(API_URL, json={"headlines": HEADLINES})
    res.raise_for_status()
    sentiments = res.json()["results"]
except Exception as e:
    print(f"❌ Failed to get sentiment: {e}")
    sentiments = []

# --- Print Sentiment Results ---
if sentiments:
    print("📊 Prediction Results:")
    for item in sentiments:
        score = item.get("probabilities", {}).get("positive", 0.0) - item.get("probabilities", {}).get("negative", 0.0)
        item["sentiment_score"] = round(score, 4)
        print(f"📰 {item['headline']:<70} | Sentiment: {item['sentiment_score']:.4f}")

    # --- Compute average sentiment ---
    avg_sentiment = sum(item["sentiment_score"] for item in sentiments) / len(sentiments)

    # --- Make decision ---
    if avg_sentiment > 0.5:
        decision = "BUY"
    elif avg_sentiment < -0.5:
        decision = "SELL"
    else:
        decision = "HOLD"

    # --- Get current price of NIFTY ---
    price = None
    try:
        price_series = yf.download(TICKER, period="1d", interval="1m", progress=False)["Close"]
        if not price_series.empty:
            price = float(price_series.iloc[-1])
        else:
            fallback_series = yf.download(TICKER, period="2d", interval="1d", progress=False)["Close"]
            if not fallback_series.empty:
                price = float(fallback_series.iloc[-1])
    except Exception as e:
        print(f"⚠️ Failed to fetch price: {e}")

    # --- Final Output ---
    print(f"\n✅ Decision: {decision} | Avg Sentiment: {avg_sentiment:.2f} | Price: ₹{price:.2f}" if price else
          f"\n✅ Decision: {decision} | Avg Sentiment: {avg_sentiment:.2f} | Price: Not available")

    # --- MLflow Logging ---
    with mlflow.start_run(run_name="TradingBotSentimentRun") as run:
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("decision", decision)
        mlflow.log_metric("avg_sentiment", avg_sentiment)
        mlflow.log_metric("price", price if price is not None else -1)

        for i, item in enumerate(sentiments):
            mlflow.log_metric(f"sentiment_{i}", item["sentiment_score"])

        # Save raw API response as JSON
        artifact_path = f"sentiment_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(artifact_path, "w") as f:
            json.dump(sentiments, f, indent=2)
        mlflow.log_artifact(artifact_path, artifact_path="api_response")

else:
    print("❌ No sentiment results to process.")
