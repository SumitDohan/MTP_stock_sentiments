# Main training entry point


from src.ingestion import get_stock_data, get_news
from src.preprocessing import analyze_sentiment, merge_with_stock, prepare_dataset
from src.model import train_model

def main():
    api_key = "YOUR_NEWSAPI_KEY"
    news = get_news("NIFTY", api_key)
    stock = get_stock_data("^NSEI")
    news = analyze_sentiment(news)
    merged = merge_with_stock(news, stock)
    dataset = prepare_dataset(merged)
    run_id = train_model(dataset)

    print(f"Model training complete. Run ID: {run_id}")

if __name__ == "__main__":
    main()
