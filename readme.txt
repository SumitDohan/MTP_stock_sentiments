 data/ (optional)
This folder is for storing raw or preprocessed data locally.

Example files:

stock_data.csv

news_articles.json

processed_dataset.csv
Useful for:

Re-running the pipeline without fetching data again.

Keeping debug/training datasets.




 src/ — Source Code
Holds all your Python modules organized by task.

📄 ingestion.py
Purpose: Collect data

📥 Gets stock price data using yfinance

📰 Fetches financial news articles using NewsAPI

Stores or returns this data as DataFrames





 preprocessing.py
Purpose: Process and combine datasets

🧠 Performs sentiment analysis on news headlines using FinBERT or FinGPT

📊 Aggregates daily sentiment scores

🔗 Merges news + stock data into one feature set

🏷️ Adds target labels (whether price will rise next day)






 model.py
Purpose: Model training + MLflow logging

📈 Trains a model (e.g. RandomForestClassifier or LSTM)

✅ Splits into train/test sets

📊 Evaluates using metrics like accuracy

🔐 Logs model and metrics to MLflow






 app.py
Purpose: FastAPI inference server

🔄 Loads the trained model using MLflow

🌐 Hosts a REST API (on /predict)

Accepts JSON with features like:





pipeline.py (optional if using main.py)
Purpose: Wrap the entire ML pipeline as a single callable function.

Often used if you integrate with Prefect or Airflow






 main.py
Purpose: Your entry-point script to run the entire pipeline manually.

It typically does:

Call get_stock_data and get_news

Run analyze_sentiment and merge_with_stock

Prepare dataset

Train and log model with train_model







# 📈 Stock Sentiment Prediction with News + MLOps

This project predicts stock movement (rise/fall) using financial news sentiment and stock price data.

### 🔧 Stack

- Python, FastAPI
- MLflow (for experiment tracking)
- Transformers (FinBERT)
- Docker
- GCP Cloud Run (deployment)
- Prefect (optional pipeline orchestration)

### 🚀 Run Locally

```bash
# Create venv + activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py

# Run API server
uvicorn src.app:app --reload
