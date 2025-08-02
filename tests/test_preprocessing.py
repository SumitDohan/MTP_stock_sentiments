import pandas as pd
from src.preprocessing import prepare_dataset

def test_prepare_dataset():
    df = pd.DataFrame({
        "Close": [100, 102, 101],
        "sentiment": [1, -1, 1]
    })
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    result = prepare_dataset(df)
    assert "target" in result.columns
    assert len(result) > 0
