import pandas as pd
from src.model import train_model

def test_model_train():
    df = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104],
        "sentiment": [1, -1, 1, -1, 0],
        "target": [1, 1, 1, 0, 0]
    })
    run_id = train_model(df)
    assert isinstance(run_id, str)
