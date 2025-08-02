from src.ingestion import get_stock_data

def test_stock_data_download():
    df = get_stock_data("^NSEI", "2023-01-01", "2023-01-10")
    assert not df.empty
    assert "Close" in df.columns
