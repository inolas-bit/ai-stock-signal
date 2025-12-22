import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start="2019-01-01", end=None):
    """
    Fetch historical stock data from Yahoo Finance
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)

    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df




from src.data_loader import fetch_stock_data, save_data
import os

# Print current working directory
print("Current Directory:", os.getcwd())

df = fetch_stock_data("AAPL", start="2020-01-01")

# Force save inside project/data folder
save_data(df, os.path.join("data", "raw_data.csv"))

print("File saved successfully!")
