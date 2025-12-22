import pandas as pd
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, df):
        """
        Initializes with a dataframe containing OHLCV data.
        """
        self.df = df.copy()

    def add_technical_indicators(self):
        """
        Adds Momentum, Trend, and Volatility indicators.
        """
        # 1. RSI (Relative Strength Index) - Overbought/Oversold
        self.df['RSI'] = ta.rsi(self.df['Close'], length=14)

        # 2. MACD (Moving Average Convergence Divergence)
        macd = ta.macd(self.df['Close'])
        self.df['MACD'] = macd['MACD_12_26_9']
        self.df['MACD_Signal'] = macd['MACDs_12_26_9']
        self.df['MACD_Hist'] = macd['MACDh_12_26_9']

        # 3. EMA (Exponential Moving Averages) - Trend
        self.df['EMA_20'] = ta.ema(self.df['Close'], length=20)
        self.df['EMA_50'] = ta.ema(self.df['Close'], length=50)

        # 4. ATR (Average True Range) - Volatility
        self.df['ATR'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)

        # 5. Volume Change
        self.df['Vol_Pct_Change'] = self.df['Volume'].pct_change() * 100
        
        return self.df

    def add_custom_features(self):
        """
        Adds engineered features like returns and price-to-ema ratios.
        """
        # Daily Returns
        self.df['Daily_Return'] = self.df['Close'].pct_change() * 100
        
        # Distance from EMA (How overextended is the price?)
        self.df['Dist_EMA_20'] = (self.df['Close'] - self.df['EMA_20']) / self.df['EMA_20']
        
        # Target Log Returns (Optional, helpful for some models)
        import numpy as np
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        return self.df

    def clean_data(self):
        """
        Removes the 'NaN' rows created by rolling indicators.
        """
        self.df.dropna(inplace=True)
        return self.df

    def get_engineered_df(self):
        self.add_technical_indicators()
        self.add_custom_features()
        return self.clean_data()

if __name__ == "__main__":
    # Example usage for testing
    df_raw = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)
    fe = FeatureEngineer(df_raw)
    df_final = fe.get_engineered_df()
    print(df_final.tail())
    # Save for next step
    df_final.to_csv("data/processed_data.csv")

    