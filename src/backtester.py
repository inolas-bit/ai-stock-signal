import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, df_test, predictions):
        """
        df_test: The slice of the dataframe used for testing (X_test)
        predictions: The model's Buy/Sell/Hold outputs
        """
        self.results = df_test.copy()
        self.results['Signal'] = predictions
        # Calculate daily market returns
        self.results['Market_Returns'] = self.results['Close'].pct_change()

    def run_strategy(self):
        """
        Simple Strategy: 
        - Long (1) when signal is 'Buy'
        - Flat (0) when signal is 'Hold' or 'Sell'
        """
        # We shift the signal by 1 day because we trade on the NEXT open
        self.results['Strategy_Returns'] = self.results['Market_Returns'] * self.results['Signal'].shift(1).replace(2, 0)
        
        # Calculate Cumulative Returns
        self.results['Cum_Market_Returns'] = (1 + self.results['Market_Returns']).cumprod()
        self.results['Cum_Strategy_Returns'] = (1 + self.results['Strategy_Returns']).cumprod()
        
        return self.results

    def calculate_metrics(self):
        # 1. Total Return
        total_ret = self.results['Cum_Strategy_Returns'].iloc[-1] - 1
        
        # 2. Max Drawdown (Biggest peak-to-trough drop)
        rolling_max = self.results['Cum_Strategy_Returns'].cummax()
        drawdown = (self.results['Cum_Strategy_Returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 3. Win Rate
        win_rate = len(self.results[self.results['Strategy_Returns'] > 0]) / \
                   len(self.results[self.results['Strategy_Returns'] != 0])

        print(f"--- Backtest Results ---")
        print(f"Total Strategy Return: {total_ret*100:.2f}%")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['Cum_Market_Returns'], label='Benchmark (Buy & Hold)')
        plt.plot(self.results['Cum_Strategy_Returns'], label='AI Signal Strategy', color='orange')
        plt.title("Strategy vs Market Cumulative Returns")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage for testing
    df = pd.read_csv("data/processed_data.csv")
    from src.model import StockModel

    sm = StockModel()
    X_train, X_test, y_train, y_test = sm.prepare_data(df)
    sm.train(X_train, y_train)
    predictions = sm.evaluate(X_test, y_test)

    # Backtesting
    bt = Backtester(df_test=df.iloc[int(len(df)*0.8):], predictions=predictions)
    bt.run_strategy()
    bt.calculate_metrics()
    bt.plot_results()