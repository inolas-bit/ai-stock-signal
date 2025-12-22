def create_labels(df, horizon=5, threshold=0.02):
    """
    Creates Buy(1), Hold(0), Sell(2) labels based on future returns.
    """
    # Calculate returns over the next 'n' days
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    def labeling_logic(ret):
        if ret > threshold:
            return 1 # Buy
        elif ret < -threshold:
            return 2 # Sell
        else:
            return 0 # Hold

    df['Target'] = df['Future_Return'].apply(labeling_logic)
    
    # Drop rows where we don't have future data (the last 'horizon' rows)
    df.dropna(inplace=True)
    return df
