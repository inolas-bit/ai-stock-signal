import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
import jobstate  # To save the model

class StockModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='multi:softprob', # Multi-class classification
            random_state=42
        )
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        """
        Splits data into features (X) and target (y).
        Ensures we don't use 'Future_Return' as a feature.
        """
        # Features are everything except the target, future returns, and dates
        features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'EMA_20', 
                    'EMA_50', 'ATR', 'Vol_Pct_Change', 'Daily_Return', 'Dist_EMA_20']
        
        X = df[features]
        y = df['Target']
        
        # Time-series split: No shuffling! (Important for finance)
        # We take the first 80% for training, last 20% for testing
        split_index = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print("\n--- Model Performance Report ---")
        print(classification_report(y_test, predictions, target_names=['Hold', 'Buy', 'Sell']))
        
        # Precision is key: We want to be sure when we say 'Buy'
        prec = precision_score(y_test, predictions, average=None)
        print(f"Precision for Buy Signals: {prec[1]:.2f}")
        
        return predictions

    def save_model(self, path="models/xgb_stock_model.json"):
        self.model.save_model(path)

if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")
    sm = StockModel()
    X_train, X_test, y_train, y_test = sm.prepare_data(df)
    sm.train(X_train, y_train)
    sm.evaluate(X_test, y_test)


def explain_model(model, X_train, X_test): # Added X_test here
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    