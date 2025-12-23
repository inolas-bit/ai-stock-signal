AI-Driven Stock Signal System 

Project Overview
This repository contains an end-to-end Machine Learning Decision-Support System that analyzes historical stock market data to generate Buy, Sell, or Hold signals. Unlike "black-box" prediction tools, this system uses industry-standard technical indicators and SHAP explainability to provide transparent logic behind every signal.

The Problem
Financial markets are noisy and non-stationary. Retail traders often struggle with emotional bias and "information overload" from too many technical indicators.

The Solution
A systematic pipeline that filters market noise through an XGBoost Classifier, providing traders with high-precision signals and backtested performance metrics to manage risk effectively.

 System Architecture
The project follows a production-ready modular structure:

Plaintext

ai-stock-signal/
│── data/               # Raw and processed OHLCV data
│── notebooks/          # Exploratory Data Analysis (01_eda.ipynb)
│── src/                # Core Logic (Modular Scripts)
│   ├── data_loader.py          # Yahoo Finance API integration
│   ├── feature_engineering.py  # Technical indicator logic (RSI, MACD, etc.)
│   ├── model.py                # XGBoost training & evaluation
│   └── backtester.py           # Strategy simulation engine
│── app.py              # Interactive Streamlit Dashboard
│── requirements.txt    # Production dependencies
│── .gitignore          # Prevents uploading large data/cache files
└── README.md           # Project documentation

Key Features
Dynamic Data Ingestion: Fetches real-time market data (AAPL, TSLA, RELIANCE, etc.) via the yfinance API.

Elite Feature Engineering: Implements 10+ indicators including RSI (Momentum), MACD (Trend), ATR (Volatility), and EMA Crosses.

Look-Ahead Bias Protection: All labels are generated using forward-looking returns, while features are strictly historical to ensure the model doesn't "cheat."

Backtesting Engine: Simulates the strategy against a "Buy & Hold" benchmark, calculating Max Drawdown and Win Rate.

Model Explainability: Uses SHAP (SHapley Additive exPlanations) to visualize which features (e.g., a specific Volume spike) influenced a "Buy" signal.

Technical Stack
Data: Pandas, NumPy, YFinance

ML: XGBoost, Scikit-Learn

Technical Analysis: Pandas-TA

Interpretability: SHAP

Dashboard: Streamlit, Plotly

Setup & Installation
Clone the Repository

Bash

git clone https://github.com/your-username/ai-stock-signal.git
cd ai-stock-signal
Install Requirements

Bash

pip install -r requirements.txt
Run Locally

Bash

streamlit run app.py

Performance & Results
Model Goal: Maximize Precision for "Buy" signals to ensure a high probability of success for traders.

Insight: The model demonstrated that combining Volatility (ATR) with Trend (EMA) significantly reduced false signals during market regime shifts.

Interview Discussion Points
Why XGBoost? "It handles non-linear interactions between indicators better than traditional linear models and is robust to the outliers common in financial data."

Handling Non-Stationarity: "I avoided using raw price as a feature. Instead, I used stationary features like Percentage Returns and RSI to help the model generalize across different price levels."

Risk Management: "The dashboard focuses on Max Drawdown, teaching the user that avoiding big losses is more important than chasing maximum profit."


The Problem
Financial markets are highly non-stationary and noisy. Retail traders often face "analysis paralysis" due to conflicting technical indicators and emotional bias.

The Solution
A systematic, modular pipeline that automates data ingestion, processes raw OHLCV data into meaningful features, and provides a clear, backtested signal via an interactive dashboard.

Methodology & Results
Labeling: Signals are generated based on a 5-day future return threshold. If price increases >2%, it's a Buy; if it drops >2%, it's a Sell.

Model Choice: XGBoost was selected for its ability to handle non-linear interactions between noisy features.

Metric focus: Prioritized Precision for "Buy" signals to minimize false entries in a trading scenario.

🤝 Let's Connect
Saloni Kumari Singh 👩‍💻 GitHub: inolas-bit

🔗 LinkedIn: saloni-singh1329
