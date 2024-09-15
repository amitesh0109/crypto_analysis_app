import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from textblob import TextBlob
import tweepy

# ... (keep the previous imports and functions) ...

def handle_nan_values(data: pd.DataFrame) -> pd.DataFrame:
    # Fill NaN values in 'Daily_Return' and 'Volatility' columns with 0
    data['Daily_Return'] = data['Daily_Return'].fillna(0)
    data['Volatility'] = data['Volatility'].fillna(0)
    
    # For 'Close' and 'Volume', use forward fill method
    data['Close'] = data['Close'].fillna(method='ffill')
    data['Volume'] = data['Volume'].fillna(method='ffill')
    
    return data

def detect_anomalies(data: pd.DataFrame) -> pd.Series:
    # Handle NaN values before anomaly detection
    data_cleaned = handle_nan_values(data)
    
    clf = IsolationForest(contamination=0.01, random_state=42)
    anomalies = clf.fit_predict(data_cleaned[['Close', 'Volume', 'Daily_Return', 'Volatility']])
    return pd.Series(anomalies, index=data.index)

def main():
    st.set_page_config(page_title="AI-Enhanced Crypto Analysis Dashboard", layout="wide")
    st.title("AI-Enhanced Cryptocurrency Analysis Dashboard")

    # Sidebar for user input
    st.sidebar.header("User Input")
    crypto_symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "XRP", "LTC", "ADA"])
    days = st.sidebar.slider("Number of days to analyze", 30, 365, 90)
    short_ma = st.sidebar.slider("Short-term MA window", 5, 50, 20)
    long_ma = st.sidebar.slider("Long-term MA window", 20, 200, 50)
    volatility_window = st.sidebar.slider("Volatility window", 5, 30, 14)

    # Fetch and process data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = fetch_crypto_data(crypto_symbol, start_date, end_date)
    data = calculate_indicators(data, short_ma, long_ma, volatility_window)

    # Handle NaN values
    data = handle_nan_values(data)

    # AI-powered price prediction
    X, y, scaler = prepare_data_for_lstm(data)
    model = train_lstm_model(X, y)
    future_predictions = predict_future_prices(model, X[-1], scaler)

    # Anomaly detection
    anomalies = detect_anomalies(data)
    data['Anomaly'] = anomalies

    # Sentiment analysis
    try:
        sentiment = get_twitter_sentiment(crypto_symbol)
        st.sidebar.metric("Twitter Sentiment", f"{sentiment:.2f}", "-1 (Negative) to 1 (Positive)")
    except Exception as e:
        st.sidebar.warning(f"Unable to fetch Twitter sentiment: {str(e)}")

    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_price_chart(data, crypto_symbol, future_predictions), use_container_width=True)
    with col2:
        st.plotly_chart(plot_returns_distribution(data, crypto_symbol), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_volatility(data, crypto_symbol, volatility_window), use_container_width=True)
    with col4:
        st.plotly_chart(plot_rsi(data, crypto_symbol), use_container_width=True)

    # Anomaly detection plot
    st.subheader("Anomaly Detection")
    fig_anomaly = px.scatter(data, x=data.index, y='Close', color='Anomaly',
                             title=f"{crypto_symbol} Price with Anomalies Highlighted")
    st.plotly_chart(fig_anomaly, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = data['Close'].describe()
    st.dataframe(summary_stats)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data[['Close', 'Volume', 'Daily_Return', 'Volatility', 'RSI']].corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                         title="Correlation Matrix of Key Metrics")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(data.tail())

if __name__ == "__main__":
    main()
