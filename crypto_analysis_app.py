import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# Function to fetch cryptocurrency data
def fetch_crypto_data(symbol, start_date, end_date):
    data = yf.download(f"{symbol}-USD", start=start_date, end=end_date)
    return data

# Function to calculate moving averages
def calculate_moving_averages(data, short_window, long_window):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    return data

# Function to calculate daily returns
def calculate_daily_returns(data):
    data['Daily_Return'] = data['Close'].pct_change()
    return data

# Function to calculate volatility
def calculate_volatility(data, window):
    data['Volatility'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(window)
    return data

# Streamlit app
def main():
    st.title("Cryptocurrency Analysis Dashboard")

    # Sidebar for user input
    st.sidebar.header("User Input")
    crypto_symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "XRP", "LTC", "ADA"])
    days = st.sidebar.slider("Number of days to analyze", 30, 365, 90)
    short_ma = st.sidebar.slider("Short-term MA window", 5, 50, 20)
    long_ma = st.sidebar.slider("Long-term MA window", 20, 200, 50)
    volatility_window = st.sidebar.slider("Volatility window", 5, 30, 14)

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = fetch_crypto_data(crypto_symbol, start_date, end_date)

    # Perform analysis
    data = calculate_moving_averages(data, short_ma, long_ma)
    data = calculate_daily_returns(data)
    data = calculate_volatility(data, volatility_window)

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(data.tail())

    # Price chart with moving averages
    st.subheader("Price Chart with Moving Averages")
    fig_price = px.line(data, x=data.index, y=['Close', 'SMA_short', 'SMA_long'],
                        labels={'value': 'Price', 'variable': 'Metric'},
                        title=f"{crypto_symbol} Price and Moving Averages")
    st.plotly_chart(fig_price)

    # Daily returns histogram
    st.subheader("Daily Returns Distribution")
    fig_returns = px.histogram(data, x='Daily_Return', nbins=50,
                               title=f"{crypto_symbol} Daily Returns Distribution")
    st.plotly_chart(fig_returns)

    # Volatility chart
    st.subheader("Historical Volatility")
    fig_volatility = px.line(data, x=data.index, y='Volatility',
                             title=f"{crypto_symbol} {volatility_window}-Day Historical Volatility")
    st.plotly_chart(fig_volatility)

    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = data['Close'].describe()
    st.dataframe(summary_stats)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data[['Close', 'Volume', 'Daily_Return', 'Volatility']].corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                         title="Correlation Matrix of Key Metrics")
    st.plotly_chart(fig_corr)

if __name__ == "__main__":
    main()
