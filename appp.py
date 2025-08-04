# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

st.set_page_config(layout="wide")
st.title("📈 Live AI Trading Bot with LSTM")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")
days = st.sidebar.slider("Days of historical data to use", min_value=60, max_value=365*2, value=180)
epochs = st.sidebar.slider("Training epochs", 5, 50, 10)

# Load live data
@st.cache_data
def load_data(ticker, days):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']].dropna()

df = load_data(ticker, days)
st.line_chart(df, use_container_width=True)

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_dataset(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_dataset(scaled_data, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
with st.spinner("Training model..."):
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

# Predict
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot prediction
st.subheader("📊 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(real_prices, label="Actual")
ax1.plot(predicted_prices, label="Predicted")
ax1.legend()
st.pyplot(fig1)

# Simulate trades
balance = 100000
stock = 0
trades = []

for i in range(1, len(predicted_prices)):
    if predicted_prices[i] > real_prices[i] and balance >= real_prices[i]:
        stock = balance // real_prices[i]
        balance -= stock * real_prices[i]
        trades.append((i, 'BUY', stock, real_prices[i][0]))
    elif predicted_prices[i] < real_prices[i] and stock > 0:
        balance += stock * real_prices[i]
        trades.append((i, 'SELL', stock, real_prices[i][0]))
        stock = 0

final_value = balance + stock * real_prices[-1]
profit = final_value - 100000

st.subheader("📈 Strategy Summary")
st.write(f"**Final balance:** ₹{final_value[0]:,.2f}")
st.write(f"**Profit:** ₹{profit[0]:,.2f}")
st.write(f"**Trades executed:** {len(trades)}")

# Show trade signals
buy_signals = [i for i, action, _, _ in trades if action == 'BUY']
sell_signals = [i for i, action, _, _ in trades if action == 'SELL']

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(real_prices, label="Price", alpha=0.6)
ax2.scatter(buy_signals, [real_prices[i][0] for i in buy_signals], color='green', marker='^', label='Buy')
ax2.scatter(sell_signals, [real_prices[i][0] for i in sell_signals], color='red', marker='v', label='Sell')
ax2.legend()
st.subheader("📍 Buy & Sell Points")
st.pyplot(fig2)
