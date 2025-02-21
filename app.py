import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.techindicators import TechIndicators

# Set page title and layout
st.set_page_config(page_title="Wildwood Hedging", layout="wide")
st.title("Advanced Oil & Gas Hedging Tool")

# ===========================================
# 1. Fetch Historical Data (Alpha Vantage)
# ===========================================
@st.cache_data
def get_historical_data(commodity, api_key):
    symbol_map = {
        "Brent": "BRENT",
        "WTI": "WTI",
        "Natural Gas": "NATGAS"
    }
    symbol = symbol_map.get(commodity, "BRENT")
    
    ti = TechIndicators(key=api_key)
    data, _ = ti.get_bbands(symbol=symbol, interval='daily', time_period=60)
    df = pd.DataFrame(data).iloc[::-1]
    df['close'] = (df['Real Upper Band'].astype(float) + df['Real Lower Band'].astype(float)) / 2
    return df['close'].values

# ===========================================
# 2. Preprocess Data for LSTM
# ===========================================
def prepare_data(data, n_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# ===========================================
# 3. Build LSTM Model
# ===========================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ===========================================
# 4. Streamlit App Logic
# ===========================================
# Sidebar Inputs
with st.sidebar:
    st.header("Inputs")
    commodity = st.selectbox("Commodity", ["Brent", "WTI", "Natural Gas"])
    production = st.number_input("Monthly Production", 1000, 1000000, 10000)
    cost = st.number_input("Breakeven Cost ($/unit)", 30, 200, 50)
    risk = st.selectbox("Risk Tolerance", ["High", "Medium", "Low"])
    years = st.slider("Hedging Duration (years)", 1, 5, 1)
    carbon_footprint = st.number_input("Carbon Footprint (tons CO2/month)", 100, 10000, 1000)

# Fetch live price
api_key = st.secrets["IJPZKPCXDAYWYW59"]  # Replace with your key
current_price = get_historical_data(commodity, api_key)[-1]
st.write(f"Current {commodity} price: **${current_price:.2f}**")

# AI Forecast Button
if st.button("Generate AI Forecast"):
    with st.spinner("Training AI model..."):
        # Get data
        historical_data = get_historical_data(commodity, api_key)
        n_steps = 60
        
        # Prepare data
        X, y, scaler = prepare_data(historical_data, n_steps)
        X = X.reshape(X.shape[0], n_steps, 1)
        
        # Train model
        model = build_lstm_model((n_steps, 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Forecast
        last_sequence = historical_data[-n_steps:]
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        forecast = model.predict(last_sequence_scaled.reshape(1, n_steps, 1))
        forecast = scaler.inverse_transform(forecast)
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(historical_data[-100:], label="Historical Prices")
        ax.scatter(len(historical_data), forecast[0][0], color='red', label="Forecast")
        ax.set_title(f"{commodity} Price Forecast")
        ax.legend()
        st.pyplot(fig)

# Rest of your app (hedging logic, ESG metrics, etc.) goes here...
