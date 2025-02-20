import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Set page title and layout
st.set_page_config(page_title="Hedging Tool", layout="wide")
st.title("🛢️ Wildwood Trading Company Hedging Tool")
st.write("Enter your details below to get started!")

# Sidebar Inputs
with st.sidebar:
    st.header("Inputs")
    production = st.number_input("Monthly Production (barrels)", min_value=1000, value=10000)
    cost = st.number_input("Breakeven Cost ($/barrel)", min_value=30, value=50)
    risk = st.selectbox("Risk Tolerance", ["High", "Medium", "Low"])
    years = st.slider("Hedging Duration (years)", min_value=1, max_value=5, value=1)

# Fetch live oil prices
api_key = "IJPZKPCXDAYWYW59"  # Replace with your Alpha Vantage API key
url = f"https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey={api_key}"
response = requests.get(url).json()
current_price = float(response["data"][0]["value"])

# Simulate prices over the hedging duration
def simulate_prices(current_price, months=12, n_simulations=1000):
    prices = np.zeros((n_simulations, months))
    prices[:, 0] = current_price
    for t in range(1, months):
        drift = 0  # Risk-neutral assumption
        volatility = 0.3  # Annualized volatility
        shock = np.random.normal(0, 1, n_simulations)
        prices[:, t] = prices[:, t-1] * np.exp((drift - 0.5*volatility**2)*(1/12) + volatility*np.sqrt(1/12)*shock)
    return prices

months = years * 12
prices = simulate_prices(current_price, months=months)

# Calculate hedge ratio
if risk == "High":
    hedge_ratio = 0.6
elif risk == "Medium":
    hedge_ratio = 0.4
else:
    hedge_ratio = 0.2

# Calculate hedged cash flows
hedged_cash_flows = hedge_ratio * (75 - cost) * production + (1 - hedge_ratio) * (prices - cost) * production

# Calculate risk metrics
def calculate_risk(cash_flows):
    var = np.percentile(cash_flows, 5)  # 5th percentile
    cvar = cash_flows[cash_flows <= var].mean()  # Average of worst 5%
    return var, cvar

var, cvar = calculate_risk(hedged_cash_flows)

# Display results
st.subheader("Results")
st.write(f"Current Brent crude price: **${current_price:.2f}**")
if current_price > cost:
    st.success(f"Hedge **{hedge_ratio*100:.0f}%** to lock in profits.")
else:
    st.warning(f"Hedge **{hedge_ratio*100:.0f}%** to minimize losses.")
st.write(f"Value-at-Risk (5%): **${var:,.2f}**")
st.write(f"Conditional Value-at-Risk: **${cvar:,.2f}**")

# Create a chart
st.subheader("Price Scenarios")
fig, ax = plt.subplots()
ax.plot(prices.mean(axis=0), label="Average Price Scenario")
ax.axhline(cost, color="red", linestyle="--", label="Breakeven Cost")
ax.set_xlabel("Time (Months)")
ax.set_ylabel("Price ($)")
ax.set_title("Hedging Strategy Over Time")
ax.legend()
st.pyplot(fig)

# Save results to CSV
results = {
    "Production (barrels)": [production],
    "Breakeven Cost ($/barrel)": [cost],
    "Risk Tolerance": [risk],
    "Hedge Ratio": [hedge_ratio],
    "Current Price ($)": [current_price],
    "Value-at-Risk (5%)": [var],
    "Conditional Value-at-Risk": [cvar],
}
df = pd.DataFrame(results)
df.to_csv("hedging_recommendation.csv", index=False)
st.download_button(
    label="Download Results",
    data=df.to_csv(),
    file_name="hedging_recommendation.csv",
    mime="text/csv",
)