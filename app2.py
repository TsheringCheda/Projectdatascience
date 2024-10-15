import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('bhutan_tourism_data.csv')
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Year', inplace=True)
    return data

# Forecast function
def forecast_tourists(data, year):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)  # Forecasting for the next 5 years
    return forecast

# Streamlit app
st.title("Tourist Arrival Prediction in Bhutan")
st.write("This app predicts the number of tourists arriving in Bhutan based on historical data.")

# Load data
data = load_data()
st.line_chart(data)

# User input for year
year_input = st.number_input("Enter the year for prediction:", min_value=2018, max_value=2030, value=2023)

if st.button("Predict"):
    forecast = forecast_tourists(data['Arrivals'], year_input)
    st.write(f"Predicted tourist arrivals for the year {year_input}: {forecast[year_input - 2023]:.0f}")

    # Plotting the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Arrivals'], label='Historical Data', color='blue')
    plt.axvline(x=pd.to_datetime(f"{year_input}-01-01"), color='red', linestyle='--', label='Prediction Year')
    plt.plot(pd.date_range(start=f"{year_input}-01-01", periods=5, freq='Y'), forecast, label='Forecast', color='orange')
    plt.title('Tourist Arrivals Forecast')
    plt.xlabel('Year')
    plt.ylabel('Number of Tourists')
    plt.legend()
    plt.grid()
    st.pyplot(plt)
