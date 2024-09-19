import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('temperatures.csv')

# Streamlit UI Setup
st.title("Temperature Analysis and Forecasting")

# Display Summary Statistics
st.header("Summary Statistics")
st.write(data.describe())

# Line plot of annual temperatures
st.header("Annual Temperature Over Years")
plt.figure(figsize=(14,7))
plt.plot(data['YEAR'], data['ANNUAL'])
plt.xlabel('Year')
plt.ylabel('Annual Temperature')
st.pyplot(plt)

# Heatmap of monthly temperatures
st.header("Monthly Temperatures Heatmap")
plt.figure(figsize=(14, 7))
sns.heatmap(data.iloc[:, 1:13].transpose(), cmap='coolwarm', cbar_kws={'label': 'Temperature'})
st.pyplot(plt)

# Time Series Decomposition
st.header("Time Series Decomposition")
data.set_index('YEAR', inplace=True)
result = seasonal_decompose(data['ANNUAL'], model='multiplicative', period=12)
st.pyplot(result.plot())

# Check for stationarity
result = adfuller(data['ANNUAL'])
st.write(f"ADF Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")

# Plot ACF and PACF
st.header("Autocorrelation and Partial Autocorrelation")
plot_acf(data['ANNUAL'], lags=20)
plot_pacf(data['ANNUAL'], lags=20)
st.pyplot(plt)

# ARIMA Model for Forecasting
st.header("ARIMA Model Forecast")
arima_model = ARIMA(data['ANNUAL'], order=(5, 1, 0))
model_fit = arima_model.fit()
st.write(model_fit.summary())

# ARIMA Forecast Plot
forecast = model_fit.forecast(steps=10)
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['ANNUAL'], label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=11, freq='Y')[1:], forecast, label='Forecast', color='red')
st.pyplot(plt)

# SARIMA Model for Seasonality
st.header("SARIMA Model Forecast")
model_sarima = SARIMAX(data['ANNUAL'], order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
model_fit_sarima = model_sarima.fit()
st.write(model_fit_sarima.summary())

# SARIMA Forecast Plot
forecast_sarima = model_fit_sarima.forecast(steps=10)
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['ANNUAL'], label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=11, freq='Y')[1:], forecast_sarima, label='Forecast', color='red')
st.pyplot(plt)

# Prophet Model for Forecasting
st.header("Prophet Model Forecast")
prophet_data = data.reset_index().rename(columns={'YEAR': 'ds', 'ANNUAL': 'y'})
model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(prophet_data)

# Prophet Forecast Plot
future = model_prophet.make_future_dataframe(periods=10, freq='Y')
forecast_prophet = model_prophet.predict(future)
fig = model_prophet.plot(forecast_prophet)
st.pyplot(fig)

# LSTM for Time Series Prediction
st.header("LSTM Model Forecast")
data_values = data['ANNUAL'].values.reshape((len(data['ANNUAL']), 1))
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_values)

X, y = [], []
for i in range(5, len(scaled_data)):
    X.append(scaled_data[i-5:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=50, verbose=1)

# LSTM Forecast Plot
forecast_input = scaled_data[-5:]
forecast_output = []
for _ in range(10):
    forecast_input_reshaped = forecast_input.reshape((1, 5, 1))
    pred = model_lstm.predict(forecast_input_reshaped)
    forecast_output.append(pred[0, 0])
    forecast_input = np.append(forecast_input[1:], pred[0, 0])
forecast_output = scaler.inverse_transform(np.array(forecast_output).reshape(-1, 1))

plt.figure(figsize=(14, 7))
plt.plot(data.index, data_values, label='Historical Data')
forecast_dates = pd.date_range(start=data.index[-1], periods=11, freq='Y')[1:]
plt.plot(forecast_dates, forecast_output, label='Forecast', color='red')
st.pyplot(plt)

# Display Model Comparison
st.header("Model Comparison")
X = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
y = data['ANNUAL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Random Forest Model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# KNN Model
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# SVR Model
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train, y_train)
y_pred_svr = model_svr.predict(X_test)

# Plotting Model Comparisons
results = {
    'Linear Regression': y_pred_lr,
    'Random Forest': y_pred_rf,
    'KNN': y_pred_knn,
    'SVR': y_pred_svr
}

plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual', color='black')
for name, y_pred in results.items():
    plt.plot(y_pred, label=name)
plt.xlabel("Sample")
plt.ylabel("Annual Temperature")
plt.title("Model Comparison: Actual vs Predicted Annual Temperature")
plt.legend()
st.pyplot(plt)