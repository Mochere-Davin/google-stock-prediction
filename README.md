# Google Stock Price Prediction

A Streamlit-based web application for forecasting Google (GOOG) stock prices using an XGBoost model. The app fetches real-time stock data from Yahoo Finance, trains a predictive model, and allows users to input recent stock metrics to predict the next day's closing price.

## Features
- **Real-Time Data**: Fetches historical and recent GOOG stock data using `yfinance`.
- **XGBoost Model**: Trains an XGBoost regressor with engineered features (lagged prices, moving averages, returns, volume).
- **User-Friendly Interface**: Select date ranges, train models, input metrics, and view predictions.
- **Data Visualization**: Displays closing price trends and actual vs. predicted prices.
- **Data Export**: Download processed stock data as a CSV.

## Project Structure