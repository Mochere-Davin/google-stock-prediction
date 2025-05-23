# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import yfinance as yf
from datetime import datetime, timedelta
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Google Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data
@st.cache_data
def load_data(start_date, end_date):
    try:
        # Convert date_input objects to strings in the format yfinance expects
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"Attempting to download data for GOOG from {start_str} to {end_str}")
        
        data = yf.download('GOOG', start=start_str, end=end_str, progress=False)
        if data.empty:
            logger.error("No data retrieved from Yahoo Finance for the given date range")
            st.error("Error loading data: No data retrieved from Yahoo Finance. "
                     "Please try a different date range (e.g., 2015-01-01 to 2020-12-31).")
            # Fallback: Try a default range if the requested range fails
            default_start = datetime(2015, 1, 1).strftime('%Y-%m-%d')
            default_end = datetime(2020, 12, 31).strftime('%Y-%m-%d')
            logger.info(f"Falling back to default range: {default_start} to {default_end}")
            data = yf.download('GOOG', start=default_start, end=default_end, progress=False)
            if data.empty:
                logger.error("Fallback data retrieval also failed")
                st.error("Fallback data retrieval failed. Check your internet connection or Yahoo Finance availability.")
                return None
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        # Feature engineering
        data['lag1_close'] = data['Close'].shift(1)
        data['lag2_close'] = data['Close'].shift(2)
        data['sma10'] = data['Close'].rolling(window=10).mean()
        data['return'] = data['Close'].pct_change()
        data['volume_lag1'] = data['Volume'].shift(1)
        data = data.dropna()
        logger.info("Data loaded and preprocessed successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

# Train XGBoost model
@st.cache_resource
def train_xgboost(data):
    try:
        features = ['lag1_close', 'lag2_close', 'sma10', 'volume_lag1', 'return']
        target = 'Close'
        X = data[features]
        y = data[target]
        train_size = int(0.8 * len(data))
        
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        test_dates = data['Date'][train_size:]
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'model': model,
            'scaler': scaler,
            'y_pred': y_pred,
            'y_test': y_test,
            'test_dates': test_dates
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"Error training model: {e}")
        return None

# Prediction function
def predict_close_price(model, scaler, lag1_close, lag2_close, sma10, volume_lag1, return_val):
    try:
        input_data = pd.DataFrame({
            'lag1_close': [lag1_close],
            'lag2_close': [lag2_close],
            'sma10': [sma10],
            'volume_lag1': [volume_lag1],
            'return': [return_val]
        })
        input_data_scaled = scaler.transform(input_data)
        return model.predict(input_data_scaled)[0]
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        st.error(f"Error making prediction: {e}")
        return None

# Sidebar
st.sidebar.header("Prediction Settings")
start_date = st.sidebar.date_input("Start Date", value=datetime(2016, 6, 14), min_value=datetime(2004, 8, 19))
end_date = st.sidebar.date_input("End Date", value=datetime(2021, 6, 11), max_value=datetime.now())

# Main page
st.title("📈 Google Stock Price Prediction")
st.markdown("### Forecasting Google Stock Prices Using XGBoost")

# Load data
data = load_data(start_date, end_date)
if data is None:
    st.stop()

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

# EDA
st.header("Data Exploration")
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
ax.set_title('GOOG Closing Price Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price ($)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Train model
st.header("Model Training")
if st.button("Train XGBoost Model"):
    with st.spinner("Training model..."):
        metrics = train_xgboost(data)
        if metrics:
            st.session_state['metrics'] = metrics
            st.success("Model trained successfully!")
        else:
            st.error("Failed to train model. Please check logs.")

# Display metrics
if 'metrics' in st.session_state:
    metrics = st.session_state['metrics']
    st.subheader("Model Performance")
    st.write(f"**XGBoost** - MAE: ${metrics['MAE']:.2f}, RMSE: ${metrics['RMSE']:.2f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics['test_dates'], metrics['y_test'], label='Actual', color='blue')
    ax.plot(metrics['test_dates'], metrics['y_pred'], label='Predicted', color='red')
    ax.set_title('XGBoost Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price ($)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Prediction interface
st.header("Make a Prediction")
st.markdown("Enter recent stock metrics to predict the next day's closing price.")
col1, col2, col3 = st.columns(3)
with col1:
    lag1_close = st.number_input("Previous Day's Close", min_value=0.0, value=1350.34)
    lag2_close = st.number_input("Two Days Ago Close", min_value=0.0, value=1345.12)
with col2:
    sma10 = st.number_input("10-Day SMA", min_value=0.0, value=1342.0)
    volume_lag1 = st.number_input("Previous Day's Volume", min_value=0.0, value=1500000.0)
with col3:
    return_val = st.number_input("Daily Return (e.g., 0.0039 for 0.39%)", value=0.0039)

if st.button("Predict"):
    if 'metrics' not in st.session_state:
        st.error("Please train the model first!")
    else:
        predicted_price = predict_close_price(
            st.session_state['metrics']['model'],
            st.session_state['metrics']['scaler'],
            lag1_close, lag2_close, sma10, volume_lag1, return_val
        )
        if predicted_price is not None:
            st.write(f"**Predicted Close Price**: ${predicted_price:.2f}")

# Download data
st.header("Download Data")
csv = data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="goog_stock_data.csv">Download CSV File</a>'
st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Prepared by Davin Nyanchama**  
**Contact**: 0708260067 | davinmochere456@gmail.com  
**Data Source**: Yahoo Finance (GOOG)  
**Model**: XGBoost
""")
