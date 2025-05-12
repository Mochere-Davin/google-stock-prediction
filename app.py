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
            default_end = datetime(2020, 12, 31
