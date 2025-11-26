"""
Configuration file for Retail Sales Forecasting Project
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
TRAIN_FILE = RAW_DATA_DIR / 'train.csv'
TEST_FILE = RAW_DATA_DIR / 'test.csv'
STORES_FILE = RAW_DATA_DIR / 'stores.csv'
HOLIDAYS_FILE = RAW_DATA_DIR / 'holidays_events.csv'
OIL_FILE = RAW_DATA_DIR / 'oil.csv'
TRANSACTIONS_FILE = RAW_DATA_DIR / 'transactions.csv'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Forecasting parameters
FORECAST_HORIZON = 16  # Days to forecast ahead
MIN_TRAIN_DATE = '2013-01-01'
MAX_TRAIN_DATE = '2017-08-15'

# Feature engineering parameters
LAG_DAYS = [1, 7, 14, 28]  # Lag features to create
ROLLING_WINDOWS = [7, 14, 28]  # Rolling window sizes
TARGET_COL = 'sales'
DATE_COL = 'date'

# Model training parameters
MODELS_TO_TRAIN = ['baseline', 'prophet', 'lightgbm', 'xgboost', 'lstm', 'tft']

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 1000,
    'random_state': RANDOM_STATE
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist'
}

# LSTM parameters
LSTM_PARAMS = {
    'sequence_length': 60,
    'lstm_units': 64,
    'dropout': 0.2,
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001
}

# TFT parameters
TFT_PARAMS = {
    'max_epochs': 20,
    'batch_size': 128,
    'hidden_size': 32,
    'lstm_layers': 2,
    'attention_head_size': 4,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'gradient_clip_val': 0.1
}

# Evaluation metrics
METRICS = ['rmse', 'mae', 'mape', 'wape', 'smape']

# Top product families by revenue (for focused analysis if needed)
TOP_FAMILIES = [
    'GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY',
    'BREAD/BAKERY', 'POULTRY', 'MEATS', 'PERSONAL CARE', 'DELI'
]

# Store types
STORE_TYPES = ['A', 'B', 'C', 'D', 'E']

# Categorical features
CATEGORICAL_FEATURES = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster',
    'day_of_week', 'month', 'year', 'is_weekend', 'is_month_start',
    'is_month_end', 'holiday_type', 'locale'
]

# Numerical features (will be populated during feature engineering)
NUMERICAL_FEATURES = []

print(f"Configuration loaded. Base directory: {BASE_DIR}")
