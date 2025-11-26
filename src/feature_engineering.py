"""
Feature Engineering for Retail Sales Forecasting
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import config

def create_temporal_features(df):
    """Create time-based features"""
    print("Creating temporal features...")
    
    df['day_of_week'] = df[config.DATE_COL].dt.dayofweek.astype('int8')
    df['day_of_month'] = df[config.DATE_COL].dt.day.astype('int8')
    df['week_of_year'] = df[config.DATE_COL].dt.isocalendar().week.astype('int8')
    df['month'] = df[config.DATE_COL].dt.month.astype('int8')
    df['quarter'] = df[config.DATE_COL].dt.quarter.astype('int8')
    df['year'] = df[config.DATE_COL].dt.year.astype('int16')
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['is_month_start'] = df[config.DATE_COL].dt.is_month_start.astype('int8')
    df['is_month_end'] = df[config.DATE_COL].dt.is_month_end.astype('int8')
    df['is_quarter_start'] = df[config.DATE_COL].dt.is_quarter_start.astype('int8')
    df['is_quarter_end'] = df[config.DATE_COL].dt.is_quarter_end.astype('int8')
    df['is_year_start'] = df[config.DATE_COL].dt.is_year_start.astype('int8')
    df['is_year_end'] = df[config.DATE_COL].dt.is_year_end.astype('int8')
    
    # Day of year (useful for yearly seasonality)
    df['day_of_year'] = df[config.DATE_COL].dt.dayofyear.astype('int16')
    
    print(f"Added {14} temporal features")
    return df

def create_lag_features(df, lags=None):
    """Create lag features for sales"""
    if lags is None:
        lags = config.LAG_DAYS
    
    print(f"Creating lag features for lags: {lags}...")
    
    # Use family_code which is already created in engineer_all_features
    df = df.sort_values(['store_nbr', 'family_code', config.DATE_COL])
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family_code'])[config.TARGET_COL].shift(lag).astype('float32')
        df[f'onpromotion_lag_{lag}'] = df.groupby(['store_nbr', 'family_code'])['onpromotion'].shift(lag).astype('float32')
    
    print(f"Added {len(lags) * 2} lag features")
    return df

def create_rolling_features(df, windows=None):
    """Create rolling window statistics"""
    if windows is None:
        windows = config.ROLLING_WINDOWS
    
    print(f"Creating rolling features for windows: {windows}...")
    
    # family_code is already created in engineer_all_features
    df = df.sort_values(['store_nbr', 'family_code', config.DATE_COL])
    
    for window in windows:
        # Rolling mean
        df[f'sales_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family_code'])[config.TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        ).astype('float32')
        
        # Rolling std
        df[f'sales_rolling_std_{window}'] = df.groupby(['store_nbr', 'family_code'])[config.TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        ).astype('float32')
        
        # Rolling min
        df[f'sales_rolling_min_{window}'] = df.groupby(['store_nbr', 'family_code'])[config.TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        ).astype('float32')
        
        # Rolling max
        df[f'sales_rolling_max_{window}'] = df.groupby(['store_nbr', 'family_code'])[config.TARGET_COL].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        ).astype('float32')
        
        # Promotion rolling mean
        df[f'onpromotion_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family_code'])['onpromotion'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        ).astype('float32')
    
    print(f"Added {len(windows) * 5} rolling features")
    return df

def create_oil_features(df):
    """Create oil price features"""
    print("Creating oil price features...")
    
    # Oil price lags
    df = df.sort_values(config.DATE_COL)
    df['oil_lag_7'] = df['dcoilwtico'].shift(7).astype('float32')
    df['oil_lag_14'] = df['dcoilwtico'].shift(14).astype('float32')
    
    # Oil price rolling statistics
    df['oil_rolling_mean_7'] = df['dcoilwtico'].shift(1).rolling(window=7, min_periods=1).mean().astype('float32')
    df['oil_rolling_std_7'] = df['dcoilwtico'].shift(1).rolling(window=7, min_periods=1).std().astype('float32')
    df['oil_rolling_mean_28'] = df['dcoilwtico'].shift(1).rolling(window=28, min_periods=1).mean().astype('float32')
    
    # Oil price change
    df['oil_change'] = (df['dcoilwtico'] - df['dcoilwtico'].shift(1)).astype('float32')
    df['oil_change_pct'] = (df['oil_change'] / df['dcoilwtico'].shift(1) * 100).astype('float32')
    
    print(f"Added 7 oil price features")
    return df

def create_transaction_features(df):
    """Create transaction-based features"""
    print("Creating transaction features...")
    
    # Transaction lags
    df = df.sort_values(['store_nbr', config.DATE_COL])
    df['transactions_lag_7'] = df.groupby('store_nbr')['transactions'].shift(7).astype('float32')
    df['transactions_lag_14'] = df.groupby('store_nbr')['transactions'].shift(14).astype('float32')
    
    # Transaction rolling mean
    df['transactions_rolling_mean_7'] = df.groupby('store_nbr')['transactions'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    ).astype('float32')
    
    # Transaction change
    df['transactions_change'] = df.groupby('store_nbr')['transactions'].diff().astype('float32')
    
    print(f"Added 4 transaction features")
    return df

def create_holiday_features(df):
    """Create holiday-related features"""
    print("Creating holiday features...")
    
    # Already have is_holiday from data loader
    # Add days to/from nearest holiday
    df = df.sort_values(config.DATE_COL)
    
    # This is a simplified version - in production, you'd calculate actual days to/from holidays
    # For now, we'll use the holiday indicators we already have
    
    # Keep holiday_type as string, don't convert to category
    if 'holiday_type' in df.columns:
        df['holiday_type'] = df['holiday_type'].fillna('None').astype(str)
    
    print(f"Holiday features processed")
    return df

def create_promotion_features(df):
    """Create promotion-related features"""
    print("Creating promotion features...")
    
    # family_code is already created in engineer_all_features
    df = df.sort_values(['store_nbr', 'family_code', config.DATE_COL])
    
    # Promotion ratio (items on promotion)
    df['promotion_ratio'] = df['onpromotion'].astype('float32')
    
    # Promotion change (momentum)
    df['promotion_change'] = df.groupby(['store_nbr', 'family_code'])['onpromotion'].diff().astype('float32')
    
    # Days since promotion started/ended (simplified)
    df['promotion_active'] = (df['onpromotion'] > 0).astype('int8')
    
    print(f"Added 3 promotion features")
    return df

def create_aggregated_features(df):
    """Create aggregated features at different levels"""
    print("Creating aggregated features...")
    
    # Store-level daily sales using transform (avoids merge issues)
    df['store_daily_sales'] = df.groupby(['store_nbr', config.DATE_COL])[config.TARGET_COL].transform('sum').astype('float32')
    
    # Family-level daily sales using transform with family_code (already created)
    df['family_daily_sales'] = df.groupby(['family_code', config.DATE_COL])[config.TARGET_COL].transform('sum').astype('float32')
    
    print(f"Added 2 aggregated features")
    return df

def create_interaction_features(df):
    """Create interaction features"""
    print("Creating interaction features...")
    
    # Store type × family (keep as string, don't convert to category)
    df['store_type_family'] = (df['type'].astype(str) + '_' + df['family'].astype(str))
    
    # Cluster × family (keep as string, don't convert to category)
    df['cluster_family'] = (df['cluster'].astype(str) + '_' + df['family'].astype(str))
    
    print(f"Added 2 interaction features")
    return df

def engineer_all_features(df):
    """Apply all feature engineering steps"""
    print("\n" + "="*50)
    print("Starting Feature Engineering")
    print("="*50)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # Create family_code using simple mapping (avoid pd.Categorical completely)
    unique_families = df['family'].unique()
    family_mapping = {family: idx for idx, family in enumerate(unique_families)}
    df['family_code'] = df['family'].map(family_mapping).astype('int16')
    print(f"Created family_code with {len(unique_families)} unique families")
    
    # Create all features
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_oil_features(df)
    df = create_transaction_features(df)
    df = create_holiday_features(df)
    df = create_promotion_features(df)
    df = create_aggregated_features(df)
    df = create_interaction_features(df)
    
    final_shape = df.shape
    print("\n" + "="*50)
    print(f"Feature Engineering Complete!")
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Features added: {final_shape[1] - initial_shape[1]}")
    print("="*50)
    
    return df

def prepare_for_modeling(df, is_train=True):
    """Prepare data for modeling"""
    print("\nPreparing data for modeling...")
    
    if is_train:
        # Remove rows with NaN in target
        initial_len = len(df)
        df = df.dropna(subset=[config.TARGET_COL])
        print(f"Removed {initial_len - len(df)} rows with missing target values")
    
    # Fill remaining NaNs in features with 0 (for lag/rolling features at the start)
    feature_cols = [col for col in df.columns if col not in [config.DATE_COL, config.TARGET_COL, 'id']]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    # Test feature engineering
    import data_loader
    
    print("Loading data...")
    df = data_loader.load_and_merge_all()
    
    print("\nEngineering features...")
    df_features = engineer_all_features(df)
    
    print("\nPreparing for modeling...")
    df_final = prepare_for_modeling(df_features, is_train=True)
    
    print("\n" + "="*50)
    print("Feature Engineering Test Complete!")
    print("="*50)
    print(f"\nFinal columns ({len(df_final.columns)}):")
    print(df_final.columns.tolist())
    print(f"\nSample data:")
    print(df_final.head())
    print(f"\nData types:")
    print(df_final.dtypes)
