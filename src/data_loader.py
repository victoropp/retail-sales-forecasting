"""
Data loading utilities for Retail Sales Forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path
import config

def load_train_data():
    """Load training data"""
    print("Loading training data...")
    df = pd.read_csv(config.TRAIN_FILE, parse_dates=[config.DATE_COL])
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Optimize data types - use string instead of category to avoid categorical validation errors
    df['store_nbr'] = df['store_nbr'].astype('int16')
    df['family'] = df['family'].astype(str)  # Changed from category to string
    df['sales'] = df['sales'].astype('float32')
    df['onpromotion'] = df['onpromotion'].astype('int8')
    
    print(f"Training data shape: {df.shape}")
    print(f"Date range: {df[config.DATE_COL].min()} to {df[config.DATE_COL].max()}")
    return df

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    df = pd.read_csv(config.TEST_FILE, parse_dates=[config.DATE_COL])
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    
    # Optimize data types
    df['store_nbr'] = df['store_nbr'].astype('int16')
    df['family'] = df['family'].astype('category')
    df['onpromotion'] = df['onpromotion'].astype('int8')
    
    print(f"Test data shape: {df.shape}")
    return df

def load_stores():
    """Load store metadata"""
    print("Loading store metadata...")
    df = pd.read_csv(config.STORES_FILE)
    
    # Optimize data types - use string instead of category
    df['store_nbr'] = df['store_nbr'].astype('int16')
    df['city'] = df['city'].astype(str)  # Changed from category
    df['state'] = df['state'].astype(str)  # Changed from category
    df['type'] = df['type'].astype(str)  # Changed from category
    df['cluster'] = df['cluster'].astype('int8')
    
    print(f"Stores data shape: {df.shape}")
    return df

def load_holidays():
    """Load holidays and events data"""
    print("Loading holidays data...")
    df = pd.read_csv(config.HOLIDAYS_FILE, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Optimize data types - use string instead of category
    df['type'] = df['type'].astype(str)  # Changed from category
    df['locale'] = df['locale'].astype(str)  # Changed from category
    df['locale_name'] = df['locale_name'].astype(str)  # Changed from category
    df['description'] = df['description'].astype(str)  # Changed from category
    df['transferred'] = df['transferred'].astype('bool')
    
    print(f"Holidays data shape: {df.shape}")
    return df

def load_oil():
    """Load oil prices data"""
    print("Loading oil prices data...")
    df = pd.read_csv(config.OIL_FILE, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Forward fill missing oil prices
    df['dcoilwtico'] = df['dcoilwtico'].ffill()
    # Backward fill any remaining NaNs at the start
    df['dcoilwtico'] = df['dcoilwtico'].bfill()
    
    df['dcoilwtico'] = df['dcoilwtico'].astype('float32')
    
    print(f"Oil prices data shape: {df.shape}")
    print(f"Missing values after filling: {df['dcoilwtico'].isna().sum()}")
    return df

def load_transactions():
    """Load transactions data"""
    print("Loading transactions data...")
    df = pd.read_csv(config.TRANSACTIONS_FILE, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Optimize data types
    df['store_nbr'] = df['store_nbr'].astype('int16')
    df['transactions'] = df['transactions'].astype('int32')
    
    print(f"Transactions data shape: {df.shape}")
    return df

def merge_all_data(train_df, stores_df, holidays_df, oil_df, transactions_df):
    """Merge all data sources"""
    print("\nMerging all data sources...")
    
    # Merge with stores
    df = train_df.merge(stores_df, on='store_nbr', how='left')
    print(f"After merging stores: {df.shape}")
    
    # Merge with oil prices
    df = df.merge(oil_df, left_on=config.DATE_COL, right_on='date', how='left', suffixes=('', '_oil'))
    df = df.drop('date_oil', axis=1, errors='ignore')
    print(f"After merging oil: {df.shape}")
    
    # Merge with transactions
    df = df.merge(transactions_df, on=[config.DATE_COL, 'store_nbr'], how='left')
    print(f"After merging transactions: {df.shape}")
    
    # Merge with holidays (this is more complex due to locale matching)
    # National holidays
    national_holidays = holidays_df[holidays_df['locale'] == 'National'][['date', 'type', 'transferred']].copy()
    national_holidays.columns = ['date', 'holiday_type_national', 'holiday_transferred_national']
    df = df.merge(national_holidays, left_on=config.DATE_COL, right_on='date', how='left', suffixes=('', '_hol'))
    df = df.drop('date_hol', axis=1, errors='ignore')
    
    # Regional holidays (match by state) - convert to string to avoid categorical issues
    regional_holidays = holidays_df[holidays_df['locale'] == 'Regional'][['date', 'locale_name', 'type']].copy()
    regional_holidays.columns = ['date', 'state', 'holiday_type_regional']
    regional_holidays['state'] = regional_holidays['state'].astype(str)
    df['state_str'] = df['state'].astype(str)
    df = df.merge(regional_holidays, left_on=[config.DATE_COL, 'state_str'], right_on=['date', 'state'], how='left', suffixes=('', '_reg'))
    df = df.drop(['date_reg', 'state_str', 'state_reg'], axis=1, errors='ignore')
    
    # Local holidays (match by city) - convert to string to avoid categorical issues
    local_holidays = holidays_df[holidays_df['locale'] == 'Local'][['date', 'locale_name', 'type']].copy()
    local_holidays.columns = ['date', 'city', 'holiday_type_local']
    local_holidays['city'] = local_holidays['city'].astype(str)
    df['city_str'] = df['city'].astype(str)
    df = df.merge(local_holidays, left_on=[config.DATE_COL, 'city_str'], right_on=['date', 'city'], how='left', suffixes=('', '_loc'))
    df = df.drop(['date_loc', 'city_str', 'city_loc'], axis=1, errors='ignore')
    
    # Create unified holiday indicator
    df['is_holiday'] = (
        df['holiday_type_national'].notna() | 
        df['holiday_type_regional'].notna() | 
        df['holiday_type_local'].notna()
    ).astype('int8')
    
    # Combine holiday types (prioritize national > regional > local)
    df['holiday_type'] = df['holiday_type_national'].fillna(
        df['holiday_type_regional'].fillna(df['holiday_type_local'])
    )
    
    print(f"Final merged data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def load_and_merge_all():
    """Load and merge all data sources"""
    train = load_train_data()
    stores = load_stores()
    holidays = load_holidays()
    oil = load_oil()
    transactions = load_transactions()
    
    merged_data = merge_all_data(train, stores, holidays, oil, transactions)
    
    return merged_data

if __name__ == "__main__":
    # Test data loading
    df = load_and_merge_all()
    print("\n" + "="*50)
    print("Data loading successful!")
    print("="*50)
    print(f"\nFinal dataset info:")
    print(df.info())
    print(f"\nSample data:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
