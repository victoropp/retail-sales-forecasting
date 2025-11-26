"""
Baseline models for time series forecasting
"""
import numpy as np
import pandas as pd
import config

class NaiveForecaster:
    """Naive forecast: use last observed value"""
    
    def __init__(self):
        self.name = "Naive"
        self.last_values = {}
    
    def fit(self, X, y):
        """Store last values for each store-family combination"""
        if 'store_nbr' in X.columns and 'family' in X.columns:
            for (store, family), group in X.groupby(['store_nbr', 'family']):
                last_idx = group.index[-1]
                self.last_values[(store, family)] = y.loc[last_idx]
        else:
            self.last_values['global'] = y.iloc[-1]
        return self
    
    def predict(self, X):
        """Predict using last observed value"""
        predictions = np.zeros(len(X))
        
        if 'store_nbr' in X.columns and 'family' in X.columns:
            for i, (idx, row) in enumerate(X.iterrows()):
                key = (row['store_nbr'], row['family'])
                predictions[i] = self.last_values.get(key, 0)
        else:
            predictions[:] = self.last_values.get('global', 0)
        
        return predictions

class SeasonalNaiveForecaster:
    """Seasonal naive: use value from same day last week"""
    
    def __init__(self, seasonal_period=7):
        self.name = f"Seasonal_Naive_{seasonal_period}d"
        self.seasonal_period = seasonal_period
        self.historical_data = None
    
    def fit(self, X, y):
        """Store historical data"""
        self.historical_data = X.copy()
        self.historical_data[config.TARGET_COL] = y
        return self
    
    def predict(self, X):
        """Predict using seasonal lag"""
        predictions = np.zeros(len(X))
        
        for i, (idx, row) in enumerate(X.iterrows()):
            # Find the value from seasonal_period days ago
            target_date = row[config.DATE_COL] - pd.Timedelta(days=self.seasonal_period)
            
            if 'store_nbr' in X.columns and 'family' in X.columns:
                mask = (
                    (self.historical_data[config.DATE_COL] == target_date) &
                    (self.historical_data['store_nbr'] == row['store_nbr']) &
                    (self.historical_data['family'] == row['family'])
                )
            else:
                mask = self.historical_data[config.DATE_COL] == target_date
            
            if mask.any():
                predictions[i] = self.historical_data.loc[mask, config.TARGET_COL].values[0]
            else:
                predictions[i] = 0
        
        return predictions

class MovingAverageForecaster:
    """Moving average forecast"""
    
    def __init__(self, window=7):
        self.name = f"Moving_Average_{window}d"
        self.window = window
        self.historical_data = None
    
    def fit(self, X, y):
        """Store historical data"""
        self.historical_data = X.copy()
        self.historical_data[config.TARGET_COL] = y
        return self
    
    def predict(self, X):
        """Predict using moving average"""
        predictions = np.zeros(len(X))
        
        for i, (idx, row) in enumerate(X.iterrows()):
            # Get last window days of data
            end_date = row[config.DATE_COL] - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=self.window)
            
            if 'store_nbr' in X.columns and 'family' in X.columns:
                mask = (
                    (self.historical_data[config.DATE_COL] > start_date) &
                    (self.historical_data[config.DATE_COL] <= end_date) &
                    (self.historical_data['store_nbr'] == row['store_nbr']) &
                    (self.historical_data['family'] == row['family'])
                )
            else:
                mask = (
                    (self.historical_data[config.DATE_COL] > start_date) &
                    (self.historical_data[config.DATE_COL] <= end_date)
                )
            
            if mask.any():
                predictions[i] = self.historical_data.loc[mask, config.TARGET_COL].mean()
            else:
                predictions[i] = 0
        
        return predictions

def get_baseline_models():
    """Get all baseline models"""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonal_period=7),
        MovingAverageForecaster(window=7),
        MovingAverageForecaster(window=28)
    ]

if __name__ == "__main__":
    print("Baseline models module loaded successfully")
    models = get_baseline_models()
    print(f"Available baseline models: {[m.name for m in models]}")
