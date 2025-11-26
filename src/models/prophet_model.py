"""
Prophet model for time series forecasting
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import config

class ProphetForecaster:
    """Facebook Prophet model for forecasting"""
    
    def __init__(self, **kwargs):
        self.name = "Prophet"
        self.models = {}  # Store separate model for each store-family combination
        self.prophet_params = kwargs if kwargs else {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05
        }
    
    def fit(self, X, y, store_family_col=None):
        """Train Prophet models
        
        Args:
            X: Features DataFrame (must include 'date' column)
            y: Target values
            store_family_col: Column name for store-family grouping (optional)
        """
        print(f"Training {self.name}...")
        
        # Prepare data
        df = X.copy()
        df['y'] = y
        
        # Train separate models for each store-family or one global model
        if store_family_col and store_family_col in df.columns:
            unique_groups = df[store_family_col].unique()
            print(f"Training {len(unique_groups)} Prophet models (one per group)...")
            
            for i, group in enumerate(unique_groups):
                if i % 10 == 0:
                    print(f"Training model {i+1}/{len(unique_groups)}...")
                
                group_data = df[df[store_family_col] == group][[config.DATE_COL, 'y']].copy()
                group_data.columns = ['ds', 'y']
                
                # Train Prophet model
                model = Prophet(**self.prophet_params)
                model.fit(group_data, verbose=False)
                self.models[group] = model
        else:
            # Train single global model
            print("Training single global Prophet model...")
            prophet_df = df[[config.DATE_COL, 'y']].copy()
            prophet_df.columns = ['ds', 'y']
            
            model = Prophet(**self.prophet_params)
            model.fit(prophet_df)
            self.models['global'] = model
        
        print(f"{self.name} training complete!")
        return self
    
    def predict(self, X, store_family_col=None):
        """Make predictions"""
        if not self.models:
            raise ValueError("Model not trained yet!")
        
        predictions = np.zeros(len(X))
        
        if store_family_col and store_family_col in X.columns:
            # Predict using group-specific models
            for group, model in self.models.items():
                mask = X[store_family_col] == group
                if mask.any():
                    future_df = X.loc[mask, [config.DATE_COL]].copy()
                    future_df.columns = ['ds']
                    
                    forecast = model.predict(future_df)
                    predictions[mask] = forecast['yhat'].values
        else:
            # Predict using global model
            future_df = X[[config.DATE_COL]].copy()
            future_df.columns = ['ds']
            
            forecast = self.models['global'].predict(future_df)
            predictions = forecast['yhat'].values
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        return predictions
    
    def save(self, filepath):
        """Save models"""
        joblib.dump({
            'models': self.models,
            'prophet_params': self.prophet_params
        }, filepath)
        print(f"Models saved to {filepath}")
    
    def load(self, filepath):
        """Load models"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.prophet_params = data['prophet_params']
        print(f"Models loaded from {filepath}")
        return self

if __name__ == "__main__":
    print("Prophet model module loaded successfully")
