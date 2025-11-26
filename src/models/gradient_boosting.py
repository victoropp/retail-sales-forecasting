"""
Gradient Boosting models for time series forecasting
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import config

class LightGBMForecaster:
    """LightGBM model for forecasting"""
    
    def __init__(self, params=None):
        self.name = "LightGBM"
        self.params = params if params is not None else config.LIGHTGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.categorical_features = None
    
    def fit(self, X, y, categorical_features=None, eval_set=None):
        """Train LightGBM model"""
        print(f"Training {self.name}...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        self.categorical_features = categorical_features
        
        # Prepare categorical features
        cat_features_idx = []
        if categorical_features:
            cat_features_idx = [i for i, col in enumerate(self.feature_names) 
                               if col in categorical_features]
        
        # Create dataset
        train_data = lgb.Dataset(
            X, label=y,
            categorical_feature=cat_features_idx,
            free_raw_data=False
        )
        
        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=cat_features_idx,
                reference=train_data,
                free_raw_data=False
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"{self.name} training complete!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure columns match training
        X = X[self.feature_names]
        
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self, filepath):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'params': self.params
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.categorical_features = data['categorical_features']
        self.params = data['params']
        print(f"Model loaded from {filepath}")
        return self

class XGBoostForecaster:
    """XGBoost model for forecasting"""
    
    def __init__(self, params=None):
        self.name = "XGBoost"
        self.params = params if params is not None else config.XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = None
    
    def fit(self, X, y, eval_set=None):
        """Train XGBoost model"""
        print(f"Training {self.name}...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['category', 'object']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        # Prepare eval set
        eval_list = []
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_encoded = X_val.copy()
            for col in X_val_encoded.select_dtypes(include=['category', 'object']).columns:
                X_val_encoded[col] = X_val_encoded[col].astype('category').cat.codes
            eval_list = [(X_val_encoded, y_val)]
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_encoded, y,
            eval_set=eval_list if eval_list else None,
            verbose=100
        )
        
        print(f"{self.name} training complete!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure columns match training
        X = X[self.feature_names]
        
        # Encode categorical features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['category', 'object']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        predictions = self.model.predict(X_encoded)
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self, filepath):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.params = data['params']
        print(f"Model loaded from {filepath}")
        return self

if __name__ == "__main__":
    print("Gradient boosting models module loaded successfully")
    print("Available models: LightGBM, XGBoost")
