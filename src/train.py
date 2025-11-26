"""
Training script for all forecasting models
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
import data_loader
import feature_engineering
import evaluation
from models.baseline import get_baseline_models
from models.prophet_model import ProphetForecaster
from models.gradient_boosting import LightGBMForecaster, XGBoostForecaster
from models.lstm_model import LSTMForecaster
from models.tft_model import TemporalFusionTransformer

def prepare_train_test_split(df, test_size=0.2):
    """Split data into train and test sets (time-based split)"""
    print("\nSplitting data into train and test sets...")
    
    # Sort by date
    df = df.sort_values(config.DATE_COL)
    
    # Time-based split
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_idx][config.DATE_COL]
    
    train_df = df[df[config.DATE_COL] < split_date].copy()
    test_df = df[df[config.DATE_COL] >= split_date].copy()
    
    print(f"Train set: {len(train_df)} samples ({train_df[config.DATE_COL].min()} to {train_df[config.DATE_COL].max()})")
    print(f"Test set: {len(test_df)} samples ({test_df[config.DATE_COL].min()} to {test_df[config.DATE_COL].max()})")
    
    return train_df, test_df

def prepare_features_target(df, exclude_cols=None, encode_categoricals=True):
    """Prepare features and target with proper categorical encoding"""
    if exclude_cols is None:
        exclude_cols = [config.TARGET_COL, config.DATE_COL, 'id', 'family_code']
    
    # Identify categorical columns (string dtype)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    print(f"\nCategorical columns found: {categorical_cols}")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[config.TARGET_COL].copy()
    
    # Encode categorical features if requested
    if encode_categoricals and len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        
        print(f"Encoding {len(categorical_cols)} categorical features...")
        label_encoders = {}
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"  - Encoded {col}: {len(le.classes_)} unique values")
        
        # Store encoders for later use
        X.label_encoders = label_encoders
    
    return X, y

def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train and evaluate baseline models"""
    print("\n" + "="*80)
    print("Training Baseline Models")
    print("="*80)
    
    results = {}
    
    models = get_baseline_models()
    
    for model in models:
        print(f"\nTraining {model.name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Ensure same length
            min_len = min(len(y_test), len(y_pred))
            y_test_subset = y_test.iloc[:min_len].values
            y_pred_subset = y_pred[:min_len]
            
            metrics = evaluation.calculate_all_metrics(y_test_subset, y_pred_subset)
            evaluation.print_metrics(metrics, model.name)
            
            results[model.name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
        except Exception as e:
            print(f"Error training {model.name}: {str(e)}")
    
    return results

def train_prophet(train_df, test_df, sample_frac=0.05):
    """Train and evaluate Prophet model with sampling for memory efficiency"""
    print("\n" + "="*80)
    print("Training Prophet Model")
    print("="*80)
    
    try:
        # Sample data for memory efficiency (Prophet is memory-intensive)
        print(f"\nSampling {sample_frac*100}% of training data for Prophet (memory optimization)...")
        sample_size = int(len(train_df) * sample_frac)
        sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
        sample_indices = np.sort(sample_indices)  # Keep time order
        
        train_sample = train_df.iloc[sample_indices].copy()
        
        print(f"Training on {len(train_sample)} samples (sampled from {len(train_df)})")
        
        # Prophet expects X (with date column) and y separately
        model = ProphetForecaster()
        model.fit(train_sample, train_sample[config.TARGET_COL])
        
        # Prepare test DataFrame for prediction
        # model.predict expects X with date column
        y_pred = model.predict(test_df)
        y_test = test_df[config.TARGET_COL]
        
        metrics = evaluation.calculate_all_metrics(y_test.values, y_pred)
        evaluation.print_metrics(metrics, "Prophet")
        
        # Save model
        model.save(config.MODELS_DIR / 'prophet_model.pkl')
        evaluation.save_metrics(metrics, 'prophet')
        
        return {
            'Prophet': {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
        }
    except Exception as e:
        print(f"Error training Prophet: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train and evaluate gradient boosting models"""
    print("\n" + "="*80)
    
    results = {}  # Initialize results dictionary
    
    if hasattr(X_train, 'label_encoders'):
        categorical_features = list(X_train.label_encoders.keys())
        print(f"\nUsing {len(categorical_features)} categorical features: {categorical_features}")
    
    # LightGBM
    print("\n--- LightGBM ---")
    try:
        lgb_model = LightGBMForecaster()
        
        # LightGBM can handle categorical features natively if we specify them
        # Since we've label-encoded them, we'll pass them as regular features
        # but LightGBM will still benefit from knowing they're categorical
        lgb_model.fit(X_train, y_train, 
                     categorical_features=categorical_features,
                     eval_set=(X_test, y_test))
        
        y_pred_lgb = lgb_model.predict(X_test)
        metrics_lgb = evaluation.calculate_all_metrics(y_test.values, y_pred_lgb)
        evaluation.print_metrics(metrics_lgb, "LightGBM")
        
        # Save model
        lgb_model.save(config.MODELS_DIR / 'lightgbm_model.pkl')
        evaluation.save_metrics(metrics_lgb, 'lightgbm')
        
        # Save feature importance
        feat_imp = lgb_model.get_feature_importance()
        feat_imp.to_csv(config.MODELS_DIR / 'lightgbm_feature_importance.csv', index=False)
        
        results['LightGBM'] = {
            'model': lgb_model,
            'metrics': metrics_lgb,
            'predictions': y_pred_lgb,
            'feature_importance': feat_imp
        }
    except Exception as e:
        print(f"Error training LightGBM: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # XGBoost
    print("\n--- XGBoost ---")
    try:
        xgb_model = XGBoostForecaster()
        xgb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = evaluation.calculate_all_metrics(y_test.values, y_pred_xgb)
        evaluation.print_metrics(metrics_xgb, "XGBoost")
        
        # Save model
        xgb_model.save(config.MODELS_DIR / 'xgboost_model.pkl')
        evaluation.save_metrics(metrics_xgb, 'xgboost')
        
        # Save feature importance
        feat_imp = xgb_model.get_feature_importance()
        feat_imp.to_csv(config.MODELS_DIR / 'xgboost_feature_importance.csv', index=False)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'metrics': metrics_xgb,
            'predictions': y_pred_xgb,
            'feature_importance': feat_imp
        }
    except Exception as e:
        print(f"Error training XGBoost: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results

def train_deep_learning(X_train, y_train, X_test, y_test, lstm_sample_frac=0.05, tft_sample_frac=0.10):
    """Train and evaluate deep learning models with optimized sampling
    
    Args:
        lstm_sample_frac: Sampling fraction for LSTM (5% = ~150k samples)
        tft_sample_frac: Sampling fraction for TFT (10% = ~300k samples)
    """
    print("\n" + "="*80)
    print("Training Deep Learning Models")
    print("="*80)
    
    results = {}
    
    # Sample test set for prediction if it's too large (to avoid MemoryError)
    if len(X_test) > 50000:
        print(f"\nSampling test set for prediction (memory optimization, max 50k samples)...")
        test_sample_indices = np.random.choice(len(X_test), 50000, replace=False)
        test_sample_indices = np.sort(test_sample_indices)
        X_test_dl = X_test.iloc[test_sample_indices]
        y_test_dl = y_test.iloc[test_sample_indices]
    else:
        X_test_dl = X_test
        y_test_dl = y_test

    # LSTM with 5% sampling
    print("\n--- LSTM ---")
    print(f"Sampling {lstm_sample_frac*100}% of training data for LSTM...")
    lstm_sample_size = int(len(X_train) * lstm_sample_frac)
    lstm_indices = np.random.choice(len(X_train), lstm_sample_size, replace=False)
    lstm_indices = np.sort(lstm_indices)
    
    X_train_lstm = X_train.iloc[lstm_indices]
    y_train_lstm = y_train.iloc[lstm_indices]
    print(f"LSTM training on {len(X_train_lstm)} samples")
    
    try:
        lstm_model = LSTMForecaster(
            sequence_length=30,  # Reduced from 60 for memory
            lstm_units=64,  # Reduced from config
            dropout=0.2
        )
        
        lstm_model.fit(X_train_lstm, y_train_lstm, validation_split=0.1)
        
        y_pred_lstm = lstm_model.predict(X_test_dl)
        
        # Handle length mismatch due to sequence creation
        min_len = min(len(y_test_dl), len(y_pred_lstm))
        metrics_lstm = evaluation.calculate_all_metrics(
            y_test_dl.iloc[:min_len].values, 
            y_pred_lstm[:min_len]
        )
        evaluation.print_metrics(metrics_lstm, "LSTM")
        
        # Save model
        lstm_model.save(config.MODELS_DIR / 'lstm_model.pkl')
        evaluation.save_metrics(metrics_lstm, 'lstm')
        
        results['LSTM'] = {
            'model': lstm_model,
            'metrics': metrics_lstm,
            'predictions': y_pred_lstm
        }
    except Exception as e:
        print(f"Error training LSTM: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # TFT with 10% sampling (state-of-the-art implementation)
    print("\n--- Temporal Fusion Transformer (State-of-the-Art) ---")
    print(f"Sampling {tft_sample_frac*100}% of training data for TFT...")
    tft_sample_size = int(len(X_train) * tft_sample_frac)
    tft_indices = np.random.choice(len(X_train), tft_sample_size, replace=False)
    tft_indices = np.sort(tft_indices)
    
    X_train_tft = X_train.iloc[tft_indices]
    y_train_tft = y_train.iloc[tft_indices]
    print(f"TFT training on {len(X_train_tft)} samples")
    
    try:
        tft_model = TemporalFusionTransformer(
            hidden_size=128,  # Optimized for memory
            num_heads=2,
            dropout=0.2,
            forecast_horizon=config.FORECAST_HORIZON,
            use_mixed_precision=True
        )
        
        # Use shorter lookback and smaller batch for memory
        tft_model.fit(
            X_train_tft, y_train_tft,
            validation_split=0.1,
            lookback=30,  # Reduced from 60
            epochs=15,  # Reduced for faster training
            batch_size=64
        )
        
        y_pred_tft = tft_model.predict(X_test_dl, lookback=30, batch_size=64)
        
        # Handle length mismatch
        min_len = min(len(y_test_dl), len(y_pred_tft))
        metrics_tft = evaluation.calculate_all_metrics(
            y_test_dl.iloc[:min_len].values,
            y_pred_tft[:min_len]
        )
        evaluation.print_metrics(metrics_tft, "TFT")
        
        # Save model
        tft_model.save(config.MODELS_DIR / 'tft_model.pkl')
        evaluation.save_metrics(metrics_tft, 'tft')
        
        results['TFT'] = {
            'model': tft_model,
            'metrics': metrics_tft,
            'predictions': y_pred_tft
        }
    except Exception as e:
        print(f"Error training TFT: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results

def main(quick_test=False, models_to_train=None, pre_sample_frac=0.2):
    """Main training pipeline
    
    Args:
        quick_test: Use 20% of data for quick testing
        models_to_train: List of model types to train
        pre_sample_frac: Pre-sample raw data before feature engineering (for memory)
    """
    print("\n" + "="*80)
    print("RETAIL SALES FORECASTING - TRAINING PIPELINE")
    print("="*80)
    
    if models_to_train is None:
        models_to_train = ['baseline', 'prophet', 'gradient_boosting', 'deep_learning']
    
    # Load and merge data
    print("\n1. Loading data...")
    df = data_loader.load_and_merge_all()
    
    # PRE-SAMPLE for memory efficiency (before feature engineering)
    if pre_sample_frac < 1.0:
        print(f"\n⚠️  PRE-SAMPLING {pre_sample_frac*100}% of data BEFORE feature engineering (memory optimization)")
        print(f"Original size: {len(df)} records")
        df = df.sample(frac=pre_sample_frac, random_state=config.RANDOM_STATE)
        print(f"Sampled size: {len(df)} records")
    
    # Engineer features
    print("\n2. Engineering features...")
    df = feature_engineering.engineer_all_features(df)
    df = feature_engineering.prepare_for_modeling(df, is_train=True)
    
    # For quick testing, use a subset
    if quick_test:
        print("\nQUICK TEST MODE: Using 20% of data")
        df = df.sample(frac=0.2, random_state=config.RANDOM_STATE)
    
    # Split data
    print("\n3. Splitting data...")
    train_df, test_df = prepare_train_test_split(df, test_size=0.2)
    
    X_train, y_train = prepare_features_target(train_df)
    X_test, y_test = prepare_features_target(test_df)
    
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Train models
    all_results = {}
    
    if 'baseline' in models_to_train:
        baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
        all_results.update(baseline_results)
    
    if 'prophet' in models_to_train:
        prophet_results = train_prophet(train_df, test_df)
        all_results.update(prophet_results)
    
    if 'gradient_boosting' in models_to_train:
        gb_results = train_gradient_boosting(X_train, y_train, X_test, y_test)
        all_results.update(gb_results)
    
    if 'deep_learning' in models_to_train:
        # Optimized sampling: 5% for LSTM, 10% for TFT
        import gc
        gc.collect()
        dl_results = train_deep_learning(
            X_train, y_train, X_test, y_test,
            lstm_sample_frac=0.05,
            tft_sample_frac=0.10
        )
        all_results.update(dl_results)
    
    # Compare all models
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    if len(all_results) == 0:
        print("\nNo models were trained successfully.")
        return all_results
    
    model_metrics = {name: results['metrics'] for name, results in all_results.items()}
    df_comparison, fig = evaluation.compare_models(
        model_metrics,
        save_path=config.REPORTS_DIR / 'model_comparison.png'
    )
    
    # Save comparison
    df_comparison.to_csv(config.MODELS_DIR / 'model_comparison.csv')
    
    # Find best model
    best_model_name = df_comparison['wape'].idxmin()
    print(f"\n🏆 Best Model: {best_model_name} (WAPE: {df_comparison.loc[best_model_name, 'wape']:.2f}%)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to: {config.MODELS_DIR}")
    print(f"Reports saved to: {config.REPORTS_DIR}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train forecasting models')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode with subset of data')
    parser.add_argument('--models', nargs='+', default=None,
                       choices=['baseline', 'prophet', 'gradient_boosting', 'deep_learning'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    results = main(quick_test=args.quick_test, models_to_train=args.models)
