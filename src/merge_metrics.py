import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import config

def merge_metrics():
    print("Merging all model metrics...")
    models_dir = config.MODELS_DIR
    metrics_files = list(models_dir.glob('*_metrics.json'))
    
    all_metrics = {}
    
    for file in metrics_files:
        model_name = file.name.replace('_metrics.json', '')
        # Capitalize for display
        if model_name == 'tft': model_name = 'TFT'
        elif model_name == 'lstm': model_name = 'LSTM'
        elif model_name == 'lightgbm': model_name = 'LightGBM'
        elif model_name == 'xgboost': model_name = 'XGBoost'
        elif model_name == 'prophet': model_name = 'Prophet'
        else: model_name = model_name.capitalize()
            
        try:
            with open(file, 'r') as f:
                metrics = json.load(f)
                all_metrics[model_name] = metrics
        except json.JSONDecodeError:
            print(f"Warning: Skipping corrupted metrics file: {file.name}")
        except Exception as e:
            print(f"Warning: Error reading {file.name}: {str(e)}")
            
    if not all_metrics:
        print("No metrics found.")
        return

    # Create DataFrame
    df_comparison = pd.DataFrame(all_metrics).T
    
    # Sort by WAPE (if available) or RMSE
    if 'wape' in df_comparison.columns:
        df_comparison = df_comparison.sort_values('wape')
    
    # Save
    output_file = models_dir / 'model_comparison.csv'
    df_comparison.to_csv(output_file)
    print(f"Saved merged comparison to {output_file}")
    print(df_comparison)

if __name__ == "__main__":
    merge_metrics()
