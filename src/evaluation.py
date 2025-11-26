"""
Evaluation metrics for time series forecasting
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import config

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_wape(y_true, y_pred):
    """Calculate Weighted Absolute Percentage Error (WAPE)
    
    WAPE is preferred in retail as it's less sensitive to small denominators
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(numerator[mask] / denominator[mask]) * 100

def calculate_all_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'wape': calculate_wape(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred)
    }
    return metrics

def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way"""
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"MAPE:  {metrics['mape']:.2f}%")
    print(f"WAPE:  {metrics['wape']:.2f}%")
    print(f"SMAPE: {metrics['smape']:.2f}%")
    print("=" * 50)

def save_metrics(metrics, model_name, filepath=None):
    """Save metrics to JSON file"""
    if filepath is None:
        filepath = config.MODELS_DIR / f"{model_name}_metrics.json"
    
    # Convert numpy types to native python types for JSON serialization
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Metrics saved to {filepath}")

def load_metrics(model_name, filepath=None):
    """Load metrics from JSON file"""
    if filepath is None:
        filepath = config.MODELS_DIR / f"{model_name}_metrics.json"
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_predictions(y_true, y_pred, dates=None, title="Predictions vs Actual", 
                     save_path=None, show_last_n=None):
    """Plot predictions vs actual values"""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    if show_last_n is not None:
        y_true = y_true[-show_last_n:]
        y_pred = y_pred[-show_last_n:]
        if dates is not None:
            dates = dates[-show_last_n:]
    
    x = dates if dates is not None else range(len(y_true))
    
    ax.plot(x, y_true, label='Actual', linewidth=2, alpha=0.7)
    ax.plot(x, y_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax.set_xlabel('Date' if dates is not None else 'Index')
    ax.set_ylabel('Sales')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return fig

def plot_residuals(y_true, y_pred, title="Residual Analysis", save_path=None):
    """Plot residual analysis"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predicted vs Residuals
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Predicted vs Residuals')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
    
    plt.close()
    return fig

def compare_models(model_metrics_dict, save_path=None):
    """Compare multiple models"""
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(model_metrics_dict).T
    df_comparison = df_comparison.round(4)
    
    print("\nModel Comparison:")
    print("=" * 80)
    print(df_comparison.to_string())
    print("=" * 80)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Error metrics (lower is better)
    error_metrics = ['rmse', 'mae']
    df_comparison[error_metrics].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Error Metrics (Lower is Better)')
    axes[0].set_ylabel('Value')
    axes[0].set_xlabel('Model')
    axes[0].legend(title='Metric')
    axes[0].grid(True, alpha=0.3)
    
    # Percentage metrics (lower is better)
    pct_metrics = ['mape', 'wape', 'smape']
    df_comparison[pct_metrics].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Percentage Metrics (Lower is Better)')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xlabel('Model')
    axes[1].legend(title='Metric')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()
    return df_comparison, fig

if __name__ == "__main__":
    # Test metrics calculation
    np.random.seed(42)
    y_true = np.random.rand(100) * 100
    y_pred = y_true + np.random.randn(100) * 10
    
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")
    
    # Test plotting
    plot_predictions(y_true, y_pred, title="Test Predictions", 
                    save_path=config.REPORTS_DIR / "test_predictions.png")
    plot_residuals(y_true, y_pred, title="Test Residuals",
                  save_path=config.REPORTS_DIR / "test_residuals.png")
    
    print("\nEvaluation module test complete!")
