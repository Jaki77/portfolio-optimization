"""
Model Evaluation and Comparison Module
Calculates performance metrics and compares different forecasting models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    """Evaluates and compares time series forecasting models"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_train: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training values (for MASE calculation)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        
        # MAPE (handle zero values)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['MAPE'] = mape
        
        # SMAPE
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        metrics['SMAPE'] = smape
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['R2'] = 1 - (ss_res / ss_tot)
        
        # MASE (if training data provided)
        if y_train is not None:
            naive_forecast_errors = np.abs(np.diff(y_train))
            if len(naive_forecast_errors) > 0:
                mean_naive_error = np.mean(naive_forecast_errors)
                if mean_naive_error != 0:
                    metrics['MASE'] = metrics['MAE'] / mean_naive_error
        
        return metrics
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy metrics
        """
        # Calculate returns
        true_returns = np.diff(y_true)
        pred_returns = np.diff(y_pred)
        
        # Directional accuracy
        correct_direction = np.sign(true_returns) == np.sign(pred_returns)
        directional_accuracy = np.mean(correct_direction) * 100
        
        # Up/Down prediction accuracy
        true_up = true_returns > 0
        pred_up = pred_returns > 0
        
        up_accuracy = np.mean(pred_up[true_up]) * 100 if true_up.any() else 0
        down_accuracy = np.mean(~pred_up[~true_up]) * 100 if (~true_up).any() else 0
        
        return {
            'Directional_Accuracy': directional_accuracy,
            'Up_Accuracy': up_accuracy,
            'Down_Accuracy': down_accuracy
        }
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and create a comparison dataframe
        
        Args:
            models_results: Dictionary with model names as keys and 
                          metrics dictionaries as values
                          
        Returns:
            Comparison dataframe
        """
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        if 'RMSE' in df.columns:
            df = df.sort_values('RMSE')
        
        return df
    
    @staticmethod
    def plot_predictions(y_true: pd.Series, predictions_dict: Dict[str, pd.Series], 
                         title: str = "Model Predictions Comparison"):
        """
        Plot actual vs predicted values for multiple models
        
        Args:
            y_true: Actual values
            predictions_dict: Dictionary with model names and predictions
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot actual values
        plt.plot(y_true.index, y_true.values, 'k-', linewidth=2, label='Actual', alpha=0.7)
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = colors[idx % len(colors)]
            plt.plot(predictions.index, predictions.values, '--', 
                    color=color, linewidth=1.5, label=f'{model_name} Predictions')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model"):
        """
        Plot residual analysis
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, 'o', alpha=0.5, markersize=3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[0, 0].set_title(f'{model_name} - Residuals over Time')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title(f'{model_name} - Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[1, 1].set_title(f'{model_name} - Residuals vs Predicted')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        from scipy.stats import shapiro, jarque_bera
        
        # Normality tests
        if len(residuals) < 5000:
            shapiro_stat, shapiro_p = shapiro(residuals)
            print(f"Shapiro-Wilk Test: Stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        
        jb_stat, jb_p = jarque_bera(residuals)
        print(f"Jarque-Bera Test: Stat={jb_stat:.4f}, p-value={jb_p:.4f}")
        
        # Autocorrelation test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        print(f"Ljung-Box Test (lag=10): p-value={lb_test['lb_pvalue'].iloc[0]:.4f}")
        
        return residuals
    
    @staticmethod
    def create_model_report(models_metrics: Dict[str, Dict]) -> str:
        """
        Create a comprehensive model comparison report
        
        Args:
            models_metrics: Dictionary with model metrics
            
        Returns:
            Formatted report string
        """
        report = "=" * 70 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Find best model for each metric
        metrics_to_minimize = ['MAE', 'RMSE', 'MSE', 'MAPE', 'SMAPE']
        metrics_to_maximize = ['R2', 'Directional_Accuracy']
        
        for metric in metrics_to_minimize:
            if any(metric in metrics for metrics in models_metrics.values()):
                best_model = min(models_metrics.items(), 
                                key=lambda x: x[1].get(metric, float('inf')))[0]
                best_value = min(metrics.get(metric, float('inf')) 
                               for metrics in models_metrics.values())
                report += f"Best {metric}: {best_model} ({best_value:.4f})\n"
        
        for metric in metrics_to_maximize:
            if any(metric in metrics for metrics in models_metrics.values()):
                best_model = max(models_metrics.items(), 
                                key=lambda x: x[1].get(metric, float('-inf')))[0]
                best_value = max(metrics.get(metric, float('-inf')) 
                               for metrics in models_metrics.values())
                report += f"Best {metric}: {best_model} ({best_value:.2f}%)\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report