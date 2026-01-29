"""
Forecast Generation Module for Task 3
Generates multi-step forecasts with confidence intervals and uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Statistical methods
from scipy import stats
from sklearn.utils import resample

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Custom imports
from .forecasting_models import TimeSeriesForecaster


class ForecastGenerator:
    """Generates and analyzes multi-step time series forecasts"""
    
    def __init__(self, model: TimeSeriesForecaster, scaler: Any = None):
        """
        Initialize forecast generator with trained model
        
        Args:
            model: Trained forecasting model (ARIMA, LSTM, etc.)
            scaler: Data scaler for inverse transformation
        """
        self.model = model
        self.scaler = scaler
        self.forecast_history = {}
        self.confidence_intervals = {}
        
    def generate_forecast(self, last_data: np.ndarray, 
                         horizon: int,
                         confidence_level: float = 0.95,
                         method: str = 'parametric') -> Dict[str, Any]:
        """
        Generate multi-step forecast with confidence intervals
        
        Args:
            last_data: Most recent data points (shape depends on model)
            horizon: Number of steps to forecast
            confidence_level: Confidence level for intervals (0-1)
            method: 'parametric' (assuming normal) or 'bootstrap'
            
        Returns:
            Dictionary with forecasts, intervals, and metadata
        """
        print(f"Generating {horizon}-step forecast with {confidence_level*100:.0f}% confidence...")
        
        # Generate point forecast
        if hasattr(self.model, 'predict_future'):
            # LSTM-style multi-step forecasting
            point_forecast = self.model.predict_future(
                last_sequence=last_data,
                steps=horizon
            )
        elif hasattr(self.model, 'predict'):
            # ARIMA-style forecasting
            point_forecast = self.model.predict(steps=horizon)
            
            # Convert to array if it's a pandas series
            if hasattr(point_forecast, 'values'):
                point_forecast = point_forecast.values
        else:
            raise ValueError("Model doesn't have required prediction method")
        
        # Generate confidence intervals
        if method == 'parametric':
            ci_lower, ci_upper = self._parametric_confidence_intervals(
                point_forecast, confidence_level
            )
        elif method == 'bootstrap':
            ci_lower, ci_upper = self._bootstrap_confidence_intervals(
                last_data, horizon, confidence_level, n_samples=1000
            )
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")
        
        # Create forecast dates
        last_date = pd.Timestamp.now() if not hasattr(self, 'last_date') else self.last_date
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='B'  # Business days
        )
        
        # Store results
        forecast_results = {
            'point_forecast': point_forecast,
            'confidence_lower': ci_lower,
            'confidence_upper': ci_upper,
            'forecast_dates': forecast_dates,
            'horizon': horizon,
            'confidence_level': confidence_level,
            'method': method,
            'generation_date': pd.Timestamp.now()
        }
        
        self.forecast_history[f'forecast_{len(self.forecast_history)}'] = forecast_results
        
        return forecast_results
    
    def _parametric_confidence_intervals(self, forecast: np.ndarray, 
                                        confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate parametric confidence intervals assuming normal distribution
        
        Args:
            forecast: Point forecast values
            confidence_level: Confidence level (0-1)
            
        Returns:
            Lower and upper confidence bounds
        """
        # Calculate forecast error (simplified - would use model residuals in practice)
        # In practice, you would use the model's estimated residual variance
        residual_std = np.std(forecast) * 0.1  # Simplified assumption
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Calculate confidence intervals
        ci_width = z_score * residual_std * np.sqrt(np.arange(1, len(forecast) + 1))
        
        ci_lower = forecast - ci_width
        ci_upper = forecast + ci_width
        
        return ci_lower, ci_upper
    
    def _bootstrap_confidence_intervals(self, last_data: np.ndarray,
                                       horizon: int,
                                       confidence_level: float,
                                       n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate confidence intervals using bootstrap resampling
        
        Args:
            last_data: Most recent data for forecasting
            horizon: Forecast horizon
            confidence_level: Confidence level (0-1)
            n_samples: Number of bootstrap samples
            
        Returns:
            Lower and upper confidence bounds
        """
        # This is a simplified bootstrap implementation
        # In practice, you would bootstrap model residuals
        
        bootstrap_forecasts = []
        
        for _ in range(n_samples):
            # Add small noise to simulate uncertainty
            noisy_data = last_data + np.random.normal(0, last_data.std() * 0.01, size=last_data.shape)
            
            # Generate forecast with noisy data
            if hasattr(self.model, 'predict_future'):
                forecast = self.model.predict_future(noisy_data, steps=horizon)
            else:
                # Simplified forecast for demonstration
                forecast = self.model.predict(steps=horizon)
                if hasattr(forecast, 'values'):
                    forecast = forecast.values
            
            bootstrap_forecasts.append(forecast)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # Calculate percentiles
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        ci_lower = np.percentile(bootstrap_forecasts, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_forecasts, upper_percentile, axis=0)
        
        return ci_lower, ci_upper
    
    def plot_forecast(self, historical_data: pd.Series,
                     forecast_results: Dict[str, Any],
                     title: str = "Time Series Forecast",
                     save_path: Optional[str] = None):
        """
        Visualize forecast with historical data and confidence intervals
        
        Args:
            historical_data: Historical time series data
            forecast_results: Forecast results from generate_forecast
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, 
                'k-', linewidth=2, label='Historical Data', alpha=0.8)
        
        # Plot forecast
        forecast_dates = forecast_results['forecast_dates']
        point_forecast = forecast_results['point_forecast']
        ci_lower = forecast_results['confidence_lower']
        ci_upper = forecast_results['confidence_upper']
        
        ax.plot(forecast_dates, point_forecast, 'b--', 
                linewidth=2, label='Forecast', alpha=0.9)
        
        # Plot confidence intervals
        ax.fill_between(forecast_dates, ci_lower, ci_upper,
                       color='blue', alpha=0.2,
                       label=f"{forecast_results['confidence_level']*100:.0f}% Confidence Interval")
        
        # Add vertical line at forecast start
        forecast_start = forecast_dates[0]
        ax.axvline(x=forecast_start, color='red', linestyle='--', 
                  alpha=0.7, linewidth=1, label='Forecast Start')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Add forecast info text
        forecast_info = (
            f"Forecast Horizon: {forecast_results['horizon']} days\n"
            f"Confidence Level: {forecast_results['confidence_level']*100:.0f}%\n"
            f"Generated: {forecast_results['generation_date'].strftime('%Y-%m-%d')}"
        )
        
        ax.text(0.02, 0.98, forecast_info,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_forecast_components(self, forecast_results: Dict[str, Any],
                                save_path: Optional[str] = None):
        """
        Plot forecast components: point forecast and confidence intervals separately
        
        Args:
            forecast_results: Forecast results dictionary
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        forecast_dates = forecast_results['forecast_dates']
        point_forecast = forecast_results['point_forecast']
        ci_lower = forecast_results['confidence_lower']
        ci_upper = forecast_results['confidence_upper']
        
        # 1. Point forecast with confidence bands
        axes[0, 0].plot(forecast_dates, point_forecast, 'b-', linewidth=2, label='Point Forecast')
        axes[0, 0].fill_between(forecast_dates, ci_lower, ci_upper,
                               color='blue', alpha=0.2, label='Confidence Interval')
        axes[0, 0].set_title('Point Forecast with Confidence Bands', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Forecast returns (percentage change)
        forecast_returns = np.diff(point_forecast) / point_forecast[:-1] * 100
        return_dates = forecast_dates[1:]
        
        axes[0, 1].bar(return_dates, forecast_returns, color='green', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].set_title('Forecast Daily Returns (%)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Daily Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Confidence interval width over time
        ci_width = ci_upper - ci_lower
        ci_width_pct = (ci_width / point_forecast) * 100
        
        axes[1, 0].plot(forecast_dates, ci_width_pct, 'r-', linewidth=2)
        axes[1, 0].set_title('Confidence Interval Width (% of Forecast)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('CI Width (% of Value)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Cumulative forecast
        cumulative_forecast = np.cumsum(np.diff(point_forecast))
        axes[1, 1].plot(forecast_dates[1:], cumulative_forecast, 'purple-', linewidth=2)
        axes[1, 1].set_title('Cumulative Forecast Returns', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Forecast Components Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_forecast_metrics(self, forecast_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate key metrics for the forecast
        
        Args:
            forecast_results: Forecast results dictionary
            
        Returns:
            Dictionary of forecast metrics
        """
        point_forecast = forecast_results['point_forecast']
        ci_lower = forecast_results['confidence_lower']
        ci_upper = forecast_results['confidence_upper']
        
        metrics = {
            'forecast_horizon': len(point_forecast),
            'final_forecast_value': point_forecast[-1],
            'total_forecast_change': ((point_forecast[-1] - point_forecast[0]) / point_forecast[0]) * 100,
            'avg_forecast_value': np.mean(point_forecast),
            'forecast_volatility': np.std(np.diff(point_forecast) / point_forecast[:-1]) * 100,
            'avg_ci_width': np.mean(ci_upper - ci_lower),
            'max_ci_width': np.max(ci_upper - ci_lower),
            'ci_width_growth': ((ci_upper[-1] - ci_lower[-1]) - (ci_upper[0] - ci_lower[0])) / (ci_upper[0] - ci_lower[0]) * 100
        }
        
        # Calculate trend metrics
        if len(point_forecast) > 1:
            returns = np.diff(point_forecast) / point_forecast[:-1]
            metrics['positive_days_pct'] = (returns > 0).sum() / len(returns) * 100
            metrics['avg_daily_return'] = np.mean(returns) * 100
            metrics['sharpe_ratio_forecast'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return metrics
    
    def save_forecast_results(self, forecast_results: Dict[str, Any], 
                            filepath: str):
        """
        Save forecast results to file
        
        Args:
            forecast_results: Forecast results dictionary
            filepath: Path to save results
        """
        import pickle
        
        # Convert numpy arrays to lists for serialization
        serializable_results = forecast_results.copy()
        
        for key, value in serializable_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, pd.DatetimeIndex):
                serializable_results[key] = value.strftime('%Y-%m-%d').tolist()
            elif isinstance(value, pd.Timestamp):
                serializable_results[key] = value.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_results, f)
        
        print(f"Forecast results saved to {filepath}")
    
    def load_forecast_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load forecast results from file
        
        Args:
            filepath: Path to saved results
            
        Returns:
            Forecast results dictionary
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            loaded_results = pickle.load(f)
        
        # Convert back to appropriate types
        if 'forecast_dates' in loaded_results:
            loaded_results['forecast_dates'] = pd.to_datetime(loaded_results['forecast_dates'])
        
        for key in ['point_forecast', 'confidence_lower', 'confidence_upper']:
            if key in loaded_results:
                loaded_results[key] = np.array(loaded_results[key])
        
        return loaded_results