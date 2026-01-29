"""
Trend Analysis Module for Task 3
Analyzes forecast trends, identifies patterns, and assesses market opportunities/risks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class TrendAnalyzer:
    """Analyzes trends, patterns, and market implications of forecasts"""
    
    def __init__(self):
        """Initialize trend analyzer"""
        self.trend_results = {}
        self.risk_assessment = {}
        
    def analyze_trends(self, forecast: np.ndarray, 
                      forecast_dates: pd.DatetimeIndex,
                      window_sizes: List[int] = [5, 20, 60]) -> Dict[str, Any]:
        """
        Analyze trends in the forecast
        
        Args:
            forecast: Forecast values
            forecast_dates: Corresponding dates
            window_sizes: Window sizes for moving averages
            
        Returns:
            Dictionary with trend analysis results
        """
        print("Analyzing forecast trends...")
        
        results = {
            'raw_forecast': forecast,
            'forecast_dates': forecast_dates,
            'moving_averages': {},
            'trend_direction': None,
            'trend_strength': None,
            'momentum_indicators': {},
            'breakpoints': []
        }
        
        # Calculate moving averages
        for window in window_sizes:
            if len(forecast) >= window:
                ma = pd.Series(forecast).rolling(window=window).mean().values
                results['moving_averages'][f'ma_{window}'] = ma
        
        # Determine overall trend direction
        if len(forecast) > 1:
            # Linear regression for trend
            x = np.arange(len(forecast))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, forecast)
            
            results['trend_slope'] = slope
            results['trend_intercept'] = intercept
            results['trend_r_squared'] = r_value ** 2
            results['trend_p_value'] = p_value
            
            # Categorize trend
            if slope > 0:
                results['trend_direction'] = 'upward'
            elif slope < 0:
                results['trend_direction'] = 'downward'
            else:
                results['trend_direction'] = 'stable'
            
            # Trend strength
            if abs(slope / forecast.mean()) > 0.001:
                results['trend_strength'] = 'strong'
            else:
                results['trend_strength'] = 'weak'
        
        # Calculate momentum indicators
        if len(forecast) > 5:
            returns = np.diff(forecast) / forecast[:-1]
            
            results['momentum_indicators']['avg_return'] = np.mean(returns) * 100
            results['momentum_indicators']['return_volatility'] = np.std(returns) * 100
            results['momentum_indicators']['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # RSI-like indicator (simplified)
            gains = returns[returns > 0]
            losses = returns[returns < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                results['momentum_indicators']['rs_index'] = 100 - (100 / (1 + rs))
            else:
                results['momentum_indicators']['rs_index'] = 100
        
        # Detect structural breaks (simplified)
        if len(forecast) > 30:
            breakpoints = self._detect_structural_breaks(forecast)
            results['breakpoints'] = breakpoints
        
        self.trend_results['latest_analysis'] = results
        
        return results
    
    def _detect_structural_breaks(self, series: np.ndarray, 
                                 min_segment_length: int = 10) -> List[int]:
        """
        Detect structural breaks in time series
        
        Args:
            series: Time series data
            min_segment_length: Minimum length for a segment
            
        Returns:
            List of breakpoint indices
        """
        breakpoints = []
        
        # Simplified breakpoint detection using rolling statistics
        window = min(20, len(series) // 4)
        
        if len(series) > window * 2:
            rolling_mean = pd.Series(series).rolling(window=window).mean().dropna().values
            rolling_std = pd.Series(series).rolling(window=window).std().dropna().values
            
            # Look for significant changes in rolling statistics
            for i in range(1, len(rolling_mean)):
                mean_change = abs(rolling_mean[i] - rolling_mean[i-1]) / rolling_mean[i-1]
                std_change = abs(rolling_std[i] - rolling_std[i-1]) / rolling_std[i-1] if rolling_std[i-1] > 0 else 0
                
                if mean_change > 0.05 or std_change > 0.2:
                    breakpoint_idx = i + window
                    if breakpoint_idx not in breakpoints:
                        breakpoints.append(breakpoint_idx)
        
        return breakpoints
    
    def assess_opportunities_risks(self, forecast: np.ndarray,
                                  ci_lower: np.ndarray,
                                  ci_upper: np.ndarray,
                                  historical_volatility: float = 0.02) -> Dict[str, Any]:
        """
        Assess market opportunities and risks from forecast
        
        Args:
            forecast: Point forecast values
            ci_lower: Lower confidence bounds
            ci_upper: Upper confidence bounds
            historical_volatility: Historical volatility for comparison
            
        Returns:
            Dictionary with opportunities and risks assessment
        """
        print("Assessing market opportunities and risks...")
        
        assessment = {
            'opportunities': [],
            'risks': [],
            'uncertainty_metrics': {},
            'scenario_analysis': {}
        }
        
        # Calculate basic statistics
        forecast_returns = np.diff(forecast) / forecast[:-1]
        
        if len(forecast_returns) > 0:
            avg_return = np.mean(forecast_returns)
            volatility = np.std(forecast_returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Opportunities assessment
            if avg_return > 0:
                assessment['opportunities'].append({
                    'type': 'positive_expected_return',
                    'description': f'Positive expected daily return: {avg_return*100:.2f}%',
                    'confidence': 'high' if sharpe_ratio > 0.5 else 'medium'
                })
            
            if forecast[-1] > forecast[0] * 1.1:  # 10% increase
                assessment['opportunities'].append({
                    'type': 'significant_appreciation',
                    'description': f'Significant price appreciation forecast: {(forecast[-1]/forecast[0]-1)*100:.1f}%',
                    'confidence': 'medium'
                })
            
            # Risks assessment
            if volatility > historical_volatility:
                assessment['risks'].append({
                    'type': 'high_volatility',
                    'description': f'Forecast volatility ({volatility*100:.2f}%) exceeds historical ({historical_volatility*100:.2f}%)',
                    'severity': 'high' if volatility > historical_volatility * 1.5 else 'medium'
                })
            
            max_drawdown = self._calculate_max_drawdown(forecast)
            if max_drawdown > 0.1:  # 10% max drawdown
                assessment['risks'].append({
                    'type': 'significant_drawdown_risk',
                    'description': f'Maximum forecast drawdown: {max_drawdown*100:.1f}%',
                    'severity': 'high' if max_drawdown > 0.2 else 'medium'
                })
            
            # Uncertainty metrics
            ci_width = ci_upper - ci_lower
            ci_width_pct = ci_width / forecast * 100
            
            assessment['uncertainty_metrics'] = {
                'avg_ci_width_pct': np.mean(ci_width_pct),
                'ci_width_growth': ((ci_width_pct[-1] - ci_width_pct[0]) / ci_width_pct[0] * 100) if ci_width_pct[0] > 0 else 0,
                'max_ci_width_pct': np.max(ci_width_pct),
                'uncertainty_trend': 'increasing' if ci_width_pct[-1] > ci_width_pct[0] else 'decreasing'
            }
            
            # Scenario analysis
            assessment['scenario_analysis'] = self._generate_scenarios(
                forecast, ci_lower, ci_upper
            )
        
        self.risk_assessment['latest_assessment'] = assessment
        
        return assessment
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown (0-1)
        """
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _generate_scenarios(self, forecast: np.ndarray,
                          ci_lower: np.ndarray,
                          ci_upper: np.ndarray) -> Dict[str, Dict]:
        """
        Generate different market scenarios
        
        Args:
            forecast: Point forecast
            ci_lower: Lower confidence bound
            ci_upper: Upper confidence bound
            
        Returns:
            Dictionary of scenarios
        """
        scenarios = {
            'bull': {
                'probability': 0.3,
                'description': 'Optimistic growth scenario',
                'final_price': ci_upper[-1],
                'total_return': (ci_upper[-1] - forecast[0]) / forecast[0] * 100,
                'key_drivers': ['Strong earnings', 'Market optimism', 'Economic growth']
            },
            'base': {
                'probability': 0.5,
                'description': 'Base case scenario',
                'final_price': forecast[-1],
                'total_return': (forecast[-1] - forecast[0]) / forecast[0] * 100,
                'key_drivers': ['Moderate growth', 'Stable market conditions', 'Expected performance']
            },
            'bear': {
                'probability': 0.2,
                'description': 'Pessimistic downturn scenario',
                'final_price': ci_lower[-1],
                'total_return': (ci_lower[-1] - forecast[0]) / forecast[0] * 100,
                'key_drivers': ['Market correction', 'Economic slowdown', 'Increased volatility']
            }
        }
        
        return scenarios
    
    def plot_trend_analysis(self, forecast_results: Dict[str, Any],
                           trend_results: Dict[str, Any],
                           save_path: Optional[str] = None):
        """
        Visualize trend analysis results
        
        Args:
            forecast_results: Forecast results dictionary
            trend_results: Trend analysis results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        forecast = forecast_results['point_forecast']
        forecast_dates = forecast_results['forecast_dates']
        
        # 1. Forecast with moving averages
        ax = axes[0, 0]
        ax.plot(forecast_dates, forecast, 'b-', linewidth=2, label='Forecast', alpha=0.8)
        
        # Plot moving averages
        colors = ['green', 'orange', 'red']
        for idx, (ma_name, ma_values) in enumerate(trend_results['moving_averages'].items()):
            if len(ma_values) == len(forecast_dates):
                ax.plot(forecast_dates, ma_values, '--', color=colors[idx % len(colors)],
                       linewidth=1.5, label=f'{ma_name}', alpha=0.7)
        
        ax.set_title('Forecast with Moving Averages', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Trend line
        ax = axes[0, 1]
        ax.plot(forecast_dates, forecast, 'b-', linewidth=1, alpha=0.5, label='Forecast')
        
        # Plot trend line
        if 'trend_slope' in trend_results:
            x = np.arange(len(forecast))
            trend_line = trend_results['trend_slope'] * x + trend_results['trend_intercept']
            ax.plot(forecast_dates, trend_line, 'r--', linewidth=2, 
                   label=f"Trend (slope: {trend_results['trend_slope']:.4f})")
            
            # Add trend direction annotation
            direction = trend_results['trend_direction']
            strength = trend_results['trend_strength']
            ax.text(0.02, 0.98, f'Trend: {strength} {direction}',
                   transform=ax.transAxes,
                   fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('Trend Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Returns distribution
        ax = axes[1, 0]
        if len(forecast) > 1:
            returns = np.diff(forecast) / forecast[:-1] * 100
            ax.hist(returns, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.set_title('Forecast Returns Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 4. Cumulative returns
        ax = axes[1, 1]
        if len(forecast) > 1:
            cumulative_returns = np.cumsum(np.diff(forecast))
            ax.plot(forecast_dates[1:], cumulative_returns, 'purple-', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Confidence interval analysis
        ax = axes[2, 0]
        ci_width = forecast_results['confidence_upper'] - forecast_results['confidence_lower']
        ci_width_pct = ci_width / forecast * 100
        
        ax.plot(forecast_dates, ci_width_pct, 'r-', linewidth=2)
        ax.set_title('Confidence Interval Width Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('CI Width (% of Value)')
        ax.grid(True, alpha=0.3)
        
        # Add uncertainty trend annotation
        if ci_width_pct[-1] > ci_width_pct[0]:
            trend_text = 'Uncertainty Increasing'
            color = 'red'
        else:
            trend_text = 'Uncertainty Decreasing'
            color = 'green'
        
        ax.text(0.02, 0.98, trend_text,
               transform=ax.transAxes,
               fontsize=12,
               color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Scenario probabilities
        ax = axes[2, 1]
        if 'scenario_analysis' in self.risk_assessment.get('latest_assessment', {}):
            scenarios = self.risk_assessment['latest_assessment']['scenario_analysis']
            
            scenario_names = list(scenarios.keys())
            probabilities = [scenarios[s]['probability'] for s in scenario_names]
            colors_scenarios = ['green', 'blue', 'red']
            
            bars = ax.bar(scenario_names, probabilities, color=colors_scenarios, alpha=0.7)
            ax.set_title('Scenario Probability Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Scenario')
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
            
            # Add value labels
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.0%}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Comprehensive Trend Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trend analysis figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, forecast_results: Dict[str, Any],
                       trend_results: Dict[str, Any],
                       risk_assessment: Dict[str, Any]) -> str:
        """
        Generate a comprehensive trend analysis report
        
        Args:
            forecast_results: Forecast results
            trend_results: Trend analysis results
            risk_assessment: Risk assessment results
            
        Returns:
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "FORECAST TREND ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Executive Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 40 + "\n\n"
        
        if 'trend_direction' in trend_results:
            direction = trend_results['trend_direction']
            strength = trend_results['trend_strength']
            report += f"Primary Trend: {strength.upper()} {direction.upper()}\n"
        
        if 'total_forecast_change' in forecast_results.get('metrics', {}):
            change = forecast_results['metrics']['total_forecast_change']
            report += f"Total Forecast Change: {change:+.2f}%\n"
        
        report += f"Forecast Horizon: {len(forecast_results['point_forecast'])} days\n"
        report += f"Confidence Level: {forecast_results['confidence_level']*100:.0f}%\n\n"
        
        # Trend Analysis
        report += "TREND ANALYSIS\n"
        report += "-" * 40 + "\n\n"
        
        if 'trend_slope' in trend_results:
            report += f"Trend Slope: {trend_results['trend_slope']:.6f} (units/day)\n"
            report += f"Trend R-squared: {trend_results['trend_r_squared']:.4f}\n"
            report += f"Trend P-value: {trend_results['trend_p_value']:.6f}\n\n"
        
        if 'momentum_indicators' in trend_results:
            momentum = trend_results['momentum_indicators']
            report += "Momentum Indicators:\n"
            for key, value in momentum.items():
                report += f"  - {key.replace('_', ' ').title()}: {value:.4f}\n"
            report += "\n"
        
        # Opportunities and Risks
        report += "OPPORTUNITIES AND RISKS ASSESSMENT\n"
        report += "-" * 40 + "\n\n"
        
        if 'opportunities' in risk_assessment:
            report += "OPPORTUNITIES:\n"
            for opp in risk_assessment['opportunities']:
                report += f"  • {opp['description']} (Confidence: {opp['confidence'].upper()})\n"
            report += "\n"
        
        if 'risks' in risk_assessment:
            report += "RISKS:\n"
            for risk in risk_assessment['risks']:
                report += f"  • {risk['description']} (Severity: {risk['severity'].upper()})\n"
            report += "\n"
        
        # Uncertainty Analysis
        report += "UNCERTAINTY ANALYSIS\n"
        report += "-" * 40 + "\n\n"
        
        if 'uncertainty_metrics' in risk_assessment:
            uncertainty = risk_assessment['uncertainty_metrics']
            report += "Confidence Interval Metrics:\n"
            for key, value in uncertainty.items():
                report += f"  - {key.replace('_', ' ').title()}: {value:.4f}\n"
            report += "\n"
        
        # Scenario Analysis
        report += "SCENARIO ANALYSIS\n"
        report += "-" * 40 + "\n\n"
        
        if 'scenario_analysis' in risk_assessment:
            scenarios = risk_assessment['scenario_analysis']
            
            for scenario_name, scenario in scenarios.items():
                report += f"{scenario_name.upper()} SCENARIO ({scenario['probability']:.0%}):\n"
                report += f"  Description: {scenario['description']}\n"
                report += f"  Final Price: ${scenario['final_price']:.2f}\n"
                report += f"  Total Return: {scenario['total_return']:+.2f}%\n"
                report += f"  Key Drivers: {', '.join(scenario['key_drivers'])}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-" * 40 + "\n\n"
        
        # Generate recommendations based on analysis
        if trend_results.get('trend_direction') == 'upward':
            report += "1. Consider increasing exposure to capture upward trend\n"
        elif trend_results.get('trend_direction') == 'downward':
            report += "1. Consider reducing exposure or implementing hedging strategies\n"
        
        if risk_assessment.get('risks'):
            report += "2. Monitor identified risks closely and implement risk controls\n"
        
        if risk_assessment.get('uncertainty_metrics', {}).get('uncertainty_trend') == 'increasing':
            report += "3. Uncertainty increasing with forecast horizon - maintain flexible positioning\n"
        
        report += "4. Regularly update forecasts as new data becomes available\n"
        report += "5. Use forecasts as one input among many in investment decision process\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 80
        
        return report