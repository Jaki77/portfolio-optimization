"""
Portfolio Backtesting Engine
Simulates and compares portfolio performance against benchmark
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# Financial calculations
from scipy import stats
import math

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import PercentFormatter


class PortfolioBacktester:
    """Backtesting engine for portfolio strategies"""
    
    def __init__(self, prices: pd.DataFrame, 
                 initial_capital: float = 100000,
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtester
        
        Args:
            prices: DataFrame of asset prices (columns: assets, index: dates)
            initial_capital: Initial investment capital
            risk_free_rate: Annual risk-free rate
            transaction_cost: Percentage transaction cost per trade
            slippage: Percentage slippage per trade
        """
        self.prices = prices
        self.assets = prices.columns.tolist()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        
    def run_backtest(self, strategy_weights: Dict[str, float],
                     benchmark_weights: Dict[str, float],
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     rebalancing_freq: str = 'monthly',
                     rebalancing_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Run backtest for strategy vs benchmark
        
        Args:
            strategy_weights: Dictionary of strategy portfolio weights
            benchmark_weights: Dictionary of benchmark portfolio weights
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)
            rebalancing_freq: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly', 'none')
            rebalancing_threshold: Threshold for threshold-based rebalancing
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Running backtest from {start_date or self.prices.index[0]} "
              f"to {end_date or self.prices.index[-1]}...")
        
        # Filter data for backtest period
        if start_date:
            prices_subset = self.prices.loc[start_date:end_date] if end_date else self.prices.loc[start_date:]
        else:
            prices_subset = self.prices
        
        if len(prices_subset) == 0:
            raise ValueError("No data available for backtest period")
        
        print(f"Backtest period: {len(prices_subset)} trading days")
        
        # Validate weights
        strategy_weights = self._validate_weights(strategy_weights)
        benchmark_weights = self._validate_weights(benchmark_weights)
        
        # Run strategy backtest
        strategy_results = self._run_portfolio_backtest(
            prices_subset, strategy_weights, 'strategy',
            rebalancing_freq, rebalancing_threshold
        )
        
        # Run benchmark backtest
        benchmark_results = self._run_portfolio_backtest(
            prices_subset, benchmark_weights, 'benchmark',
            'none', 0  # Benchmark typically not rebalanced
        )
        
        # Calculate performance metrics
        strategy_metrics = self._calculate_performance_metrics(
            strategy_results['portfolio_value'],
            strategy_results['returns'],
            strategy_results['dates']
        )
        
        benchmark_metrics = self._calculate_performance_metrics(
            benchmark_results['portfolio_value'],
            benchmark_results['returns'],
            benchmark_results['dates']
        )
        
        # Calculate relative metrics
        relative_metrics = self._calculate_relative_metrics(
            strategy_results, benchmark_results
        )
        
        # Store results
        results = {
            'strategy': {
                'results': strategy_results,
                'metrics': strategy_metrics,
                'weights': strategy_weights
            },
            'benchmark': {
                'results': benchmark_results,
                'metrics': benchmark_metrics,
                'weights': benchmark_weights
            },
            'relative': relative_metrics,
            'backtest_period': {
                'start': prices_subset.index[0],
                'end': prices_subset.index[-1],
                'days': len(prices_subset)
            },
            'parameters': {
                'initial_capital': self.initial_capital,
                'rebalancing_freq': rebalancing_freq,
                'rebalancing_threshold': rebalancing_threshold,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage
            }
        }
        
        self.backtest_results = results
        print("Backtest completed successfully.")
        
        return results
    
    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize portfolio weights"""
        # Check if all assets exist
        for asset in weights.keys():
            if asset not in self.assets:
                raise ValueError(f"Asset {asset} not in price data")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            print(f"Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _run_portfolio_backtest(self, prices: pd.DataFrame,
                               weights: Dict[str, float],
                               portfolio_name: str,
                               rebalancing_freq: str,
                               rebalancing_threshold: float) -> Dict[str, Any]:
        """
        Run backtest for a single portfolio
        
        Args:
            prices: Price data for backtest period
            weights: Portfolio weights
            portfolio_name: Name of portfolio for logging
            rebalancing_freq: Rebalancing frequency
            rebalancing_threshold: Rebalancing threshold
            
        Returns:
            Portfolio backtest results
        """
        print(f"  Running {portfolio_name} portfolio backtest...")
        
        # Initialize tracking variables
        n_days = len(prices)
        dates = prices.index
        asset_prices = prices.values
        
        # Convert weights to array in correct asset order
        weight_array = np.array([weights.get(asset, 0) for asset in self.assets])
        
        # Initialize portfolio
        portfolio_value = np.zeros(n_days)
        portfolio_returns = np.zeros(n_days)
        current_weights = weight_array.copy()
        cash = self.initial_capital
        
        # Track transactions
        transactions = []
        rebalance_dates = []
        
        # Day 0: Initial investment
        portfolio_value[0] = self.initial_capital
        portfolio_returns[0] = 0
        
        # Track asset values
        asset_values = np.zeros((n_days, len(self.assets)))
        asset_values[0] = weight_array * self.initial_capital
        
        # Main backtest loop
        for day in range(1, n_days):
            # Calculate portfolio value from previous day's weights
            daily_returns = (asset_prices[day] / asset_prices[day-1]) - 1
            
            # Update asset values
            asset_values[day] = asset_values[day-1] * (1 + daily_returns)
            
            # Calculate total portfolio value
            portfolio_value[day] = np.sum(asset_values[day]) + cash
            
            # Calculate daily return
            portfolio_returns[day] = (portfolio_value[day] / portfolio_value[day-1]) - 1
            
            # Check for rebalancing
            if self._should_rebalance(day, dates[day], current_weights, 
                                     asset_values[day], portfolio_value[day],
                                     rebalancing_freq, rebalancing_threshold):
                
                rebalance_dates.append(dates[day])
                
                # Calculate target asset values
                target_asset_values = weight_array * portfolio_value[day]
                
                # Calculate trades needed
                trades = target_asset_values - asset_values[day]
                
                # Apply transaction costs and slippage
                trade_costs = np.abs(trades) * (self.transaction_cost + self.slippage)
                total_cost = np.sum(trade_costs)
                
                # Execute rebalancing
                if total_cost > 0:
                    # Adjust cash for transaction costs
                    cash -= total_cost
                    
                    # Record transaction
                    transactions.append({
                        'date': dates[day],
                        'trades': trades.copy(),
                        'cost': total_cost,
                        'portfolio_value': portfolio_value[day]
                    })
                
                # Update asset values and weights
                asset_values[day] = target_asset_values
                current_weights = asset_values[day] / portfolio_value[day]
        
        # Compile results
        results = {
            'portfolio_value': portfolio_value,
            'returns': portfolio_returns,
            'dates': dates,
            'asset_values': asset_values,
            'transactions': transactions,
            'rebalance_dates': rebalance_dates,
            'final_weights': current_weights,
            'cash': cash,
            'total_transaction_costs': sum(t['cost'] for t in transactions)
        }
        
        print(f"    Final portfolio value: ${portfolio_value[-1]:,.2f}")
        print(f"    Total return: {(portfolio_value[-1]/self.initial_capital - 1)*100:.2f}%")
        print(f"    Transaction costs: ${results['total_transaction_costs']:,.2f}")
        print(f"    Rebalances: {len(rebalance_dates)}")
        
        return results
    
    def _should_rebalance(self, day: int, date: pd.Timestamp,
                         current_weights: np.ndarray,
                         asset_values: np.ndarray,
                         portfolio_value: float,
                         rebalancing_freq: str,
                         rebalancing_threshold: float) -> bool:
        """
        Determine if portfolio should be rebalanced
        
        Args:
            day: Current day index
            date: Current date
            current_weights: Current portfolio weights
            asset_values: Current asset values
            portfolio_value: Current portfolio value
            rebalancing_freq: Rebalancing frequency
            rebalancing_threshold: Rebalancing threshold
            
        Returns:
            True if portfolio should be rebalanced
        """
        if rebalancing_freq == 'none':
            return False
        
        # Time-based rebalancing
        if rebalancing_freq == 'daily':
            return True
        elif rebalancing_freq == 'weekly':
            return date.weekday() == 0  # Monday
        elif rebalancing_freq == 'monthly':
            return date.day == 1
        elif rebalancing_freq == 'quarterly':
            return date.month in [1, 4, 7, 10] and date.day == 1
        
        # Threshold-based rebalancing
        elif rebalancing_freq == 'threshold':
            # Calculate target weights from asset_values and portfolio_value
            target_weights = asset_values / portfolio_value
            
            # Check if any weight deviates beyond threshold
            max_deviation = np.max(np.abs(current_weights - target_weights))
            return max_deviation > rebalancing_threshold
        
        return False
    
    def _calculate_performance_metrics(self, portfolio_value: np.ndarray,
                                      portfolio_returns: np.ndarray,
                                      dates: pd.DatetimeIndex) -> Dict[str, float]:
        """
        Calculate performance metrics for a portfolio
        
        Args:
            portfolio_value: Array of portfolio values
            portfolio_returns: Array of portfolio returns
            dates: Corresponding dates
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio_value) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_value[-1] / portfolio_value[0]) - 1
        
        # Annualized metrics
        days_in_period = len(portfolio_value)
        years = days_in_period / 252  # Approximate trading days in a year
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
            annualized_volatility = np.std(portfolio_returns[1:]) * np.sqrt(252)
        else:
            annualized_return = total_return
            annualized_volatility = 0
        
        # Risk-adjusted metrics
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown, drawdown_start, drawdown_end = self._calculate_max_drawdown(portfolio_value)
        
        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0
        
        # Sortino ratio (downside risk)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # Value at Risk (95%)
        var_95 = -np.percentile(portfolio_returns, 5) * 100
        
        # Win rate
        positive_days = np.sum(portfolio_returns > 0)
        total_days = len(portfolio_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(portfolio_returns[portfolio_returns > 0])
        gross_loss = abs(np.sum(portfolio_returns[portfolio_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'value_at_risk_95': var_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'positive_days': positive_days,
            'total_days': total_days,
            'final_value': portfolio_value[-1],
            'peak_value': np.max(portfolio_value),
            'trough_value': np.min(portfolio_value),
            'drawdown_start': dates[drawdown_start] if drawdown_start is not None else None,
            'drawdown_end': dates[drawdown_end] if drawdown_end is not None else None,
            'drawdown_duration': (drawdown_end - drawdown_start) if drawdown_start is not None and drawdown_end is not None else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Args:
            values: Array of portfolio values
            
        Returns:
            Maximum drawdown, start index, end index
        """
        peak = values[0]
        max_dd = 0
        peak_idx = 0
        trough_idx = 0
        
        for i in range(1, len(values)):
            if values[i] > peak:
                peak = values[i]
                peak_idx = i
            
            dd = (peak - values[i]) / peak
            
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
        
        return max_dd, peak_idx, trough_idx
    
    def _calculate_relative_metrics(self, strategy_results: Dict[str, Any],
                                   benchmark_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate relative performance metrics
        
        Args:
            strategy_results: Strategy backtest results
            benchmark_results: Benchmark backtest results
            
        Returns:
            Dictionary of relative metrics
        """
        strategy_value = strategy_results['portfolio_value']
        benchmark_value = benchmark_results['portfolio_value']
        strategy_returns = strategy_results['returns']
        benchmark_returns = benchmark_results['returns']
        
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Alpha and Beta
        if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = 0
            
            # Annualize returns for alpha calculation
            strategy_annual_return = (1 + np.mean(strategy_returns)) ** 252 - 1
            benchmark_annual_return = (1 + np.mean(benchmark_returns)) ** 252 - 1
            
            alpha = strategy_annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        else:
            beta = 0
            alpha = 0
        
        # Tracking error
        tracking_error = np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0
        
        # Information ratio
        avg_excess_return = np.mean(excess_returns) * 252 if len(excess_returns) > 0 else 0
        information_ratio = avg_excess_return / tracking_error if tracking_error > 0 else 0
        
        # Relative return
        relative_return = (strategy_value[-1] / benchmark_value[-1] - 1) * 100
        
        # Capture ratios
        up_market_returns_benchmark = benchmark_returns[benchmark_returns > 0]
        down_market_returns_benchmark = benchmark_returns[benchmark_returns < 0]
        
        if len(up_market_returns_benchmark) > 0:
            up_capture = np.mean(strategy_returns[benchmark_returns > 0]) / np.mean(up_market_returns_benchmark)
        else:
            up_capture = 0
        
        if len(down_market_returns_benchmark) > 0:
            down_capture = np.mean(strategy_returns[benchmark_returns < 0]) / np.mean(down_market_returns_benchmark)
        else:
            down_capture = 0
        
        relative_metrics = {
            'excess_return': relative_return,
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'capture_ratio': up_capture / abs(down_capture) if down_capture != 0 else np.inf
        }
        
        return relative_metrics
    
    def plot_cumulative_returns(self, save_path: Optional[str] = None):
        """
        Plot cumulative returns comparison
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        results = self.backtest_results
        strategy_value = results['strategy']['results']['portfolio_value']
        benchmark_value = results['benchmark']['results']['portfolio_value']
        dates = results['strategy']['results']['dates']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot cumulative returns
        ax1.plot(dates, strategy_value, 
                label=f"Strategy (Max Sharpe)", 
                color='#2E86AB', linewidth=2)
        
        ax1.plot(dates, benchmark_value, 
                label=f"Benchmark (60/40)", 
                color='#A23B72', linewidth=2)
        
        ax1.set_title('Portfolio Value: Strategy vs Benchmark', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add final value annotations
        ax1.text(0.02, 0.98, 
                f"Strategy Final: ${strategy_value[-1]:,.0f}\n"
                f"Benchmark Final: ${benchmark_value[-1]:,.0f}",
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        # Plot relative performance (Strategy/Benchmark)
        relative_performance = (strategy_value / benchmark_value - 1) * 100
        
        ax2.fill_between(dates, relative_performance, 0,
                        where=relative_performance >= 0,
                        color='green', alpha=0.3, label='Outperformance')
        
        ax2.fill_between(dates, relative_performance, 0,
                        where=relative_performance < 0,
                        color='red', alpha=0.3, label='Underperformance')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Relative Performance: Strategy vs Benchmark', 
                     fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Outperformance (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """
        Plot drawdown comparison
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        results = self.backtest_results
        strategy_value = results['strategy']['results']['portfolio_value']
        benchmark_value = results['benchmark']['results']['portfolio_value']
        dates = results['strategy']['results']['dates']
        
        # Calculate drawdowns
        strategy_dd = self._calculate_drawdown_series(strategy_value)
        benchmark_dd = self._calculate_drawdown_series(benchmark_value)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot drawdowns
        ax.fill_between(dates, 0, strategy_dd * 100,
                       color='#2E86AB', alpha=0.5, label='Strategy Drawdown')
        
        ax.fill_between(dates, 0, benchmark_dd * 100,
                       color='#A23B72', alpha=0.3, label='Benchmark Drawdown')
        
        ax.set_title('Maximum Drawdown Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add maximum drawdown annotations
        strategy_max_dd = results['strategy']['metrics']['max_drawdown'] * 100
        benchmark_max_dd = results['benchmark']['metrics']['max_drawdown'] * 100
        
        ax.text(0.02, 0.98,
               f"Strategy Max DD: {strategy_max_dd:.1f}%\n"
               f"Benchmark Max DD: {benchmark_max_dd:.1f}%",
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_drawdown_series(self, values: np.ndarray) -> np.ndarray:
        """Calculate drawdown series for plotting"""
        drawdown = np.zeros(len(values))
        peak = values[0]
        
        for i in range(len(values)):
            if values[i] > peak:
                peak = values[i]
            
            drawdown[i] = (peak - values[i]) / peak if peak > 0 else 0
        
        return drawdown
    
    def plot_rolling_metrics(self, window: int = 60, save_path: Optional[str] = None):
        """
        Plot rolling performance metrics
        
        Args:
            window: Rolling window size (in days)
            save_path: Optional path to save figure
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        results = self.backtest_results
        strategy_returns = results['strategy']['results']['returns']
        benchmark_returns = results['benchmark']['results']['returns']
        dates = results['strategy']['results']['dates'][1:]  # Exclude first day (0 return)
        
        # Calculate rolling metrics
        strategy_rolling_sharpe = self._calculate_rolling_sharpe(strategy_returns, window)
        benchmark_rolling_sharpe = self._calculate_rolling_sharpe(benchmark_returns, window)
        
        strategy_rolling_vol = pd.Series(strategy_returns).rolling(window=window).std() * np.sqrt(252) * 100
        benchmark_rolling_vol = pd.Series(benchmark_returns).rolling(window=window).std() * np.sqrt(252) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Rolling Sharpe Ratio
        axes[0, 0].plot(dates[window-1:], strategy_rolling_sharpe[window-1:], 
                       label='Strategy', color='#2E86AB', linewidth=2)
        axes[0, 0].plot(dates[window-1:], benchmark_rolling_sharpe[window-1:], 
                       label='Benchmark', color='#A23B72', linewidth=2)
        axes[0, 0].set_title(f'Rolling Sharpe Ratio ({window}-day window)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling Volatility
        axes[0, 1].plot(dates[window-1:], strategy_rolling_vol[window-1:], 
                       label='Strategy', color='#2E86AB', linewidth=2)
        axes[0, 1].plot(dates[window-1:], benchmark_rolling_vol[window-1:], 
                       label='Benchmark', color='#A23B72', linewidth=2)
        axes[0, 1].set_title(f'Rolling Annualized Volatility ({window}-day window)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Volatility (%)', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Alpha
        rolling_alpha = self._calculate_rolling_alpha(strategy_returns, 
                                                     benchmark_returns, 
                                                     window)
        axes[1, 0].plot(dates[window-1:], rolling_alpha[window-1:], 
                       color='#C73E1D', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 0].set_title(f'Rolling Alpha ({window}-day window)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Alpha', fontsize=12)
        axes[1, 0].set_xlabel('Date', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Beta
        rolling_beta = self._calculate_rolling_beta(strategy_returns, 
                                                   benchmark_returns, 
                                                   window)
        axes[1, 1].plot(dates[window-1:], rolling_beta[window-1:], 
                       color='#F18F01', linewidth=2)
        axes[1, 1].axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].set_title(f'Rolling Beta ({window}-day window)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Beta', fontsize=12)
        axes[1, 1].set_xlabel('Date', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Format x-axis dates
        for ax in axes.flatten():
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = pd.Series(returns).rolling(window=window).mean() * 252
        rolling_std = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_std
        
        return rolling_sharpe.fillna(0).values
    
    def _calculate_rolling_alpha(self, strategy_returns: np.ndarray,
                                benchmark_returns: np.ndarray,
                                window: int) -> np.ndarray:
        """Calculate rolling alpha"""
        rolling_alpha = np.zeros(len(strategy_returns))
        
        for i in range(window, len(strategy_returns)):
            # Get window returns
            strat_window = strategy_returns[i-window:i]
            bench_window = benchmark_returns[i-window:i]
            
            # Calculate beta for window
            covariance = np.cov(strat_window, bench_window)[0, 1]
            bench_variance = np.var(bench_window)
            
            if bench_variance > 0:
                beta = covariance / bench_variance
            else:
                beta = 0
            
            # Calculate alpha
            strat_mean = np.mean(strat_window) * 252
            bench_mean = np.mean(bench_window) * 252
            alpha = strat_mean - (self.risk_free_rate + beta * (bench_mean - self.risk_free_rate))
            
            rolling_alpha[i] = alpha
        
        return rolling_alpha
    
    def _calculate_rolling_beta(self, strategy_returns: np.ndarray,
                               benchmark_returns: np.ndarray,
                               window: int) -> np.ndarray:
        """Calculate rolling beta"""
        rolling_beta = np.zeros(len(strategy_returns))
        
        for i in range(window, len(strategy_returns)):
            # Get window returns
            strat_window = strategy_returns[i-window:i]
            bench_window = benchmark_returns[i-window:i]
            
            # Calculate beta for window
            covariance = np.cov(strat_window, bench_window)[0, 1]
            bench_variance = np.var(bench_window)
            
            if bench_variance > 0:
                beta = covariance / bench_variance
            else:
                beta = 0
            
            rolling_beta[i] = beta
        
        return rolling_beta
    
    def plot_performance_summary(self, save_path: Optional[str] = None):
        """
        Plot performance summary with key metrics
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        results = self.backtest_results
        
        # Extract key metrics
        strategy_metrics = results['strategy']['metrics']
        benchmark_metrics = results['benchmark']['metrics']
        relative_metrics = results['relative']
        
        # Prepare data for bar chart
        metrics_to_plot = [
            'total_return',
            'annualized_return', 
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'calmar_ratio',
            'information_ratio'
        ]
        
        metric_names = [
            'Total Return',
            'Annualized Return',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown',
            'Calmar Ratio',
            'Information Ratio'
        ]
        
        strategy_values = []
        benchmark_values = []
        
        for metric in metrics_to_plot:
            if metric in strategy_metrics:
                strategy_values.append(strategy_metrics[metric])
            else:
                strategy_values.append(0)
            
            if metric in benchmark_metrics:
                benchmark_values.append(benchmark_metrics[metric])
            else:
                benchmark_values.append(0)
        
        # Convert percentages for display
        for i, metric in enumerate(metrics_to_plot):
            if metric in ['total_return', 'annualized_return', 'max_drawdown']:
                strategy_values[i] *= 100
                benchmark_values[i] *= 100
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strategy_values, width, 
                      label='Strategy', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, benchmark_values, width, 
                      label='Benchmark', color='#A23B72', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height) and not np.isinf(height):
                    if abs(height) < 0.01:  # Very small values
                        label = f'{height:.2e}'
                    elif abs(height) < 1:   # Ratios
                        label = f'{height:.2f}'
                    else:                   # Percentages
                        label = f'{height:.1f}%'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Performance Metric', fontsize=12)
        ax.set_title('Performance Metrics Comparison: Strategy vs Benchmark', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add outperformance annotation
        excess_return = relative_metrics.get('excess_return', 0)
        alpha = relative_metrics.get('alpha', 0)
        
        annotation_text = (
            f"Strategy Outperformance: {excess_return:.1f}%\n"
            f"Alpha: {alpha:.2%}"
        )
        
        ax.text(0.02, 0.98, annotation_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_backtest_report(self) -> str:
        """
        Generate comprehensive backtest report
        
        Returns:
            Formatted report string
        """
        if not self.backtest_results:
            return "No backtest results available. Run backtest first."
        
        results = self.backtest_results
        strategy_metrics = results['strategy']['metrics']
        benchmark_metrics = results['benchmark']['metrics']
        relative_metrics = results['relative']
        
        report = "=" * 80 + "\n"
        report += "BACKTESTING RESULTS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Backtest Summary
        report += "BACKTEST SUMMARY\n"
        report += "-" * 40 + "\n\n"
        
        period = results['backtest_period']
        report += f"Period: {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}\n"
        report += f"Duration: {period['days']} trading days\n"
        report += f"Initial Capital: ${self.initial_capital:,.0f}\n"
        report += f"Risk-Free Rate: {self.risk_free_rate:.1%}\n\n"
        
        # Strategy vs Benchmark Results
        report += "STRATEGY vs BENCHMARK PERFORMANCE\n"
        report += "-" * 40 + "\n\n"
        
        report += f"{'Metric':<25} {'Strategy':>15} {'Benchmark':>15} {'Difference':>15}\n"
        report += "-" * 70 + "\n"
        
        # Format and add each metric
        metrics_display = [
            ('Final Value', f"${strategy_metrics['final_value']:,.0f}", 
             f"${benchmark_metrics['final_value']:,.0f}",
             f"${strategy_metrics['final_value'] - benchmark_metrics['final_value']:,.0f}"),
            
            ('Total Return', f"{strategy_metrics['total_return']:.2%}", 
             f"{benchmark_metrics['total_return']:.2%}",
             f"{strategy_metrics['total_return'] - benchmark_metrics['total_return']:+.2%}"),
            
            ('Annualized Return', f"{strategy_metrics['annualized_return']:.2%}", 
             f"{benchmark_metrics['annualized_return']:.2%}",
             f"{strategy_metrics['annualized_return'] - benchmark_metrics['annualized_return']:+.2%}"),
            
            ('Annualized Volatility', f"{strategy_metrics['annualized_volatility']:.2%}", 
             f"{benchmark_metrics['annualized_volatility']:.2%}",
             f"{strategy_metrics['annualized_volatility'] - benchmark_metrics['annualized_volatility']:+.2%}"),
            
            ('Sharpe Ratio', f"{strategy_metrics['sharpe_ratio']:.3f}", 
             f"{benchmark_metrics['sharpe_ratio']:.3f}",
             f"{strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']:+.3f}"),
            
            ('Sortino Ratio', f"{strategy_metrics['sortino_ratio']:.3f}", 
             f"{benchmark_metrics['sortino_ratio']:.3f}",
             f"{strategy_metrics['sortino_ratio'] - benchmark_metrics['sortino_ratio']:+.3f}"),
            
            ('Max Drawdown', f"{strategy_metrics['max_drawdown']:.2%}", 
             f"{benchmark_metrics['max_drawdown']:.2%}",
             f"{strategy_metrics['max_drawdown'] - benchmark_metrics['max_drawdown']:+.2%}"),
            
            ('Calmar Ratio', f"{strategy_metrics['calmar_ratio']:.3f}", 
             f"{benchmark_metrics['calmar_ratio']:.3f}",
             f"{strategy_metrics['calmar_ratio'] - benchmark_metrics['calmar_ratio']:+.3f}"),
            
            ('Value at Risk (95%)', f"{strategy_metrics['value_at_risk_95']:.2f}%", 
             f"{benchmark_metrics['value_at_risk_95']:.2f}%",
             f"{strategy_metrics['value_at_risk_95'] - benchmark_metrics['value_at_risk_95']:+.2f}%"),
            
            ('Win Rate', f"{strategy_metrics['win_rate']:.1%}", 
             f"{benchmark_metrics['win_rate']:.1%}",
             f"{strategy_metrics['win_rate'] - benchmark_metrics['win_rate']:+.1%}"),
        ]
        
        for name, strategy_val, benchmark_val, diff in metrics_display:
            report += f"{name:<25} {strategy_val:>15} {benchmark_val:>15} {diff:>15}\n"
        
        # Relative Performance Metrics
        report += "\nRELATIVE PERFORMANCE METRICS\n"
        report += "-" * 40 + "\n\n"
        
        report += f"Alpha: {relative_metrics['alpha']:.2%}\n"
        report += f"Beta: {relative_metrics['beta']:.3f}\n"
        report += f"Tracking Error: {relative_metrics['tracking_error']:.2%}\n"
        report += f"Information Ratio: {relative_metrics['information_ratio']:.3f}\n"
        report += f"Up Capture Ratio: {relative_metrics['up_capture_ratio']:.3f}\n"
        report += f"Down Capture Ratio: {relative_metrics['down_capture_ratio']:.3f}\n"
        report += f"Capture Ratio: {relative_metrics['capture_ratio']:.3f}\n"
        
        # Strategy Details
        report += "\nSTRATEGY DETAILS\n"
        report += "-" * 40 + "\n\n"
        
        strategy_weights = results['strategy']['weights']
        report += "Portfolio Weights:\n"
        for asset, weight in strategy_weights.items():
            report += f"  {asset}: {weight:.1%}\n"
        
        report += f"\nRebalancing Frequency: {results['parameters']['rebalancing_freq']}\n"
        report += f"Transaction Costs: ${results['strategy']['results']['total_transaction_costs']:,.2f}\n"
        report += f"Number of Rebalances: {len(results['strategy']['results']['rebalance_dates'])}\n"
        
        # Drawdown Analysis
        report += "\nDRAWDOWN ANALYSIS\n"
        report += "-" * 40 + "\n\n"
        
        if strategy_metrics['drawdown_start'] and strategy_metrics['drawdown_end']:
            report += f"Maximum Drawdown Period:\n"
            report += f"  Start: {strategy_metrics['drawdown_start'].strftime('%Y-%m-%d')}\n"
            report += f"  End: {strategy_metrics['drawdown_end'].strftime('%Y-%m-%d')}\n"
            report += f"  Duration: {strategy_metrics['drawdown_duration']} days\n"
            report += f"  Depth: {strategy_metrics['max_drawdown']:.2%}\n"
        
        # Conclusions
        report += "\nCONCLUSIONS AND RECOMMENDATIONS\n"
        report += "-" * 40 + "\n\n"
        
        # Determine if strategy outperformed
        excess_return = relative_metrics['excess_return']
        alpha = relative_metrics['alpha']
        information_ratio = relative_metrics['information_ratio']
        
        if excess_return > 0 and alpha > 0:
            report += "✅ STRATEGY OUTPERFORMED BENCHMARK\n\n"
            report += "Key Strengths:\n"
            report += "1. Positive excess returns achieved\n"
            report += "2. Positive alpha indicates skill, not just risk-taking\n"
            
            if information_ratio > 0.5:
                report += "3. Strong information ratio indicates consistent outperformance\n"
            
            if relative_metrics['up_capture_ratio'] > 1 and relative_metrics['down_capture_ratio'] < 1:
                report += "4. Favorable capture ratios: participating in upsides, protecting in downsides\n"
            
            report += "\nRECOMMENDATION: Implement strategy with monitoring\n"
            
        elif excess_return > 0:
            report += "⚠️ STRATEGY MODESTLY OUTPERFORMED\n\n"
            report += "Analysis:\n"
            report += "1. Strategy generated excess returns but with minimal alpha\n"
            report += "2. Outperformance may be due to risk factors rather than skill\n"
            report += "3. Further analysis needed to validate edge\n"
            
            report += "\nRECOMMENDATION: Implement with caution, monitor closely\n"
            
        else:
            report += "❌ STRATEGY UNDERPERFORMED BENCHMARK\n\n"
            report += "Analysis:\n"
            report += "1. Strategy failed to beat traditional 60/40 portfolio\n"
            report += "2. Negative alpha suggests implementation issues\n"
            
            if relative_metrics['down_capture_ratio'] > 1:
                report += "3. Poor downside protection during market declines\n"
            
            report += "\nRECOMMENDATION: Review and refine strategy before implementation\n"
        
        # Limitations
        report += "\nLIMITATIONS OF BACKTEST\n"
        report += "-" * 40 + "\n\n"
        
        report += "1. Historical performance does not guarantee future results\n"
        report += "2. Assumes perfect execution with modeled transaction costs\n"
        report += "3. Does not account for market impact of large trades\n"
        report += "4. Uses simplified slippage and transaction cost models\n"
        report += "5. Single historical period may not represent all market regimes\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 80
        
        return report
    
    def save_backtest_results(self, filepath: str):
        """
        Save backtest results to file
        
        Args:
            filepath: Path to save results
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.backtest_results, f)
        
        print(f"Backtest results saved to {filepath}")
    
    def load_backtest_results(self, filepath: str):
        """
        Load backtest results from file
        
        Args:
            filepath: Path to saved results
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.backtest_results = pickle.load(f)
        
        print(f"Backtest results loaded from {filepath}")