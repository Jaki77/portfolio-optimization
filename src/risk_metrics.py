"""
Portfolio Risk Metrics Calculation
Advanced risk metrics beyond standard deviation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PortfolioRiskMetrics:
    """Calculate advanced risk metrics for portfolios"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level (0-1)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR at specified confidence level
        """
        if method == 'historical':
            # Historical VaR
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
        elif method == 'monte_carlo':
            # Monte Carlo VaR (simplified)
            n_simulations = 10000
            simulated_returns = np.random.normal(
                np.mean(returns), np.std(returns), n_simulations
            )
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level (0-1)
            
        Returns:
            CVaR at specified confidence level
        """
        var = PortfolioRiskMetrics.calculate_var(returns, confidence_level, 'historical')
        
        # Calculate average of losses beyond VaR
        losses_beyond_var = returns[returns < -var]
        
        if len(losses_beyond_var) > 0:
            cvar = -np.mean(losses_beyond_var)
        else:
            cvar = var  # If no losses beyond VaR, use VaR
        
        return cvar
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown (as decimal), start index, end index
        """
        peak = prices[0]
        max_dd = 0
        peak_idx = 0
        trough_idx = 0
        
        for i in range(1, len(prices)):
            if prices[i] > peak:
                peak = prices[i]
                peak_idx = i
            
            dd = (peak - prices[i]) / peak
            
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
        
        return max_dd, peak_idx, trough_idx
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                               target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return)
        
        Args:
            returns: Portfolio returns
            risk_free_rate: Risk-free rate
            target_return: Minimum acceptable return
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            downside_deviation = 0
        else:
            downside_deviation = np.std(downside_returns)
        
        avg_excess_return = np.mean(excess_returns)
        
        if downside_deviation > 0:
            sortino_ratio = avg_excess_return / downside_deviation
        else:
            sortino_ratio = np.inf if avg_excess_return > 0 else 0
        
        return sortino_ratio
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, prices: np.ndarray,
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate Calmar ratio (return to max drawdown ratio)
        
        Args:
            returns: Portfolio returns
            prices: Price series
            risk_free_rate: Risk-free rate
            
        Returns:
            Calmar ratio
        """
        avg_return = np.mean(returns)
        max_dd, _, _ = PortfolioRiskMetrics.calculate_max_drawdown(prices)
        
        if max_dd > 0:
            calmar_ratio = (avg_return - risk_free_rate) / max_dd
        else:
            calmar_ratio = np.inf if avg_return > risk_free_rate else 0
        
        return calmar_ratio
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns: np.ndarray,
                                benchmark_returns: np.ndarray) -> float:
        """
        Calculate tracking error
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tracking error (standard deviation of active returns)
        """
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        return tracking_error
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: np.ndarray,
                                   benchmark_returns: np.ndarray) -> float:
        """
        Calculate information ratio
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        active_returns = portfolio_returns - benchmark_returns
        avg_active_return = np.mean(active_returns)
        tracking_error = np.std(active_returns)
        
        if tracking_error > 0:
            information_ratio = avg_active_return / tracking_error
        else:
            information_ratio = np.inf if avg_active_return > 0 else 0
        
        return information_ratio
    
    @staticmethod
    def calculate_beta(portfolio_returns: np.ndarray,
                      market_returns: np.ndarray) -> float:
        """
        Calculate beta (systematic risk)
        
        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns
            
        Returns:
            Beta coefficient
        """
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance > 0:
            beta = covariance / market_variance
        else:
            beta = 0
        
        return beta
    
    @staticmethod
    def calculate_alpha(portfolio_returns: np.ndarray,
                       market_returns: np.ndarray,
                       risk_free_rate: float = 0.02) -> float:
        """
        Calculate alpha (excess risk-adjusted return)
        
        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Alpha
        """
        beta = PortfolioRiskMetrics.calculate_beta(portfolio_returns, market_returns)
        
        avg_portfolio_return = np.mean(portfolio_returns)
        avg_market_return = np.mean(market_returns)
        
        # Annualize returns if they are daily
        if len(portfolio_returns) > 252:  # More than a year of daily data
            avg_portfolio_return = (1 + avg_portfolio_return) ** 252 - 1
            avg_market_return = (1 + avg_market_return) ** 252 - 1
        
        expected_return = risk_free_rate + beta * (avg_market_return - risk_free_rate)
        alpha = avg_portfolio_return - expected_return
        
        return alpha
    
    @staticmethod
    def calculate_all_metrics(portfolio_returns: np.ndarray,
                            portfolio_prices: np.ndarray,
                            benchmark_returns: Optional[np.ndarray] = None,
                            market_returns: Optional[np.ndarray] = None,
                            risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate all risk metrics
        
        Args:
            portfolio_returns: Portfolio returns
            portfolio_prices: Portfolio prices
            benchmark_returns: Benchmark returns (optional)
            market_returns: Market returns (optional)
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of all risk metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['avg_return'] = np.mean(portfolio_returns)
        metrics['volatility'] = np.std(portfolio_returns)
        metrics['sharpe_ratio'] = (metrics['avg_return'] - risk_free_rate) / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        # Downside risk metrics
        metrics['var_95'] = PortfolioRiskMetrics.calculate_var(portfolio_returns, 0.95)
        metrics['cvar_95'] = PortfolioRiskMetrics.calculate_cvar(portfolio_returns, 0.95)
        metrics['sortino_ratio'] = PortfolioRiskMetrics.calculate_sortino_ratio(
            portfolio_returns, risk_free_rate
        )
        
        # Drawdown metrics
        max_dd, peak_idx, trough_idx = PortfolioRiskMetrics.calculate_max_drawdown(portfolio_prices)
        metrics['max_drawdown'] = max_dd
        metrics['calmar_ratio'] = PortfolioRiskMetrics.calculate_calmar_ratio(
            portfolio_returns, portfolio_prices, risk_free_rate
        )
        
        # Benchmark-relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            metrics['tracking_error'] = PortfolioRiskMetrics.calculate_tracking_error(
                portfolio_returns, benchmark_returns
            )
            metrics['information_ratio'] = PortfolioRiskMetrics.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )
        
        # Market-relative metrics (if market returns provided)
        if market_returns is not None:
            metrics['beta'] = PortfolioRiskMetrics.calculate_beta(
                portfolio_returns, market_returns
            )
            metrics['alpha'] = PortfolioRiskMetrics.calculate_alpha(
                portfolio_returns, market_returns, risk_free_rate
            )
        
        return metrics