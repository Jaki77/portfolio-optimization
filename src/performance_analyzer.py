"""
Advanced Performance Analysis
Statistical tests and robustness checks for backtest results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Advanced performance analysis and statistical testing"""
    
    @staticmethod
    def calculate_mar_ratio(returns: np.ndarray, 
                           minimum_acceptable_return: float = 0.0) -> float:
        """
        Calculate MAR (Minimum Acceptable Return) ratio
        
        Args:
            returns: Portfolio returns
            minimum_acceptable_return: Minimum acceptable return
            
        Returns:
            MAR ratio
        """
        avg_return = np.mean(returns) * 252  # Annualize
        
        if minimum_acceptable_return > 0:
            mar_ratio = (avg_return - minimum_acceptable_return) / abs(minimum_acceptable_return)
        else:
            mar_ratio = avg_return
        
        return mar_ratio
    
    @staticmethod
    def calculate_ulcer_index(returns: np.ndarray) -> float:
        """
        Calculate Ulcer Index (measure of downside volatility)
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Ulcer Index
        """
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate drawdowns
        peak = cumulative[0]
        squared_drawdowns = []
        
        for value in cumulative:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            squared_drawdowns.append(drawdown ** 2)
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    @staticmethod
    def calculate_tail_ratio(returns: np.ndarray, 
                            percentile: float = 0.95) -> float:
        """
        Calculate Tail Ratio (ratio of right tail to left tail)
        
        Args:
            returns: Portfolio returns
            percentile: Percentile for tail calculation (e.g., 0.95 for 95th percentile)
            
        Returns:
            Tail ratio
        """
        right_tail = np.percentile(returns, percentile * 100)
        left_tail = np.percentile(returns, (1 - percentile) * 100)
        
        if left_tail != 0:
            tail_ratio = abs(right_tail / left_tail)
        else:
            tail_ratio = np.inf if right_tail > 0 else 0
        
        return tail_ratio
    
    @staticmethod
    def calculate_skewness_kurtosis(returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate skewness and kurtosis of returns
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Skewness and kurtosis
        """
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return skewness, kurtosis
    
    @staticmethod
    def calculate_hurst_exponent(returns: np.ndarray) -> float:
        """
        Calculate Hurst exponent (measure of long-term memory)
        
        Args:
            returns: Time series of returns
            
        Returns:
            Hurst exponent
        """
        # Simplified R/S analysis
        lags = range(2, min(100, len(returns) // 4))
        tau = []
        rs = []
        
        for lag in lags:
            # Create subseries
            k = len(returns) // lag
            sub_series = returns[:k * lag].reshape(k, lag)
            
            # Calculate R/S for each subseries
            rs_values = []
            for series in sub_series:
                mean_adj = series - np.mean(series)
                cumulative = np.cumsum(mean_adj)
                r = np.max(cumulative) - np.min(cumulative)
                s = np.std(series)
                
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(lag)
                rs.append(np.mean(rs_values))
        
        if len(tau) > 1:
            # Fit linear regression to log-log plot
            log_tau = np.log(tau)
            log_rs = np.log(rs)
            
            slope, _, _, _, _ = stats.linregress(log_tau, log_rs)
            hurst = slope
        else:
            hurst = 0.5  # Random walk assumption
        
        return hurst
    
    @staticmethod
    def perform_wilcoxon_test(strategy_returns: np.ndarray,
                             benchmark_returns: np.ndarray) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test for paired differences
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with test results
        """
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Return series must have same length")
        
        # Calculate paired differences
        differences = strategy_returns - benchmark_returns
        
        # Perform Wilcoxon test
        stat, p_value = stats.wilcoxon(differences)
        
        # Calculate effect size
        n = len(differences)
        effect_size = stat / (n * (n + 1) / 2)
        
        results = {
            'test_statistic': stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant_at_5%': p_value < 0.05,
            'significant_at_1%': p_value < 0.01
        }
        
        return results
    
    @staticmethod
    def perform_bootstrap_test(strategy_returns: np.ndarray,
                              benchmark_returns: np.ndarray,
                              n_bootstrap: int = 1000,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Perform bootstrap test for difference in means
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with bootstrap results
        """
        differences = strategy_returns - benchmark_returns
        original_mean_diff = np.mean(differences)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(differences, size=len(differences), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence interval
        alpha = (1 - confidence_level) / 2
        lower_bound = np.percentile(bootstrap_means, alpha * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        # Calculate p-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_means <= 0),
            np.mean(bootstrap_means >= 0)
        )
        
        results = {
            'original_mean_diff': original_mean_diff,
            'bootstrap_mean': np.mean(bootstrap_means),
            'bootstrap_std': np.std(bootstrap_means),
            'confidence_interval_lower': lower_bound,
            'confidence_interval_upper': upper_bound,
            'p_value': p_value,
            'significant': p_value < (1 - confidence_level)
        }
        
        return results
    
    @staticmethod
    def calculate_omega_ratio(returns: np.ndarray,
                             threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio (ratio of gains to losses relative to threshold)
        
        Args:
            returns: Portfolio returns
            threshold: Return threshold (e.g., risk-free rate)
            
        Returns:
            Omega ratio
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        if len(losses) > 0 and np.sum(losses) > 0:
            omega_ratio = np.sum(gains) / np.sum(losses)
        else:
            omega_ratio = np.inf if len(gains) > 0 else 0
        
        return omega_ratio
    
    @staticmethod
    def calculate_pain_index(returns: np.ndarray) -> float:
        """
        Calculate Pain Index (average drawdown)
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Pain Index
        """
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate drawdowns
        peak = cumulative[0]
        drawdowns = []
        
        for value in cumulative:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        # Calculate average drawdown
        pain_index = np.mean(drawdowns)
        
        return pain_index
    
    @staticmethod
    def calculate_gain_to_pain_ratio(returns: np.ndarray) -> float:
        """
        Calculate Gain to Pain ratio (Cem Karsan's metric)
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Gain to Pain ratio
        """
        cumulative = np.cumsum(returns)
        total_gain = np.sum(returns[returns > 0])
        total_pain = abs(np.sum(returns[returns < 0]))
        
        if total_pain > 0:
            gain_to_pain = total_gain / total_pain
        else:
            gain_to_pain = np.inf if total_gain > 0 else 0
        
        return gain_to_pain
    
    @staticmethod
    def perform_regime_analysis(returns: np.ndarray,
                               window: int = 60) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes
        
        Args:
            returns: Portfolio returns
            window: Window for regime detection
            
        Returns:
            Dictionary with regime analysis
        """
        # Simple regime detection based on volatility
        rolling_vol = pd.Series(returns).rolling(window=window).std()
        
        # Define regimes based on volatility percentiles
        low_threshold = np.percentile(rolling_vol.dropna(), 33)
        high_threshold = np.percentile(rolling_vol.dropna(), 67)
        
        regimes = []
        for i in range(window, len(returns)):
            if rolling_vol.iloc[i] <= low_threshold:
                regimes.append('low_vol')
            elif rolling_vol.iloc[i] >= high_threshold:
                regimes.append('high_vol')
            else:
                regimes.append('medium_vol')
        
        # Pad beginning with NaN
        regimes = [np.nan] * window + regimes
        
        # Calculate performance by regime
        regime_returns = {}
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            mask = [r == regime for r in regimes]
            if any(mask):
                regime_returns[regime] = {
                    'mean_return': np.mean(returns[mask]),
                    'std_return': np.std(returns[mask]),
                    'sharpe_ratio': np.mean(returns[mask]) / np.std(returns[mask]) if np.std(returns[mask]) > 0 else 0,
                    'count': sum(mask)
                }
        
        results = {
            'regime_returns': regime_returns,
            'regime_labels': regimes,
            'low_vol_threshold': low_threshold,
            'high_vol_threshold': high_threshold
        }
        
        return results
    
    @staticmethod
    def calculate_all_advanced_metrics(returns: np.ndarray,
                                      benchmark_returns: Optional[np.ndarray] = None,
                                      risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate all advanced performance metrics
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of advanced metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = np.mean(returns)
        metrics['std_return'] = np.std(returns)
        metrics['skewness'], metrics['kurtosis'] = PerformanceAnalyzer.calculate_skewness_kurtosis(returns)
        
        # Downside risk metrics
        metrics['ulcer_index'] = PerformanceAnalyzer.calculate_ulcer_index(returns)
        metrics['tail_ratio_95'] = PerformanceAnalyzer.calculate_tail_ratio(returns, 0.95)
        metrics['pain_index'] = PerformanceAnalyzer.calculate_pain_index(returns)
        
        # Risk-adjusted metrics
        metrics['omega_ratio'] = PerformanceAnalyzer.calculate_omega_ratio(returns, risk_free_rate / 252)
        metrics['mar_ratio'] = PerformanceAnalyzer.calculate_mar_ratio(returns, risk_free_rate / 252)
        metrics['gain_to_pain_ratio'] = PerformanceAnalyzer.calculate_gain_to_pain_ratio(returns)
        
        # Time series properties
        metrics['hurst_exponent'] = PerformanceAnalyzer.calculate_hurst_exponent(returns)
        
        # Benchmark-relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            # Statistical tests
            wilcoxon_results = PerformanceAnalyzer.perform_wilcoxon_test(returns, benchmark_returns)
            metrics.update({f'wilcoxon_{k}': v for k, v in wilcoxon_results.items()})
            
            # Regime analysis
            regime_results = PerformanceAnalyzer.perform_regime_analysis(returns - benchmark_returns)
            metrics['regime_analysis'] = regime_results
        
        return metrics