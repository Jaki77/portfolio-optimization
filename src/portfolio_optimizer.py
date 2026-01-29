"""
Modern Portfolio Theory (MPT) Implementation
Portfolio optimization using Efficient Frontier and risk-return analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
import cvxpy as cp
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


class PortfolioOptimizer:
    """Modern Portfolio Theory implementation for portfolio optimization"""
    
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            returns_data: DataFrame of asset returns (columns: assets, index: dates)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.returns = returns_data
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        
        # Calculate statistics
        self.expected_returns = self._calculate_expected_returns()
        self.covariance_matrix = self._calculate_covariance_matrix()
        self.correlation_matrix = self._calculate_correlation_matrix()
        
        # Results storage
        self.optimization_results = {}
        self.efficient_frontier = None
        self.key_portfolios = {}
        
    def _calculate_expected_returns(self) -> pd.Series:
        """Calculate annualized expected returns for each asset"""
        # Annualize daily returns: (1 + mean_daily_return)^252 - 1
        annual_returns = (1 + self.returns.mean()) ** 252 - 1
        return annual_returns
    
    def _calculate_covariance_matrix(self) -> pd.DataFrame:
        """Calculate annualized covariance matrix"""
        # Annualize daily covariance: covariance_daily * 252
        annual_covariance = self.returns.cov() * 252
        return annual_covariance
    
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return self.returns.corr()
    
    def generate_random_portfolios(self, n_portfolios: int = 10000, 
                                  min_weight: float = 0.0, 
                                  max_weight: float = 1.0) -> pd.DataFrame:
        """
        Generate random portfolios for visualization
        
        Args:
            n_portfolios: Number of random portfolios to generate
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            DataFrame with portfolio weights, returns, and risks
        """
        print(f"Generating {n_portfolios} random portfolios...")
        
        portfolio_results = []
        
        for _ in range(n_portfolios):
            # Generate random weights that sum to 1
            weights = np.random.random(self.n_assets)
            weights = weights / weights.sum()
            
            # Apply weight constraints
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()  # Re-normalize
            
            # Calculate portfolio return
            port_return = np.sum(self.expected_returns * weights)
            
            # Calculate portfolio risk (standard deviation)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            
            # Calculate Sharpe ratio
            sharpe_ratio = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
            
            portfolio_results.append({
                'weights': weights.copy(),
                'return': port_return,
                'risk': port_risk,
                'sharpe_ratio': sharpe_ratio
            })
        
        # Convert to DataFrame
        portfolios_df = pd.DataFrame(portfolio_results)
        
        # Store for later use
        self.random_portfolios = portfolios_df
        
        return portfolios_df
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate key metrics for a given portfolio
        
        Args:
            weights: Portfolio weights (must sum to 1)
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Portfolio return
        port_return = np.sum(self.expected_returns * weights)
        
        # Portfolio risk (standard deviation)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
        
        # Value at Risk (95% confidence, parametric)
        var_95 = port_return - 1.645 * port_risk
        
        # Maximum drawdown approximation
        # Simplified: assume worst case based on individual asset drawdowns
        max_dd = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix * 2, weights)))
        
        # Diversification ratio
        weighted_vol = np.sum([weights[i] * np.sqrt(self.covariance_matrix.iloc[i, i]) 
                              for i in range(self.n_assets)])
        diversification_ratio = weighted_vol / port_risk if port_risk > 0 else 1
        
        metrics = {
            'weights': weights,
            'expected_return': port_return,
            'volatility': port_risk,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown_est': max_dd,
            'diversification_ratio': diversification_ratio,
            'risk_free_rate': self.risk_free_rate
        }
        
        return metrics
    
    def optimize_max_sharpe(self) -> Dict[str, Any]:
        """
        Optimize for maximum Sharpe ratio (Tangency portfolio)
        
        Returns:
            Dictionary with optimal portfolio metrics
        """
        print("Optimizing for maximum Sharpe ratio...")
        
        # Objective function: maximize Sharpe ratio = minimize negative Sharpe
        def negative_sharpe(weights):
            port_return = np.sum(self.expected_returns * weights)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            self.key_portfolios['max_sharpe'] = portfolio_metrics
            print(f"Maximum Sharpe ratio portfolio found: {portfolio_metrics['sharpe_ratio']:.4f}")
            
            return portfolio_metrics
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def optimize_min_volatility(self) -> Dict[str, Any]:
        """
        Optimize for minimum volatility
        
        Returns:
            Dictionary with optimal portfolio metrics
        """
        print("Optimizing for minimum volatility...")
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            self.key_portfolios['min_volatility'] = portfolio_metrics
            print(f"Minimum volatility portfolio found: {portfolio_metrics['volatility']:.4f}")
            
            return portfolio_metrics
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def optimize_target_return(self, target_return: float) -> Dict[str, Any]:
        """
        Optimize for minimum volatility given target return
        
        Args:
            target_return: Target annual return
            
        Returns:
            Dictionary with optimal portfolio metrics
        """
        print(f"Optimizing for target return: {target_return:.1%}...")
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
            {'type': 'eq', 'fun': lambda w: np.sum(self.expected_returns * w) - target_return}  # Target return
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimization
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_metrics = self.calculate_portfolio_metrics(optimal_weights)
            
            # Store with target return as key
            key = f"target_return_{target_return:.3f}"
            self.key_portfolios[key] = portfolio_metrics
            
            return portfolio_metrics
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def generate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Generate the efficient frontier
        
        Args:
            n_points: Number of points on efficient frontier
            
        Returns:
            DataFrame of efficient portfolios
        """
        print(f"Generating efficient frontier with {n_points} points...")
        
        # Get minimum and maximum achievable returns
        min_return_portfolio = self.optimize_min_volatility()
        min_return = min_return_portfolio['expected_return']
        
        # Maximum return portfolio (100% in highest returning asset)
        max_return_idx = self.expected_returns.idxmax()
        max_return_weights = np.zeros(self.n_assets)
        max_return_weights[self.assets.index(max_return_idx)] = 1.0
        max_return_portfolio = self.calculate_portfolio_metrics(max_return_weights)
        max_return = max_return_portfolio['expected_return']
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        
        for target in target_returns:
            try:
                portfolio = self.optimize_target_return(target)
                efficient_portfolios.append({
                    'target_return': target,
                    'expected_return': portfolio['expected_return'],
                    'volatility': portfolio['volatility'],
                    'sharpe_ratio': portfolio['sharpe_ratio'],
                    'weights': portfolio['weights']
                })
            except:
                # Skip if optimization fails for this target
                continue
        
        # Create DataFrame
        efficient_df = pd.DataFrame(efficient_portfolios)
        
        # Store for later use
        self.efficient_frontier = efficient_df
        
        return efficient_df
    
    def calculate_equal_weight_portfolio(self) -> Dict[str, Any]:
        """
        Calculate metrics for equal-weight portfolio
        
        Returns:
            Dictionary with portfolio metrics
        """
        print("Calculating equal-weight portfolio...")
        
        equal_weights = np.ones(self.n_assets) / self.n_assets
        portfolio_metrics = self.calculate_portfolio_metrics(equal_weights)
        
        self.key_portfolios['equal_weight'] = portfolio_metrics
        
        return portfolio_metrics
    
    def plot_efficient_frontier(self, save_path: Optional[str] = None):
        """
        Plot the efficient frontier with key portfolios
        
        Args:
            save_path: Optional path to save figure
        """
        if not hasattr(self, 'random_portfolios'):
            self.generate_random_portfolios()
        
        if self.efficient_frontier is None:
            self.generate_efficient_frontier()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot random portfolios
        ax.scatter(self.random_portfolios['risk'], 
                  self.random_portfolios['return'],
                  c=self.random_portfolios['sharpe_ratio'],
                  cmap='viridis', alpha=0.3, s=10,
                  label='Random Portfolios')
        
        # Plot efficient frontier
        ax.plot(self.efficient_frontier['volatility'],
               self.efficient_frontier['expected_return'],
               'b-', linewidth=3, label='Efficient Frontier')
        
        # Plot key portfolios
        colors = {'max_sharpe': 'green', 'min_volatility': 'red', 
                 'equal_weight': 'purple', 'max_return': 'orange'}
        markers = {'max_sharpe': '^', 'min_volatility': 'v', 
                  'equal_weight': 's', 'max_return': 'o'}
        
        for portfolio_name, portfolio in self.key_portfolios.items():
            if portfolio_name in colors:
                ax.scatter(portfolio['volatility'], portfolio['expected_return'],
                          color=colors[portfolio_name], marker=markers[portfolio_name],
                          s=200, edgecolors='black', linewidth=2,
                          label=portfolio_name.replace('_', ' ').title())
        
        # Add risk-free rate line (Capital Market Line)
        if 'max_sharpe' in self.key_portfolios:
            max_sharpe = self.key_portfolios['max_sharpe']
            x_range = np.linspace(0, max(self.random_portfolios['risk']) * 1.1, 100)
            cml = self.risk_free_rate + (max_sharpe['sharpe_ratio'] * x_range)
            
            ax.plot(x_range, cml, 'k--', alpha=0.5, linewidth=1.5, label='Capital Market Line')
        
        # Formatting
        ax.set_title('Efficient Frontier with Key Portfolios', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Portfolio Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Portfolio Expected Return', fontsize=12)
        
        # Add Sharpe ratio colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=self.random_portfolios['sharpe_ratio'].min(),
                                                    vmax=self.random_portfolios['sharpe_ratio'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Add portfolio composition info
        info_text = f"Assets: {', '.join(self.assets)}\nRisk-free rate: {self.risk_free_rate:.1%}"
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_portfolio_composition(self, save_path: Optional[str] = None):
        """
        Plot portfolio composition for key portfolios
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.key_portfolios:
            print("No key portfolios calculated. Running optimizations...")
            self.optimize_max_sharpe()
            self.optimize_min_volatility()
            self.calculate_equal_weight_portfolio()
        
        # Prepare data for plotting
        portfolio_names = []
        weights_data = []
        
        for port_name, port_metrics in self.key_portfolios.items():
            if port_name in ['max_sharpe', 'min_volatility', 'equal_weight']:
                portfolio_names.append(port_name.replace('_', ' ').title())
                weights_data.append(port_metrics['weights'])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bottom = np.zeros(len(portfolio_names))
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_assets))
        
        for i, asset in enumerate(self.assets):
            weights = [w[i] for w in weights_data]
            ax.bar(portfolio_names, weights, bottom=bottom, 
                  color=colors[i], edgecolor='black', label=asset)
            bottom += weights
        
        # Formatting
        ax.set_title('Portfolio Composition - Key Portfolios', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Weight (%)', fontsize=12)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for i, port_name in enumerate(portfolio_names):
            total_height = 0
            for j, asset in enumerate(self.assets):
                weight = weights_data[i][j]
                if weight > 0.05:  # Only label significant allocations
                    ax.text(i, total_height + weight/2, f'{weight:.1%}',
                           ha='center', va='center', fontsize=9,
                           color='black', fontweight='bold')
                total_height += weight
        
        ax.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Plot correlation matrix heatmap
        
        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(self.correlation_matrix, 
                   annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0,
                   square=True, linewidths=1,
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title('Asset Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_portfolio_report(self) -> str:
        """
        Generate comprehensive portfolio optimization report
        
        Returns:
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "PORTFOLIO OPTIMIZATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Assets Information
        report += "ASSETS IN PORTFOLIO\n"
        report += "-" * 40 + "\n\n"
        
        for asset in self.assets:
            exp_return = self.expected_returns[asset]
            volatility = np.sqrt(self.covariance_matrix.loc[asset, asset])
            sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            report += f"{asset}:\n"
            report += f"  Expected Return: {exp_return:.2%}\n"
            report += f"  Volatility: {volatility:.2%}\n"
            report += f"  Sharpe Ratio: {sharpe:.3f}\n\n"
        
        # Key Portfolios
        report += "KEY OPTIMAL PORTFOLIOS\n"
        report += "-" * 40 + "\n\n"
        
        for port_name, port_metrics in self.key_portfolios.items():
            if port_name in ['max_sharpe', 'min_volatility', 'equal_weight']:
                display_name = port_name.replace('_', ' ').title()
                
                report += f"{display_name} Portfolio:\n"
                report += f"  Expected Return: {port_metrics['expected_return']:.2%}\n"
                report += f"  Volatility: {port_metrics['volatility']:.2%}\n"
                report += f"  Sharpe Ratio: {port_metrics['sharpe_ratio']:.3f}\n"
                report += f"  Value at Risk (95%): {port_metrics['var_95']:.2%}\n"
                
                # Portfolio weights
                report += "  Portfolio Weights:\n"
                for i, asset in enumerate(self.assets):
                    weight = port_metrics['weights'][i]
                    if weight > 0.001:  # Only show non-zero weights
                        report += f"    {asset}: {weight:.1%}\n"
                
                report += "\n"
        
        # Efficient Frontier Summary
        if self.efficient_frontier is not None:
            report += "EFFICIENT FRONTIER SUMMARY\n"
            report += "-" * 40 + "\n\n"
            
            min_risk_port = self.efficient_frontier.loc[self.efficient_frontier['volatility'].idxmin()]
            max_sharpe_port = self.efficient_frontier.loc[self.efficient_frontier['sharpe_ratio'].idxmax()]
            
            report += f"Minimum Risk Portfolio:\n"
            report += f"  Return: {min_risk_port['expected_return']:.2%}\n"
            report += f"  Risk: {min_risk_port['volatility']:.2%}\n\n"
            
            report += f"Maximum Sharpe Portfolio:\n"
            report += f"  Return: {max_sharpe_port['expected_return']:.2%}\n"
            report += f"  Risk: {max_sharpe_port['volatility']:.2%}\n"
            report += f"  Sharpe Ratio: {max_sharpe_port['sharpe_ratio']:.3f}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-" * 40 + "\n\n"
        
        if 'max_sharpe' in self.key_portfolios:
            max_sharpe = self.key_portfolios['max_sharpe']
            report += "1. PRIMARY RECOMMENDATION: Maximum Sharpe Ratio Portfolio\n"
            report += f"   • Expected Return: {max_sharpe['expected_return']:.2%}\n"
            report += f"   • Risk (Volatility): {max_sharpe['volatility']:.2%}\n"
            report += f"   • Risk-Adjusted Return (Sharpe): {max_sharpe['sharpe_ratio']:.3f}\n"
            report += f"   • Diversification Benefit: {max_sharpe['diversification_ratio']:.2f}x\n\n"
            
            report += "   Optimal Allocation:\n"
            for i, asset in enumerate(self.assets):
                weight = max_sharpe['weights'][i]
                if weight > 0.001:
                    report += f"   • {asset}: {weight:.1%}\n"
        
        report += "\n2. RISK MANAGEMENT CONSIDERATIONS:\n"
        report += "   • Use minimum volatility portfolio for capital preservation\n"
        report += "   • Consider equal-weight portfolio for maximum diversification\n"
        report += "   • Implement stop-loss mechanisms for high-risk assets\n"
        
        report += "\n3. IMPLEMENTATION GUIDANCE:\n"
        report += "   • Rebalance portfolio quarterly\n"
        report += "   • Monitor correlation changes between assets\n"
        report += "   • Update expected returns based on new forecasts\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 80
        
        return report
    
    def save_optimization_results(self, filepath: str):
        """
        Save optimization results to file
        
        Args:
            filepath: Path to save results
        """
        import pickle
        
        results = {
            'assets': self.assets,
            'expected_returns': self.expected_returns,
            'covariance_matrix': self.covariance_matrix,
            'key_portfolios': self.key_portfolios,
            'efficient_frontier': self.efficient_frontier,
            'risk_free_rate': self.risk_free_rate,
            'generation_date': pd.Timestamp.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """
        Load optimization results from file
        
        Args:
            filepath: Path to saved results
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.assets = results['assets']
        self.expected_returns = results['expected_returns']
        self.covariance_matrix = results['covariance_matrix']
        self.key_portfolios = results['key_portfolios']
        self.efficient_frontier = results['efficient_frontier']
        self.risk_free_rate = results['risk_free_rate']
        
        print(f"Optimization results loaded from {filepath}")