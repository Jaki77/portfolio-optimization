"""
Task 4 Execution Script
Runs portfolio optimization using Modern Portfolio Theory
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run Task 4: Portfolio Optimization')
    parser.add_argument('--n-portfolios', type=int, default=10000,
                       help='Number of random portfolios to generate')
    parser.add_argument('--risk-free', type=float, default=0.02,
                       help='Risk-free rate (default: 0.02)')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all visualizations and reports')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick optimization with fewer portfolios')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TASK 4: PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../figures/task4', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    # Import and run the optimization
    from task4_portfolio_optimization import run_optimization_pipeline
    
    run_optimization_pipeline(
        n_portfolios=1000 if args.quick else args.n_portfolios,
        risk_free_rate=args.risk_free,
        save_all=args.save_all
    )
    
    print("\n" + "=" * 80)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()