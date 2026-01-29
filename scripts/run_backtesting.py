"""
Task 5 Execution Script
Runs portfolio backtesting against benchmark
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run Task 5: Strategy Backtesting')
    parser.add_argument('--rebalancing', type=str, default='monthly',
                       choices=['daily', 'weekly', 'monthly', 'quarterly', 'none', 'threshold'],
                       help='Rebalancing frequency')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Rebalancing threshold (for threshold-based rebalancing)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick backtest with reduced analysis')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all visualizations and reports')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TASK 5: STRATEGY BACKTESTING")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../figures/task5', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    # Import and run the backtesting
    from task5_backtesting import run_backtesting_pipeline
    
    run_backtesting_pipeline(
        rebalancing_freq=args.rebalancing,
        rebalancing_threshold=args.threshold,
        quick_mode=args.quick,
        save_all=args.save_all
    )
    
    print("\n" + "=" * 80)
    print("TASK 5 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()