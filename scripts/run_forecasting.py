"""
Task 3 Execution Script
Generates 6-12 month forecasts and analyzes trends
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run Task 3: Forecasting')
    parser.add_argument('--horizon', type=int, default=12,
                       help='Forecast horizon in months (default: 12)')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95)')
    parser.add_argument('--method', type=str, default='parametric',
                       choices=['parametric', 'bootstrap'],
                       help='Confidence interval method')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all visualizations and reports')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TASK 3: FORECAST FUTURE MARKET TRENDS")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../figures/task3', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    # Import and run the forecasting
    from task3_forecasting import run_forecasting_pipeline
    
    run_forecasting_pipeline(
        horizon_months=args.horizon,
        confidence_level=args.confidence,
        method=args.method,
        save_all=args.save_all
    )
    
    print("\n" + "=" * 80)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()