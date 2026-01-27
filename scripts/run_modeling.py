#!/usr/bin/env python3
"""
Task 2 Execution Script
Runs the complete time series modeling pipeline
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run Task 2: Time Series Modeling')
    parser.add_argument('--model', type=str, default='all',
                       choices=['arima', 'sarima', 'lstm', 'all'],
                       help='Which model to run')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to figures directory')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with reduced parameters')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TASK 2: TIME SERIES FORECASTING MODELS")
    print("=" * 70)
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../figures/task2', exist_ok=True)
    
    # Import and run the modeling
    from task2_modeling import run_complete_pipeline
    
    run_complete_pipeline(
        model_type=args.model,
        save_plots=args.save_plots,
        quick_mode=args.quick
    )
    
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()