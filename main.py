"""Main entry point for stock prediction system"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.single_step_pipeline import SingleStepPipeline
from pipelines.multi_horizon_pipeline import MultiHorizonPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-step prediction (next day)
  python main.py META
  python main.py AAPL --model ridge
  
  # Multi-horizon prediction (1, 7, 30 days)
  python main.py META --multi-horizon
  python main.py TSLA --multi-horizon --skip-backtest
  
  # Force refresh and retrain
  python main.py GOOGL --force-refresh
  
  # Skip plots for faster execution
  python main.py NVDA --skip-plots
        """
    )
    
    # Required arguments
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (e.g., META, AAPL, TSLA, GOOGL)'
    )
    
    # Pipeline selection
    parser.add_argument(
        '--multi-horizon',
        action='store_true',
        help='Use multi-horizon pipeline (1, 7, 30-day predictions)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'ridge', 'random_forest', 'lstm'],
        help='Model type to use (default: xgboost)'
    )
    
    # Directory configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache',
        help='Cache directory for models and data (default: cache)'
    )
    
    # Execution options
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh data and retrain models (ignore cache)'
    )
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation (faster execution)'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtesting (multi-horizon only)'
    )
    
    parser.add_argument(
        '--use-quantiles',
        action='store_true',
        help='Use quantile heads for LSTM to output prediction bands (default: off)'
    )

    args = parser.parse_args()
    # Apply runtime flags to settings (non-breaking)
    try:
        from config import settings as _settings
        if getattr(args, 'use_quantiles', False):
            _settings.USE_QUANTILES = True
    except Exception:
        pass
    
    # Select and run pipeline
    try:
        if args.multi_horizon:
            print("\n🚀 Running MULTI-HORIZON prediction pipeline...\n")
            pipeline = MultiHorizonPipeline(
                ticker=args.ticker,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                force_refresh=args.force_refresh
            )
            pipeline.run(
                model_type=args.model,
                skip_plots=args.skip_plots,
                skip_backtest=args.skip_backtest
            )
        else:
            print("\n🚀 Running SINGLE-STEP prediction pipeline...\n")
            pipeline = SingleStepPipeline(
                ticker=args.ticker,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                force_refresh=args.force_refresh
            )
            pipeline.run(
                model_type=args.model,
                skip_plots=args.skip_plots
            )
        
        print("\n✅ SUCCESS! Check the output directory for results.\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()