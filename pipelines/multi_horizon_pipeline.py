"""Multi-horizon (1, 7, 30-day) prediction pipeline"""
from pathlib import Path
from .single_step_pipeline import SingleStepPipeline
from models.multi_horizon import MultiHorizonModel
from models.model_trainer import ModelTrainer
from visualization.price_plots import PricePlotter
from visualization.report_generator import ReportGenerator
from config.settings import FEATURE_COLS, PREDICTION_HORIZONS
from models.single_step import get_model
import pandas as pd
import numpy as np
import pickle

class MultiHorizonPipeline(SingleStepPipeline):
    """Pipeline for multi-horizon predictions"""
    
    def __init__(self, ticker, output_dir="output", cache_dir="cache", force_refresh=False):
        super().__init__(ticker, output_dir, cache_dir, force_refresh)
        self.multi_model = None
        self.multi_metrics = {}
        self.predictions = {}
        self.backtest_predictions = {}
        self.model_type = 'xgboost'
    
    def _get_multihorizon_cache_path(self, horizon):
        """Get cache path for specific horizon model"""
        return self.cache_dir / f"{self.ticker}_multihorizon_{horizon}d_{self.model_type}.pkl"
    
    def _save_horizon_model(self, horizon, model, metrics):
        """Save individual horizon model to cache"""
        cache_path = self._get_multihorizon_cache_path(horizon)
        cache_data = {
            'model': model,
            'metrics': metrics,
            'model_type': self.model_type
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  ✓ Model cached: {cache_path}")
    
    def _load_horizon_model(self, horizon):
        """Load individual horizon model from cache"""
        cache_path = self._get_multihorizon_cache_path(horizon)
        if not self.force_refresh and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data['model_type'] == self.model_type:
                        return cache_data['model'], cache_data['metrics']
            except Exception as e:
                print(f"  ⚠ Cache load failed: {e}")
        return None, None
    
    def train_multihorizon_models(self, model_type='xgboost', use_grid_search=False):
        """Train models for multiple prediction horizons with caching"""
        
        self.model_type = model_type
        
        print(f"\n{'='*70}")
        print(f"TRAINING MULTI-HORIZON MODELS ({', '.join([f'{h}d' for h in PREDICTION_HORIZONS])})")
        print(f"{'='*70}\n")
        
        self.multi_model = MultiHorizonModel(model_type)
        
        # Create targets for each horizon
        df_with_targets = self.multi_model.create_targets(self.df)
        
        # Train model for each horizon
        for horizon in PREDICTION_HORIZONS:
            print(f"\n--- Training {horizon}-Day Model ---")
            
            # Try loading from cache first
            cached_model, cached_metrics = self._load_horizon_model(horizon)
            if cached_model is not None:
                print(f"  ✓ Loaded from cache")
                model_instance = get_model(model_type)
                model_instance.model = cached_model
                self.multi_model.models[f'{horizon}d'] = model_instance
                self.multi_metrics[f'{horizon}d'] = cached_metrics
                
                # Still need to generate test predictions for detailed analysis
                target_col = f'Target_{horizon}d'
                df_clean = df_with_targets.dropna(subset=[target_col])
                X = df_clean[FEATURE_COLS]
                y = df_clean[target_col]
                
                # Use same train/test split
                train_size = int(0.8 * len(X))
                X_test = X.iloc[train_size:]
                y_test = y.iloc[train_size:]
                test_index = X_test.index
                y_pred = cached_model.predict(X_test)
                
                # Store test predictions
                self.backtest_predictions[f'{horizon}d'] = {
                    'predictions': y_pred,
                    'actuals': y_test,
                    'dates': test_index,
                    'index': test_index
                }
                
                # Store detailed recent predictions
                recent_preds = y_pred[-10:]
                recent_actuals = y_test.iloc[-10:]
                recent_dates = test_index[-10:]
                
                detailed_predictions = []
                for date, pred, actual in zip(recent_dates, recent_preds, recent_actuals):
                    error = abs(actual - pred)
                    error_pct = (error / actual) * 100
                    detailed_predictions.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_price': float(pred),
                        'actual_price': float(actual),
                        'error': float(error),
                        'error_percent': float(error_pct)
                    })
                
                self.backtest_predictions[f'{horizon}d']['detailed'] = detailed_predictions
                
                print(f"  ✓ {horizon}-day model: MAE=${cached_metrics['MAE']:.2f}, RMSE=${cached_metrics['RMSE']:.2f}, R²={cached_metrics['R2']:.4f}")
                continue
            
            # Train new model
            target_col = f'Target_{horizon}d'
            df_clean = df_with_targets.dropna(subset=[target_col])
            
            X = df_clean[FEATURE_COLS]
            y = df_clean[target_col]
            
            # Get model instance
            model_instance = self.multi_model.get_models_dict()[f'{horizon}d']
            trainer = ModelTrainer(model_instance, verbose=0)
            
            # Train
            param_grid = model_instance.get_param_grid() if use_grid_search else None
            trained_model, metrics, test_index, y_pred = trainer.train(X, y, param_grid)
            
            # Store results
            self.multi_model.models[f'{horizon}d'] = model_instance
            self.multi_metrics[f'{horizon}d'] = metrics
            
            # Save to cache
            self._save_horizon_model(horizon, model_instance.model, metrics)
            
            # Store test predictions with actual values
            self.backtest_predictions[f'{horizon}d'] = {
                'predictions': y_pred,
                'actuals': y.loc[test_index],
                'dates': test_index,
                'index': test_index
            }
            
            # Calculate detailed error metrics for most recent predictions
            recent_preds = y_pred[-10:]
            recent_actuals = y.loc[test_index[-10:]]
            recent_dates = test_index[-10:]
            
            # Store detailed recent predictions
            detailed_predictions = []
            for date, pred, actual in zip(recent_dates, recent_preds, recent_actuals):
                error = abs(actual - pred)
                error_pct = (error / actual) * 100
                detailed_predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': float(pred),
                    'actual_price': float(actual),
                    'error': float(error),
                    'error_percent': float(error_pct)
                })
            
            self.backtest_predictions[f'{horizon}d']['detailed'] = detailed_predictions
            
            print(f"  ✓ {horizon}-day model: MAE=${metrics['MAE']:.2f}, RMSE=${metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
        
        # Print summary
        self.multi_model.metrics = self.multi_metrics
        self.multi_model.print_all_metrics()
    
    def predict_future(self):
        """Make predictions for all horizons using latest data"""
        
        if not self.multi_model or not self.multi_model.models:
            raise ValueError("Models not trained. Call train_multihorizon_models() first.")
        
        # Get latest data point
        latest_data = self.df[FEATURE_COLS].iloc[-1:].copy()
        current_price = self.df['Close'].iloc[-1]
        current_date = self.df.index[-1]
        
        predictions = {}
        
        print(f"\n{'='*70}")
        print("FUTURE PRICE PREDICTIONS")
        print(f"{'='*70}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Current Date: {current_date.date()}\n")
        
        for horizon in PREDICTION_HORIZONS:
            model = self.multi_model.models[f'{horizon}d'].model
            if model is None:
                continue
            
            pred = model.predict(latest_data)[0]
            change = pred - current_price
            change_pct = (change / current_price) * 100
            
            # Calculate target date
            target_date = current_date + pd.Timedelta(days=horizon)
            
            # Get historical performance for confidence
            if f'{horizon}d' in self.backtest_predictions:
                recent_errors = []
                for item in self.backtest_predictions[f'{horizon}d'].get('detailed', []):
                    recent_errors.append(item['error_percent'])
                
                avg_error = np.mean(recent_errors) if recent_errors else 0
                std_error = np.std(recent_errors) if recent_errors else 0
                
                # 95% confidence interval
                confidence_lower = pred * (1 - 2 * std_error / 100)
                confidence_upper = pred * (1 + 2 * std_error / 100)
            else:
                avg_error = 0
                confidence_lower = pred * 0.95
                confidence_upper = pred * 1.05
            
            predictions[f'{horizon}d'] = {
                'predicted_price': float(pred),
                'change': float(change),
                'change_percent': float(change_pct),
                'date': target_date.strftime('%Y-%m-%d'),
                'confidence_interval': {
                    'lower': float(confidence_lower),
                    'upper': float(confidence_upper)
                },
                'historical_avg_error_percent': float(avg_error) if avg_error else None
            }
            
            direction = "📈" if change > 0 else "📉"
            print(f"{horizon}-Day Prediction:")
            print(f"  Target Date: {target_date.strftime('%Y-%m-%d')}")
            print(f"  Predicted Price: ${pred:.2f} ({change:+.2f}, {change_pct:+.2f}%) {direction}")
            if avg_error:
                print(f"  Historical Avg Error: {avg_error:.2f}%")
            print(f"  95% Confidence Range: ${confidence_lower:.2f} - ${confidence_upper:.2f}")
            print()
        
        print(f"{'='*70}\n")
        
        return predictions
    
    def generate_multihorizon_plots(self):
        """Generate plots for multi-horizon predictions"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plotter = PricePlotter(self.df, self.ticker)
        
        # Multi-horizon prediction overlay
        print("Creating multi-horizon prediction plot...")
        filepath = plots_dir / "multi_horizon_predictions.png"
        plotter.plot_multi_horizon_predictions(filepath, self.backtest_predictions)
        print(f"✓ Saved: {filepath}")
        
        # Detailed backtest comparison
        if self.backtest_predictions:
            print("Creating backtest comparison plot...")
            filepath = plots_dir / "backtest_comparison.png"
            plotter.plot_backtest_comparison(filepath, self.backtest_predictions)
            print(f"✓ Saved: {filepath}")
    
    def backtest_multihorizon(self, n_splits=5):
        """Backtest multi-horizon predictions using walk-forward validation"""
        from evaluation.backtester import Backtester
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING MULTI-HORIZON MODELS (Walk-Forward Validation)")
        print(f"{'='*70}\n")
        
        backtest_results = {}
        
        # Create targets for backtesting
        df_with_targets = self.multi_model.create_targets(self.df)
        
        for horizon in PREDICTION_HORIZONS:
            print(f"\n--- Backtesting {horizon}-Day Model ---")
            
            # Get the trained model for this horizon
            if f'{horizon}d' not in self.multi_model.models:
                print(f"  ⚠ Skipping - model not trained")
                continue
            
            trained_model_instance = self.multi_model.models[f'{horizon}d']
            
            # Prepare data with correct target
            target_col = f'Target_{horizon}d'
            df_clean = df_with_targets.dropna(subset=[target_col])
            
            # Create backtester with the trained model (it will create fresh instances internally)
            backtester = Backtester(df_clean, trained_model_instance.model, FEATURE_COLS)
            
            # Set the target column for backtester
            backtester.target_col = target_col
            
            # Run walk-forward validation
            avg_metrics = backtester.walk_forward_validation(n_splits=n_splits, horizon=horizon)
            backtester.print_summary()
            
            backtest_results[f'{horizon}d'] = avg_metrics
        
        return backtest_results
    
    def generate_reports(self):
        """Generate enhanced reports with multi-horizon predictions"""
        print("\nGenerating comprehensive analysis reports...")
        
        # Prepare detailed prediction data for report
        detailed_predictions = {}
        
        for horizon in PREDICTION_HORIZONS:
            horizon_key = f'{horizon}d'
            if horizon_key in self.backtest_predictions:
                backtest_data = self.backtest_predictions[horizon_key]
                
                # Get the most recent prediction from test set
                if len(backtest_data['dates']) > 0:
                    last_idx = -1
                    last_date = backtest_data['dates'][last_idx]
                    last_pred = backtest_data['predictions'][last_idx]
                    last_actual = backtest_data['actuals'].iloc[last_idx]
                    
                    error = abs(last_actual - last_pred)
                    error_pct = (error / last_actual) * 100
                    
                    detailed_predictions[horizon_key] = {
                        'date': last_date.strftime('%Y-%m-%d'),
                        'predicted_price': float(last_pred),
                        'actual_price': float(last_actual),
                        'error': float(error),
                        'error_percent': float(error_pct),
                        'all_predictions': backtest_data.get('detailed', [])
                    }
        
        # Add future predictions
        if self.predictions:
            for horizon_key, pred_data in self.predictions.items():
                if horizon_key in detailed_predictions:
                    detailed_predictions[horizon_key]['future_prediction'] = pred_data
                else:
                    detailed_predictions[horizon_key] = {
                        'future_prediction': pred_data
                    }
        
        # Generate reports with all data
        reporter = ReportGenerator(
            self.df, 
            self.ticker, 
            self.output_dir,
            self.multi_metrics,  # Pass all horizon metrics
            self.sentiment_info,
            detailed_predictions
        )
        reporter.model = self.model  # Pass the model for next-day forecast
        reporter.generate_all()
    
    def run(self, model_type='xgboost', skip_plots=False, skip_backtest=False):
        """Run complete multi-horizon pipeline"""
        print(f"\n{'='*70}")
        print(f"MULTI-HORIZON PREDICTION PIPELINE - {self.ticker}")
        print(f"{'='*70}\n")
        
        # Run base pipeline steps
        self.download_data()
        self.calculate_indicators()
        self.analyze_sentiment()
        
        # Train multi-horizon models
        self.train_multihorizon_models(model_type=model_type)
        
        # Make future predictions
        self.predictions = self.predict_future()
        
        # Generate visualizations
        if not skip_plots:
            self.generate_visualizations(skip_if_exists=True)
            self.generate_multihorizon_plots()
        else:
            print("\n⊘ Skipping plot generation")
        
        # Backtest
        if not skip_backtest:
            self.backtest_multihorizon()
        
        # Generate reports with all prediction data
        self.generate_reports()
        
        print(f"\n{'='*70}")
        print(f"✓ Multi-horizon analysis complete for {self.ticker}!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return self.df, self.multi_model, self.multi_metrics, self.predictions