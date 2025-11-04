"""Multi-horizon (1, 7, 30-day) prediction pipeline"""
from pathlib import Path
from .single_step_pipeline import SingleStepPipeline
from models.multi_horizon import MultiHorizonModel
from models.model_trainer import ModelTrainer
from features.sequence_builder import (
    build_supervised_sequences,
    apply_feature_scaler,
    fit_feature_scaler,
)
from visualization.price_plots import PricePlotter
from visualization.report_generator import ReportGenerator
from config.settings import (
    FEATURE_COLS,
    PREDICTION_HORIZONS,
    LSTM_LOOKBACK,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_LR,
    LSTM_EPOCHS,
    LSTM_BATCH,
    USE_QUANTILES,
)
from models.single_step import get_model
import pandas as pd
import numpy as np
import pickle
import os
import torch

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
        if self.model_type == 'lstm':
            base = self.cache_dir / f"{self.ticker}_multihorizon_{horizon}d_{self.model_type}"
            return base.with_suffix('.pt')  # weights file; meta stored as .meta.pkl
        return self.cache_dir / f"{self.ticker}_multihorizon_{horizon}d_{self.model_type}.pkl"
    
    def _save_horizon_model(self, horizon, model, metrics, scaler=None):
        """Save individual horizon model to cache"""
        cache_path = self._get_multihorizon_cache_path(horizon)
        if self.model_type == 'lstm':
            # Save torch weights
            torch.save(model.state_dict(), cache_path)
            # Save meta (metrics + scaler path + feature_cols)
            meta_path = str(cache_path).replace('.pt', '.meta.pkl')
            scaler_path = None
            if scaler is not None:
                scaler_path = str(cache_path).replace('.pt', '.scaler.pkl')
                with open(scaler_path, 'wb') as sf:
                    pickle.dump(scaler, sf)
            from config.settings import FEATURE_COLS
            with open(meta_path, 'wb') as mf:
                pickle.dump({
                    'metrics': metrics, 
                    'model_type': self.model_type, 
                    'scaler_path': scaler_path,
                    'feature_cols': FEATURE_COLS
                }, mf)
            print(f"  ✓ Model cached: {cache_path} (+meta)")
            return
        # Non-LSTM path (sklearn/xgboost)
        from config.settings import FEATURE_COLS
        cache_data = {
            'model': model,
            'metrics': metrics,
            'model_type': self.model_type,
            'feature_cols': FEATURE_COLS
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  ✓ Model cached: {cache_path}")
    
    def _load_horizon_model(self, horizon):
        """Load individual horizon model from cache"""
        cache_path = self._get_multihorizon_cache_path(horizon)
        if not self.force_refresh and os.path.exists(cache_path):
            try:
                if self.model_type == 'lstm':
                    meta_path = str(cache_path).replace('.pt', '.meta.pkl')
                    with open(meta_path, 'rb') as mf:
                        meta = pickle.load(mf)
                    return cache_path, meta.get('metrics', {})
                else:
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
                # Validate feature columns for both LSTM and non-LSTM
                cache_valid = True
                if self.model_type == 'lstm':
                    # For LSTM, cached_model is actually the weights path
                    meta_path = str(cached_model).replace('.pt', '.meta.pkl')
                    try:
                        with open(meta_path, 'rb') as mf:
                            meta = pickle.load(mf)
                            cached_features = meta.get('feature_cols')
                            if cached_features is not None and cached_features != FEATURE_COLS:
                                print(f"  ⚠ Cache invalid: Feature columns changed, retraining...")
                                cache_valid = False
                    except Exception:
                        cache_valid = False
                else:
                    # For non-LSTM, validate from pickle
                    cache_path = self._get_multihorizon_cache_path(horizon)
                    try:
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                            cached_features = cache_data.get('feature_cols')
                            if cached_features is not None and cached_features != FEATURE_COLS:
                                print(f"  ⚠ Cache invalid: Feature columns changed, retraining...")
                                cache_valid = False
                    except Exception:
                        cache_valid = False
                
                if cache_valid:
                    print(f"  ✓ Loaded from cache")
                    if self.model_type == 'lstm':
                        # Keep path to weights; will be used for inference later
                        self.multi_model.set_horizon_artifacts(horizon, weights_path=str(cached_model))
                    else:
                        model_instance = get_model(model_type)
                        model_instance.model = cached_model
                        self.multi_model.models[f'{horizon}d'] = model_instance
                    
                    self.multi_metrics[f'{horizon}d'] = cached_metrics
                    
                    # Still need to generate test predictions for detailed analysis
                    target_col = f'Target_{horizon}d'
                    # Ensure no NaNs in features or target (indicators create early NaNs)
                    df_clean = df_with_targets.dropna(subset=[target_col] + FEATURE_COLS)
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
            
            # Only reach here if cache load failed or was invalid
            
            # Train new model
            target_col = f'Target_{horizon}d'
            # Ensure no NaNs in features or target (indicators create early NaNs)
            df_clean = df_with_targets.dropna(subset=[target_col] + FEATURE_COLS)
            
            X = df_clean[FEATURE_COLS]
            y = df_clean[target_col]

            if self.model_type == 'lstm':
                # Sequence-aware path with proper data leakage prevention
                # Build sequences first, then split, then refit scaler on training portion only
                
                # Temporary scaler to build sequences
                scaler = fit_feature_scaler(X, train_end_index=len(X))
                X_scaled = apply_feature_scaler(X, scaler)
                X_seq, y_seq, meta = build_supervised_sequences(X_scaled, y, lookback=LSTM_LOOKBACK)
                
                # Calculate proper 80/20 split on sequences
                sequences_train_size = int(0.8 * len(X_seq))
                sequences_train_size = max(10, min(sequences_train_size, len(X_seq) - 10))
                
                # Split sequences
                X_train, y_train = X_seq[:sequences_train_size], y_seq[:sequences_train_size]
                X_test, y_test = X_seq[sequences_train_size:], y_seq[sequences_train_size:]
                
                # CRITICAL FIX: Re-fit scaler on ONLY original rows used by training sequences
                scaler_train_end = sequences_train_size + LSTM_LOOKBACK
                scaler = fit_feature_scaler(X, train_end_index=scaler_train_end)
                X_scaled = apply_feature_scaler(X, scaler)
                # Rebuild sequences with properly fitted scaler
                X_seq, y_seq, meta = build_supervised_sequences(X_scaled, y, lookback=LSTM_LOOKBACK)
                X_train, y_train = X_seq[:sequences_train_size], y_seq[:sequences_train_size]
                X_test, y_test = X_seq[sequences_train_size:], y_seq[sequences_train_size:]

                # Standardize target on train only to stabilize training; store stats for inverse transform
                y_train_mean = float(np.mean(y_train))
                y_train_std = float(np.std(y_train)) if float(np.std(y_train)) > 1e-8 else 1.0
                y_train_stdized = (y_train - y_train_mean) / y_train_std
                y_test_stdized = (y_test - y_train_mean) / y_train_std

                # Torch tensors
                Xt = torch.tensor(X_train, dtype=torch.float32)
                yt = torch.tensor(y_train_stdized, dtype=torch.float32)
                Xtt = torch.tensor(X_test, dtype=torch.float32)
                ytt = torch.tensor(y_test_stdized, dtype=torch.float32)

                # Lazy import to avoid torch requirement unless LSTM is selected
                from models.lstm import LSTMTrainer, LSTMConfig
                cfg = LSTMConfig(
                    input_size=X.shape[1],
                    hidden_size=LSTM_HIDDEN,
                    num_layers=LSTM_LAYERS,
                    dropout=LSTM_DROPOUT,
                    lr=LSTM_LR,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH,
                    use_quantiles=USE_QUANTILES,
                )
                lstm_trainer = LSTMTrainer(cfg)
                lstm_trainer.fit(Xt, yt, X_val=None, y_val=None)

                # Evaluate on test
                yp, _, _ = lstm_trainer.predict(Xtt)
                y_pred_std = yp.numpy()
                # Inverse standardization back to price scale
                y_pred = (y_pred_std * y_train_std) + y_train_mean
                from evaluation.metrics import MetricsCalculator
                metrics = MetricsCalculator.calculate_all(y_test, y_pred)

                # Save to cache (weights + scaler + target stats)
                # augment meta later when saving
                self._save_horizon_model(horizon, lstm_trainer.model, metrics, scaler=scaler)
                # Save target stats alongside meta
                meta_path = str(self._get_multihorizon_cache_path(horizon)).replace('.pt', '.meta.pkl')
                try:
                    with open(meta_path, 'rb') as mf:
                        meta = pickle.load(mf)
                except Exception:
                    meta = {}
                meta['y_mean'] = y_train_mean
                meta['y_std'] = y_train_std
                with open(meta_path, 'wb') as mf:
                    pickle.dump(meta, mf)
                self.multi_metrics[f'{horizon}d'] = metrics
                self.multi_model.set_horizon_artifacts(horizon, weights_path=str(self._get_multihorizon_cache_path(horizon)))

                # CRITICAL FIX: Correct index alignment for test sequences
                # Test sequences start at sequences_train_size in sequence array
                # This maps to original position: LSTM_LOOKBACK + sequences_train_size
                test_start_in_original = LSTM_LOOKBACK + sequences_train_size
                test_index = X.index[test_start_in_original:test_start_in_original + len(y_pred)]
                self.backtest_predictions[f'{horizon}d'] = {
                    'predictions': y_pred,
                    'actuals': y.loc[test_index],  # keep as Series for downstream plotting
                    'dates': test_index,
                    'index': test_index
                }
            else:
                # Sklearn/xgboost path
                model_instance = self.multi_model.get_models_dict()[f'{horizon}d']
                trainer = ModelTrainer(model_instance, verbose=0)
                param_grid = model_instance.get_param_grid() if use_grid_search else None
                trained_model, metrics, test_index, y_pred = trainer.train(X, y, param_grid)
                self.multi_model.models[f'{horizon}d'] = model_instance
                self.multi_metrics[f'{horizon}d'] = metrics
                self._save_horizon_model(horizon, model_instance.model, metrics)
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
            # For LSTM path, models dict is not used; artifacts hold weights
            if self.model_type != 'lstm':
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
            if self.model_type == 'lstm':
                # Load scaler and weights
                weights_path = self.multi_model.get_horizon_artifacts(horizon).get('weights_path')
                if not weights_path:
                    continue
                meta_path = weights_path.replace('.pt', '.meta.pkl')
                with open(meta_path, 'rb') as mf:
                    meta = pickle.load(mf)
                scaler_path = meta.get('scaler_path')
                from sklearn.preprocessing import StandardScaler
                scaler = None
                if scaler_path and os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as sf:
                        scaler = pickle.load(sf)
                # Scale recent window
                X_all = self.df[FEATURE_COLS]
                if scaler is None:
                    scaler = StandardScaler().fit(X_all.values)
                X_scaled = apply_feature_scaler(X_all, scaler)
                # Build last window
                from config.settings import LSTM_LOOKBACK as LB
                window = X_scaled.values[-LB:]
                if len(window) < LB:
                    continue
                xw = torch.tensor(window[None, ...], dtype=torch.float32)
                from models.lstm import LSTMTrainer, LSTMConfig
                cfg = LSTMConfig(input_size=X_all.shape[1], hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, lr=LSTM_LR, epochs=1, batch_size=1, use_quantiles=USE_QUANTILES)
                lstm_trainer = LSTMTrainer(cfg)
                lstm_trainer.load(weights_path)
                # Load target stats for inverse transform
                y_mean = meta.get('y_mean')
                y_std = meta.get('y_std') or 1.0
                yp, y10, y90 = lstm_trainer.predict(xw)
                pred_std = float(yp.numpy().ravel()[0])
                pred = float(pred_std * y_std + y_mean)
                # Optionally store bands
                lower = float(y10.numpy().ravel()[0] * y_std + y_mean) if y10 is not None else None
                upper = float(y90.numpy().ravel()[0] * y_std + y_mean) if y90 is not None else None
            else:
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
            
            entry = {
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
            if self.model_type == 'lstm' and (y10 is not None and y90 is not None):
                entry['quantile_band'] = {'p10': lower, 'p90': upper}
            predictions[f'{horizon}d'] = entry
            
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
        # Pass future predictions for overlay if available
        future_preds = {
            k: v if 'predicted_price' in v else v.get('future_prediction', {})
            for k, v in (self.predictions or {}).items()
        } if self.predictions else None
        plotter.plot_multi_horizon_predictions(filepath, self.backtest_predictions, future_predictions=future_preds)
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
        from config.settings import MH_BACKTEST_SPLITS_LSTM
        
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
            
            # Run walk-forward validation (fewer splits for LSTM/low data)
            splits = n_splits
            if self.model_type == 'lstm':
                splits = MH_BACKTEST_SPLITS_LSTM
                # Reduce splits if dataset is small
                min_points_per_split = 50
                max_splits = max(2, len(df_clean) // min_points_per_split)
                splits = max(2, min(splits, max_splits))
            avg_metrics = backtester.walk_forward_validation(n_splits=splits, horizon=horizon)
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