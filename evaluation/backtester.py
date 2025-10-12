"""Backtesting functionality for model validation"""
import pandas as pd
import numpy as np
from .metrics import MetricsCalculator

class Backtester:
    """Backtest prediction models"""
    
    def __init__(self, df, model, feature_cols):
        self.df = df
        self.model = model
        self.feature_cols = feature_cols
        self.results = []
        self.target_col = 'Target'  # Default target column
    
    def walk_forward_validation(self, n_splits=5, horizon=1):
        """
        Perform walk-forward validation
        
        Args:
            n_splits: Number of validation splits
            horizon: Prediction horizon in days
        """
        print(f"\nPerforming walk-forward validation ({n_splits} splits, {horizon}-day horizon)...")
        
        # Use the dataframe that already has the correct target column
        df_clean = self.df.dropna(subset=[self.target_col])
        
        total_size = len(df_clean)
        test_size = total_size // (n_splits + 1)
        
        for i in range(n_splits):
            split_end = total_size - (n_splits - i - 1) * test_size
            split_start = max(0, split_end - 2 * test_size)
            
            train_end = split_end - test_size
            
            # Split data
            train_df = df_clean.iloc[split_start:train_end]
            test_df = df_clean.iloc[train_end:split_end]
            
            if len(train_df) < 50 or len(test_df) < 10:
                print(f"  ⚠ Split {i+1}: Insufficient data (train={len(train_df)}, test={len(test_df)})")
                continue
            
            X_train = train_df[self.feature_cols]
            y_train = train_df[self.target_col]
            X_test = test_df[self.feature_cols]
            y_test = test_df[self.target_col]
            
            # Create a fresh model for this split
            from models.single_step import get_model
            
            # Detect model type from self.model
            model_type = 'xgboost'  # default
            if self.model is not None:
                model_name = type(self.model).__name__
                if 'Ridge' in model_name:
                    model_type = 'ridge'
                elif 'RandomForest' in model_name:
                    model_type = 'random_forest'
            
            # Create fresh model instance
            fresh_model_wrapper = get_model(model_type)
            fresh_model = fresh_model_wrapper.get_model()  # Get the actual sklearn/xgboost model
            
            # Train
            fresh_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fresh_model.predict(X_test)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_all(y_test, y_pred)
            
            self.results.append({
                'split': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_period': f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
                'test_period': f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
                'metrics': metrics
            })
            
            print(f"  Split {i+1}/{n_splits}: MAE=${metrics['MAE']:.2f}, RMSE=${metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
        
        return self.get_average_metrics()
    
    def get_average_metrics(self):
        """Calculate average metrics across all splits"""
        if not self.results:
            return {}
        
        avg_metrics = {}
        metric_keys = self.results[0]['metrics'].keys()
        
        for key in metric_keys:
            values = [r['metrics'][key] for r in self.results]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics
    
    def print_summary(self):
        """Print backtest summary"""
        avg_metrics = self.get_average_metrics()
        
        if not avg_metrics:
            print("No backtest results available")
            return
        
        print(f"\n{'='*60}")
        print("BACKTEST SUMMARY (Walk-Forward Validation)")
        print(f"{'='*60}")
        print(f"Number of splits: {len(self.results)}")
        print(f"\nAverage Metrics:")
        print(f"  MAE: ${avg_metrics['avg_MAE']:.2f} (±${avg_metrics['std_MAE']:.2f})")
        print(f"  RMSE: ${avg_metrics['avg_RMSE']:.2f} (±${avg_metrics['std_RMSE']:.2f})")
        print(f"  R²: {avg_metrics['avg_R2']:.4f} (±{avg_metrics['std_R2']:.4f})")
        print(f"  Direction Accuracy: {avg_metrics['avg_Direction_Accuracy']:.2f}%")
        print(f"{'='*60}\n")