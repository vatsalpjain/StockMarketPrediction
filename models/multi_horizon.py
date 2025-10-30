"""Multi-horizon prediction models"""
import pandas as pd
import numpy as np
from .single_step import get_model
from config.settings import PREDICTION_HORIZONS

class MultiHorizonModel:
    """Train and manage multiple models for different prediction horizons"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.models = {}
        self.metrics = {}
        self.horizons = PREDICTION_HORIZONS
        # Optional per-horizon auxiliary artifacts (e.g., scalers, checkpoints for LSTM)
        self.artifacts = {}
    
    def create_targets(self, df):
        """Create target variables for each horizon"""
        df_copy = df.copy()
        
        for horizon in self.horizons:
            df_copy[f'Target_{horizon}d'] = df_copy['Close'].shift(-horizon)
        
        return df_copy
    
    def get_models_dict(self):
        """Get dictionary of models for each horizon"""
        if not self.models:
            for horizon in self.horizons:
                # For classic sklearn/xgboost models, use registry; LSTM handled in pipelines
                if self.model_type != 'lstm':
                    self.models[f'{horizon}d'] = get_model(self.model_type)
        
        return self.models

    def set_horizon_artifacts(self, horizon: int, **kwargs):
        """Store auxiliary artifacts for a horizon (e.g., scaler path, checkpoint path)."""
        key = f'{horizon}d'
        self.artifacts.setdefault(key, {}).update(kwargs)

    def get_horizon_artifacts(self, horizon: int):
        """Retrieve stored artifacts dict for a horizon; returns empty dict if none."""
        return self.artifacts.get(f'{horizon}d', {})
    
    def get_metrics_summary(self):
        """Get summary of all models' performance"""
        summary = {}
        for horizon, metrics in self.metrics.items():
            summary[horizon] = {
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            }
        return summary
    
    def print_all_metrics(self):
        """Print metrics for all horizon models"""
        print(f"\n{'='*70}")
        print("MULTI-HORIZON MODEL PERFORMANCE")
        print(f"{'='*70}")
        
        for horizon in self.horizons:
            key = f'{horizon}d'
            if key in self.metrics:
                metrics = self.metrics[key]
                print(f"\n{horizon}-Day Ahead Prediction:")
                print(f"  MAE: ${metrics['MAE']:.2f}")
                print(f"  RMSE: ${metrics['RMSE']:.2f}")
                print(f"  R²: {metrics['R2']:.4f}")
        
        print(f"{'='*70}\n")