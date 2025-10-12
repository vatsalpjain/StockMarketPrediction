"""Evaluation metrics for stock predictions"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def calculate_all(y_true, y_pred):
        """Calculate all available metrics"""
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': MetricsCalculator.calculate_mape(y_true, y_pred),
            'Direction_Accuracy': MetricsCalculator.calculate_direction_accuracy(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def calculate_direction_accuracy(y_true, y_pred):
        """Calculate directional accuracy (up/down prediction)"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate direction (1 for up, -1 for down)
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        return (correct / total) * 100 if total > 0 else 0
    
    @staticmethod
    def print_metrics(metrics, title="Model Performance"):
        """Print formatted metrics"""
        print(f"\n{'='*50}")
        print(title.upper())
        print(f"{'='*50}")
        
        for metric, value in metrics.items():
            if metric in ['MSE', 'RMSE', 'MAE']:
                print(f"{metric}: ${value:.2f}")
            elif metric in ['R2']:
                print(f"{metric}: {value:.4f}")
            elif metric in ['MAPE', 'Direction_Accuracy']:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value}")
        
        print(f"{'='*50}\n")