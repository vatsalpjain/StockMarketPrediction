"""Base model class for stock prediction"""
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class BaseStockModel(ABC):
    """Abstract base class for stock prediction models"""
    
    def __init__(self):
        self.model = None
        self.metrics = {}
        self.is_trained = False
    
    @abstractmethod
    def get_model(self):
        """Return the model instance"""
        pass
    
    @abstractmethod
    def get_param_grid(self):
        """Return parameter grid for GridSearchCV"""
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        self.metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        return self.metrics
    
    def print_metrics(self):
        """Print model performance metrics"""
        if not self.metrics:
            print("No metrics available. Train the model first.")
            return
        
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE): {self.metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {self.metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {self.metrics['MAE']:.4f}")
        print(f"R-squared (R2): {self.metrics['R2']:.4f}")
        print(f"{'='*50}")