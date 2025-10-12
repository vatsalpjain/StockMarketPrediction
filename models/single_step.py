"""Single-step (1-day ahead) prediction model"""
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseStockModel

class XGBoostModel(BaseStockModel):
    """XGBoost model for stock prediction"""
    
    def get_model(self):
        return XGBRegressor(objective='reg:squarederror', random_state=42)
    
    def get_param_grid(self):
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 10, 100]
        }


class RidgeModel(BaseStockModel):
    """Ridge regression model for stock prediction"""
    
    def get_model(self):
        return Ridge()
    
    def get_param_grid(self):
        return {
            'alpha': [0.1, 1.0, 10, 50, 100, 200, 500, 1000],
            'solver': ['auto', 'svd', 'cholesky']
        }


class RandomForestModel(BaseStockModel):
    """Random Forest model for stock prediction"""
    
    def get_model(self):
        return RandomForestRegressor(random_state=42)
    
    def get_param_grid(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }


def get_model(model_type='xgboost'):
    """Factory function to get model instance"""
    models = {
        'xgboost': XGBoostModel,
        'ridge': RidgeModel,
        'random_forest': RandomForestModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type]()