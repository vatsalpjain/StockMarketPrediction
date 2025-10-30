"""Model package"""
from .base_model import BaseStockModel
from .single_step import get_model, XGBoostModel, RidgeModel, RandomForestModel
from .multi_horizon import MultiHorizonModel
try:
    from .lstm import LSTMTrainer, LSTMConfig
except Exception:
    # Optional dependency; available when torch is installed
    LSTMTrainer = None
    LSTMConfig = None
from .model_trainer import ModelTrainer