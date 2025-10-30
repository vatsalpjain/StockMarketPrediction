"""Model training logic with GridSearchCV"""
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import numpy as np
from config.settings import TRAIN_TEST_SPLIT, RANDOM_STATE, CV_FOLDS

class ModelTrainer:
    """Handles model training with hyperparameter tuning"""
    
    def __init__(self, model, verbose=1):
        self.model = model
        self.verbose = verbose
        self.best_estimator = None
        self.best_params = None
        self.cv_score = None
    
    def train(self, X, y, param_grid=None):
        """Train model with GridSearchCV"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=TRAIN_TEST_SPLIT, 
            shuffle=False,
            random_state=RANDOM_STATE
        )
        
        if param_grid:
            print(f"Training with GridSearchCV (CV={CV_FOLDS})...")
            
            grid_search = GridSearchCV(
                self.model.get_model(),
                param_grid=param_grid,
                # Use time-aware CV for time series data
                cv=TimeSeriesSplit(n_splits=CV_FOLDS),
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=self.verbose
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_estimator = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_score = -grid_search.best_score_
            
            print("\nBest Parameters:", self.best_params)
            print(f"Best CV Score (MSE): {self.cv_score:.4f}")

            # For XGBoost, refit with early stopping on a validation tail to reduce overfitting
            try:
                from xgboost import XGBRegressor  # local import to avoid hard dependency for non-XGB runs
                if isinstance(self.best_estimator, XGBRegressor):
                    # Create validation split from the tail of training data (time-aware)
                    val_size = max(1, int(0.2 * len(X_train)))
                    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
                    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

                    # Rebuild the model with best params to enable early stopping cleanly
                    best_params = {**self.best_estimator.get_params()}
                    # Ensure objective and random_state are set
                    best_params['objective'] = 'reg:squarederror'
                    best_params['random_state'] = RANDOM_STATE
                    model_es = XGBRegressor(**best_params)
                    # Early stopping on validation tail
                    model_es.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        eval_metric=['rmse', 'mae'],  # monitor both magnitude and stability
                        verbose=False
                    )
                    self.best_estimator = model_es
            except Exception:
                # If XGBoost is unavailable or any issue arises, proceed with grid_search estimator
                pass
            
        else:
            print("Training without hyperparameter tuning...")
            model_instance = self.model.get_model()
            # Add early stopping for XGBoost using a small validation tail
            try:
                from xgboost import XGBRegressor  # local import
                if isinstance(model_instance, XGBRegressor):
                    val_size = max(1, int(0.2 * len(X_train)))
                    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
                    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
                    model_instance.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        eval_metric=['rmse', 'mae'],  # monitor both magnitude and stability
                        verbose=False
                    )
                else:
                    model_instance.fit(X_train, y_train)
            except Exception:
                model_instance.fit(X_train, y_train)
            self.best_estimator = model_instance
        
        # Make predictions
        y_pred = self.best_estimator.predict(X_test)
        
        # Calculate metrics
        metrics = self.model.calculate_metrics(y_test, y_pred)
        
        return self.best_estimator, metrics, X_test.index, y_pred