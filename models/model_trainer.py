"""Model training logic with GridSearchCV"""
from sklearn.model_selection import train_test_split, GridSearchCV
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
                cv=CV_FOLDS,
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
            
        else:
            print("Training without hyperparameter tuning...")
            model_instance = self.model.get_model()
            model_instance.fit(X_train, y_train)
            self.best_estimator = model_instance
        
        # Make predictions
        y_pred = self.best_estimator.predict(X_test)
        
        # Calculate metrics
        metrics = self.model.calculate_metrics(y_test, y_pred)
        
        return self.best_estimator, metrics, X_test.index, y_pred