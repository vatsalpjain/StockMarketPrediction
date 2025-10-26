"""Cache management for models"""
import pickle
from pathlib import Path
from datetime import datetime

class CacheManager:
    """Handles caching of trained models"""
    
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, ticker, model_type):
        """Generate cache file path for model"""
        return self.cache_dir / f"{ticker}_model_{model_type}.pkl"
    
    def save_model(self, ticker, model, metrics, data_hash, model_type="xgboost", data_info=None):
        """Save trained model to cache with metadata"""
        cache_file = self._get_cache_path(ticker, model_type)
        cache_metadata = {
            'model': model,
            'metrics': metrics,
            'data_hash': data_hash,
            'cached_at': datetime.now().isoformat(),
            'model_type': model_type
        }
        
        # Add data range information for validation
        if data_info:
            cache_metadata.update(data_info)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_metadata, f)
        return cache_file
    
    def load_model(self, ticker, model_type="xgboost"):
        """Load model from cache"""
        cache_file = self._get_cache_path(ticker, model_type)
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached
    
    def is_cache_valid(self, cached_model, current_hash, max_age_days=2):
        """
        Check if cached model is valid based on:
        1. Hash matching (training data hasn't changed)
        2. Cache age (not too old)
        """
        if cached_model is None:
            return False
        
        # Check hash
        if cached_model.get('data_hash') != current_hash:
            return False
        
        # Check age
        if 'cached_at' in cached_model:
            try:
                cached_at = datetime.fromisoformat(cached_model['cached_at'])
                age_days = (datetime.now() - cached_at).days
                if age_days > max_age_days:
                    print(f"⚠ Model cache is {age_days} days old (max: {max_age_days} days). Will retrain.")
                    return False
            except:
                pass
        
        return True