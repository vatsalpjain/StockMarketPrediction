"""API key management"""
import os
from dotenv import load_dotenv

load_dotenv()

class APIKeys:
    """Centralized API key management"""
    
    @staticmethod
    def get_marketaux_key():
        return os.getenv("MARKETAUX_API_KEY")
    
    @staticmethod
    def get_finnhub_key():
        return os.getenv("FINNHUB_API_KEY")
    
    @staticmethod
    def get_alpha_vantage_key():
        return os.getenv("ALPHA_VANTAGE_API_KEY")
    
    @staticmethod
    def has_any_sentiment_api():
        """Check if at least one sentiment API is available"""
        return any([
            APIKeys.get_marketaux_key(),
            APIKeys.get_finnhub_key(),
            APIKeys.get_alpha_vantage_key()
        ])