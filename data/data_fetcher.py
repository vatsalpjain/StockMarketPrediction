"""Stock data fetching from Yahoo Finance"""
import yfinance as yf
from config.settings import DEFAULT_PERIOD

class StockDataFetcher:
    """Handles downloading stock data"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
    
    def fetch(self, period=DEFAULT_PERIOD):
        """Fetch stock data (no caching - only models are cached)"""
        
        # Download fresh data
        print(f"Downloading data for {self.ticker}...")
        ticker_obj = yf.Ticker(self.ticker)
        data = ticker_obj.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker}")
        
        print(f"✓ Downloaded {len(data)} days of historical data")
        
        return data