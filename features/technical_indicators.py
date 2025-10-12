"""Technical indicator calculations"""
import numpy as np
import pandas as pd
from config.settings import (
    SMA_PERIOD, EMA_PERIOD, RSI_PERIOD,
    BOLLINGER_BANDS_PERIOD, BOLLINGER_BANDS_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, HMA_PERIOD
)

class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    @staticmethod
    def calculate_sma(df, period=SMA_PERIOD):
        """Calculate Simple Moving Average"""
        return df['Close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(df, period=EMA_PERIOD):
        """Calculate Exponential Moving Average"""
        return df['Close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(df, period=BOLLINGER_BANDS_PERIOD, std_dev=BOLLINGER_BANDS_STD):
        """Calculate Bollinger Bands"""
        sma = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * rolling_std)
        lower_band = sma - (std_dev * rolling_std)
        return upper_band, lower_band, sma
    
    @staticmethod
    def calculate_rsi(df, period=RSI_PERIOD):
        """Calculate Relative Strength Index"""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
        """Calculate MACD indicator"""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def calculate_hma(df, period=HMA_PERIOD):
        """Calculate Hull Moving Average"""
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(
                lambda prices: np.dot(prices, weights) / weights.sum(), 
                raw=True
            )
        
        half_length = period // 2
        sqrt_length = int(np.sqrt(period))
        wma_half = wma(df['Close'], half_length)
        wma_full = wma(df['Close'], period)
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_length)
    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to dataframe"""
        # SMA & EMA
        df['SMA_20'] = TechnicalIndicators.calculate_sma(df)
        df['EMA_20'] = TechnicalIndicators.calculate_ema(df)
        
        # Bollinger Bands
        upper, lower, sma = TechnicalIndicators.calculate_bollinger_bands(df)
        df['UPPER_BAND'] = upper
        df['LOWER_BAND'] = lower
        
        # RSI
        df['RSI_14'] = TechnicalIndicators.calculate_rsi(df)
        
        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # HMA
        df['HMA_20'] = TechnicalIndicators.calculate_hma(df)
        
        # Clean up unnecessary columns
        if 'Dividends' in df.columns:
            df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')
        
        return df