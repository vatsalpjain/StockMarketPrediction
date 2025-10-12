"""Configuration settings for the stock prediction system"""
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

# Data settings
DEFAULT_PERIOD = "2y"
TRAIN_TEST_SPLIT = 0.8

# Technical indicators
SMA_PERIOD = 20
EMA_PERIOD = 20
RSI_PERIOD = 14
BOLLINGER_BANDS_PERIOD = 20
BOLLINGER_BANDS_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
HMA_PERIOD = 20

# Model settings
RANDOM_STATE = 42
CV_FOLDS = 5

# Multi-horizon settings
PREDICTION_HORIZONS = [1, 7, 30]

# Plot settings
PLOT_MONTHS = 4
PLOT_DPI = 300

# Feature columns
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'EMA_20', 'UPPER_BAND', 'LOWER_BAND',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'HMA_20', 'Sentiment_Score'
]