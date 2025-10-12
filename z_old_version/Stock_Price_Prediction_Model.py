import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for environments without display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mplfinance as mpf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import timedelta
from matplotlib.dates import WeekdayLocator, MO
import matplotlib.dates as mdates
import os
import pickle
import argparse
import json
from pathlib import Path
import hashlib


class StockPredictionPipeline:
    """Complete pipeline for stock price prediction with caching"""
    
    def __init__(self, ticker_symbol, output_dir="output", cache_dir="cache", force_refresh=False):
        self.ticker_symbol = ticker_symbol.upper()
        self.output_dir = Path(output_dir) / self.ticker_symbol
        self.cache_dir = Path(cache_dir)
        self.force_refresh = force_refresh
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.model = None
        self.metrics = {}
        
    def _get_cache_path(self, data_type):
        """Generate cache file path"""
        return self.cache_dir / f"{self.ticker_symbol}_{data_type}.pkl"
    
    def _get_data_hash(self, df):
        """Generate hash of dataframe for cache validation"""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def download_data(self, period="2y"):
        """Download stock data with caching"""
        cache_file = self._get_cache_path("raw_data")
        
        if not self.force_refresh and cache_file.exists():
            print(f"Loading cached data for {self.ticker_symbol}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.df = cached_data['data']
                print(f"Loaded {len(self.df)} days from cache")
                return
        
        print(f"Downloading data for {self.ticker_symbol}...")
        ticker = yf.Ticker(self.ticker_symbol)
        historical_data = ticker.history(period=period)
        
        if historical_data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker_symbol}")
        
        self.df = historical_data.copy()
        print(f"Downloaded {len(self.df)} days of historical data")
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': self.df, 'hash': self._get_data_hash(self.df)}, f)
        print(f"Data cached to {cache_file}")
    
    def calculate_indicators(self):
        """Calculate all technical indicators efficiently"""
        print("Calculating technical indicators...")
        
        # SMA & EMA
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        rolling_std = self.df['Close'].rolling(window=20).std()
        self.df['UPPER_BAND'] = self.df['SMA_20'] + (2 * rolling_std)
        self.df['LOWER_BAND'] = self.df['SMA_20'] - (2 * rolling_std)
        
        # RSI
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_short = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema_long = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_short - ema_long
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # Hull Moving Average
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        
        half_length = 10
        sqrt_length = int(np.sqrt(20))
        wma_half = wma(self.df['Close'], half_length)
        wma_full = wma(self.df['Close'], 20)
        diff = 2 * wma_half - wma_full
        self.df['HMA_20'] = wma(diff, sqrt_length)
        
        # Drop unnecessary columns
        if 'Dividends' in self.df.columns and 'Stock Splits' in self.df.columns:
            self.df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        
        print("Indicators calculated successfully")

    def fetch_and_analyze_sentiment(self):
        """Fetch news sentiment from Marketaux, Finnhub, and Alpha Vantage APIs"""
        import requests
        from dotenv import load_dotenv
        import finnhub
        
        load_dotenv()
        
        sentiment_scores = []
        sources = []
        detailed_info = {}
        
        # Marketaux API
        MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
        marketaux_sentiment = None
        
        if MARKETAUX_API_KEY:
            print(f"Fetching sentiment from Marketaux for {self.ticker_symbol}...")
            try:
                url = f"https://api.marketaux.com/v1/news/all?symbols={self.ticker_symbol}&filter_entities=true&language=en&api_token={MARKETAUX_API_KEY}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    article_sentiments = []
                    for article in data['data'][:10]:
                        if 'entities' in article:
                            for entity in article['entities']:
                                if entity['symbol'] == self.ticker_symbol and 'sentiment_score' in entity:
                                    article_sentiments.append(entity['sentiment_score'])
                    
                    if article_sentiments:
                        marketaux_sentiment = sum(article_sentiments) / len(article_sentiments)
                        sentiment_scores.append(marketaux_sentiment)
                        sources.append('Marketaux')
                        detailed_info['marketaux'] = {
                            'score': float(marketaux_sentiment),
                            'num_articles': len(article_sentiments)
                        }
                        print(f"✓ Marketaux Sentiment: {marketaux_sentiment:.3f} from {len(article_sentiments)} articles")
            except Exception as e:
                print(f"Marketaux Error: {e}")
        else:
            print("Marketaux API key not found in environment variables.")
        
        # Finnhub API
        FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
        finnhub_sentiment = None
        
        if FINNHUB_API_KEY:
            print(f"Fetching sentiment from Finnhub for {self.ticker_symbol}...")
            try:
                finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
                sentiment_data = finnhub_client.news_sentiment(self.ticker_symbol)
                
                if sentiment_data and 'sentiment' in sentiment_data:
                    finnhub_sentiment = (sentiment_data['sentiment']['bullishPercent'] - sentiment_data['sentiment']['bearishPercent']) / 100
                    sentiment_scores.append(finnhub_sentiment)
                    sources.append('Finnhub')
                    detailed_info['finnhub'] = {
                        'score': float(finnhub_sentiment),
                        'bullish_percent': sentiment_data['sentiment']['bullishPercent'],
                        'bearish_percent': sentiment_data['sentiment']['bearishPercent']
                    }
                    print(f"✓ Finnhub Sentiment: {finnhub_sentiment:.3f}")
                    print(f"  Bullish: {sentiment_data['sentiment']['bullishPercent']:.1f}%")
                    print(f"  Bearish: {sentiment_data['sentiment']['bearishPercent']:.1f}%")
            except Exception as e:
                print(f"Finnhub Error: {e}")
        else:
            print("Finnhub API key not found in environment variables.")
        
        # Alpha Vantage API
        ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        alpha_vantage_sentiment = None
        
        if ALPHA_VANTAGE_API_KEY:
            print(f"Fetching sentiment from Alpha Vantage for {self.ticker_symbol}...")
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if 'Error Message' in data:
                    print(f"⚠ Alpha Vantage API Error: {data['Error Message']}")
                elif 'Note' in data:
                    print(f"⚠ Alpha Vantage API Limit: {data['Note']} (Free tier: 25 calls/day)")
                elif 'feed' in data and len(data['feed']) > 0:
                    av_sentiment_scores = []
                    
                    for article in data['feed'][:10]:
                        ticker_sentiments = article.get('ticker_sentiment', [])
                        ticker_score = None
                        
                        for ticker_sent in ticker_sentiments:
                            if ticker_sent.get('ticker') == self.ticker_symbol:
                                ticker_score = float(ticker_sent.get('ticker_sentiment_score', 0))
                                break
                        
                        if ticker_score is None:
                            ticker_score = float(article.get('overall_sentiment_score', 0))
                        
                        av_sentiment_scores.append(ticker_score)
                    
                    if av_sentiment_scores:
                        alpha_vantage_sentiment = sum(av_sentiment_scores) / len(av_sentiment_scores)
                        sentiment_scores.append(alpha_vantage_sentiment)
                        sources.append('Alpha Vantage')
                        
                        # Determine overall label
                        if alpha_vantage_sentiment >= 0.15:
                            av_label = "BULLISH 📈"
                        elif alpha_vantage_sentiment <= -0.15:
                            av_label = "BEARISH 📉"
                        else:
                            av_label = "NEUTRAL ➖"
                        
                        detailed_info['alpha_vantage'] = {
                            'score': float(alpha_vantage_sentiment),
                            'num_articles': len(av_sentiment_scores),
                            'label': av_label
                        }
                        print(f"✓ Alpha Vantage Sentiment: {alpha_vantage_sentiment:.3f} ({av_label}) from {len(av_sentiment_scores)} articles")
            except Exception as e:
                print(f"Alpha Vantage Error: {e}")
        else:
            print("Alpha Vantage API key not found in environment variables.")
        
        # Calculate combined sentiment
        if sentiment_scores:
            combined_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Determine combined label
            if combined_sentiment >= 0.15:
                combined_label = "BULLISH 📈"
            elif combined_sentiment <= -0.15:
                combined_label = "BEARISH 📉"
            else:
                combined_label = "NEUTRAL ➖"
            
            print(f"\n{'='*60}")
            print(f"✓ COMBINED SENTIMENT: {combined_sentiment:.3f} ({combined_label})")
            print(f"  Sources: {', '.join(sources)} ({len(sources)} APIs)")
            print(f"{'='*60}\n")
            
            self.df['Sentiment_Score'] = combined_sentiment
            self.sentiment_info = {
                'average_score': float(combined_sentiment),
                'label': combined_label,
                'sources': sources,
                'num_sources': len(sources),
                'detailed_scores': detailed_info
            }
        else:
            print("\n⚠ No sentiment data available from any API")
            print("Using neutral sentiment (0.0)\n")
            self.df['Sentiment_Score'] = 0.0
            self.sentiment_info = {
                'average_score': 0.0,
                'label': 'NEUTRAL ➖',
                'sources': [],
                'num_sources': 0
            }


    def train_model(self, model_type='xgboost'):
        """Train prediction model with caching"""
        
        cache_file = self._get_cache_path(f"model_{model_type}")
        data_hash = self._get_data_hash(self.df)
        
        # Check if cached model is valid
        if not self.force_refresh and cache_file.exists():
            print(f"Loading cached model for {self.ticker_symbol}...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                if cached.get('data_hash') == data_hash:
                    self.model = cached['model']
                    self.metrics = cached['metrics']
                    print("Model loaded from cache")
                    return
                else:
                    print("Cache invalid, retraining model...")
        
        print(f"Training {model_type} model with GridSearchCV...")
        
        # Prepare data
        self.df['Target'] = self.df['Close'].shift(-1)
        df_clean = self.df.dropna()
        
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20',
                       'UPPER_BAND', 'LOWER_BAND', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'HMA_20','Sentiment_Score']
        
        X = df_clean[feature_cols]
        y = df_clean['Target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        
        # Model selection
        if model_type == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror', random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 10, 100]
            }
        elif model_type == 'ridge':
            model = Ridge()
            param_grid = {
                'alpha': [0.1, 1.0, 10, 50, 100, 200, 500, 1000],
                'solver': ['auto', 'svd', 'cholesky']
            }
        else:  # random forest
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Grid search
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print("\nBest Parameters:", grid_search.best_params_)
        print("Best CV Score (MSE):", -grid_search.best_score_)
        
        # Predictions
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'best_params': grid_search.best_params_
        }
        
        # Add predictions to dataframe
        self.df.loc[X_test.index, 'Predicted_Close'] = y_pred
        
        # Cache model
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics,
                'data_hash': data_hash
            }, f)
        print(f"Model cached to {cache_file}")
        
        self._print_metrics()
    
    def _print_metrics(self):
        """Print model performance metrics"""
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE): {self.metrics['MSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {self.metrics['MAE']:.4f}")
        print(f"R-squared (R2): {self.metrics['R2']:.4f}")
        print(f"{'='*50}")
    
    def generate_plots(self, skip_if_exists=True):
        """Generate all plots with option to skip if already exists"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_files = {
            'price_predictions.png': self._plot_price_predictions,
            'rsi_indicator.png': self._plot_rsi,
            'macd_indicator.png': self._plot_macd
        }
        
        for filename, plot_func in plot_files.items():
            filepath = plots_dir / filename
            if skip_if_exists and filepath.exists():
                print(f"Skipping {filename} (already exists)")
                continue
            
            print(f"Creating {filename}...")
            plot_func(filepath)
            print(f"Saved: {filepath}")
    
    def _plot_price_predictions(self, filepath):
        """Plot price predictions with indicators"""
        last_4_months_start = self.df.index.max() - pd.DateOffset(months=4)
        df_recent = self.df.loc[self.df.index >= last_4_months_start].copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12),
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
        
        # Price chart
        ax1.plot(df_recent.index, df_recent['Close'], color='black', linewidth=2, label='Close Price', alpha=0.8)
        ax1.plot(df_recent.index, df_recent['SMA_20'], color='purple', linewidth=2, linestyle='--', label='SMA 20', alpha=0.9)
        ax1.plot(df_recent.index, df_recent['EMA_20'], color='orange', linewidth=2, linestyle='--', label='EMA 20', alpha=0.9)
        ax1.plot(df_recent.index, df_recent['UPPER_BAND'], color='gray', linewidth=1.5, linestyle=':', alpha=0.7)
        ax1.plot(df_recent.index, df_recent['LOWER_BAND'], color='gray', linewidth=1.5, linestyle=':', alpha=0.7)
        ax1.fill_between(df_recent.index, df_recent['UPPER_BAND'], df_recent['LOWER_BAND'],
                         color='lightgray', alpha=0.2, label='Bollinger Bands')
        ax1.plot(df_recent.index, df_recent['HMA_20'], color='cyan', linewidth=2, label='HMA 20', alpha=0.9)
        ax1.plot(df_recent.index, df_recent['Predicted_Close'], color='red', linewidth=3, label='Predictions', alpha=0.9)
        
        textstr = f'Model Performance:\nR² = {self.metrics["R2"]:.3f}\nMAE = ${self.metrics["MAE"]:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        ax1.set_title(f'{self.ticker_symbol} Stock Price with Technical Indicators & Predictions (Last 4 Months)',
                      fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Date', fontsize=14)
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax1.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Volume chart
        colors = ['green' if close >= open_price else 'red'
                  for open_price, close in zip(df_recent['Open'], df_recent['Close'])]
        ax2.bar(df_recent.index, df_recent['Volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rsi(self, filepath):
        """Plot RSI indicator"""
        last_4_months_start = self.df.index.max() - pd.DateOffset(months=4)
        df_recent = self.df.loc[self.df.index >= last_4_months_start].copy()
        
        plt.figure(figsize=(20, 6))
        plt.plot(df_recent.index, df_recent['RSI_14'], color='purple', linewidth=3, label='RSI (14-day)')
        plt.axhline(70, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Oversold (30)')
        plt.axhline(50, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (50)')
        plt.fill_between(df_recent.index, 70, 100, color='red', alpha=0.1, label='Overbought Zone')
        plt.fill_between(df_recent.index, 0, 30, color='green', alpha=0.1, label='Oversold Zone')
        
        current_rsi = df_recent['RSI_14'].iloc[-1]
        status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        status_text = f'Current RSI: {current_rsi:.1f} ({status})'
        plt.text(0.98, 0.95, status_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.title(f'{self.ticker_symbol} - Relative Strength Index (RSI) - Last 4 Months', 
                  fontsize=18, fontweight='bold', pad=15)
        plt.ylabel('RSI Value', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.gca().xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_macd(self, filepath):
        """Plot MACD indicator"""
        last_4_months_start = self.df.index.max() - pd.DateOffset(months=4)
        df_recent = self.df.loc[self.df.index >= last_4_months_start].copy()
        
        plt.figure(figsize=(20, 6))
        plt.plot(df_recent.index, df_recent['MACD'], color='blue', linewidth=3, label='MACD Line')
        plt.plot(df_recent.index, df_recent['MACD_Signal'], color='orange', linewidth=3, label='Signal Line')
        
        hist_colors = ['green' if x > 0 else 'red' for x in df_recent['MACD_Hist']]
        plt.bar(df_recent.index, df_recent['MACD_Hist'], color=hist_colors, alpha=0.6, width=0.7, label='Histogram')
        plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero Line')
        
        current_macd = df_recent['MACD'].iloc[-1]
        current_signal = df_recent['MACD_Signal'].iloc[-1]
        signal_status = "Bullish" if current_macd > current_signal else "Bearish"
        macd_text = f'Current Signal: {signal_status}'
        plt.text(0.98, 0.95, macd_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightgreen' if signal_status == 'Bullish' else 'lightcoral', alpha=0.8))
        
        plt.title(f'{self.ticker_symbol} - MACD Indicator - Last 4 Months', 
                  fontsize=18, fontweight='bold', pad=15)
        plt.ylabel('MACD Value', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.gca().xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary(self):
        """Generate and save analysis summary"""
        last_4_months_start = self.df.index.max() - pd.DateOffset(months=4)
        df_recent = self.df.loc[self.df.index >= last_4_months_start].copy()
        
        print("\n" + "="*50)
        print("PREDICTION ANALYSIS")
        print("="*50)
        
        try:
            if 'Predicted_Close' in self.df.columns:
                last_prediction = self.df['Predicted_Close'].dropna()
                if not last_prediction.empty:
                    last_pred_date = last_prediction.index[-1]
                    last_pred_value = last_prediction.iloc[-1]
                    actual_price = self.df.loc[last_pred_date, 'Close']
                    
                    print(f"Most recent prediction date: {last_pred_date.date()}")
                    print(f"Predicted price: ${last_pred_value:.2f}")
                    print(f"Actual price: ${actual_price:.2f}")
                    
                    prediction_error = abs(actual_price - last_pred_value)
                    error_percentage = (prediction_error / actual_price) * 100
                    print(f"Prediction error: ${prediction_error:.2f} ({prediction_error:.1f}%)")
                
                prediction_coverage = (self.df['Predicted_Close'].notna().sum() / len(self.df)) * 100
                print(f"Prediction coverage: {prediction_coverage:.1f}% of total data points")
        except Exception as e:
            print(f"Error in prediction analysis: {e}")
        
        print("\n" + "="*50)
        print("TECHNICAL ANALYSIS SUMMARY")
        print("="*50)
        
        try:
            rsi_current = df_recent['RSI_14'].iloc[-1]
            macd_current = df_recent['MACD'].iloc[-1]
            signal_current = df_recent['MACD_Signal'].iloc[-1]
            close_current = df_recent['Close'].iloc[-1]
            sma_current = df_recent['SMA_20'].iloc[-1]
            hma_current = df_recent['HMA_20'].iloc[-1]
            
            summary = {
                'ticker': self.ticker_symbol,
                'current_price': float(close_current),
                'sma_20': float(sma_current),
                'hma_20': float(hma_current),
                'rsi_14': float(rsi_current),
                'macd_signal': 'Bullish' if macd_current > signal_current else 'Bearish',
                'sentiment': self.sentiment_info if hasattr(self, 'sentiment_info') else None,
                'metrics': self.metrics
            }
            
            print(f"Ticker: {self.ticker_symbol}")
            print(f"Current Close Price: ${close_current:.2f}")
            print(f"20-day SMA: ${sma_current:.2f} ({'Above' if close_current > sma_current else 'Below'} SMA)")
            print(f"20-day HMA: ${hma_current:.2f} ({'Above' if close_current > hma_current else 'Below'} HMA)")
            print(f"RSI (14): {rsi_current:.1f} ({'Overbought' if rsi_current > 70 else 'Oversold' if rsi_current < 30 else 'Neutral'})")
            print(f"MACD Signal: {summary['macd_signal']}")
            if hasattr(self, 'sentiment_info') and self.sentiment_info['num_sources'] > 0:
                sent_score = self.sentiment_info['average_score']
                sent_label = self.sentiment_info.get('label', 'NEUTRAL ➖')
                num_sources = self.sentiment_info['num_sources']
                sources_str = ', '.join(self.sentiment_info['sources'])
                print(f"News Sentiment: {sent_label} (Score: {sent_score:.3f})")
                print(f"  Combined from {num_sources} source(s): {sources_str}")
            
            # Save summary to JSON
            summary_file = self.output_dir / "analysis_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            print(f"\nSummary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Could not generate technical analysis summary: {e}")
        
        print("="*50)
    
    def run_pipeline(self, model_type='xgboost', skip_plots=False):
        """Run complete pipeline"""
        print(f"\n{'='*60}")
        print(f"STOCK PREDICTION PIPELINE - {self.ticker_symbol}")
        print(f"{'='*60}\n")
        
        self.download_data()
        self.calculate_indicators()
        self.fetch_and_analyze_sentiment() 
        self.train_model(model_type=model_type)
        
        if not skip_plots:
            self.generate_plots(skip_if_exists=True)
        else:
            print("\nSkipping plot generation (--skip-plots enabled)")
        
        self.generate_summary()
        
        print(f"\n{'='*60}")
        print(f"Analysis complete for {self.ticker_symbol}!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Pipeline')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., META, AAPL, TSLA)')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'ridge', 'random_forest'],
                       help='Model type to use (default: xgboost)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Cache directory for models and data (default: cache)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh data and retrain model')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    pipeline = StockPredictionPipeline(
        ticker_symbol=args.ticker,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh
    )
    
    pipeline.run_pipeline(model_type=args.model, skip_plots=args.skip_plots)


if __name__ == "__main__":
    main()


##To run this 
# # Basic usage - predict META stock
# python Stock_Price_Prediction_Model.py META

# # Predict different stocks
# python Stock_Price_Prediction_Model.py AAPL
# python Stock_Price_Prediction_Model.py TSLA
# python Stock_Price_Prediction_Model.py GOOGL

# # Use different model
# python Stock_Price_Prediction_Model.py META --model ridge

# # Force refresh (ignore cache)
# python Stock_Price_Prediction_Model.py META --force-refresh

# # Skip plots (faster)
# python Stock_Price_Prediction_Model.py META --skip-plots

# # Custom output directory
# python Stock_Price_Prediction_Model.py META --output-dir my_analysis