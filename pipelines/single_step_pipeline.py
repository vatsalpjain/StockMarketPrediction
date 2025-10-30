"""Single-step (1-day ahead) prediction pipeline"""
from pathlib import Path
from data.data_fetcher import StockDataFetcher
from data.cache_manager import CacheManager
from features.technical_indicators import TechnicalIndicators
from features.sentiment_analyzer import SentimentAnalyzer
from models.single_step import get_model
from features.sequence_builder import build_supervised_sequences, fit_feature_scaler, apply_feature_scaler
from config.settings import (
    FEATURE_COLS, DEFAULT_PERIOD,
    LSTM_LOOKBACK, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, LSTM_LR, LSTM_EPOCHS, LSTM_BATCH, USE_QUANTILES
)
from models.model_trainer import ModelTrainer
from visualization.price_plots import PricePlotter
from visualization.indicator_plots import IndicatorPlotter
from visualization.report_generator import ReportGenerator
from config.settings import FEATURE_COLS, DEFAULT_PERIOD
import numpy as np
import pandas as pd

class SingleStepPipeline:
    """Complete pipeline for single-step stock prediction"""
    
    def __init__(self, ticker, output_dir="output", cache_dir="cache", force_refresh=False):
        self.ticker = ticker.upper()
        self.output_dir = Path(output_dir) / self.ticker
        self.cache_dir = Path(cache_dir)
        self.force_refresh = force_refresh
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cache_manager = CacheManager(self.cache_dir)
        self.df = None
        self.model = None
        self.metrics = {}
        self.sentiment_info = {}
    
    def download_data(self, period=DEFAULT_PERIOD):
        """Step 1: Download stock data"""
        fetcher = StockDataFetcher(self.ticker)
        self.df = fetcher.fetch(period)
    
    def calculate_indicators(self):
        """Step 2: Calculate technical indicators"""
        print("Calculating technical indicators...")
        self.df = TechnicalIndicators.add_all_indicators(self.df)
        print("✓ Indicators calculated successfully")
    
    def analyze_sentiment(self):
        """Step 3: Analyze news sentiment"""
        analyzer = SentimentAnalyzer(self.ticker)
        self.sentiment_info = analyzer.get_combined_sentiment()
        
        # Add sentiment to dataframe
        if self.sentiment_info['num_sources'] > 0:
            self.df['Sentiment_Score'] = self.sentiment_info['average_score']
        else:
            self.df['Sentiment_Score'] = 0.0
    
    def train_model(self, model_type='xgboost', use_grid_search=True):
        """Step 4: Train prediction model"""
        
        # Check cache
        import hashlib
        data_hash = hashlib.md5(pd.util.hash_pandas_object(self.df, index=True).values).hexdigest()
        cached_model = self.cache_manager.load_model(self.ticker, model_type)
        
        if not self.force_refresh and self.cache_manager.is_cache_valid(cached_model, data_hash):
            print(f"✓ Loaded cached model for {self.ticker}")
            self.model = cached_model['model']
            self.metrics = cached_model['metrics']
            return
        
        # Prepare data
        print(f"Training {model_type} model...")

        # TARGETS
        if model_type == 'lstm':
            # Predict next-day closing price directly for LSTM
            self.df['Target'] = self.df['Close'].shift(-1)
        else:
            # r_{t+1} = (Close_{t+1} - Close_t) / Close_t
            self.df['Target'] = (self.df['Close'].shift(-1) - self.df['Close']) / self.df['Close']

        # MINIMAL FEATURE BLOCK: add a few robust, time-series friendly features
        # - Lagged returns, rolling volatility, overnight gap, intraday range
        self.df['Ret_1'] = self.df['Close'].pct_change(1)
        self.df['Ret_5'] = self.df['Close'].pct_change(5)
        self.df['Vol_5'] = self.df['Ret_1'].rolling(5).std()
        self.df['Gap'] = (self.df['Open'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        self.df['Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']

        df_clean = self.df.dropna()

        # Use existing FEATURE_COLS + the small feature block (local-only, no global setting change)
        feature_cols_local = FEATURE_COLS + ['Ret_1', 'Ret_5', 'Vol_5', 'Gap', 'Range']
        X = df_clean[feature_cols_local]
        y = df_clean['Target']
        
        if model_type == 'lstm':
            # LSTM path (sequence-aware)
            train_size = int(0.8 * len(X))
            scaler = fit_feature_scaler(X, train_end_index=train_size)
            X_scaled = apply_feature_scaler(X, scaler)
            X_seq, y_seq, _ = build_supervised_sequences(X_scaled, y, lookback=LSTM_LOOKBACK)
            adj_train_size = max(1, train_size - LSTM_LOOKBACK)
            X_train, y_train = X_seq[:adj_train_size], y_seq[:adj_train_size]
            X_test, y_test = X_seq[adj_train_size:], y_seq[adj_train_size:]

            # Standardize target (train only), then inverse for predictions
            y_train_mean = float(np.mean(y_train))
            y_train_std = float(np.std(y_train)) if float(np.std(y_train)) > 1e-8 else 1.0
            y_train_stdized = (y_train - y_train_mean) / y_train_std

            import torch
            Xt = torch.tensor(X_train, dtype=torch.float32)
            yt = torch.tensor(y_train_stdized, dtype=torch.float32)
            Xtt = torch.tensor(X_test, dtype=torch.float32)

            # Lazy import to avoid torch requirement unless selected
            from models.lstm import LSTMTrainer, LSTMConfig
            cfg = LSTMConfig(
                input_size=X.shape[1],
                hidden_size=LSTM_HIDDEN,
                num_layers=LSTM_LAYERS,
                dropout=LSTM_DROPOUT,
                lr=LSTM_LR,
                epochs=LSTM_EPOCHS,
                batch_size=LSTM_BATCH,
                use_quantiles=USE_QUANTILES,
            )
            lstm_trainer = LSTMTrainer(cfg)
            lstm_trainer.fit(Xt, yt, X_val=None, y_val=None)
            yp, _, _ = lstm_trainer.predict(Xtt)
            y_pred = (yp.numpy() * y_train_std) + y_train_mean

            # Align test index and write predictions
            test_index = X.index[train_size:][: len(y_pred)]
            self.df.loc[test_index, 'Predicted_Close'] = y_pred

            # Compute simple metrics vs true next-day Close on test_index
            from evaluation.metrics import MetricsCalculator
            y_true = y.loc[test_index].values
            self.metrics = MetricsCalculator.calculate_all(y_true, y_pred)
            self.model = None  # LSTM model handled separately; not used by report
        else:
            # Sklearn/XGBoost path
            model_instance = get_model(model_type)
            trainer = ModelTrainer(model_instance, verbose=1)
            param_grid = model_instance.get_param_grid() if use_grid_search else None
            self.model, self.metrics, test_index, y_pred = trainer.train(X, y, param_grid)

            # RECONSTRUCT NEXT-DAY PRICE from predicted return and today's Close
            pred_close = self.df.loc[test_index, 'Close'] * (1 + y_pred)
            # Store next-day price prediction at index t
            self.df.loc[test_index, 'Predicted_Close'] = pred_close
        
        # Cache model
        self.cache_manager.save_model(self.ticker, self.model, self.metrics, data_hash, model_type)

        # Print metrics if available from tree-based trainer
        if model_type != 'lstm':
            model_instance.print_metrics()
    
    def generate_visualizations(self, skip_if_exists=True):
        """Step 5: Generate plots"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        price_plotter = PricePlotter(self.df, self.ticker)
        indicator_plotter = IndicatorPlotter(self.df, self.ticker)
        
        plots = {
            'price_predictions.png': lambda f: price_plotter.plot_price_predictions(f, self.metrics),
            'rsi_indicator.png': indicator_plotter.plot_rsi,
            'macd_indicator.png': indicator_plotter.plot_macd
        }
        
        for filename, plot_func in plots.items():
            filepath = plots_dir / filename
            if skip_if_exists and filepath.exists():
                print(f"⊘ Skipping {filename} (already exists)")
                continue
            
            print(f"Creating {filename}...")
            plot_func(filepath)
            print(f"✓ Saved: {filepath}")
    
    def generate_reports(self):
        """Step 6: Generate analysis reports"""
        print("\nGenerating analysis reports...")
        reporter = ReportGenerator(
            self.df, 
            self.ticker, 
            self.output_dir,
            self.metrics,
            self.sentiment_info
        )
        # Pass trained model so report can forecast using returns and build walk-forward summary
        reporter.model = self.model
        reporter.generate_all()
    
    def run(self, model_type='xgboost', skip_plots=False):
        """Run complete pipeline"""
        print(f"\n{'='*70}")
        print(f"SINGLE-STEP PREDICTION PIPELINE - {self.ticker}")
        print(f"{'='*70}\n")
        
        self.download_data()
        self.calculate_indicators()
        self.analyze_sentiment()
        self.train_model(model_type=model_type)
        
        if not skip_plots:
            self.generate_visualizations(skip_if_exists=True)
        else:
            print("\n⊘ Skipping plot generation (--skip-plots enabled)")
        
        self.generate_reports()
        
        print(f"\n{'='*70}")
        print(f"✓ Analysis complete for {self.ticker}!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return self.df, self.model, self.metrics