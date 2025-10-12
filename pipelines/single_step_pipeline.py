"""Single-step (1-day ahead) prediction pipeline"""
from pathlib import Path
from data.data_fetcher import StockDataFetcher
from data.cache_manager import CacheManager
from features.technical_indicators import TechnicalIndicators
from features.sentiment_analyzer import SentimentAnalyzer
from models.single_step import get_model
from models.model_trainer import ModelTrainer
from visualization.price_plots import PricePlotter
from visualization.indicator_plots import IndicatorPlotter
from visualization.report_generator import ReportGenerator
from config.settings import FEATURE_COLS, DEFAULT_PERIOD
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
        fetcher = StockDataFetcher(self.ticker, self.cache_manager, self.force_refresh)
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
        data_hash = self.cache_manager._get_data_hash(self.df)
        cached_model = self.cache_manager.load_model(self.ticker, model_type)
        
        if not self.force_refresh and cached_model and \
           self.cache_manager.is_cache_valid(cached_model, data_hash):
            print(f"✓ Loaded cached model for {self.ticker}")
            self.model = cached_model['model']
            self.metrics = cached_model['metrics']
            return
        
        # Prepare data
        print(f"Training {model_type} model...")
        self.df['Target'] = self.df['Close'].shift(-1)
        df_clean = self.df.dropna()
        
        X = df_clean[FEATURE_COLS]
        y = df_clean['Target']
        
        # Train model
        model_instance = get_model(model_type)
        trainer = ModelTrainer(model_instance, verbose=1)
        
        param_grid = model_instance.get_param_grid() if use_grid_search else None
        self.model, self.metrics, test_index, y_pred = trainer.train(X, y, param_grid)
        
        # Add predictions to dataframe
        self.df.loc[test_index, 'Predicted_Close'] = y_pred
        
        # Cache model
        self.cache_manager.save_model(self.ticker, self.model, self.metrics, data_hash, model_type)
        
        # Print metrics
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