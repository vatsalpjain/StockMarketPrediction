# 📈 Advanced Stock Price Prediction System

> **A professional-grade ML platform for stock market analysis featuring advanced technical indicators, multi-horizon forecasting, and real-time sentiment analysis.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This comprehensive stock prediction platform combines **cutting-edge machine learning algorithms** with **sophisticated technical analysis** to deliver accurate, multi-horizon price forecasts. Built for traders, quantitative analysts, and financial researchers, the system leverages 15+ technical indicators and state-of-the-art deep learning models to provide actionable market insights.

### ✨ Key Highlights

- **🖥️ Professional Web Interface** - Enterprise-grade Streamlit dashboard with real-time analytics
- **📊 Advanced Technical Analysis** - 15+ professional-grade indicators including SMA, EMA, HMA, RSI, MACD, and Bollinger Bands
- **🤖 4 ML Models** - XGBoost, Ridge Regression, Random Forest, and LSTM neural networks
- **🔮 Multi-Horizon Forecasting** - Simultaneous predictions for 1, 7, and 30-day horizons
- **💭 Real-time Sentiment Analysis** - Multi-source news sentiment integration (Finnhub, MarketAux, Alpha Vantage)
- **📉 Walk-Forward Backtesting** - Time-series cross-validation for robust performance evaluation
- **⚡ Smart Caching** - Intelligent model caching with automatic invalidation

---

## 🎨 Interactive Web Interface

Launch the full-featured web application:

```bash
streamlit run app.py
```

Or on Windows: **double-click** `run_streamlit.bat`

📖 See [STREAMLIT_README.md](STREAMLIT_README.md) for complete UI documentation.

---

## � Technical Indicators - Core Feature Set

Our platform implements a comprehensive suite of **15+ technical indicators** for robust feature engineering:

### 📈 Trend Indicators

| Indicator | Period | Description | Use Case |
|-----------|--------|-------------|----------|
| **SMA (Simple Moving Average)** | 20 | Arithmetic mean of closing prices | Identify trend direction and support/resistance |
| **EMA (Exponential Moving Average)** | 20 | Weighted average favoring recent prices | More responsive to recent price changes |
| **HMA (Hull Moving Average)** | 20 | Weighted MA with reduced lag | Superior trend identification with minimal delay |

### 📊 Volatility Indicators

| Indicator | Configuration | Description | Application |
|-----------|--------------|-------------|-------------|
| **Bollinger Bands** | 20-period, 2σ | Standard deviation bands around SMA | Volatility measurement, overbought/oversold detection |
| **ATR (Average True Range)** | 14-period | Measure of price volatility | Risk management and position sizing |

### ⚡ Momentum Indicators

| Indicator | Period | Range | Interpretation |
|-----------|--------|-------|----------------|
| **RSI (Relative Strength Index)** | 14 | 0-100 | >70: Overbought, <30: Oversold |
| **MACD** | 12/26/9 | Unbounded | Signal line crossovers indicate buy/sell |
| **MACD Signal** | 9 | Unbounded | Smoothed MACD for confirmation |
| **MACD Histogram** | Derived | Unbounded | Distance between MACD and signal |

### 📉 Custom Time-Series Features

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| **Ret_1** | 1-day return | Short-term momentum |
| **Ret_5** | 5-day return | Medium-term momentum |
| **Vol_5** | 5-day volatility | Recent volatility measurement |
| **Gap** | Open vs. previous close | Overnight sentiment shift |
| **Range** | High-Low spread | Intraday volatility |

### 💭 Sentiment Score

- **Multi-source aggregation** from news APIs
- **Real-time sentiment analysis** using NLP
- **Weighted averaging** across sources
- **Range:** -1.0 (bearish) to +1.0 (bullish)

---

## 🛠️ Technology Stack

### Core Framework
```
Python 3.8+          │ Primary development language
Streamlit 1.28+      │ Interactive web application framework
NumPy 1.24+          │ Numerical computing and array operations
Pandas 2.0+          │ Data manipulation and time-series analysis
```

### Machine Learning
```
Scikit-learn 1.3+    │ Classical ML algorithms and preprocessing
XGBoost 2.0+         │ Gradient boosting framework
PyTorch 2.2+         │ Deep learning for LSTM networks
```

### Data & Visualization
```
yfinance 0.2.28+     │ Yahoo Finance API for historical data
Matplotlib 3.7+      │ Static plotting and visualizations
Seaborn 0.12+        │ Statistical data visualization
Plotly 5.17+         │ Interactive charts and dashboards
```

### APIs & Integration
```
Finnhub 2.4+         │ Financial news and sentiment data
MarketAux            │ Market news aggregation
Alpha Vantage        │ Additional sentiment analysis
python-dotenv 1.0+   │ Environment variable management
```

---

## 🚀 Features & Capabilities

---

## � Features & Capabilities

### 📈 Single-Step Prediction
- **Next-day price forecasting** with high accuracy
- **Return-based modeling** for consistent performance metrics
- **Multiple model support** - Choose optimal algorithm for your data
- **Real-time progress tracking** in web interface
- **Automatic hyperparameter tuning** via GridSearchCV with time-series CV
- **Early stopping** for gradient boosting models to prevent overfitting

### 🔮 Multi-Horizon Forecasting
- **Simultaneous predictions** for 1, 7, and 30-day horizons
- **Confidence intervals** with historical error analysis
- **Quantile predictions** (LSTM) for probabilistic forecasting
- **Independent model training** per horizon for optimal performance
- **Backtest comparison plots** showing actual vs. predicted prices

### 🧪 Robust Evaluation
- **Walk-forward validation** - Time-series aware cross-validation
- **5-fold time-series splits** maintaining temporal order
- **Comprehensive metrics**: MAE, RMSE, R², MAPE, Direction Accuracy
- **Visual backtesting** with detailed error analysis
- **Historical performance tracking** across multiple runs

### 💭 Sentiment Analysis
- **Multi-source aggregation** from 3 financial news APIs
- **Weighted sentiment scores** (-1.0 to +1.0)
- **Real-time news analysis** integrated into predictions
- **Sentiment-based features** for enhanced model performance
- **Article count tracking** and source-level breakdowns

### ⚡ Performance Optimizations
- **Smart caching system** with automatic invalidation
- **Feature-aware cache validation** prevents stale model usage
- **Parallel processing** for batch analysis
- **Efficient data pipelines** with minimal memory footprint
- **GPU acceleration** support for LSTM training (CUDA)

### 📊 Visualization Suite
- **Price charts** with overlaid predictions and indicators
- **Technical indicator plots** (RSI, MACD with interpretations)
- **Multi-horizon comparison charts** with Plotly interactivity
- **Backtest performance plots** showing prediction accuracy
- **Volume analysis** with buy/sell pressure visualization
- **High-DPI exports** (300 DPI) for publication-quality charts

### 🎯 Model Comparison & Analytics
- **Side-by-side model comparison** on same stock
- **Batch processing** for multiple tickers
- **Performance dashboards** with radar charts
- **Model selection guidance** based on metrics
- **Cross-stock performance analysis**

---

## 📂 Project Architecture

```
StockPricePrediction/
│
├── 🌐 app.py                          # Streamlit web application
├── 💻 main.py                         # CLI interface
│
├── 📊 streamlit_pages/                # Web UI components
│   ├── single_step_page.py           # Single-day prediction interface
│   ├── multi_horizon_page.py         # Multi-horizon forecasting UI
│   ├── results_page.py               # Analysis results viewer
│   └── comparison_page.py            # Model comparison dashboard
│
├── 🔄 pipelines/                      # ML pipelines
│   ├── single_step_pipeline.py       # 1-day prediction workflow
│   └── multi_horizon_pipeline.py     # Multi-horizon workflow
│
├── 🤖 models/                         # ML model implementations
│   ├── base_model.py                 # Abstract base class
│   ├── single_step.py                # XGBoost, Ridge, RF models
│   ├── multi_horizon.py              # Multi-output wrappers
│   ├── lstm.py                       # PyTorch LSTM implementation
│   └── model_trainer.py              # Training orchestration
│
├── 📦 data/                           # Data acquisition & management
│   ├── data_fetcher.py               # Yahoo Finance integration
│   ├── cache_manager.py              # Model & data caching
│   └── data_validator.py             # Data quality checks
│
├── 🔧 features/                       # Feature engineering
│   ├── technical_indicators.py       # 15+ indicator calculations
│   ├── sentiment_analyzer.py         # Multi-source sentiment
│   └── sequence_builder.py           # LSTM sequence preparation
│
├── 📏 evaluation/                     # Model evaluation
│   ├── metrics.py                    # Performance metrics
│   ├── backtester.py                 # Walk-forward validation
│   └── ts_backtester.py              # Time-series backtesting
│
├── 📈 visualization/                  # Plotting & reporting
│   ├── price_plots.py                # Price & prediction charts
│   ├── indicator_plots.py            # Technical indicator plots
│   └── report_generator.py           # Comprehensive reports
│
├── ⚙️ config/                         # Configuration
│   ├── settings.py                   # Global parameters
│   └── api_keys.py                   # API credentials
│
├── 🛠️ utils/                          # Utilities
│   ├── logger.py                     # Logging configuration
│   ├── helpers.py                    # Helper functions
│   └── file_utils.py                 # File operations
│
├── 📁 output/                         # Analysis results (auto-generated)
├── 💾 cache/                          # Model cache (auto-generated)
├── 📋 requirements.txt                # Python dependencies
└── 📖 README.md                       # This document
```

---

## 🚦 Quick Start Guide

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package installer)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vatsalpjain/StockPricePrediction.git
   cd StockPricePrediction/StockPricePrediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (Optional - for sentiment analysis)
   
   Edit `config/api_keys.py`:
   ```python
   FINNHUB_API_KEY = "your_finnhub_api_key"
   MARKETAUX_API_KEY = "your_marketaux_api_key"  # Optional
   ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"  # Optional
   ```
   
   Get free API keys:
   - [Finnhub](https://finnhub.io/) - 60 calls/minute (free tier)
   - [MarketAux](https://www.marketaux.com/) - Optional
   - [Alpha Vantage](https://www.alphavantage.co/) - Optional

### Option 1: Web Interface (Recommended)

```bash
streamlit run app.py
```

🌐 Navigate to `http://localhost:8501` in your browser

**Windows shortcut:** Double-click `run_streamlit.bat`

### Option 2: Command Line Interface

```bash
# Basic single-step prediction
python main.py AAPL

# Multi-horizon prediction
python main.py AAPL --multi-horizon

# With specific model
python main.py TSLA --model lstm

# Force refresh (ignore cache)
python main.py GOOGL --force-refresh

# Skip plots for faster execution
python main.py NVDA --skip-plots --skip-backtest

# LSTM with quantile predictions
python main.py META --model lstm --use-quantiles
```

---

## 🤖 Model Specifications

### XGBoost (Default - Recommended)
```
Type:           Gradient Boosting Decision Trees
Training:       GridSearchCV with TimeSeriesSplit
Parameters:     300-600 estimators, depth 2-4, early stopping
Performance:    High accuracy, fast training
Best for:       General-purpose prediction, noisy data
Speed:          ⚡⚡⚡⚡ Very Fast
```

### Ridge Regression
```
Type:           Regularized Linear Regression
Training:       L2 regularization with cross-validation
Parameters:     Alpha tuning (0.1-1000)
Performance:    Good baseline, interpretable
Best for:       Linear relationships, quick analysis
Speed:          ⚡⚡⚡⚡⚡ Extremely Fast
```

### Random Forest
```
Type:           Ensemble of Decision Trees
Training:       Bootstrap aggregating with feature sampling
Parameters:     50-200 estimators, depth tuning
Performance:    Robust to outliers and noise
Best for:       Non-linear patterns, feature importance
Speed:          ⚡⚡⚡ Fast
```

### LSTM (Deep Learning)
```
Type:           Recurrent Neural Network
Architecture:   1-2 layers, 64 hidden units, dropout 0.1
Training:       Adam optimizer, early stopping, AMP
Lookback:       60 timesteps
Performance:    Excellent for sequential patterns
Best for:       Long-term dependencies, complex patterns
Speed:          ⚡⚡ Moderate (GPU recommended)
Features:       Optional quantile heads (P10/P90) for uncertainty
```

---

## 📊 Performance Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average prediction error in dollars/percentage |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more heavily |
| **R² Score** | Coefficient of Determination | Variance explained (0-1, higher is better) |
| **MAPE** | Mean Absolute Percentage Error | Error as percentage of actual value |
| **Direction Accuracy** | Up/Down prediction correctness | Trading signal accuracy (%) |

---

## 📸 Screenshots & Results

### Single-Step Prediction Dashboard
- Real-time prediction with technical indicators
- Model performance metrics
- Sentiment analysis integration
- Next-day forecast with confidence

### Multi-Horizon Forecasting
- Simultaneous 1/7/30-day predictions
- Confidence intervals per horizon
- Backtest comparison charts
- Historical accuracy tracking

### Model Comparison Analytics
- Side-by-side performance comparison
- Radar charts for multi-metric analysis
- Batch processing results
- Performance distribution histograms

---

## 🔧 Advanced Configuration

### Technical Indicator Settings

Edit `config/settings.py` to customize indicators:

```python
# Moving Averages
SMA_PERIOD = 20              # Simple Moving Average period
EMA_PERIOD = 20              # Exponential Moving Average period
HMA_PERIOD = 20              # Hull Moving Average period

# Momentum Indicators
RSI_PERIOD = 14              # Relative Strength Index period
MACD_FAST = 12               # MACD fast period
MACD_SLOW = 26               # MACD slow period
MACD_SIGNAL = 9              # MACD signal line period

# Volatility Indicators
BOLLINGER_BANDS_PERIOD = 20  # Bollinger Bands period
BOLLINGER_BANDS_STD = 2      # Standard deviation multiplier

# Model Settings
TRAIN_TEST_SPLIT = 0.8       # 80% train, 20% test
CV_FOLDS = 5                 # Time-series cross-validation folds
PREDICTION_HORIZONS = [1, 7, 30]  # Multi-horizon targets

# LSTM Configuration
LSTM_LOOKBACK = 60           # Sequence length
LSTM_HIDDEN = 64             # Hidden layer size
LSTM_LAYERS = 1              # Number of LSTM layers
LSTM_DROPOUT = 0.1           # Dropout rate
LSTM_EPOCHS = 30             # Max training epochs
LSTM_BATCH = 128             # Batch size
USE_QUANTILES = False        # Enable quantile predictions
```

---

## 📚 Documentation

- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Complete web interface guide
- **[main.py](main.py)** - CLI usage and examples
- **[config/settings.py](config/settings.py)** - All configuration options

---

## 🎯 Usage Examples

### Example 1: Quick Analysis
```bash
python main.py AAPL
```
Generates next-day prediction with XGBoost, technical indicators, and sentiment analysis.

### Example 2: Multi-Horizon with LSTM
```bash
python main.py TSLA --multi-horizon --model lstm --use-quantiles
```
Forecasts 1/7/30-day prices with LSTM neural network and uncertainty bands.

### Example 3: Model Comparison
Use the web interface:
1. Navigate to "📉 Model Comparison"
2. Enter ticker symbol
3. Compare all 4 models side-by-side
4. View radar chart and performance metrics

### Example 4: Batch Processing
Use the web interface:
1. Navigate to "📉 Model Comparison" → "Batch Analysis"
2. Enter multiple tickers (e.g., `AAPL,GOOGL,MSFT,TSLA`)
3. Select model type
4. Run batch analysis
5. Download combined results

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** 

Stock market prediction is inherently uncertain and past performance does not guarantee future results. This tool should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and conduct your own research before making investment decisions.

The developers and contributors of this project are not responsible for any financial losses incurred through the use of this software.

---

## 🙏 Acknowledgments

- **yfinance** - Yahoo Finance data integration
- **Finnhub** - Financial news and sentiment API
- **Streamlit** - Interactive web framework
- **Scikit-learn** - Machine learning foundation
- **PyTorch** - Deep learning framework
- **XGBoost** - Gradient boosting excellence

---

## � Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/vatsalpjain/StockPricePrediction/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/vatsalpjain/StockPricePrediction/discussions)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ by [Vatsal Jain](https://github.com/vatsalpjain)

</div>

