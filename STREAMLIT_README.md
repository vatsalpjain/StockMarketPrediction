# 📈 Stock Price Prediction - Streamlit Web Application

## 🎯 Overview

This is a comprehensive web-based interface for the Stock Price Prediction system, built with Streamlit. The application provides an intuitive, interactive way to run stock predictions, visualize results, and compare different machine learning models.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📋 Features

### 🏠 Home Page
- Overview of the system
- Quick start guide
- Supported models information
- Technical indicators reference

### 📈 Single-Step Prediction
- Predict next-day stock prices
- Support for multiple ML models (XGBoost, Ridge, Random Forest, LSTM)
- Real-time progress tracking
- Interactive visualizations
- Comprehensive metrics display
- Sentiment analysis integration

### 🔮 Multi-Horizon Prediction
- Forecast prices for 1, 7, and 30 days ahead
- Confidence intervals for predictions
- Walk-forward backtesting
- Multi-horizon performance comparison
- Interactive Plotly charts

### 📊 View Results
- Browse previously generated analysis results
- View all plots and visualizations
- Read detailed analysis reports
- Download reports in JSON/TXT format
- File browser for output directory

### 📉 Model Comparison
- **Model Comparison**: Compare different models on the same stock
- **Batch Analysis**: Analyze multiple stocks with the same model
- **Performance Analytics**: Dashboard for historical performance analysis
- Interactive comparison charts
- Radar charts for overall performance
- Performance distribution analysis

## 🎨 User Interface Features

### Navigation
- Sidebar navigation for easy page switching
- Quick info panel
- Responsive design

### Interactive Elements
- Progress bars for long-running operations
- Real-time status updates
- Expandable sections for advanced options
- Tabbed interfaces for organized content
- Download buttons for reports

### Visualizations
- Static plots (Matplotlib/Seaborn)
- Interactive charts (Plotly)
- Multi-panel comparisons
- Radar charts
- Distribution histograms

## 🔧 Configuration Options

### Single-Step Prediction Options
- **Ticker**: Stock symbol (e.g., AAPL, TSLA)
- **Data Period**: Historical data range (1y, 2y, 3y, 5y, 10y, max)
- **Model Type**: xgboost, ridge, random_forest, lstm
- **Force Refresh**: Ignore cache and retrain
- **Skip Plots**: Faster execution without visualizations
- **Output Directory**: Custom output location
- **Cache Directory**: Custom cache location
- **Use Quantiles** (LSTM only): Enable prediction bands

### Multi-Horizon Prediction Options
- All single-step options plus:
- **Skip Backtest**: Skip walk-forward validation
- **Prediction Horizons**: 1, 7, 30 days (configurable in settings)

## 📊 Model Information

### Supported Models

| Model | Type | Best For | Speed |
|-------|------|----------|-------|
| XGBoost | Gradient Boosting | General purpose, high accuracy | Fast |
| Ridge Regression | Linear | Fast training, interpretable | Very Fast |
| Random Forest | Ensemble | Robust, handles noise well | Medium |
| LSTM | Deep Learning | Sequential patterns, long-term | Slow |

### Performance Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes larger errors more
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

## 🎯 Usage Examples

### Example 1: Quick Single-Step Prediction

1. Navigate to "📈 Single-Step Prediction"
2. Enter ticker: `AAPL`
3. Select model: `xgboost`
4. Click "🚀 Run Single-Step Prediction"
5. View results and visualizations

### Example 2: Multi-Horizon Analysis

1. Navigate to "🔮 Multi-Horizon Prediction"
2. Enter ticker: `TSLA`
3. Select model: `xgboost`
4. Uncheck "Skip Backtest" for full validation
5. Click "🚀 Run Multi-Horizon Analysis"
6. View predictions for 1, 7, and 30 days

### Example 3: Compare Models

1. Navigate to "📉 Model Comparison"
2. Go to "🔬 Model Comparison" tab
3. Enter ticker: `GOOGL`
4. Select models to compare (e.g., XGBoost, Ridge, Random Forest)
5. Click "🚀 Run Model Comparison"
6. View side-by-side performance metrics

### Example 4: Batch Analysis

1. Navigate to "📉 Model Comparison"
2. Go to "📊 Batch Analysis" tab
3. Enter multiple tickers (one per line):
   ```
   AAPL
   TSLA
   GOOGL
   META
   ```
4. Select model: `xgboost`
5. Click "🚀 Run Batch Analysis"
6. Compare performance across stocks

## 📁 Output Structure

```
output/
├── AAPL/
│   ├── plots/
│   │   ├── price_predictions.png
│   │   ├── rsi_indicator.png
│   │   ├── macd_indicator.png
│   │   ├── multi_horizon_predictions.png
│   │   └── backtest_comparison.png
│   ├── analysis_summary.json
│   └── analysis_summary.txt
├── TSLA/
│   └── ...
└── ...
```

## 🔐 API Keys

The application uses the Finnhub API for sentiment analysis. To use this feature:

1. Get a free API key from [Finnhub](https://finnhub.io/)
2. Create/edit `config/api_keys.py`:
   ```python
   FINNHUB_API_KEY = "your_api_key_here"
   ```

If no API key is provided, the system will still work but sentiment analysis will be skipped.

## ⚙️ Advanced Configuration

### Settings File (`config/settings.py`)

You can customize various parameters:

```python
# Data settings
DEFAULT_PERIOD = "2y"
TRAIN_TEST_SPLIT = 0.8

# Technical indicators
SMA_PERIOD = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26

# Multi-horizon settings
PREDICTION_HORIZONS = [1, 7, 30]

# LSTM settings
LSTM_LOOKBACK = 60
LSTM_HIDDEN = 64
LSTM_EPOCHS = 30
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No module named 'streamlit'"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Port 8501 already in use
- **Solution**: Run with custom port: `streamlit run app.py --server.port 8502`

**Issue**: Plots not displaying
- **Solution**: Check that matplotlib backend is set correctly

**Issue**: LSTM model very slow
- **Solution**: 
  - Use smaller dataset (1y instead of 2y)
  - Reduce LSTM_EPOCHS in settings
  - Use GPU if available

**Issue**: Cache errors
- **Solution**: Delete cache directory and run with `--force-refresh`

## 🎨 Customization

### Changing Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Adding Custom Pages

1. Create new page in `streamlit_pages/`
2. Add import in `app.py`
3. Add navigation option in sidebar

## 📊 Performance Tips

1. **Use Caching**: Don't use `--force-refresh` unless necessary
2. **Skip Plots**: Use "Skip Plots" option for faster execution
3. **Smaller Datasets**: Use 1y or 2y periods for faster training
4. **Avoid LSTM for Quick Tests**: Use XGBoost or Ridge for faster results
5. **Batch Processing**: Run batch analysis during off-peak hours

## 🔄 Updates and Maintenance

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Clearing Cache

```bash
# Delete cache directory
rm -rf cache/

# Or use force refresh in the UI
```

### Backing Up Results

```bash
# Backup output directory
cp -r output/ output_backup_$(date +%Y%m%d)/
```

## 📝 Notes

- The application preserves all existing CLI functionality
- Original `main.py` still works independently
- Streamlit UI is a wrapper around existing pipelines
- No breaking changes to core codebase
- All existing features are accessible through the UI

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error details in expandable sections
3. Check console output for detailed logs
4. Verify API keys are configured correctly

## 🎉 Enjoy!

The Streamlit interface makes stock prediction accessible and interactive. Explore different models, compare results, and make informed decisions with confidence!
