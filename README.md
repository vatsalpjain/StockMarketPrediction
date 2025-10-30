# Stock Price Prediction

Welcome to **Stock Price Prediction**! This repository contains code and resources to predict stock prices using machine learning techniques. Whether you're a data enthusiast, an investor, or just curious about financial modeling, this project provides a clear and approachable way to forecast stock movements.

## ✨ NEW: Interactive Web Interface!

🎉 **Now with Streamlit UI!** - Run the entire application through an intuitive web interface:

```bash
streamlit run app.py
```

Or on Windows, simply double-click `run_streamlit.bat`

See [STREAMLIT_README.md](STREAMLIT_README.md) for detailed UI documentation.

## 🚀 Features

- **🖥️ Interactive Web UI** - Beautiful Streamlit interface for all features
- Clean and modular code for stock price prediction
- Data preprocessing and visualization
- Multiple machine learning models (XGBoost, Ridge, Random Forest, LSTM)
- Single-step and multi-horizon predictions (1, 7, 30 days)
- Sentiment analysis integration
- Walk-forward backtesting
- Model comparison tools
- Batch analysis for multiple stocks
- Easy to customize for different stocks

## 🛠️ Tech Stack

- **Python**: Core language for data analysis and modeling
- **Streamlit**: Interactive web interface
- **Pandas, NumPy**: Data manipulation and numerical operations
- **Matplotlib, Seaborn, Plotly**: Data visualization
- **Scikit-learn, XGBoost, PyTorch**: Machine learning libraries
- **yfinance**: Stock data fetching
- **Finnhub**: News sentiment analysis

## 📂 Project Structure

```
├── app.py                    # Streamlit web application entry point
├── streamlit_pages/          # Streamlit UI pages
│   ├── single_step_page.py   # Single-step prediction UI
│   ├── multi_horizon_page.py # Multi-horizon prediction UI
│   ├── results_page.py       # Results viewer
│   └── comparison_page.py    # Model comparison tools
├── pipelines/                # ML pipelines
│   ├── single_step_pipeline.py
│   └── multi_horizon_pipeline.py
├── models/                   # ML models
├── data/                     # Data fetching and processing
├── features/                 # Feature engineering
├── evaluation/               # Model evaluation
├── visualization/            # Plotting utilities
├── config/                   # Configuration
├── main.py                   # CLI entry point (still works!)
├── requirements.txt          # Python dependencies
├── STREAMLIT_README.md       # Streamlit UI documentation
└── README.md                 # This file
```

## 🚦 Quick Start

### Option 1: Web Interface (Recommended)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/vatsalpjain/StockPricePrediction.git
   cd StockPricePrediction/StockPricePrediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   
   Or on Windows, double-click `run_streamlit.bat`

4. **Open your browser** to `http://localhost:8501`

### Option 2: Command Line Interface (Original)

The original CLI still works perfectly:

```bash
# Basic usage
python main.py META

# Different model
python main.py AAPL --model ridge

# Force refresh
python main.py TSLA --force-refresh

# Skip plots (faster)
python main.py GOOGL --skip-plots

# Basic multi-horizon
python main.py META --multi-horizon

# Skip backtesting
python main.py NVDA --multi-horizon --skip-backtest

# Custom directories
python main.py AAPL --multi-horizon --output-dir my_analysis --cache-dir my_cache
```

## 🎨 Streamlit Features

The web interface provides:

- **🏠 Home**: Overview and quick start guide
- **📈 Single-Step Prediction**: Next-day price predictions with real-time progress
- **🔮 Multi-Horizon Prediction**: 1, 7, 30-day forecasts with confidence intervals
- **📊 View Results**: Browse and visualize saved analysis results
- **📉 Model Comparison**: Compare models, batch analysis, performance analytics
- **ℹ️ About**: Detailed documentation and methodology

### Key UI Features:
- Real-time progress tracking
- Interactive Plotly charts
- Model performance metrics
- Sentiment analysis display
- Downloadable reports
- Batch processing
- Model comparison tools

## 📊 Example Results

### Old Version
<img width="719" height="590" alt="image" src="https://github.com/user-attachments/assets/8138bda6-e132-4dc0-9ec7-d0475400392d" />

### New Version (with Multi-Horizon)
<img width="913" height="990" alt="image" src="https://github.com/user-attachments/assets/7c0d9c72-884f-4b0f-ae5f-a27f1cb03621" />
<img width="907" height="946" alt="image" src="https://github.com/user-attachments/assets/2d0858c2-c175-40a5-8c7c-909941145635" />
<img width="909" height="340" alt="image" src="https://github.com/user-attachments/assets/ee057255-4711-4817-8c5e-22bd8c4bcfdc" />

## 🔑 API Configuration (Optional)

For sentiment analysis, get a free API key from [Finnhub](https://finnhub.io/) and add to `config/api_keys.py`:

```python
FINNHUB_API_KEY = "your_api_key_here"
```

The system works without this, but sentiment analysis will be skipped.

## 📚 Documentation

- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Complete Streamlit UI guide
- **[main.py](main.py)** - CLI usage examples
- **[config/settings.py](config/settings.py)** - Configuration options

## 🎯 What's New

✨ **Streamlit Web Interface**
- Beautiful, interactive UI for all features
- Real-time progress tracking
- Interactive visualizations with Plotly
- Model comparison dashboard
- Batch analysis tools
- Results browser

🚀 **No Breaking Changes**
- Original CLI (`main.py`) still works
- All existing features preserved
- Backward compatible
- Same output format

**Happy predicting!** 📈
