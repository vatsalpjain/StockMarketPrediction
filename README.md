# Stock Price Prediction

Welcome to **Stock Price Prediction**! This repository contains code and resources to predict stock prices using machine learning techniques. Whether you're a data enthusiast, an investor, or just curious about financial modeling, this project provides a clear and approachable way to forecast stock movements.

## 🚀 Features

- Clean and modular code for stock price prediction
- Data preprocessing and visualization
- Multiple machine learning models (e.g., XgBoost,Ridge)
- Easy to customize for different stocks
- Example datasets and notebook walkthroughs

## 🛠️ Tech Stack

- **Python**: Core language for data analysis and modeling
- **Pandas, NumPy**: Data manipulation and numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn, TensorFlow/Keras**: Machine learning libraries

## 📂 Project Structure

```
├── data/              # Sample datasets & data loaders
├── notebooks/         # Jupyter walkthroughs
├── src/               # Source code for models & utilities
├── requirements.txt   # Python dependencies
└── README.md          # Project info
```

## 🚦 Quick Start

1. **Clone the repo:**
   ```bash
   git clone https://github.com/vatsalpjain/StockPricePrediction.git
   cd StockPricePrediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run a notebook or script:**
   - Explore `notebooks/` for step-by-step guides
   - Or run Python scripts from `src/`

## 📊 Example Results
old version
<img width="719" height="590" alt="image" src="https://github.com/user-attachments/assets/8138bda6-e132-4dc0-9ec7-d0475400392d" />

new version
<img width="913" height="990" alt="image" src="https://github.com/user-attachments/assets/7c0d9c72-884f-4b0f-ae5f-a27f1cb03621" />
<img width="907" height="946" alt="image" src="https://github.com/user-attachments/assets/2d0858c2-c175-40a5-8c7c-909941145635" />
<img width="909" height="340" alt="image" src="https://github.com/user-attachments/assets/ee057255-4711-4817-8c5e-22bd8c4bcfdc" />




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

**Happy predicting!**
