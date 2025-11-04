"""
Stock Price Prediction - Streamlit Application
Main entry point for the interactive web interface
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "📈 Single-Step Prediction",
            "🔮 Multi-Horizon Prediction",
            "📊 View Results",
            "📉 Model Comparison",
            "ℹ️ About"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Info")
    st.sidebar.info(
        "This application provides advanced stock price prediction "
        "using machine learning models including XGBoost, Ridge Regression, "
        "Random Forest, and LSTM networks."
    )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Single-Step Prediction":
        show_single_step_page()
    elif page == "🔮 Multi-Horizon Prediction":
        show_multi_horizon_page()
    elif page == "📊 View Results":
        show_results_page()
    elif page == "📉 Model Comparison":
        show_comparison_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page with overview"""
    st.markdown('<div class="main-header">📈 Stock Price Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced ML-Powered Stock Market Analysis</div>', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Single-Step Prediction")
        st.write("Predict next-day stock prices with high accuracy using multiple ML models.")
        st.markdown("**Features:**")
        st.markdown("- XGBoost, Ridge, Random Forest, LSTM")
        st.markdown("- Technical indicators (RSI, MACD, SMA)")
        st.markdown("- Sentiment analysis integration")
        
    with col2:
        st.markdown("### 🔮 Multi-Horizon Prediction")
        st.write("Forecast stock prices for 1, 7, and 30 days ahead.")
        st.markdown("**Features:**")
        st.markdown("- Multiple time horizons")
        st.markdown("- Confidence intervals")
        st.markdown("- Walk-forward backtesting")
        
    with col3:
        st.markdown("### 📊 Advanced Analytics")
        st.write("Comprehensive analysis and visualization tools.")
        st.markdown("**Features:**")
        st.markdown("- Interactive charts")
        st.markdown("- Model performance metrics")
        st.markdown("- Historical comparisons")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1️⃣ Single-Step Prediction")
        st.code("""
1. Navigate to 'Single-Step Prediction'
2. Enter stock ticker (e.g., AAPL, TSLA)
3. Select model type
4. Configure options
5. Click 'Run Prediction'
        """)
        
    with col2:
        st.markdown("### 2️⃣ Multi-Horizon Prediction")
        st.code("""
1. Navigate to 'Multi-Horizon Prediction'
2. Enter stock ticker
3. Select model type
4. Enable/disable backtesting
5. Click 'Run Multi-Horizon Analysis'
        """)
    
    st.markdown("---")
    
    # Supported models
    st.markdown("## 🤖 Supported Models")
    
    models_data = {
        "Model": ["XGBoost", "Ridge Regression", "Random Forest", "LSTM"],
        "Type": ["Gradient Boosting", "Linear", "Ensemble", "Deep Learning"],
        "Best For": [
            "General purpose, high accuracy",
            "Fast training, interpretable",
            "Robust, handles noise well",
            "Sequential patterns, long-term"
        ],
        "Speed": ["Fast", "Very Fast", "Medium", "Slow"]
    }
    
    import pandas as pd
    st.table(pd.DataFrame(models_data))
    
    st.markdown("---")
    
    # Technical indicators
    st.markdown("## 📊 Technical Indicators Used")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trend Indicators**")
        st.markdown("- SMA (Simple Moving Average)")
        st.markdown("- EMA (Exponential Moving Average)")
        st.markdown("- HMA (Hull Moving Average)")
        
    with col2:
        st.markdown("**Momentum Indicators**")
        st.markdown("- RSI (Relative Strength Index)")
        st.markdown("- MACD (Moving Average Convergence Divergence)")
        
    with col3:
        st.markdown("**Volatility Indicators**")
        st.markdown("- Bollinger Bands")
        st.markdown("- Rolling Volatility")
    
    st.success("👈 Use the sidebar to navigate to different sections!")

def show_single_step_page():
    """Single-step prediction page - imported from separate module"""
    from streamlit_pages import single_step_page
    single_step_page.show()

def show_multi_horizon_page():
    """Multi-horizon prediction page - imported from separate module"""
    from streamlit_pages import multi_horizon_page
    multi_horizon_page.show()

def show_results_page():
    """Results viewing page - imported from separate module"""
    from streamlit_pages import results_page
    results_page.show()

def show_comparison_page():
    """Model comparison page - imported from separate module"""
    from streamlit_pages import comparison_page
    comparison_page.show()

def show_about_page():
    """Display about page"""
    st.markdown('<div class="main-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📖 Overview
    
    This Stock Price Prediction System is a comprehensive machine learning application 
    designed to forecast stock prices using various advanced techniques and models.
    
    ## 🎯 Key Features
    
    - **Multiple ML Models**: XGBoost, Ridge Regression, Random Forest, and LSTM
    - **Technical Analysis**: Integration of 10+ technical indicators
    - **Sentiment Analysis**: News sentiment scoring using Finnhub API
    - **Multi-Horizon Forecasting**: Predict 1, 7, and 30 days ahead
    - **Backtesting**: Walk-forward validation for model evaluation
    - **Interactive Visualizations**: Real-time charts and analytics
    - **Caching System**: Smart caching for faster predictions
    
    ## 🛠️ Technology Stack
    
    - **Frontend**: Streamlit, Plotly
    - **ML/DL**: Scikit-learn, XGBoost, PyTorch
    - **Data**: yfinance, Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    
    ## 📊 Data Sources
    
    - **Stock Data**: Yahoo Finance (via yfinance)
    - **News Sentiment**: Finnhub API
    
    ## 🔬 Methodology
    
    1. **Data Collection**: Historical stock data and news sentiment
    2. **Feature Engineering**: Technical indicators and derived features
    3. **Model Training**: Multiple models with cross-validation
    4. **Prediction**: Single-step and multi-horizon forecasts
    5. **Evaluation**: Comprehensive metrics (MAE, RMSE, R², MAPE)
    6. **Visualization**: Interactive charts and reports
    
    ## ⚠️ Disclaimer
    
    This application is for educational and research purposes only. 
    Stock market predictions are inherently uncertain and should not be 
    used as the sole basis for investment decisions. Always consult with 
    financial professionals before making investment choices.
    

    
    ## 👨‍💻 Developer
    
    Built with ❤️ using Python and Streamlit
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start with the Single-Step Prediction page to get familiar with the system!")

if __name__ == "__main__":
    main()
