"""Single-Step Prediction Page for Streamlit"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.single_step_pipeline import SingleStepPipeline
from config.settings import DEFAULT_PERIOD

def show():
    """Display single-step prediction page"""
    st.title("📈 Single-Step Stock Price Prediction")
    st.markdown("Predict next-day stock prices using advanced machine learning models")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter a valid stock ticker (e.g., AAPL, TSLA, GOOGL, META, NVDA)"
        ).upper()
    
    with col2:
        period = st.selectbox(
            "Data Period",
            options=["1y", "2y", "3y", "5y", "10y", "max"],
            index=1,  # Default to 2y
            help="Historical data period to fetch"
        )
    
    # Model configuration
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["xgboost", "ridge", "random_forest", "lstm"],
            help="Select the machine learning model to use"
        )
    
    with col2:
        force_refresh = st.checkbox(
            "Force Refresh",
            value=False,
            help="Force refresh data and retrain model (ignore cache)"
        )
    
    with col3:
        skip_plots = st.checkbox(
            "Skip Plots",
            value=False,
            help="Skip plot generation for faster execution"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input(
                "Output Directory",
                value="output",
                help="Directory to save results"
            )
            
        with col2:
            cache_dir = st.text_input(
                "Cache Directory",
                value="cache",
                help="Directory for model and data cache"
            )
        
        if model_type == "lstm":
            st.info("🧠 LSTM model selected - Training may take longer but can capture sequential patterns better")
            use_quantiles = st.checkbox(
                "Use Quantile Heads",
                value=False,
                help="Enable quantile prediction bands (experimental)"
            )
            if use_quantiles:
                from config import settings
                settings.USE_QUANTILES = True
    
    # Run prediction button
    st.markdown("---")
    
    if st.button("🚀 Run Single-Step Prediction", type="primary", use_container_width=True):
        run_single_step_prediction(
            ticker=ticker,
            model_type=model_type,
            period=period,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            skip_plots=skip_plots
        )

def run_single_step_prediction(ticker, model_type, period, output_dir, cache_dir, force_refresh, skip_plots):
    """Execute single-step prediction pipeline"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pipeline
        status_text.text("🔧 Initializing pipeline...")
        progress_bar.progress(10)
        
        pipeline = SingleStepPipeline(
            ticker=ticker,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_refresh=force_refresh
        )
        
        # Step 1: Download data
        status_text.text("📥 Downloading stock data...")
        progress_bar.progress(20)
        pipeline.download_data(period=period)
        st.success(f"✅ Downloaded {len(pipeline.df)} days of data for {ticker}")
        
        # Step 2: Calculate indicators
        status_text.text("📊 Calculating technical indicators...")
        progress_bar.progress(35)
        pipeline.calculate_indicators()
        st.success("✅ Technical indicators calculated")
        
        # Step 3: Sentiment analysis
        status_text.text("📰 Analyzing news sentiment...")
        progress_bar.progress(50)
        pipeline.analyze_sentiment()
        
        if pipeline.sentiment_info['num_sources'] > 0:
            st.success(f"✅ Sentiment analysis complete (Score: {pipeline.sentiment_info['average_score']:.2f})")
        else:
            st.warning("⚠️ No sentiment data available")
        
        # Step 4: Train model
        status_text.text(f"🤖 Training {model_type} model...")
        progress_bar.progress(65)
        pipeline.train_model(model_type=model_type, use_grid_search=True)
        st.success(f"✅ Model trained successfully")
        
        # Display metrics
        display_metrics(pipeline.metrics)
        
        # Step 5: Generate visualizations
        if not skip_plots:
            status_text.text("📈 Generating visualizations...")
            progress_bar.progress(80)
            pipeline.generate_visualizations(skip_if_exists=False)
            st.success("✅ Visualizations generated")
            
            # Display plots
            display_plots(pipeline.output_dir / "plots", ticker)
        
        # Step 6: Generate reports
        status_text.text("📝 Generating reports...")
        progress_bar.progress(90)
        pipeline.generate_reports()
        st.success("✅ Reports generated")
        
        # Display prediction summary
        display_prediction_summary(pipeline)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✨ Analysis complete!")
        
        st.balloons()
        st.success(f"🎉 Analysis complete for {ticker}! Check the output directory: {pipeline.output_dir}")
        
        # Display dataframe preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(pipeline.df.tail(20), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 View Error Details"):
            st.code(traceback.format_exc())

def display_metrics(metrics):
    """Display model performance metrics"""
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value=f"${metrics.get('MAE', 0):.2f}",
            help="Average absolute difference between predicted and actual prices"
        )
    
    with col2:
        st.metric(
            label="RMSE (Root Mean Squared Error)",
            value=f"${metrics.get('RMSE', 0):.2f}",
            help="Square root of average squared differences"
        )
    
    with col3:
        st.metric(
            label="R² Score",
            value=f"{metrics.get('R2', 0):.4f}",
            help="Proportion of variance explained by the model (closer to 1 is better)"
        )
    
    with col4:
        mape = metrics.get('MAPE', 0)
        st.metric(
            label="MAPE",
            value=f"{mape:.2f}%",
            help="Mean Absolute Percentage Error"
        )
    
    # Interpretation
    r2 = metrics.get('R2', 0)
    if r2 > 0.8:
        st.success("🎯 Excellent model performance!")
    elif r2 > 0.6:
        st.info("👍 Good model performance")
    elif r2 > 0.4:
        st.warning("⚠️ Moderate model performance")
    else:
        st.warning("⚠️ Model may need improvement")

def display_plots(plots_dir, ticker):
    """Display generated plots"""
    st.markdown("### 📈 Visualizations")
    
    plot_files = {
        "Price Predictions": "price_predictions.png",
        "RSI Indicator": "rsi_indicator.png",
        "MACD Indicator": "macd_indicator.png"
    }
    
    tabs = st.tabs(list(plot_files.keys()))
    
    for tab, (title, filename) in zip(tabs, plot_files.items()):
        with tab:
            plot_path = plots_dir / filename
            if plot_path.exists():
                st.image(str(plot_path), use_container_width=True)
            else:
                st.warning(f"Plot not found: {filename}")

def display_prediction_summary(pipeline):
    """Display prediction summary"""
    st.markdown("### 🔮 Next-Day Prediction Summary")
    
    if 'Predicted_Close' in pipeline.df.columns:
        # Get latest prediction
        latest_actual = pipeline.df['Close'].iloc[-1]
        
        # Check if we have a prediction for the next day
        pred_series = pipeline.df['Predicted_Close'].dropna()
        if len(pred_series) > 0:
            latest_pred = pred_series.iloc[-1]
            
            change = latest_pred - latest_actual
            change_pct = (change / latest_actual) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${latest_actual:.2f}"
                )
            
            with col2:
                st.metric(
                    label="Predicted Next-Day Price",
                    value=f"${latest_pred:.2f}",
                    delta=f"{change:+.2f}"
                )
            
            with col3:
                st.metric(
                    label="Expected Change",
                    value=f"{change_pct:+.2f}%",
                    delta=f"${change:+.2f}"
                )
            
            with col4:
                direction = "📈 Bullish" if change > 0 else "📉 Bearish"
                st.metric(
                    label="Signal",
                    value=direction
                )
            
            # Confidence indicator
            if abs(change_pct) > 2:
                st.info("💡 **Strong signal detected** - Consider this prediction carefully")
            elif abs(change_pct) < 0.5:
                st.info("💡 **Weak signal** - Market may be stable")
        else:
            st.info("No prediction available for next day")
    else:
        st.warning("Prediction data not available")

if __name__ == "__main__":
    show()
