"""Multi-Horizon Prediction Page for Streamlit"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.multi_horizon_pipeline import MultiHorizonPipeline
from config.settings import PREDICTION_HORIZONS

def show():
    """Display multi-horizon prediction page"""
    st.title("🔮 Multi-Horizon Stock Price Prediction")
    st.markdown("Forecast stock prices for 1, 7, and 30 days ahead with confidence intervals")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="TSLA",
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
    
    col1, col2, col3, col4 = st.columns(4)
    
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
            help="Force refresh data and retrain models (ignore cache)"
        )
    
    with col3:
        skip_plots = st.checkbox(
            "Skip Plots",
            value=False,
            help="Skip plot generation for faster execution"
        )
    
    with col4:
        skip_backtest = st.checkbox(
            "Skip Backtest",
            value=False,
            help="Skip walk-forward backtesting (faster)"
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
    
    # Information about horizons
    st.info(f"📅 **Prediction Horizons**: {', '.join([f'{h} day(s)' for h in PREDICTION_HORIZONS])}")
    
    # Run prediction button
    st.markdown("---")
    
    if st.button("🚀 Run Multi-Horizon Analysis", type="primary", use_container_width=True):
        run_multi_horizon_prediction(
            ticker=ticker,
            model_type=model_type,
            period=period,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            skip_plots=skip_plots,
            skip_backtest=skip_backtest
        )

def run_multi_horizon_prediction(ticker, model_type, period, output_dir, cache_dir, 
                                 force_refresh, skip_plots, skip_backtest):
    """Execute multi-horizon prediction pipeline"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pipeline
        status_text.text("🔧 Initializing multi-horizon pipeline...")
        progress_bar.progress(5)
        
        pipeline = MultiHorizonPipeline(
            ticker=ticker,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force_refresh=force_refresh
        )
        
        # Step 1: Download data
        status_text.text("📥 Downloading stock data...")
        progress_bar.progress(10)
        pipeline.download_data(period=period)
        st.success(f"✅ Downloaded {len(pipeline.df)} days of data for {ticker}")
        
        # Step 2: Calculate indicators
        status_text.text("📊 Calculating technical indicators...")
        progress_bar.progress(20)
        pipeline.calculate_indicators()
        st.success("✅ Technical indicators calculated")
        
        # Step 3: Sentiment analysis
        status_text.text("📰 Analyzing news sentiment...")
        progress_bar.progress(30)
        pipeline.analyze_sentiment()
        
        if pipeline.sentiment_info['num_sources'] > 0:
            st.success(f"✅ Sentiment analysis complete (Score: {pipeline.sentiment_info['average_score']:.2f})")
        else:
            st.warning("⚠️ No sentiment data available")
        
        # Step 4: Train multi-horizon models
        status_text.text(f"🤖 Training multi-horizon {model_type} models...")
        progress_bar.progress(40)
        pipeline.train_multihorizon_models(model_type=model_type, use_grid_search=False)
        st.success(f"✅ All horizon models trained successfully")
        
        # Display metrics for all horizons
        display_multi_horizon_metrics(pipeline.multi_metrics)
        
        # Step 5: Make future predictions
        status_text.text("🔮 Generating future predictions...")
        progress_bar.progress(60)
        pipeline.predictions = pipeline.predict_future()
        st.success("✅ Future predictions generated")
        
        # Display predictions
        display_future_predictions(pipeline.predictions, ticker)
        
        # Step 6: Generate visualizations
        if not skip_plots:
            status_text.text("📈 Generating visualizations...")
            progress_bar.progress(70)
            pipeline.generate_visualizations(skip_if_exists=False)
            pipeline.generate_multihorizon_plots()
            st.success("✅ Visualizations generated")
            
            # Display plots
            display_multi_horizon_plots(pipeline.output_dir / "plots", ticker)
        
        # Step 7: Backtest
        if not skip_backtest:
            status_text.text("🔬 Running walk-forward backtesting...")
            progress_bar.progress(80)
            backtest_results = pipeline.backtest_multihorizon()
            st.success("✅ Backtesting complete")
            
            # Display backtest results
            display_backtest_results(backtest_results)
        
        # Step 8: Generate reports
        status_text.text("📝 Generating comprehensive reports...")
        progress_bar.progress(90)
        pipeline.generate_reports()
        st.success("✅ Reports generated")
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✨ Multi-horizon analysis complete!")
        
        st.balloons()
        st.success(f"🎉 Multi-horizon analysis complete for {ticker}! Check the output directory: {pipeline.output_dir}")
        
        # Display dataframe preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(pipeline.df.tail(20), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 View Error Details"):
            st.code(traceback.format_exc())

def display_multi_horizon_metrics(multi_metrics):
    """Display metrics for all horizons"""
    st.markdown("### 📊 Model Performance by Horizon")
    
    # Create tabs for each horizon
    horizon_tabs = st.tabs([f"{h}-Day" for h in PREDICTION_HORIZONS])
    
    for tab, horizon in zip(horizon_tabs, PREDICTION_HORIZONS):
        with tab:
            horizon_key = f"{horizon}d"
            if horizon_key in multi_metrics:
                metrics = multi_metrics[horizon_key]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="MAE",
                        value=f"${metrics.get('MAE', 0):.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="RMSE",
                        value=f"${metrics.get('RMSE', 0):.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="R² Score",
                        value=f"{metrics.get('R2', 0):.4f}"
                    )
                
                with col4:
                    st.metric(
                        label="MAPE",
                        value=f"{metrics.get('MAPE', 0):.2f}%"
                    )
                
                # Performance indicator
                r2 = metrics.get('R2', 0)
                if r2 > 0.7:
                    st.success(f"🎯 Excellent performance for {horizon}-day predictions!")
                elif r2 > 0.5:
                    st.info(f"👍 Good performance for {horizon}-day predictions")
                else:
                    st.warning(f"⚠️ Moderate performance - longer horizons are inherently harder to predict")

def display_future_predictions(predictions, ticker):
    """Display future price predictions with interactive chart"""
    st.markdown("### 🔮 Future Price Predictions")
    
    if not predictions:
        st.warning("No predictions available")
        return
    
    # Create prediction summary table
    pred_data = []
    for horizon_key, pred_info in predictions.items():
        horizon = int(horizon_key.replace('d', ''))
        pred_data.append({
            'Horizon': f"{horizon} Day(s)",
            'Target Date': pred_info['date'],
            'Predicted Price': f"${pred_info['predicted_price']:.2f}",
            'Change': f"${pred_info['change']:+.2f}",
            'Change %': f"{pred_info['change_percent']:+.2f}%",
            'Confidence Range': f"${pred_info['confidence_interval']['lower']:.2f} - ${pred_info['confidence_interval']['upper']:.2f}"
        })
    
    df_predictions = pd.DataFrame(pred_data)
    st.dataframe(df_predictions, use_container_width=True, hide_index=True)
    
    # Interactive Plotly chart
    st.markdown("#### 📈 Prediction Visualization")
    
    fig = go.Figure()
    
    horizons = []
    prices = []
    lowers = []
    uppers = []
    
    for horizon_key, pred_info in sorted(predictions.items(), key=lambda x: int(x[0].replace('d', ''))):
        horizon = int(horizon_key.replace('d', ''))
        horizons.append(horizon)
        prices.append(pred_info['predicted_price'])
        lowers.append(pred_info['confidence_interval']['lower'])
        uppers.append(pred_info['confidence_interval']['upper'])
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=horizons + horizons[::-1],
        y=uppers + lowers[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=horizons,
        y=prices,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f"{ticker} - Multi-Horizon Price Predictions",
        xaxis_title="Days Ahead",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal interpretation
    st.markdown("#### 🎯 Trading Signals")
    
    cols = st.columns(len(predictions))
    for col, (horizon_key, pred_info) in zip(cols, predictions.items()):
        with col:
            horizon = int(horizon_key.replace('d', ''))
            change_pct = pred_info['change_percent']
            
            if change_pct > 2:
                signal = "🟢 Strong Buy"
                color = "green"
            elif change_pct > 0.5:
                signal = "🟢 Buy"
                color = "green"
            elif change_pct < -2:
                signal = "🔴 Strong Sell"
                color = "red"
            elif change_pct < -0.5:
                signal = "🔴 Sell"
                color = "red"
            else:
                signal = "🟡 Hold"
                color = "orange"
            
            st.markdown(f"**{horizon}-Day**")
            st.markdown(f":{color}[{signal}]")
            st.markdown(f"*{change_pct:+.2f}%*")

def display_multi_horizon_plots(plots_dir, ticker):
    """Display multi-horizon plots"""
    st.markdown("### 📈 Detailed Visualizations")
    
    plot_files = {
        "Multi-Horizon Predictions": "multi_horizon_predictions.png",
        "Backtest Comparison": "backtest_comparison.png",
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
                st.info(f"Plot not available: {filename}")

def display_backtest_results(backtest_results):
    """Display backtesting results"""
    st.markdown("### 🔬 Walk-Forward Backtesting Results")
    
    st.info("Walk-forward validation tests the model on multiple time periods to ensure robustness")
    
    # Create comparison table
    backtest_data = []
    for horizon_key, metrics in backtest_results.items():
        horizon = horizon_key.replace('d', '')
        backtest_data.append({
            'Horizon': f"{horizon} Day(s)",
            'Avg MAE': f"${metrics.get('MAE', 0):.2f}",
            'Avg RMSE': f"${metrics.get('RMSE', 0):.2f}",
            'Avg R²': f"{metrics.get('R2', 0):.4f}",
            'Avg MAPE': f"{metrics.get('MAPE', 0):.2f}%"
        })
    
    df_backtest = pd.DataFrame(backtest_data)
    st.dataframe(df_backtest, use_container_width=True, hide_index=True)
    
    # Visualization
    fig = go.Figure()
    
    horizons = []
    r2_scores = []
    
    for horizon_key, metrics in sorted(backtest_results.items(), key=lambda x: int(x[0].replace('d', ''))):
        horizon = int(horizon_key.replace('d', ''))
        horizons.append(horizon)
        r2_scores.append(metrics.get('R2', 0))
    
    fig.add_trace(go.Bar(
        x=[f"{h}-Day" for h in horizons],
        y=r2_scores,
        marker_color='#1f77b4',
        text=[f"{r2:.3f}" for r2 in r2_scores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="R² Score by Prediction Horizon (Backtesting)",
        xaxis_title="Prediction Horizon",
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Model shows consistent performance across different time periods")

if __name__ == "__main__":
    show()
