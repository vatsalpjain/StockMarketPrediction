"""Model Comparison and Analytics Page for Streamlit"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.single_step_pipeline import SingleStepPipeline
from config.settings import DEFAULT_PERIOD

def show():
    """Display model comparison page"""
    st.title("📉 Model Comparison & Analytics")
    st.markdown("Compare different ML models and analyze their performance")
    st.markdown("---")
    
    # Create tabs for different comparison modes
    tabs = st.tabs(["🔬 Model Comparison", "📊 Batch Analysis", "📈 Performance Analytics"])
    
    with tabs[0]:
        show_model_comparison()
    
    with tabs[1]:
        show_batch_analysis()
    
    with tabs[2]:
        show_performance_analytics()

def show_model_comparison():
    """Compare multiple models on the same stock"""
    st.markdown("### 🔬 Compare Models on Single Stock")
    st.markdown("Train and compare different ML models on the same stock to find the best performer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock ticker to compare models"
        ).upper()
    
    with col2:
        period = st.selectbox(
            "Data Period",
            options=["1y", "2y", "3y", "5y"],
            index=1
        )
    
    # Model selection
    st.markdown("#### Select Models to Compare")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        use_xgboost = st.checkbox("XGBoost", value=True)
    with col2:
        use_ridge = st.checkbox("Ridge Regression", value=True)
    with col3:
        use_rf = st.checkbox("Random Forest", value=True)
    with col4:
        use_lstm = st.checkbox("LSTM", value=False, help="Slower but may capture patterns better")
    
    models_to_compare = []
    if use_xgboost:
        models_to_compare.append("xgboost")
    if use_ridge:
        models_to_compare.append("ridge")
    if use_rf:
        models_to_compare.append("random_forest")
    if use_lstm:
        models_to_compare.append("lstm")
    
    if not models_to_compare:
        st.warning("⚠️ Please select at least one model to compare")
        return
    
    # Options
    with st.expander("⚙️ Options"):
        force_refresh = st.checkbox("Force Refresh", value=False)
        output_dir = st.text_input("Output Directory", value="output_comparison")
        cache_dir = st.text_input("Cache Directory", value="cache")
    
    # Run comparison
    if st.button("🚀 Run Model Comparison", type="primary", use_container_width=True):
        run_model_comparison(ticker, models_to_compare, period, output_dir, cache_dir, force_refresh)

def run_model_comparison(ticker, models, period, output_dir, cache_dir, force_refresh):
    """Execute model comparison"""
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_models = len(models)
    
    for idx, model_type in enumerate(models):
        try:
            status_text.text(f"Training {model_type} model... ({idx+1}/{total_models})")
            progress_bar.progress((idx) / total_models)
            
            # Create pipeline
            pipeline = SingleStepPipeline(
                ticker=ticker,
                output_dir=f"{output_dir}/{ticker}_{model_type}",
                cache_dir=cache_dir,
                force_refresh=force_refresh
            )
            
            # Run pipeline
            pipeline.download_data(period=period)
            pipeline.calculate_indicators()
            pipeline.analyze_sentiment()
            pipeline.train_model(model_type=model_type, use_grid_search=False)
            
            # Store results
            results[model_type] = {
                'metrics': pipeline.metrics,
                'df': pipeline.df,
                'model': pipeline.model
            }
            
            st.success(f"✅ {model_type} completed")
            
        except Exception as e:
            st.error(f"❌ Error with {model_type}: {str(e)}")
            with st.expander(f"Error details for {model_type}"):
                st.code(traceback.format_exc())
    
    progress_bar.progress(1.0)
    status_text.text("✨ Comparison complete!")
    
    if results:
        display_comparison_results(results, ticker)

def display_comparison_results(results, ticker):
    """Display comparison results"""
    st.markdown("---")
    st.markdown("### 📊 Comparison Results")
    
    # Create metrics comparison table
    metrics_data = []
    for model_name, data in results.items():
        metrics = data['metrics']
        metrics_data.append({
            'Model': model_name.upper(),
            'MAE': metrics.get('MAE', 0),
            'RMSE': metrics.get('RMSE', 0),
            'R²': metrics.get('R2', 0),
            'MAPE (%)': metrics.get('MAPE', 0)
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Highlight best model for each metric
    st.dataframe(
        df_metrics.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen')
                       .highlight_max(subset=['R²'], color='lightgreen'),
        use_container_width=True,
        hide_index=True
    )
    
    # Find best model
    best_model_r2 = df_metrics.loc[df_metrics['R²'].idxmax(), 'Model']
    best_model_mae = df_metrics.loc[df_metrics['MAE'].idxmin(), 'Model']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 Best R² Score: **{best_model_r2}**")
    with col2:
        st.success(f"🏆 Lowest MAE: **{best_model_mae}**")
    
    # Visualizations
    st.markdown("### 📈 Performance Comparison Charts")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('MAE Comparison', 'RMSE Comparison', 'R² Score Comparison', 'MAPE Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    models = df_metrics['Model'].tolist()
    
    # MAE
    fig.add_trace(
        go.Bar(x=models, y=df_metrics['MAE'], name='MAE', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=models, y=df_metrics['RMSE'], name='RMSE', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # R²
    fig.add_trace(
        go.Bar(x=models, y=df_metrics['R²'], name='R²', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # MAPE
    fig.add_trace(
        go.Bar(x=models, y=df_metrics['MAPE (%)'], name='MAPE', marker_color='#d62728'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text=f"Model Performance Comparison - {ticker}",
        showlegend=False,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for overall comparison
    st.markdown("### 🎯 Overall Performance Radar")
    
    create_radar_chart(df_metrics)

def create_radar_chart(df_metrics):
    """Create radar chart for model comparison"""
    
    # Normalize metrics (0-1 scale, higher is better)
    df_norm = df_metrics.copy()
    
    # For MAE, RMSE, MAPE: lower is better, so invert
    for col in ['MAE', 'RMSE', 'MAPE (%)']:
        max_val = df_norm[col].max()
        if max_val > 0:
            df_norm[col] = 1 - (df_norm[col] / max_val)
    
    # For R²: higher is better, normalize to 0-1
    df_norm['R²'] = df_norm['R²'] / df_norm['R²'].max() if df_norm['R²'].max() > 0 else 0
    
    fig = go.Figure()
    
    categories = ['MAE', 'RMSE', 'R²', 'MAPE']
    
    for idx, row in df_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['MAE'], row['RMSE'], row['R²'], row['MAPE (%)']],
            theta=categories,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Normalized Performance Comparison (Higher is Better)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_batch_analysis():
    """Batch analysis for multiple stocks"""
    st.markdown("### 📊 Batch Stock Analysis")
    st.markdown("Analyze multiple stocks with the same model")
    
    # Input tickers
    tickers_input = st.text_area(
        "Stock Tickers (one per line or comma-separated)",
        value="AAPL\nTSLA\nGOOGL\nMETA",
        help="Enter stock tickers to analyze"
    )
    
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
    
    st.info(f"📋 {len(tickers)} stocks selected: {', '.join(tickers)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["xgboost", "ridge", "random_forest"],
            help="Select model for batch analysis"
        )
    
    with col2:
        period = st.selectbox(
            "Data Period",
            options=["1y", "2y", "3y"],
            index=1
        )
    
    if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
        run_batch_analysis(tickers, model_type, period)

def run_batch_analysis(tickers, model_type, period):
    """Execute batch analysis"""
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        try:
            status_text.text(f"Analyzing {ticker}... ({idx+1}/{total_tickers})")
            progress_bar.progress(idx / total_tickers)
            
            pipeline = SingleStepPipeline(
                ticker=ticker,
                output_dir=f"output_batch/{ticker}",
                cache_dir="cache",
                force_refresh=False
            )
            
            pipeline.download_data(period=period)
            pipeline.calculate_indicators()
            pipeline.analyze_sentiment()
            pipeline.train_model(model_type=model_type, use_grid_search=False)
            
            results[ticker] = {
                'metrics': pipeline.metrics,
                'current_price': pipeline.df['Close'].iloc[-1],
                'sentiment': pipeline.sentiment_info.get('average_score', 0)
            }
            
        except Exception as e:
            st.warning(f"⚠️ Error analyzing {ticker}: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text("✨ Batch analysis complete!")
    
    if results:
        display_batch_results(results, model_type)

def display_batch_results(results, model_type):
    """Display batch analysis results"""
    st.markdown("---")
    st.markdown(f"### 📊 Batch Analysis Results ({model_type.upper()})")
    
    # Create summary table
    summary_data = []
    for ticker, data in results.items():
        metrics = data['metrics']
        summary_data.append({
            'Ticker': ticker,
            'Current Price': f"${data['current_price']:.2f}",
            'MAE': f"${metrics.get('MAE', 0):.2f}",
            'R²': f"{metrics.get('R2', 0):.4f}",
            'MAPE': f"{metrics.get('MAPE', 0):.2f}%",
            'Sentiment': f"{data['sentiment']:.2f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Visualization
    fig = go.Figure()
    
    tickers = [d['Ticker'] for d in summary_data]
    r2_scores = [results[t]['metrics'].get('R2', 0) for t in tickers]
    
    fig.add_trace(go.Bar(
        x=tickers,
        y=r2_scores,
        marker_color='#1f77b4',
        text=[f"{r2:.3f}" for r2 in r2_scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"R² Score Comparison Across Stocks ({model_type.upper()})",
        xaxis_title="Stock Ticker",
        yaxis_title="R² Score",
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_performance_analytics():
    """Show performance analytics from saved results"""
    st.markdown("### 📈 Performance Analytics Dashboard")
    st.markdown("Analyze historical performance from saved results")
    
    output_dir = Path("output")
    
    if not output_dir.exists():
        st.warning("⚠️ No output directory found. Run some predictions first!")
        return
    
    # Get all ticker directories
    ticker_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not ticker_dirs:
        st.info("📭 No analysis results found")
        return
    
    st.success(f"📊 Found {len(ticker_dirs)} analyzed stocks")
    
    # Aggregate metrics
    all_metrics = []
    
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        summary_file = ticker_dir / "analysis_summary.json"
        
        if summary_file.exists():
            try:
                import json
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                if 'model_performance' in data:
                    metrics = data['model_performance']
                    all_metrics.append({
                        'Ticker': ticker,
                        'R²': metrics.get('R2', 0),
                        'MAE': metrics.get('MAE', 0),
                        'RMSE': metrics.get('RMSE', 0),
                        'MAPE': metrics.get('MAPE', 0)
                    })
            except:
                pass
    
    if all_metrics:
        df_all = pd.DataFrame(all_metrics)
        
        # Summary statistics
        st.markdown("#### 📊 Overall Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg R²", f"{df_all['R²'].mean():.4f}")
        with col2:
            st.metric("Avg MAE", f"${df_all['MAE'].mean():.2f}")
        with col3:
            st.metric("Avg RMSE", f"${df_all['RMSE'].mean():.2f}")
        with col4:
            st.metric("Avg MAPE", f"{df_all['MAPE'].mean():.2f}%")
        
        # Distribution charts
        st.markdown("#### 📈 Performance Distribution")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score Distribution', 'MAE Distribution')
        )
        
        fig.add_trace(
            go.Histogram(x=df_all['R²'], nbinsx=20, marker_color='#1f77b4', name='R²'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=df_all['MAE'], nbinsx=20, marker_color='#ff7f0e', name='MAE'),
            row=1, col=2
        )
        
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Full data table
        with st.expander("📋 View All Results"):
            st.dataframe(df_all, use_container_width=True, hide_index=True)
    else:
        st.info("No performance data available")

if __name__ == "__main__":
    show()
