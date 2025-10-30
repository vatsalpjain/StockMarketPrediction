"""Results Viewing Page for Streamlit"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def show():
    """Display results viewing page"""
    st.title("📊 View Analysis Results")
    st.markdown("Browse and visualize previously generated stock analysis results")
    st.markdown("---")
    
    # Get output directory
    output_dir = Path("output")
    
    if not output_dir.exists():
        st.warning("⚠️ No output directory found. Run some predictions first!")
        return
    
    # Get list of analyzed tickers
    ticker_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not ticker_dirs:
        st.info("📭 No analysis results found. Run predictions to generate results!")
        return
    
    # Ticker selection
    ticker_names = [d.name for d in ticker_dirs]
    selected_ticker = st.selectbox(
        "Select Stock Ticker",
        options=ticker_names,
        help="Choose a ticker to view its analysis results"
    )
    
    if selected_ticker:
        ticker_path = output_dir / selected_ticker
        display_ticker_results(ticker_path, selected_ticker)

def display_ticker_results(ticker_path, ticker):
    """Display results for a specific ticker"""
    
    st.markdown(f"## 📈 Results for {ticker}")
    st.markdown("---")
    
    # Check for available files
    summary_json = ticker_path / "analysis_summary.json"
    summary_txt = ticker_path / "analysis_summary.txt"
    plots_dir = ticker_path / "plots"
    
    # Create tabs for different views
    tabs = st.tabs(["📊 Summary", "📈 Visualizations", "📄 Detailed Report", "📁 Files"])
    
    # Tab 1: Summary
    with tabs[0]:
        display_summary_tab(summary_json, ticker)
    
    # Tab 2: Visualizations
    with tabs[1]:
        display_visualizations_tab(plots_dir, ticker)
    
    # Tab 3: Detailed Report
    with tabs[2]:
        display_report_tab(summary_txt, summary_json)
    
    # Tab 4: Files
    with tabs[3]:
        display_files_tab(ticker_path)

def display_summary_tab(summary_json, ticker):
    """Display summary information"""
    st.markdown("### 📊 Analysis Summary")
    
    if summary_json.exists():
        try:
            with open(summary_json, 'r') as f:
                data = json.load(f)
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Ticker",
                    value=data.get('ticker', ticker)
                )
            
            with col2:
                analysis_date = data.get('analysis_date', 'N/A')
                st.metric(
                    label="Analysis Date",
                    value=analysis_date
                )
            
            with col3:
                current_price = data.get('current_price', 0)
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}" if current_price else "N/A"
                )
            
            st.markdown("---")
            
            # Model performance
            if 'model_performance' in data:
                st.markdown("### 🎯 Model Performance")
                perf = data['model_performance']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"${perf.get('MAE', 0):.2f}")
                
                with col2:
                    st.metric("RMSE", f"${perf.get('RMSE', 0):.2f}")
                
                with col3:
                    st.metric("R² Score", f"{perf.get('R2', 0):.4f}")
                
                with col4:
                    st.metric("MAPE", f"{perf.get('MAPE', 0):.2f}%")
            
            # Multi-horizon predictions
            if 'multi_horizon_predictions' in data:
                st.markdown("---")
                st.markdown("### 🔮 Multi-Horizon Predictions")
                
                mh_preds = data['multi_horizon_predictions']
                
                pred_data = []
                for horizon_key, pred_info in mh_preds.items():
                    if 'future_prediction' in pred_info:
                        fp = pred_info['future_prediction']
                        pred_data.append({
                            'Horizon': horizon_key,
                            'Target Date': fp.get('date', 'N/A'),
                            'Predicted Price': f"${fp.get('predicted_price', 0):.2f}",
                            'Change': f"{fp.get('change_percent', 0):+.2f}%",
                            'Confidence Range': f"${fp['confidence_interval']['lower']:.2f} - ${fp['confidence_interval']['upper']:.2f}"
                        })
                
                if pred_data:
                    df_preds = pd.DataFrame(pred_data)
                    st.dataframe(df_preds, use_container_width=True, hide_index=True)
            
            # Sentiment analysis
            if 'sentiment_analysis' in data:
                st.markdown("---")
                st.markdown("### 📰 Sentiment Analysis")
                
                sentiment = data['sentiment_analysis']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    score = sentiment.get('average_score', 0)
                    st.metric("Sentiment Score", f"{score:.2f}")
                
                with col2:
                    sources = sentiment.get('num_sources', 0)
                    st.metric("News Sources", sources)
                
                with col3:
                    # Sentiment interpretation
                    if score > 0.1:
                        sentiment_label = "🟢 Positive"
                    elif score < -0.1:
                        sentiment_label = "🔴 Negative"
                    else:
                        sentiment_label = "🟡 Neutral"
                    st.metric("Sentiment", sentiment_label)
            
            # Technical indicators
            if 'technical_indicators' in data:
                st.markdown("---")
                st.markdown("### 📊 Latest Technical Indicators")
                
                indicators = data['technical_indicators']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi = indicators.get('RSI_14', 0)
                    st.metric("RSI (14)", f"{rsi:.2f}")
                    if rsi > 70:
                        st.caption("🔴 Overbought")
                    elif rsi < 30:
                        st.caption("🟢 Oversold")
                    else:
                        st.caption("🟡 Neutral")
                
                with col2:
                    macd = indicators.get('MACD', 0)
                    st.metric("MACD", f"{macd:.4f}")
                
                with col3:
                    sma = indicators.get('SMA_20', 0)
                    st.metric("SMA (20)", f"${sma:.2f}")
                
                with col4:
                    ema = indicators.get('EMA_20', 0)
                    st.metric("EMA (20)", f"${ema:.2f}")
            
        except Exception as e:
            st.error(f"Error loading summary: {str(e)}")
    else:
        st.warning("Summary file not found")

def display_visualizations_tab(plots_dir, ticker):
    """Display all available plots"""
    st.markdown("### 📈 Visualizations")
    
    if not plots_dir.exists():
        st.warning("No plots directory found")
        return
    
    # Get all PNG files
    plot_files = list(plots_dir.glob("*.png"))
    
    if not plot_files:
        st.info("No plots available")
        return
    
    # Organize plots by category
    plot_categories = {
        "Price Predictions": [],
        "Technical Indicators": [],
        "Multi-Horizon Analysis": [],
        "Other": []
    }
    
    for plot_file in plot_files:
        name = plot_file.name
        if "price" in name.lower() or "prediction" in name.lower():
            if "multi" in name.lower() or "horizon" in name.lower():
                plot_categories["Multi-Horizon Analysis"].append(plot_file)
            else:
                plot_categories["Price Predictions"].append(plot_file)
        elif "rsi" in name.lower() or "macd" in name.lower() or "indicator" in name.lower():
            plot_categories["Technical Indicators"].append(plot_file)
        elif "backtest" in name.lower():
            plot_categories["Multi-Horizon Analysis"].append(plot_file)
        else:
            plot_categories["Other"].append(plot_file)
    
    # Display plots by category
    for category, files in plot_categories.items():
        if files:
            st.markdown(f"#### {category}")
            
            # Create columns for multiple plots
            if len(files) == 1:
                st.image(str(files[0]), use_container_width=True, caption=files[0].stem.replace('_', ' ').title())
            else:
                cols = st.columns(min(2, len(files)))
                for idx, plot_file in enumerate(files):
                    with cols[idx % 2]:
                        st.image(str(plot_file), use_container_width=True, caption=plot_file.stem.replace('_', ' ').title())
            
            st.markdown("---")

def display_report_tab(summary_txt, summary_json):
    """Display detailed text report"""
    st.markdown("### 📄 Detailed Analysis Report")
    
    # Try text report first
    if summary_txt.exists():
        try:
            with open(summary_txt, 'r') as f:
                report_text = f.read()
            
            st.text_area(
                "Full Report",
                value=report_text,
                height=600,
                disabled=True
            )
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"analysis_report_{summary_txt.parent.name}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error loading report: {str(e)}")
    
    # Also show JSON if available
    elif summary_json.exists():
        try:
            with open(summary_json, 'r') as f:
                json_data = json.load(f)
            
            st.json(json_data)
            
            # Download button
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"analysis_summary_{summary_json.parent.name}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error loading JSON: {str(e)}")
    else:
        st.warning("No detailed report available")

def display_files_tab(ticker_path):
    """Display file browser"""
    st.markdown("### 📁 Output Files")
    
    # List all files
    all_files = []
    for item in ticker_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(ticker_path)
            file_size = item.stat().st_size
            modified = datetime.fromtimestamp(item.stat().st_mtime)
            
            all_files.append({
                'File': str(rel_path),
                'Size': format_file_size(file_size),
                'Modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
                'Type': item.suffix[1:].upper() if item.suffix else 'N/A'
            })
    
    if all_files:
        df_files = pd.DataFrame(all_files)
        st.dataframe(df_files, use_container_width=True, hide_index=True)
        
        st.info(f"📊 Total files: {len(all_files)}")
    else:
        st.warning("No files found")
    
    # Show directory path
    st.code(f"Output Directory: {ticker_path.absolute()}")

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

if __name__ == "__main__":
    show()
