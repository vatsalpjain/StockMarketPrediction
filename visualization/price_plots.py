"""Price and prediction visualization"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import WeekdayLocator, MO
import pandas as pd
from config.settings import PLOT_MONTHS, PLOT_DPI

class PricePlotter:
    """Create price-related plots"""
    
    def __init__(self, df, ticker):
        self.df = df
        self.ticker = ticker
    
    def plot_price_predictions(self, filepath, metrics=None):
        """Plot price with predictions and technical indicators"""
        
        # Get recent data
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        df_recent = self.df.loc[self.df.index >= last_months_start].copy()
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(20, 12),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}
        )
        
        # Price chart - Actual price (thicker, prominent)
        ax1.plot(df_recent.index, df_recent['Close'], 
                color='black', linewidth=3, label='Actual Price', alpha=1.0, zorder=5)
        
        # Predictions - MOST PROMINENT (plotted first in background, thicker line)
        if 'Predicted_Close' in df_recent.columns:
            pred_data = df_recent['Predicted_Close'].dropna()
            if not pred_data.empty:
                ax1.plot(pred_data.index, pred_data, 
                        color='#FF1744', linewidth=3.5, label='Model Predictions', 
                        alpha=0.85, linestyle='-', zorder=6, marker='o', markersize=3, 
                        markeredgecolor='white', markeredgewidth=0.5)
        
        # Technical Indicators (thinner, less prominent)
        ax1.plot(df_recent.index, df_recent['SMA_20'], 
                color='purple', linewidth=1.5, linestyle='--', label='SMA 20', alpha=0.6, zorder=3)
        ax1.plot(df_recent.index, df_recent['EMA_20'], 
                color='orange', linewidth=1.5, linestyle='--', label='EMA 20', alpha=0.6, zorder=3)
        
        # Bollinger Bands (subtle background)
        ax1.plot(df_recent.index, df_recent['UPPER_BAND'], 
                color='gray', linewidth=1, linestyle=':', alpha=0.4, zorder=1)
        ax1.plot(df_recent.index, df_recent['LOWER_BAND'], 
                color='gray', linewidth=1, linestyle=':', alpha=0.4, zorder=1)
        ax1.fill_between(df_recent.index, df_recent['UPPER_BAND'], df_recent['LOWER_BAND'],
                         color='lightgray', alpha=0.15, label='Bollinger Bands', zorder=1)
        
        # HMA
        ax1.plot(df_recent.index, df_recent['HMA_20'], 
                color='cyan', linewidth=1.5, linestyle='--', label='HMA 20', alpha=0.6, zorder=3)
        
        # Add metrics box
        if metrics:
            textstr = f'Model Performance:\nR² = {metrics["R2"]:.3f}\nMAE = ${metrics["MAE"]:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)
        
        ax1.set_title(
            f'{self.ticker} Stock Price with Technical Indicators & Predictions (Last {PLOT_MONTHS} Months)',
            fontsize=18, fontweight='bold', pad=20
        )
        ax1.set_ylabel('Price ($)', fontsize=14)
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
        ax1.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Volume chart
        colors = ['green' if close >= open_price else 'red'
                  for open_price, close in zip(df_recent['Open'], df_recent['Close'])]
        ax2.bar(df_recent.index, df_recent['Volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_multi_horizon_predictions(self, filepath, predictions_dict):
        """Plot predictions for multiple horizons with actual vs predicted - CLEAR AND VISIBLE"""
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Get recent data
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        df_recent = self.df.loc[self.df.index >= last_months_start].copy()
        
        # Plot actual price (thick, black)
        ax.plot(df_recent.index, df_recent['Close'], 
                color='black', linewidth=3.5, label='Actual Price', alpha=1.0, zorder=10)
        
        # Enhanced colors and styles for predictions
        horizon_styles = {
            '1d': {'color': '#FF1744', 'linewidth': 3, 'linestyle': '-', 'marker': 'o', 'markersize': 6, 'label': '1-Day Prediction'},
            '7d': {'color': '#FF9800', 'linewidth': 3, 'linestyle': '-', 'marker': 's', 'markersize': 6, 'label': '7-Day Prediction'},
            '30d': {'color': '#9C27B0', 'linewidth': 3, 'linestyle': '-', 'marker': '^', 'markersize': 6, 'label': '30-Day Prediction'}
        }
        
        for horizon, pred_data in predictions_dict.items():
            if 'predictions' in pred_data and 'index' in pred_data:
                pred_index = pred_data['index']
                pred_values = pred_data['predictions']
                
                # Filter for recent period
                mask = pred_index >= last_months_start
                if isinstance(mask, pd.Series):
                    recent_indices = pred_index[mask]
                    recent_preds = pred_values[mask.values] if hasattr(pred_values, '__getitem__') else [p for i, p in enumerate(pred_values) if mask.iloc[i]]
                else:
                    recent_indices = [idx for idx in pred_index if idx >= last_months_start]
                    recent_preds = [pred_values[i] for i, idx in enumerate(pred_index) if idx >= last_months_start]
                
                if len(recent_indices) > 0:
                    style = horizon_styles.get(horizon, {'color': 'blue', 'linewidth': 3, 'linestyle': '-', 'marker': 'o', 'markersize': 6, 'label': f'{horizon} Prediction'})
                    
                    # Plot prediction line with markers
                    ax.plot(recent_indices, recent_preds,
                           color=style['color'], linewidth=style['linewidth'],
                           label=style['label'], alpha=0.9, linestyle=style['linestyle'], zorder=8)
                    
                    # Add markers at every point for maximum visibility
                    ax.scatter(recent_indices, recent_preds,
                              color=style['color'], marker=style['marker'],
                              s=style['markersize']*15, alpha=0.95, zorder=9, 
                              edgecolors='white', linewidth=1.5)
        
        ax.set_title(
            f'{self.ticker} Multi-Horizon Price Predictions (Last {PLOT_MONTHS} Months)',
            fontsize=18, fontweight='bold', pad=20
        )
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc='best', fontsize=13, framealpha=0.95, markerscale=1.2)
        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_backtest_comparison(self, filepath, predictions_dict):
        """Create detailed backtest comparison showing prediction accuracy over time"""
        
        fig, axes = plt.subplots(len(predictions_dict), 1, figsize=(20, 6 * len(predictions_dict)))
        if len(predictions_dict) == 1:
            axes = [axes]
        
        # Get recent data
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        
        for idx, (horizon, pred_data) in enumerate(sorted(predictions_dict.items(), 
                                                          key=lambda x: int(x[0].replace('d', '')))):
            ax = axes[idx]
            
            if 'predictions' in pred_data and 'actuals' in pred_data and 'index' in pred_data:
                pred_index = pred_data['index']
                predictions = pred_data['predictions']
                actuals = pred_data['actuals']
                
                # Filter to recent period
                mask = pred_index >= last_months_start
                if isinstance(mask, pd.Series):
                    recent_indices = pred_index[mask]
                    recent_preds = predictions[mask.values] if hasattr(predictions, '__getitem__') else [p for i, p in enumerate(predictions) if mask.iloc[i]]
                    recent_actuals = actuals[mask]
                else:
                    recent_indices = [idx for idx in pred_index if idx >= last_months_start]
                    recent_preds = [predictions[i] for i, idx_val in enumerate(pred_index) if idx_val >= last_months_start]
                    recent_actuals = [actuals.iloc[i] for i, idx_val in enumerate(pred_index) if idx_val >= last_months_start]
                
                if len(recent_indices) > 0:
                    # Add shaded error region FIRST (background)
                    ax.fill_between(recent_indices, recent_actuals, recent_preds, 
                                   alpha=0.25, color='#FFCDD2', label='Prediction Error', zorder=1)
                    
                    # Plot actual vs predicted with THICK LINES
                    ax.plot(recent_indices, recent_actuals, color='black', linewidth=3.5, 
                           label='Actual Price', alpha=1.0, marker='o', markersize=5, 
                           markeredgecolor='white', markeredgewidth=1, zorder=5)
                    ax.plot(recent_indices, recent_preds, color='#FF1744', linewidth=3.5, 
                           label='Predicted Price', alpha=0.9, linestyle='-', marker='s', markersize=5,
                           markeredgecolor='white', markeredgewidth=1, zorder=4)
                    
                    # Calculate and show error statistics
                    errors = [abs(a - p) for a, p in zip(recent_actuals, recent_preds)]
                    avg_error = sum(errors) / len(errors) if errors else 0
                    max_error = max(errors) if errors else 0
                    
                    ax.set_title(f'{horizon} Ahead - Backtest Performance (Avg Error: ${avg_error:.2f}, Max: ${max_error:.2f})',
                               fontsize=15, fontweight='bold', pad=10)
                    ax.set_ylabel('Price ($)', fontsize=13)
                    ax.grid(True, alpha=0.3, zorder=0)
                    ax.legend(loc='best', fontsize=11, framealpha=0.95)
                    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        axes[-1].set_xlabel('Date', fontsize=13)
        plt.suptitle(f'{self.ticker} - Multi-Horizon Backtest: Actual vs Predicted Prices', 
                    fontsize=17, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath