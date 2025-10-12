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
        
        # Price chart
        ax1.plot(df_recent.index, df_recent['Close'], 
                color='black', linewidth=2, label='Close Price', alpha=0.8)
        ax1.plot(df_recent.index, df_recent['SMA_20'], 
                color='purple', linewidth=2, linestyle='--', label='SMA 20', alpha=0.9)
        ax1.plot(df_recent.index, df_recent['EMA_20'], 
                color='orange', linewidth=2, linestyle='--', label='EMA 20', alpha=0.9)
        
        # Bollinger Bands
        ax1.plot(df_recent.index, df_recent['UPPER_BAND'], 
                color='gray', linewidth=1.5, linestyle=':', alpha=0.7)
        ax1.plot(df_recent.index, df_recent['LOWER_BAND'], 
                color='gray', linewidth=1.5, linestyle=':', alpha=0.7)
        ax1.fill_between(df_recent.index, df_recent['UPPER_BAND'], df_recent['LOWER_BAND'],
                         color='lightgray', alpha=0.2, label='Bollinger Bands')
        
        # HMA
        ax1.plot(df_recent.index, df_recent['HMA_20'], 
                color='cyan', linewidth=2, label='HMA 20', alpha=0.9)
        
        # Predictions
        if 'Predicted_Close' in df_recent.columns:
            ax1.plot(df_recent.index, df_recent['Predicted_Close'], 
                    color='red', linewidth=3, label='Predictions', alpha=0.9)
        
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
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
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
        """Plot predictions for multiple horizons"""
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Get recent data
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        df_recent = self.df.loc[self.df.index >= last_months_start].copy()
        
        # Plot actual price
        ax.plot(df_recent.index, df_recent['Close'], 
                color='black', linewidth=2.5, label='Actual Price', alpha=0.9)
        
        # Plot predictions for each horizon
        colors = {'1d': 'red', '7d': 'orange', '30d': 'purple'}
        
        for horizon, pred_data in predictions_dict.items():
            if 'predictions' in pred_data and 'index' in pred_data:
                ax.plot(pred_data['index'], pred_data['predictions'],
                       color=colors.get(horizon, 'blue'), linewidth=2,
                       label=f'{horizon} Prediction', alpha=0.8, linestyle='--')
        
        ax.set_title(
            f'{self.ticker} Multi-Horizon Price Predictions (Last {PLOT_MONTHS} Months)',
            fontsize=18, fontweight='bold', pad=20
        )
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath