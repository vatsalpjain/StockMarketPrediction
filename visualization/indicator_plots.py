"""Technical indicator visualization"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import WeekdayLocator, MO
import pandas as pd
from config.settings import PLOT_MONTHS, PLOT_DPI

class IndicatorPlotter:
    """Create technical indicator plots"""
    
    def __init__(self, df, ticker):
        self.df = df
        self.ticker = ticker
    
    def plot_rsi(self, filepath):
        """Plot RSI indicator"""
        
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        df_recent = self.df.loc[self.df.index >= last_months_start].copy()
        
        plt.figure(figsize=(20, 6))
        plt.plot(df_recent.index, df_recent['RSI_14'], 
                color='purple', linewidth=3, label='RSI (14-day)')
        
        # Reference lines
        plt.axhline(70, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Oversold (30)')
        plt.axhline(50, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (50)')
        
        # Shaded zones
        plt.fill_between(df_recent.index, 70, 100, color='red', alpha=0.1, label='Overbought Zone')
        plt.fill_between(df_recent.index, 0, 30, color='green', alpha=0.1, label='Oversold Zone')
        
        # Current status
        current_rsi = df_recent['RSI_14'].iloc[-1]
        status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        status_text = f'Current RSI: {current_rsi:.1f} ({status})'
        
        plt.text(0.98, 0.95, status_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.title(f'{self.ticker} - Relative Strength Index (RSI) - Last {PLOT_MONTHS} Months', 
                  fontsize=18, fontweight='bold', pad=15)
        plt.ylabel('RSI Value', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.gca().xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_macd(self, filepath):
        """Plot MACD indicator"""
        
        last_months_start = self.df.index.max() - pd.DateOffset(months=PLOT_MONTHS)
        df_recent = self.df.loc[self.df.index >= last_months_start].copy()
        
        plt.figure(figsize=(20, 6))
        plt.plot(df_recent.index, df_recent['MACD'], 
                color='blue', linewidth=3, label='MACD Line')
        plt.plot(df_recent.index, df_recent['MACD_Signal'], 
                color='orange', linewidth=3, label='Signal Line')
        
        # Histogram
        hist_colors = ['green' if x > 0 else 'red' for x in df_recent['MACD_Hist']]
        plt.bar(df_recent.index, df_recent['MACD_Hist'], 
               color=hist_colors, alpha=0.6, width=0.7, label='Histogram')
        
        plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero Line')
        
        # Current status
        current_macd = df_recent['MACD'].iloc[-1]
        current_signal = df_recent['MACD_Signal'].iloc[-1]
        signal_status = "Bullish" if current_macd > current_signal else "Bearish"
        macd_text = f'Current Signal: {signal_status}'
        
        plt.text(0.98, 0.95, macd_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', 
                          facecolor='lightgreen' if signal_status == 'Bullish' else 'lightcoral', 
                          alpha=0.8))
        
        plt.title(f'{self.ticker} - MACD Indicator - Last {PLOT_MONTHS} Months', 
                  fontsize=18, fontweight='bold', pad=15)
        plt.ylabel('MACD Value', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.gca().xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return filepath