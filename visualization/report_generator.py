"""Generate comprehensive analysis reports"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class ReportGenerator:
    """Generate analysis reports and summaries"""
    
    def __init__(self, df, ticker, output_dir, metrics=None, sentiment_info=None, predictions=None):
        self.df = df
        self.ticker = ticker
        self.output_dir = Path(output_dir)
        self.metrics = metrics or {}
        self.sentiment_info = sentiment_info or {}
        self.predictions = predictions or {}  # For multi-horizon predictions
    
    def generate_text_summary(self):
        """Generate enhanced text-based summary"""
        
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append(f"COMPREHENSIVE STOCK ANALYSIS REPORT - {self.ticker}")
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("="*80)
        
        # Current Price Info
        current_price = self.df['Close'].iloc[-1]
        current_date = self.df.index[-1].strftime('%Y-%m-%d')
        prev_price = self.df['Close'].iloc[-2] if len(self.df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        sma_20 = self.df['SMA_20'].iloc[-1]
        ema_20 = self.df['EMA_20'].iloc[-1]
        hma_20 = self.df['HMA_20'].iloc[-1]
        
        summary_lines.append("\n📊 CURRENT MARKET STATUS")
        summary_lines.append("-"*80)
        summary_lines.append(f"Date: {current_date}")
        summary_lines.append(f"Current Price: ${current_price:.2f}")
        summary_lines.append(f"Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%) {'📈' if price_change > 0 else '📉'}")
        summary_lines.append(f"")
        summary_lines.append(f"20-day SMA: ${sma_20:.2f} ({'Above' if current_price > sma_20 else 'Below'} - {'Bullish' if current_price > sma_20 else 'Bearish'})")
        summary_lines.append(f"20-day EMA: ${ema_20:.2f} ({'Above' if current_price > ema_20 else 'Below'} - {'Bullish' if current_price > ema_20 else 'Bearish'})")
        summary_lines.append(f"20-day HMA: ${hma_20:.2f} ({'Above' if current_price > hma_20 else 'Below'} - {'Bullish' if current_price > hma_20 else 'Bearish'})")
        
        # Technical Indicators
        rsi = self.df['RSI_14'].iloc[-1]
        macd = self.df['MACD'].iloc[-1]
        macd_signal = self.df['MACD_Signal'].iloc[-1]
        
        summary_lines.append("\n📈 TECHNICAL INDICATORS")
        summary_lines.append("-"*80)
        
        rsi_status = "Overbought ⚠️" if rsi > 70 else "Oversold ⚠️" if rsi < 30 else "Neutral ✓"
        rsi_recommendation = "Consider Selling" if rsi > 70 else "Consider Buying" if rsi < 30 else "Hold Position"
        summary_lines.append(f"RSI (14): {rsi:.2f} - {rsi_status} ({rsi_recommendation})")
        
        macd_status = "Bullish Signal 📈" if macd > macd_signal else "Bearish Signal 📉"
        summary_lines.append(f"MACD: {macd:.4f}")
        summary_lines.append(f"MACD Signal: {macd_signal:.4f}")
        summary_lines.append(f"MACD Status: {macd_status}")
        summary_lines.append(f"MACD Histogram: {self.df['MACD_Hist'].iloc[-1]:.4f}")
        
        # Sentiment Analysis
        if self.sentiment_info and self.sentiment_info.get('num_sources', 0) > 0:
            summary_lines.append("\n💭 SENTIMENT ANALYSIS")
            summary_lines.append("-"*80)
            summary_lines.append(f"Overall Sentiment: {self.sentiment_info['label']}")
            summary_lines.append(f"Sentiment Score: {self.sentiment_info['average_score']:.3f} (Range: -1.0 to +1.0)")
            summary_lines.append(f"Data Sources: {', '.join(self.sentiment_info['sources'])} ({self.sentiment_info['num_sources']} APIs)")
            summary_lines.append("")
            
            for source, details in self.sentiment_info.get('detailed_scores', {}).items():
                if source == 'marketaux':
                    summary_lines.append(f"  📰 Marketaux:")
                    summary_lines.append(f"     Score: {details['score']:.3f}")
                    summary_lines.append(f"     Articles Analyzed: {details['num_articles']}")
                elif source == 'finnhub':
                    summary_lines.append(f"  📊 Finnhub:")
                    summary_lines.append(f"     Score: {details['score']:.3f}")
                    summary_lines.append(f"     Bullish: {details['bullish_percent']:.1f}%")
                    summary_lines.append(f"     Bearish: {details['bearish_percent']:.1f}%")
                elif source == 'alpha_vantage':
                    summary_lines.append(f"  📈 Alpha Vantage:")
                    summary_lines.append(f"     Score: {details['score']:.3f}")
                    summary_lines.append(f"     Label: {details['label']}")
                    summary_lines.append(f"     Articles Analyzed: {details['num_articles']}")
        
        # Single-Step Prediction Analysis
        if 'Predicted_Close' in self.df.columns:
            predictions = self.df['Predicted_Close'].dropna()
            if not predictions.empty:
                summary_lines.append("\n🔮 NEXT-DAY PREDICTION (Single-Step)")
                summary_lines.append("-"*80)
                
                # Get last prediction with actual
                last_pred_idx = predictions.index[-1]
                last_pred = predictions.iloc[-1]
                actual_price = self.df.loc[last_pred_idx, 'Close']
                pred_date = last_pred_idx.strftime('%Y-%m-%d')
                
                summary_lines.append(f"Most Recent Prediction Date: {pred_date}")
                summary_lines.append(f"Predicted Price: ${last_pred:.2f}")
                summary_lines.append(f"Actual Price: ${actual_price:.2f}")
                
                pred_error = abs(actual_price - last_pred)
                pred_error_pct = (pred_error / actual_price) * 100
                accuracy = "✅ Excellent" if pred_error_pct < 2 else "✓ Good" if pred_error_pct < 5 else "⚠️ Fair" if pred_error_pct < 10 else "❌ Needs Improvement"
                summary_lines.append(f"Prediction Error: ${pred_error:.2f} ({pred_error_pct:.2f}%)")
                summary_lines.append(f"Accuracy Rating: {accuracy}")
                
                # Prediction coverage
                prediction_coverage = (len(predictions) / len(self.df)) * 100
                summary_lines.append(f"Prediction Coverage: {prediction_coverage:.1f}% of total data points")
                
                # Next forecast
                next_pred_date = (self.df.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                summary_lines.append(f"")
                summary_lines.append(f"Next Trading Day Forecast: {next_pred_date}")
                
                # Use the last valid prediction as next day forecast
                if len(self.df) > 0:
                    latest_features_idx = self.df.dropna(subset=['SMA_20', 'EMA_20', 'RSI_14']).index[-1]
                    if hasattr(self, 'model') and self.model is not None:
                        # Make a prediction for next day
                        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20',
                                      'UPPER_BAND', 'LOWER_BAND', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'HMA_20', 'Sentiment_Score']
                        latest_features = self.df.loc[latest_features_idx, feature_cols].values.reshape(1, -1)
                        next_pred = self.model.predict(latest_features)[0]
                    else:
                        next_pred = last_pred
                    
                    summary_lines.append(f"Forecasted Close: ${next_pred:.2f}")
                    change = next_pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈 UPWARD" if change > 0 else "📉 DOWNWARD" if change < 0 else "➡️ FLAT"
                    summary_lines.append(f"Expected Movement: ${change:+.2f} ({change_pct:+.2f}%) {direction}")
        
        # Multi-Horizon Predictions
        if self.predictions:
            summary_lines.append("\n🎯 MULTI-HORIZON PREDICTIONS")
            summary_lines.append("-"*80)
            
            for horizon, pred_data in sorted(self.predictions.items(), key=lambda x: int(x[0].replace('d', ''))):
                horizon_days = int(horizon.replace('d', ''))
                pred_date = pred_data.get('date', 'N/A')
                pred_price = pred_data.get('predicted_price', 0)
                actual_price = pred_data.get('actual_price')
                confidence = pred_data.get('confidence_interval', {})
                
                summary_lines.append(f"\n{horizon_days}-Day Forecast:")
                summary_lines.append(f"  Target Date: {pred_date}")
                summary_lines.append(f"  Predicted Price: ${pred_price:.2f}")
                
                if confidence:
                    summary_lines.append(f"  Confidence Interval (95%): ${confidence.get('lower', 0):.2f} - ${confidence.get('upper', 0):.2f}")
                
                if actual_price:
                    error = abs(actual_price - pred_price)
                    error_pct = (error / actual_price) * 100
                    summary_lines.append(f"  Actual Price: ${actual_price:.2f}")
                    summary_lines.append(f"  Prediction Error: ${error:.2f} ({error_pct:.2f}%)")
                    accuracy = "✅ Excellent" if error_pct < 2 else "✓ Good" if error_pct < 5 else "⚠️ Fair" if error_pct < 10 else "❌ Needs Improvement"
                    summary_lines.append(f"  Accuracy Rating: {accuracy}")
                
                change = pred_price - current_price
                change_pct = (change / current_price) * 100
                direction = "📈 UP" if change > 0 else "📉 DOWN" if change < 0 else "➡️ FLAT"
                summary_lines.append(f"  Expected Change from Current: ${change:+.2f} ({change_pct:+.2f}%) {direction}")
        
        # Model Performance
        if self.metrics:
            summary_lines.append("\n🎯 MODEL PERFORMANCE METRICS")
            summary_lines.append("-"*80)
            r2 = self.metrics.get('R2', 0)
            rmse = self.metrics.get('RMSE', 0)
            summary_lines.append(f"R² Score: {r2:.4f} ({'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.7 else 'Fair' if r2 > 0.5 else 'Needs Improvement'})")
            summary_lines.append(f"Mean Absolute Error (MAE): ${self.metrics.get('MAE', 0):.2f}")
            summary_lines.append(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
            summary_lines.append(f"Mean Squared Error (MSE): {self.metrics.get('MSE', 0):.4f}")
            
            # RMSE interpretation
            rmse_pct = (rmse / current_price) * 100 if current_price > 0 else 0
            summary_lines.append(f"RMSE as % of Current Price: {rmse_pct:.2f}%")
            
            if 'best_params' in self.metrics:
                summary_lines.append(f"\nOptimized Model Parameters:")
                for param, value in self.metrics['best_params'].items():
                    summary_lines.append(f"  • {param}: {value}")
        
        # Trading Recommendation
        summary_lines.append("\n💡 TRADING RECOMMENDATION")
        summary_lines.append("-"*80)
        
        # Calculate recommendation score
        recommendation_score = 0
        reasons = []
        
        # Price vs Moving Averages
        if current_price > sma_20:
            recommendation_score += 1
            reasons.append("✓ Price above 20-day SMA (Bullish)")
        else:
            recommendation_score -= 1
            reasons.append("✗ Price below 20-day SMA (Bearish)")
        
        # RSI
        if rsi > 70:
            recommendation_score -= 1
            reasons.append("⚠️ RSI indicates overbought (Consider selling)")
        elif rsi < 30:
            recommendation_score += 1
            reasons.append("✓ RSI indicates oversold (Buy opportunity)")
        
        # MACD
        if macd > macd_signal:
            recommendation_score += 1
            reasons.append("✓ MACD shows bullish crossover")
        else:
            recommendation_score -= 1
            reasons.append("✗ MACD shows bearish crossover")
        
        # Sentiment
        if self.sentiment_info.get('average_score', 0) > 0.15:
            recommendation_score += 1
            reasons.append("✓ Positive market sentiment")
        elif self.sentiment_info.get('average_score', 0) < -0.15:
            recommendation_score -= 1
            reasons.append("✗ Negative market sentiment")
        
        if recommendation_score >= 2:
            recommendation = "🟢 STRONG BUY"
        elif recommendation_score == 1:
            recommendation = "🟢 BUY"
        elif recommendation_score == 0:
            recommendation = "🟡 HOLD"
        elif recommendation_score == -1:
            recommendation = "🔴 SELL"
        else:
            recommendation = "🔴 STRONG SELL"
        
        summary_lines.append(f"Overall Rating: {recommendation}")
        summary_lines.append(f"Confidence Score: {recommendation_score}/4")
        summary_lines.append(f"\nKey Factors:")
        for reason in reasons:
            summary_lines.append(f"  {reason}")
        
        summary_lines.append("\n" + "="*80)
        
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        # Save to file
        summary_file = self.output_dir / "analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        return summary_file
    
    def generate_json_summary(self):
        """Generate enhanced JSON summary"""
        
        current_price = float(self.df['Close'].iloc[-1])
        current_date = self.df.index[-1].strftime('%Y-%m-%d')
        
        summary = {
            'ticker': self.ticker,
            'generated_at': datetime.now().isoformat(),
            'report_date': current_date,
            'current_market_data': {
                'price': current_price,
                'date': current_date,
                'daily_change': float(self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]) if len(self.df) > 1 else 0,
                'daily_change_percent': float(((self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]) / self.df['Close'].iloc[-2]) * 100) if len(self.df) > 1 else 0,
                'volume': int(self.df['Volume'].iloc[-1])
            },
            'moving_averages': {
                'sma_20': float(self.df['SMA_20'].iloc[-1]),
                'ema_20': float(self.df['EMA_20'].iloc[-1]),
                'hma_20': float(self.df['HMA_20'].iloc[-1])
            },
            'technical_indicators': {
                'rsi_14': float(self.df['RSI_14'].iloc[-1]),
                'macd': float(self.df['MACD'].iloc[-1]),
                'macd_signal': float(self.df['MACD_Signal'].iloc[-1]),
                'macd_histogram': float(self.df['MACD_Hist'].iloc[-1])
            },
            'sentiment_analysis': self.sentiment_info,
            'predictions': {},
            'model_performance': self.metrics,
            'data_coverage': {
                'start_date': str(self.df.index[0].date()),
                'end_date': str(self.df.index[-1].date()),
                'total_trading_days': len(self.df)
            }
        }
        
        # Single-step predictions
        if 'Predicted_Close' in self.df.columns:
            predictions = self.df['Predicted_Close'].dropna()
            if not predictions.empty:
                last_pred_idx = predictions.index[-1]
                summary['predictions']['single_step'] = {
                    'prediction_date': last_pred_idx.strftime('%Y-%m-%d'),
                    'predicted_price': float(predictions.iloc[-1]),
                    'actual_price': float(self.df.loc[last_pred_idx, 'Close']),
                    'error': float(abs(self.df.loc[last_pred_idx, 'Close'] - predictions.iloc[-1])),
                    'error_percent': float((abs(self.df.loc[last_pred_idx, 'Close'] - predictions.iloc[-1]) / self.df.loc[last_pred_idx, 'Close']) * 100),
                    'coverage_percent': float((len(predictions) / len(self.df)) * 100)
                }
        
        # Multi-horizon predictions
        if self.predictions:
            summary['predictions']['multi_horizon'] = self.predictions
        
        # Save to file
        json_file = self.output_dir / "analysis_summary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, default=str)
        
        return json_file
    
    def generate_all(self):
        """Generate all reports"""
        text_file = self.generate_text_summary()
        json_file = self.generate_json_summary()
        
        print(f"\n✅ Reports Generated:")
        print(f"  📄 Text Report: {text_file}")
        print(f"  📊 JSON Report: {json_file}")
        
        return {'text': text_file, 'json': json_file}