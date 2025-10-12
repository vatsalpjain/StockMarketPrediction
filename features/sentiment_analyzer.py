"""News sentiment analysis from multiple APIs"""
import requests
import finnhub
from config.api_keys import APIKeys

class SentimentAnalyzer:
    """Analyze news sentiment from multiple sources"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.sentiment_scores = []
        self.sources = []
        self.detailed_info = {}
    
    def fetch_marketaux_sentiment(self):
        """Fetch sentiment from Marketaux API"""
        api_key = APIKeys.get_marketaux_key()
        if not api_key:
            return None
        
        print(f"Fetching Marketaux sentiment for {self.ticker}...")
        try:
            url = f"https://api.marketaux.com/v1/news/all?symbols={self.ticker}&filter_entities=true&language=en&api_token={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                article_sentiments = []
                for article in data['data'][:10]:
                    if 'entities' in article:
                        for entity in article['entities']:
                            if entity['symbol'] == self.ticker and 'sentiment_score' in entity:
                                article_sentiments.append(entity['sentiment_score'])
                
                if article_sentiments:
                    sentiment = sum(article_sentiments) / len(article_sentiments)
                    self.sentiment_scores.append(sentiment)
                    self.sources.append('Marketaux')
                    self.detailed_info['marketaux'] = {
                        'score': float(sentiment),
                        'num_articles': len(article_sentiments)
                    }
                    print(f"✓ Marketaux: {sentiment:.3f} from {len(article_sentiments)} articles")
                    return sentiment
        except Exception as e:
            print(f"Marketaux Error: {e}")
        return None
    
    def fetch_finnhub_sentiment(self):
        """Fetch sentiment from Finnhub API"""
        api_key = APIKeys.get_finnhub_key()
        if not api_key:
            return None
        
        print(f"Fetching Finnhub sentiment for {self.ticker}...")
        try:
            client = finnhub.Client(api_key=api_key)
            data = client.news_sentiment(self.ticker)
            
            if data and 'sentiment' in data:
                sentiment = (data['sentiment']['bullishPercent'] - 
                           data['sentiment']['bearishPercent']) / 100
                self.sentiment_scores.append(sentiment)
                self.sources.append('Finnhub')
                self.detailed_info['finnhub'] = {
                    'score': float(sentiment),
                    'bullish_percent': data['sentiment']['bullishPercent'],
                    'bearish_percent': data['sentiment']['bearishPercent']
                }
                print(f"✓ Finnhub: {sentiment:.3f}")
                print(f"  Bullish: {data['sentiment']['bullishPercent']:.1f}%")
                print(f"  Bearish: {data['sentiment']['bearishPercent']:.1f}%")
                return sentiment
        except Exception as e:
            print(f"Finnhub Error: {e}")
        return None
    
    def fetch_alpha_vantage_sentiment(self):
        """Fetch sentiment from Alpha Vantage API"""
        api_key = APIKeys.get_alpha_vantage_key()
        if not api_key:
            return None
        
        print(f"Fetching Alpha Vantage sentiment for {self.ticker}...")
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'Error Message' in data:
                print(f"⚠ Alpha Vantage Error: {data['Error Message']}")
                return None
            elif 'Note' in data:
                print(f"⚠ Alpha Vantage Limit: {data['Note']}")
                return None
            elif 'feed' in data and len(data['feed']) > 0:
                scores = []
                for article in data['feed'][:10]:
                    ticker_sentiments = article.get('ticker_sentiment', [])
                    score = None
                    for ts in ticker_sentiments:
                        if ts.get('ticker') == self.ticker:
                            score = float(ts.get('ticker_sentiment_score', 0))
                            break
                    if score is None:
                        score = float(article.get('overall_sentiment_score', 0))
                    scores.append(score)
                
                if scores:
                    sentiment = sum(scores) / len(scores)
                    self.sentiment_scores.append(sentiment)
                    self.sources.append('Alpha Vantage')
                    
                    label = ("BULLISH 📈" if sentiment >= 0.15 else 
                            "BEARISH 📉" if sentiment <= -0.15 else "NEUTRAL ➖")
                    
                    self.detailed_info['alpha_vantage'] = {
                        'score': float(sentiment),
                        'num_articles': len(scores),
                        'label': label
                    }
                    print(f"✓ Alpha Vantage: {sentiment:.3f} ({label}) from {len(scores)} articles")
                    return sentiment
        except Exception as e:
            print(f"Alpha Vantage Error: {e}")
        return None
    
    def get_combined_sentiment(self):
        """Fetch from all APIs and return combined sentiment"""
        # Fetch from all sources
        self.fetch_marketaux_sentiment()
        self.fetch_finnhub_sentiment()
        self.fetch_alpha_vantage_sentiment()
        
        # Calculate combined sentiment
        if self.sentiment_scores:
            combined = sum(self.sentiment_scores) / len(self.sentiment_scores)
            label = ("BULLISH 📈" if combined >= 0.15 else 
                    "BEARISH 📉" if combined <= -0.15 else "NEUTRAL ➖")
            
            print(f"\n{'='*60}")
            print(f"✓ COMBINED SENTIMENT: {combined:.3f} ({label})")
            print(f"  Sources: {', '.join(self.sources)} ({len(self.sources)} APIs)")
            print(f"{'='*60}\n")
            
            return {
                'average_score': float(combined),
                'label': label,
                'sources': self.sources,
                'num_sources': len(self.sources),
                'detailed_scores': self.detailed_info
            }
        else:
            print("\n⚠ No sentiment data available")
            print("Using neutral sentiment (0.0)\n")
            return {
                'average_score': 0.0,
                'label': 'NEUTRAL ➖',
                'sources': [],
                'num_sources': 0
            }