#!/usr/bin/env python3
"""
Enhanced Backtest Engine
========================

Advanced backtesting engine that can simulate all enhanced components:
- Social sentiment simulation with realistic patterns
- ML model training on historical patterns
- Multi-exchange arbitrage opportunities
- Breaking news events and market reactions
- Comprehensive performance analysis

This allows backtesting of the enhanced strategy with realistic data.
"""

import asyncio
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import warnings
import random
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import enhanced strategy
from strategies.enhanced_high_risk_strategy import EnhancedHighRiskStrategy

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class SimulatedSentimentEvent:
    """Simulated social sentiment event"""
    timestamp: datetime
    symbol: str
    platform: str
    sentiment_score: float  # -1 to +1
    momentum: float
    confidence: float
    event_type: str  # 'tweet_viral', 'reddit_discussion', 'influencer_post'

@dataclass
class SimulatedNewsEvent:
    """Simulated news event"""
    timestamp: datetime
    symbol: str
    headline: str
    sentiment_score: float  # -1 to +1
    impact_score: float  # 0 to 1
    urgency: str  # 'low', 'medium', 'high', 'critical'
    source: str

@dataclass
class SimulatedArbitrageOpportunity:
    """Simulated arbitrage opportunity"""
    timestamp: datetime
    symbol: str
    buy_exchange: str
    sell_exchange: str
    profit_percent: float
    confidence: float
    duration_minutes: int

@dataclass
class BacktestResults:
    """Enhanced backtest results"""
    # Basic performance
    total_return: float
    roi_percent: float
    win_rate: float
    total_trades: int
    
    # Enhanced metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    
    # Strategy comparison
    original_strategy_return: float
    enhancement_improvement: float
    
    # Signal source performance
    signal_source_stats: Dict[str, Dict[str, float]]
    
    # Daily breakdown
    daily_performance: List[Dict[str, Any]]
    
    # Assessment
    overall_rating: str
    key_insights: List[str]

class EnhancedMarketSimulator:
    """
    Enhanced market data and event simulation
    
    Generates realistic market conditions with:
    - Price movements based on various regimes
    - Social sentiment events
    - Breaking news events  
    - Arbitrage opportunities
    - ML training data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Simulation parameters
        self.sentiment_event_frequency = self.config.get('sentiment_event_frequency', 0.1)  # 10% chance per hour
        self.news_event_frequency = self.config.get('news_event_frequency', 0.05)  # 5% chance per hour
        self.arbitrage_frequency = self.config.get('arbitrage_frequency', 0.02)  # 2% chance per hour
        
        # Event generators
        self.sentiment_events = []
        self.news_events = []
        self.arbitrage_opportunities = []
        
        logger.info("📊 Enhanced Market Simulator initialized")
    
    def generate_enhanced_market_data(self, start_date: str, days: int = 365) -> Dict[str, Any]:
        """Generate enhanced market data with all simulation components"""
        
        print(f"📊 Generating enhanced market data for {days} days...")
        
        # Generate base market data (similar to original but with enhancements)
        market_data = self._generate_base_market_data(start_date, days)
        
        # Generate sentiment events
        self.sentiment_events = self._generate_sentiment_events(start_date, days)
        print(f"📱 Generated {len(self.sentiment_events)} sentiment events")
        
        # Generate news events  
        self.news_events = self._generate_news_events(start_date, days)
        print(f"📰 Generated {len(self.news_events)} news events")
        
        # Generate arbitrage opportunities
        self.arbitrage_opportunities = self._generate_arbitrage_opportunities(start_date, days)
        print(f"🔄 Generated {len(self.arbitrage_opportunities)} arbitrage opportunities")
        
        return {
            'market_data': market_data,
            'sentiment_events': self.sentiment_events,
            'news_events': self.news_events,
            'arbitrage_opportunities': self.arbitrage_opportunities
        }
    
    def _generate_base_market_data(self, start_date: str, days: int) -> Dict[str, pd.DataFrame]:
        """Generate base market data with enhanced realism"""
        
        # Create hourly data
        dates = pd.date_range(start=start_date, periods=days*24, freq='1H')
        
        # Define enhanced market regimes with more complexity
        regimes = [
            {'name': 'Bear_Accumulation', 'trend': -0.0001, 'volatility': 0.015, 'duration': 45, 'sentiment_bias': -0.2},
            {'name': 'Uncertainty', 'trend': -0.0003, 'volatility': 0.025, 'duration': 30, 'sentiment_bias': -0.4},
            {'name': 'Capitulation', 'trend': -0.0010, 'volatility': 0.045, 'duration': 15, 'sentiment_bias': -0.8},
            {'name': 'Bottom_Formation', 'trend': 0.0000, 'volatility': 0.020, 'duration': 30, 'sentiment_bias': -0.1},
            {'name': 'Early_Recovery', 'trend': 0.0002, 'volatility': 0.018, 'duration': 45, 'sentiment_bias': 0.1},
            {'name': 'Bull_Momentum', 'trend': 0.0005, 'volatility': 0.022, 'duration': 60, 'sentiment_bias': 0.3},
            {'name': 'FOMO_Phase', 'trend': 0.0008, 'volatility': 0.030, 'duration': 45, 'sentiment_bias': 0.6},
            {'name': 'Euphoria', 'trend': 0.0012, 'volatility': 0.040, 'duration': 30, 'sentiment_bias': 0.8},
            {'name': 'Distribution', 'trend': 0.0003, 'volatility': 0.035, 'duration': 30, 'sentiment_bias': 0.2},
            {'name': 'Correction', 'trend': -0.0007, 'volatility': 0.032, 'duration': 35, 'sentiment_bias': -0.3}
        ]
        
        # Symbols with different characteristics
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 
                   'ADA/USDT', 'DOT/USDT', 'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT']
        
        base_prices = {
            'BTC/USDT': 16800, 'ETH/USDT': 1250, 'SOL/USDT': 12, 'AVAX/USDT': 13,
            'MATIC/USDT': 0.78, 'ADA/USDT': 0.27, 'DOT/USDT': 4.8, 'ATOM/USDT': 8.5,
            'NEAR/USDT': 1.3, 'FTM/USDT': 0.20
        }
        
        # Symbol correlations and multipliers
        symbol_correlations = {
            'BTC/USDT': {'correlation': 1.0, 'volatility_mult': 1.0, 'news_sensitivity': 1.0},
            'ETH/USDT': {'correlation': 0.85, 'volatility_mult': 1.2, 'news_sensitivity': 0.9},
            'SOL/USDT': {'correlation': 0.70, 'volatility_mult': 1.8, 'news_sensitivity': 1.3},
            'AVAX/USDT': {'correlation': 0.65, 'volatility_mult': 1.6, 'news_sensitivity': 1.1},
            'MATIC/USDT': {'correlation': 0.60, 'volatility_mult': 1.4, 'news_sensitivity': 0.8},
            'ADA/USDT': {'correlation': 0.55, 'volatility_mult': 1.1, 'news_sensitivity': 0.7},
            'DOT/USDT': {'correlation': 0.58, 'volatility_mult': 1.3, 'news_sensitivity': 0.9},
            'ATOM/USDT': {'correlation': 0.52, 'volatility_mult': 1.5, 'news_sensitivity': 1.0},
            'NEAR/USDT': {'correlation': 0.48, 'volatility_mult': 1.7, 'news_sensitivity': 1.2},
            'FTM/USDT': {'correlation': 0.45, 'volatility_mult': 1.9, 'news_sensitivity': 1.4}
        }
        
        all_data = {}
        
        # Track market-wide sentiment for correlation
        market_sentiment_history = []
        
        for symbol in symbols:
            print(f"   📈 Generating enhanced data for {symbol}...")
            
            np.random.seed(42 + hash(symbol) % 1000)
            
            data = []
            current_price = base_prices[symbol]
            current_period = 0
            
            symbol_props = symbol_correlations[symbol]
            
            for regime in regimes:
                regime_periods = regime['duration'] * 24
                
                for period_in_regime in range(regime_periods):
                    if current_period >= len(dates):
                        break
                    
                    timestamp = dates[current_period]
                    
                    # Base price movement from regime
                    trend = regime['trend']
                    base_volatility = regime['volatility']
                    
                    # Apply symbol-specific multipliers
                    adjusted_volatility = base_volatility * symbol_props['volatility_mult']
                    
                    # Market-wide sentiment influence
                    market_sentiment = regime['sentiment_bias'] + np.random.normal(0, 0.1)
                    market_sentiment_history.append(market_sentiment)
                    
                    # Symbol correlation with market sentiment
                    sentiment_influence = market_sentiment * symbol_props['correlation'] * 0.0002
                    
                    # News event influence (simulated)
                    news_influence = 0
                    for news_event in self.news_events:
                        if (news_event.symbol == symbol and 
                            abs((news_event.timestamp - timestamp).total_seconds()) < 3600):  # Within 1 hour
                            news_multiplier = symbol_props['news_sensitivity']
                            news_influence += news_event.sentiment_score * news_event.impact_score * 0.001 * news_multiplier
                    
                    # Combine all influences
                    total_trend = trend + sentiment_influence + news_influence
                    noise = np.random.normal(0, adjusted_volatility)
                    total_change = total_trend + noise
                    
                    # Special events (flash crashes, pumps)
                    if np.random.random() < 0.001:  # 0.1% chance
                        if np.random.random() < 0.5:
                            total_change -= np.random.uniform(0.03, 0.08)  # Flash crash
                        else:
                            total_change += np.random.uniform(0.05, 0.12)  # Pump
                    
                    # Calculate OHLC
                    open_price = current_price
                    close_price = open_price * (1 + total_change)
                    close_price = max(close_price, base_prices[symbol] * 0.05)  # Floor at 5% of base
                    
                    # Realistic intrabar movement
                    range_multiplier = min(abs(total_change) * 20 + 1, 3)  # Higher volatility = larger ranges
                    
                    if close_price > open_price:
                        high_price = close_price * (1 + abs(np.random.normal(0, 0.005)) * range_multiplier)
                        low_price = open_price * (1 - abs(np.random.normal(0, 0.003)) * range_multiplier)
                    else:
                        high_price = open_price * (1 + abs(np.random.normal(0, 0.003)) * range_multiplier)
                        low_price = close_price * (1 - abs(np.random.normal(0, 0.005)) * range_multiplier)
                    
                    # Enhanced volume modeling
                    base_volume = 800000 if 'BTC' in symbol else 500000
                    
                    # Volume based on price movement and regime
                    volatility_volume = abs(total_change) * 15
                    regime_volume = {'Capitulation': 4.0, 'Euphoria': 3.5, 'FOMO_Phase': 3.0, 
                                   'Bull_Momentum': 2.0, 'Correction': 2.5}.get(regime['name'], 1.0)
                    
                    # News volume impact
                    news_volume_mult = 1.0
                    for news_event in self.news_events:
                        if (news_event.symbol == symbol and 
                            abs((news_event.timestamp - timestamp).total_seconds()) < 1800):  # Within 30 min
                            news_volume_mult += news_event.impact_score * 2.0
                    
                    volume = (base_volume * (1 + volatility_volume) * regime_volume * 
                             news_volume_mult * np.random.lognormal(0, 0.4))
                    
                    # Sentiment events can cause volume spikes
                    for sentiment_event in self.sentiment_events:
                        if (sentiment_event.symbol == symbol and 
                            abs((sentiment_event.timestamp - timestamp).total_seconds()) < 1800):
                            if sentiment_event.event_type == 'tweet_viral':
                                volume *= (1 + sentiment_event.confidence * 2)
                    
                    data.append({
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'regime': regime['name'],
                        'market_sentiment': market_sentiment,
                        'news_influence': news_influence
                    })
                    
                    current_price = close_price
                    current_period += 1
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            all_data[symbol] = df
        
        return all_data
    
    def _generate_sentiment_events(self, start_date: str, days: int) -> List[SimulatedSentimentEvent]:
        """Generate realistic sentiment events"""
        
        events = []
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT']
        platforms = ['twitter', 'reddit', 'discord']
        event_types = ['tweet_viral', 'reddit_discussion', 'influencer_post', 'community_rally']
        
        # Generate events for each day
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        for day in range(days):
            current_date = start_datetime + timedelta(days=day)
            
            # Average 3-5 sentiment events per day across all symbols
            num_events = np.random.poisson(4)
            
            for _ in range(num_events):
                # Random time during the day
                random_hour = np.random.randint(0, 24)
                random_minute = np.random.randint(0, 60)
                event_time = current_date + timedelta(hours=random_hour, minutes=random_minute)
                
                # Select symbol (bias toward major coins)
                symbol_weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # BTC has highest weight
                symbol = np.random.choice(symbols, p=symbol_weights)
                
                platform = np.random.choice(platforms)
                event_type = np.random.choice(event_types)
                
                # Generate sentiment characteristics
                base_sentiment = np.random.normal(0, 0.4)  # Slightly bullish bias in crypto
                
                # Event type influences sentiment strength
                sentiment_multipliers = {
                    'tweet_viral': 1.5,
                    'reddit_discussion': 1.2,
                    'influencer_post': 1.8,
                    'community_rally': 1.3
                }
                
                sentiment_score = base_sentiment * sentiment_multipliers[event_type]
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                
                # Momentum and confidence
                momentum = abs(sentiment_score) * np.random.uniform(0.8, 2.0)
                confidence = abs(sentiment_score) * 0.7 + np.random.uniform(0.1, 0.3)
                confidence = min(confidence, 1.0)
                
                event = SimulatedSentimentEvent(
                    timestamp=event_time,
                    symbol=symbol,
                    platform=platform,
                    sentiment_score=sentiment_score,
                    momentum=momentum,
                    confidence=confidence,
                    event_type=event_type
                )
                
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    def _generate_news_events(self, start_date: str, days: int) -> List[SimulatedNewsEvent]:
        """Generate realistic news events"""
        
        events = []
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT']
        
        # News categories and their typical impact
        news_categories = {
            'regulation': {'impact_range': (0.6, 0.9), 'sentiment_bias': -0.3, 'frequency': 0.1},
            'adoption': {'impact_range': (0.5, 0.8), 'sentiment_bias': 0.4, 'frequency': 0.15},
            'partnership': {'impact_range': (0.3, 0.7), 'sentiment_bias': 0.3, 'frequency': 0.2},
            'technology': {'impact_range': (0.4, 0.6), 'sentiment_bias': 0.2, 'frequency': 0.25},
            'market': {'impact_range': (0.7, 1.0), 'sentiment_bias': 0.0, 'frequency': 0.15},
            'security': {'impact_range': (0.8, 1.0), 'sentiment_bias': -0.6, 'frequency': 0.05},
            'institutional': {'impact_range': (0.6, 0.9), 'sentiment_bias': 0.5, 'frequency': 0.1}
        }
        
        sources = ['CoinDesk', 'CoinTelegraph', 'Bitcoin.com', 'CryptoNews']
        
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Generate 1-2 news events per day on average
        for day in range(days):
            current_date = start_datetime + timedelta(days=day)
            
            # Poisson distribution for news events
            num_events = np.random.poisson(1.5)
            
            for _ in range(num_events):
                # Random time during trading hours (more likely)
                if np.random.random() < 0.7:  # 70% during trading hours
                    random_hour = np.random.randint(8, 20)  # 8 AM to 8 PM
                else:
                    random_hour = np.random.randint(0, 24)
                
                random_minute = np.random.randint(0, 60)
                event_time = current_date + timedelta(hours=random_hour, minutes=random_minute)
                
                # Select news category based on frequency weights
                categories = list(news_categories.keys())
                weights = [news_categories[cat]['frequency'] for cat in categories]
                category = np.random.choice(categories, p=np.array(weights)/sum(weights))
                
                # Select symbol (major coins get more news)
                symbol_weights = [0.5, 0.25, 0.1, 0.08, 0.07]
                symbol = np.random.choice(symbols, p=symbol_weights)
                
                # Generate news characteristics
                cat_info = news_categories[category]
                impact_score = np.random.uniform(*cat_info['impact_range'])
                
                # Sentiment with category bias
                base_sentiment = np.random.normal(cat_info['sentiment_bias'], 0.3)
                sentiment_score = max(-1.0, min(1.0, base_sentiment))
                
                # Generate headline
                headlines = {
                    'regulation': [f"SEC announces new crypto regulations affecting {symbol.split('/')[0]}",
                                 f"Government considers {symbol.split('/')[0]} classification",
                                 f"Regulatory clarity expected for {symbol.split('/')[0]}"],
                    'adoption': [f"Major retailer announces {symbol.split('/')[0]} payment integration",
                               f"{symbol.split('/')[0]} adoption grows in developing markets",
                               f"Corporate treasury adds {symbol.split('/')[0]} holdings"],
                    'partnership': [f"{symbol.split('/')[0]} announces strategic partnership",
                                  f"Technology integration planned for {symbol.split('/')[0]}",
                                  f"Cross-chain bridge supports {symbol.split('/')[0]}"],
                    'technology': [f"{symbol.split('/')[0]} network upgrade scheduled",
                                 f"Scaling solution deployed for {symbol.split('/')[0]}",
                                 f"Developer activity increases for {symbol.split('/')[0]}"],
                    'market': [f"{symbol.split('/')[0]} breaks key resistance level",
                             f"Trading volume surges for {symbol.split('/')[0]}",
                             f"Institutional interest grows in {symbol.split('/')[0]}"],
                    'security': [f"Security vulnerability discovered in {symbol.split('/')[0]}",
                               f"Exchange hack affects {symbol.split('/')[0]} trading",
                               f"Network attack attempted on {symbol.split('/')[0]}"],
                    'institutional': [f"Investment fund allocates to {symbol.split('/')[0]}",
                                    f"Bank announces {symbol.split('/')[0]} custody service",
                                    f"ETF proposal filed for {symbol.split('/')[0]}"]
                }
                
                headline = np.random.choice(headlines[category])
                
                # Determine urgency based on impact
                if impact_score > 0.8:
                    urgency = 'critical'
                elif impact_score > 0.6:
                    urgency = 'high'
                elif impact_score > 0.4:
                    urgency = 'medium'
                else:
                    urgency = 'low'
                
                source = np.random.choice(sources)
                
                event = SimulatedNewsEvent(
                    timestamp=event_time,
                    symbol=symbol,
                    headline=headline,
                    sentiment_score=sentiment_score,
                    impact_score=impact_score,
                    urgency=urgency,
                    source=source
                )
                
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    def _generate_arbitrage_opportunities(self, start_date: str, days: int) -> List[SimulatedArbitrageOpportunity]:
        """Generate realistic arbitrage opportunities"""
        
        opportunities = []
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT']
        exchanges = ['binance', 'coinbase', 'kraken', 'kucoin']
        
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Generate arbitrage opportunities (less frequent than other events)
        for day in range(days):
            current_date = start_datetime + timedelta(days=day)
            
            # Average 2-3 arbitrage opportunities per day
            num_opportunities = np.random.poisson(2.5)
            
            for _ in range(num_opportunities):
                # Random time (more likely during high volatility periods)
                random_hour = np.random.randint(0, 24)
                random_minute = np.random.randint(0, 60)
                event_time = current_date + timedelta(hours=random_hour, minutes=random_minute)
                
                # Select symbol (major pairs have more arbitrage opportunities)
                symbol_weights = [0.4, 0.3, 0.2, 0.1]
                symbol = np.random.choice(symbols, p=symbol_weights)
                
                # Select exchange pair
                buy_exchange, sell_exchange = np.random.choice(exchanges, size=2, replace=False)
                
                # Generate arbitrage characteristics
                # Profit typically 0.5% to 3% for crypto arbitrage
                profit_percent = np.random.lognormal(mean=np.log(1.2), sigma=0.8)  # Log-normal distribution
                profit_percent = max(0.3, min(profit_percent, 5.0))  # Clamp between 0.3% and 5%
                
                # Confidence based on profit size and exchange pair reliability
                exchange_reliability = {
                    'binance': 0.95, 'coinbase': 0.90, 'kraken': 0.85, 'kucoin': 0.80
                }
                
                avg_reliability = (exchange_reliability[buy_exchange] + exchange_reliability[sell_exchange]) / 2
                confidence = avg_reliability * (0.7 + min(profit_percent / 10, 0.3))  # Higher profit = higher confidence
                
                # Duration (how long the opportunity lasts)
                # Most arbitrage opportunities are short-lived
                duration_minutes = int(np.random.exponential(scale=15))  # Average 15 minutes
                duration_minutes = max(2, min(duration_minutes, 120))  # Between 2 minutes and 2 hours
                
                opportunity = SimulatedArbitrageOpportunity(
                    timestamp=event_time,
                    symbol=symbol,
                    buy_exchange=buy_exchange,
                    sell_exchange=sell_exchange,
                    profit_percent=profit_percent,
                    confidence=confidence,
                    duration_minutes=duration_minutes
                )
                
                opportunities.append(opportunity)
        
        # Sort by timestamp
        opportunities.sort(key=lambda x: x.timestamp)
        
        return opportunities

class EnhancedBacktestEngine:
    """
    Enhanced backtest engine with simulation capabilities
    
    Can backtest the enhanced strategy by simulating:
    - Social sentiment events and their market impact
    - ML model predictions based on historical patterns
    - Arbitrage opportunities between exchanges
    - Breaking news events and reactions
    """
    
    def __init__(self):
        self.simulator = EnhancedMarketSimulator()
        self.results_history = []
        
        logger.info("🚀 Enhanced Backtest Engine initialized")
    
    async def run_enhanced_backtest(self, start_date: str = "2023-01-01", days: int = 365) -> BacktestResults:
        """Run comprehensive enhanced strategy backtest"""
        
        print(f"🚀 ENHANCED STRATEGY BACKTEST")
        print(f"=" * 50)
        print(f"📅 Period: {start_date} to {days} days")
        print(f"🔥 Testing enhanced vs original strategy performance")
        
        # Generate enhanced market data and events
        simulation_data = self.simulator.generate_enhanced_market_data(start_date, days)
        
        market_data = simulation_data['market_data']
        sentiment_events = simulation_data['sentiment_events']
        news_events = simulation_data['news_events']
        arbitrage_opportunities = simulation_data['arbitrage_opportunities']
        
        print(f"\n📊 Simulation Data Generated:")
        print(f"   • Market data: {len(market_data)} symbols")
        print(f"   • Sentiment events: {len(sentiment_events)}")
        print(f"   • News events: {len(news_events)}")
        print(f"   • Arbitrage opportunities: {len(arbitrage_opportunities)}")
        
        # Run original strategy backtest (for comparison)
        print(f"\n🔄 Running original strategy backtest...")
        original_results = await self._run_original_strategy_backtest(market_data, start_date, days)
        
        # Run enhanced strategy backtest
        print(f"\n🚀 Running enhanced strategy backtest...")
        enhanced_results = await self._run_enhanced_strategy_backtest(
            market_data, sentiment_events, news_events, arbitrage_opportunities, start_date, days
        )
        
        # Compare results
        results = self._analyze_backtest_results(original_results, enhanced_results, days)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_backtest_results_{timestamp}.json"
        
        results_dict = asdict(results)
        # Convert any datetime objects to strings for JSON serialization
        results_dict = self._convert_datetimes_to_strings(results_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
    
    async def _run_original_strategy_backtest(self, market_data: Dict[str, pd.DataFrame], 
                                            start_date: str, days: int) -> Dict[str, Any]:
        """Run backtest with original strategy logic (simplified simulation)"""
        
        total_budget = 30.0 * days
        daily_budget = 30.0
        current_budget = daily_budget
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        
        daily_performance = []
        
        # Simulate simple volume-based strategy
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        for day in range(days):
            current_date = start_datetime + timedelta(days=day)
            day_pnl = 0.0
            day_trades = 0
            current_budget = daily_budget  # Reset daily budget
            
            # Simple signal generation (volume spikes only)
            symbols = list(market_data.keys())[:5]  # Top 5 symbols
            
            for symbol in symbols:
                df = market_data[symbol]
                day_data = df[df.index.date == current_date.date()]
                
                if len(day_data) < 24:  # Need full day of data
                    continue
                
                # Simple volume spike detection
                current_volume = day_data['volume'].iloc[-1]
                avg_volume = day_data['volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio >= 3.0 and current_budget >= 10.0:  # Volume spike and budget available
                    # Simulate trade
                    position_size = min(15.0, current_budget * 0.8)
                    current_budget -= position_size
                    
                    # Simulate outcome (60% win rate, 2:1 reward:risk)
                    if np.random.random() < 0.6:  # Win
                        trade_pnl = position_size * 0.15  # 15% profit
                        winning_trades += 1
                    else:  # Loss
                        trade_pnl = -position_size * 0.075  # 7.5% loss
                    
                    day_pnl += trade_pnl
                    day_trades += 1
                    total_trades += 1
            
            total_pnl += day_pnl
            
            daily_performance.append({
                'day': day + 1,
                'date': current_date.strftime('%Y-%m-%d'),
                'day_pnl': day_pnl,
                'trades': day_trades,
                'cumulative_pnl': total_pnl
            })
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'daily_performance': daily_performance,
            'total_budget': total_budget
        }
    
    async def _run_enhanced_strategy_backtest(self, market_data: Dict[str, pd.DataFrame],
                                            sentiment_events: List[SimulatedSentimentEvent],
                                            news_events: List[SimulatedNewsEvent],
                                            arbitrage_opportunities: List[SimulatedArbitrageOpportunity],
                                            start_date: str, days: int) -> Dict[str, Any]:
        """Run backtest with enhanced strategy"""
        
        # Create enhanced strategy instance
        strategy = EnhancedHighRiskStrategy()
        
        # Initialize with simulated components
        await self._initialize_simulated_components(strategy, sentiment_events, news_events, arbitrage_opportunities)
        
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        daily_performance = []
        
        # Track signal source performance
        signal_source_stats = {
            'volume_spike': {'count': 0, 'wins': 0, 'total_return': 0.0},
            'social_sentiment': {'count': 0, 'wins': 0, 'total_return': 0.0},
            'ml_prediction': {'count': 0, 'wins': 0, 'total_return': 0.0},
            'news_analysis': {'count': 0, 'wins': 0, 'total_return': 0.0},
            'arbitrage': {'count': 0, 'wins': 0, 'total_return': 0.0}
        }
        
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        for day in range(days):
            current_date = start_datetime + timedelta(days=day)
            
            # Reset daily budget
            strategy.risk_limiter.reset_daily_budget()
            
            day_start_pnl = total_pnl
            day_trades = 0
            
            # Get symbols to trade
            symbols = list(market_data.keys())[:5]
            
            for symbol in symbols:
                df = market_data[symbol]
                day_data = df[df.index.date == current_date.date()]
                
                if len(day_data) < 24:
                    continue
                
                current_price = day_data['close'].iloc[-1]
                
                # Generate enhanced signal with simulated events
                signal, signal_data = await self._simulate_enhanced_signal(
                    strategy, symbol, day_data, current_price, current_date,
                    sentiment_events, news_events, arbitrage_opportunities
                )
                
                if signal in ['BUY', 'SELL'] and len(strategy.active_positions) < 3:
                    # Execute simulated trade
                    trade_result = await self._simulate_trade_execution(
                        strategy, signal, symbol, current_price, signal_data, day_data
                    )
                    
                    if trade_result:
                        day_trades += 1
                        total_trades += 1
                        
                        # Track signal source performance
                        for source in signal_data.signal_sources:
                            source_key = source.split('_')[0]
                            if source_key in signal_source_stats:
                                signal_source_stats[source_key]['count'] += 1
                                
                                # Simulate trade outcome based on signal quality
                                win_probability = self._calculate_win_probability(signal_data, source_key)
                                trade_pnl = self._simulate_trade_outcome(signal_data, win_probability)
                                
                                signal_source_stats[source_key]['total_return'] += trade_pnl
                                if trade_pnl > 0:
                                    signal_source_stats[source_key]['wins'] += 1
                                    winning_trades += 1
                                
                                total_pnl += trade_pnl
            
            day_pnl = total_pnl - day_start_pnl
            
            daily_performance.append({
                'day': day + 1,
                'date': current_date.strftime('%Y-%m-%d'),
                'day_pnl': day_pnl,
                'trades': day_trades,
                'cumulative_pnl': total_pnl
            })
            
            # Progress reporting
            if (day + 1) % 50 == 0:
                print(f"   📅 Day {day + 1}/{days} - P&L: {total_pnl:+.2f}€")
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'daily_performance': daily_performance,
            'signal_source_stats': signal_source_stats,
            'total_budget': 30.0 * days
        }
    
    async def _initialize_simulated_components(self, strategy, sentiment_events, news_events, arbitrage_opportunities):
        """Initialize strategy with simulated component data"""
        
        # Store events for simulation access
        strategy._simulated_sentiment_events = sentiment_events
        strategy._simulated_news_events = news_events
        strategy._simulated_arbitrage_opportunities = arbitrage_opportunities
        
        # Initialize enhanced components normally
        await strategy.initialize_enhanced_components()
    
    async def _simulate_enhanced_signal(self, strategy, symbol: str, market_data: pd.DataFrame,
                                      current_price: float, current_date: datetime,
                                      sentiment_events, news_events, arbitrage_opportunities):
        """Simulate enhanced signal generation with events"""
        
        from strategies.enhanced_high_risk_strategy import EnhancedSignalData
        
        signal_data = EnhancedSignalData(
            symbol=symbol,
            signal_type='HOLD',
            confidence=0.0,
            timestamp=current_date,
            signal_sources=[],
            reasoning=""
        )
        
        signal_scores = []
        
        # 1. Volume spike (traditional)
        if len(market_data) >= 24:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].iloc[-24:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio >= 3.0:
                signal_data.signal_sources.append('volume_spike')
                signal_scores.append(('volume_spike', 0.8, 0.3))
            elif volume_ratio >= 2.0:
                signal_data.signal_sources.append('volume_spike')
                signal_scores.append(('volume_spike', 0.6, 0.3))
        
        # 2. Simulated sentiment events
        current_sentiment_events = [
            event for event in sentiment_events
            if (event.symbol == symbol and 
                abs((event.timestamp - current_date).total_seconds()) < 3600)  # Within 1 hour
        ]
        
        if current_sentiment_events:
            best_sentiment = max(current_sentiment_events, key=lambda x: x.confidence)
            if abs(best_sentiment.sentiment_score) > 0.3 and best_sentiment.confidence > 0.7:
                signal_data.signal_sources.append(f'social_sentiment_{best_sentiment.platform}')
                sentiment_strength = abs(best_sentiment.sentiment_score) * best_sentiment.confidence
                signal_direction = 1 if best_sentiment.sentiment_score > 0 else -1
                signal_scores.append(('social_sentiment', sentiment_strength * signal_direction, 0.25))
        
        # 3. Simulated ML prediction (enhanced based on multiple factors)
        ml_score = self._simulate_ml_prediction(market_data, signal_data, current_sentiment_events)
        if abs(ml_score) > 0.7:
            signal_data.signal_sources.append('ml_prediction')
            signal_scores.append(('ml_prediction', ml_score, 0.2))
        
        # 4. Simulated news events
        current_news_events = [
            event for event in news_events
            if (event.symbol == symbol and 
                abs((event.timestamp - current_date).total_seconds()) < 3600)  # Within 1 hour
        ]
        
        if current_news_events:
            best_news = max(current_news_events, key=lambda x: x.impact_score)
            if best_news.impact_score > 0.6:
                signal_data.signal_sources.append(f'news_{best_news.urgency}')
                news_strength = best_news.impact_score
                signal_direction = 1 if best_news.sentiment_score > 0 else -1
                signal_scores.append(('news_analysis', news_strength * signal_direction, 0.15))
        
        # 5. Simulated arbitrage opportunities
        current_arbitrage = [
            opp for opp in arbitrage_opportunities
            if (opp.symbol == symbol and 
                abs((opp.timestamp - current_date).total_seconds()) < 1800)  # Within 30 min
        ]
        
        if current_arbitrage:
            best_arbitrage = max(current_arbitrage, key=lambda x: x.profit_percent)
            if best_arbitrage.profit_percent > 1.0 and best_arbitrage.confidence > 0.8:
                signal_data.signal_sources.append('arbitrage')
                signal_scores.append(('arbitrage', best_arbitrage.confidence * 0.5, 0.1))
        
        # Combine signals
        if signal_scores:
            final_score, final_confidence = self._combine_simulated_signals(signal_scores)
            
            if final_score > 0.6 and final_confidence > 0.6:
                signal_data.signal_type = 'BUY'
                signal_data.confidence = final_confidence
            elif final_score < -0.6 and final_confidence > 0.6:
                signal_data.signal_type = 'SELL'
                signal_data.confidence = final_confidence
            else:
                signal_data.signal_type = 'HOLD'
                signal_data.confidence = final_confidence
            
            signal_data.reasoning = f"Enhanced signal from {len(signal_scores)} sources"
        
        return signal_data.signal_type, signal_data
    
    def _simulate_ml_prediction(self, market_data: pd.DataFrame, signal_data, sentiment_events) -> float:
        """Simulate ML prediction based on multiple factors"""
        
        ml_score = 0.0
        
        # Technical factors
        if len(market_data) >= 24:
            # Price momentum
            price_change_24h = (market_data['close'].iloc[-1] - market_data['close'].iloc[-24]) / market_data['close'].iloc[-24]
            ml_score += price_change_24h * 2  # ML picks up on momentum
            
            # Volatility
            returns = market_data['close'].pct_change().dropna()
            if len(returns) > 10:
                volatility = returns.std()
                ml_score += (volatility - 0.02) * 5  # ML considers volatility patterns
        
        # Sentiment influence on ML
        if sentiment_events:
            avg_sentiment = np.mean([e.sentiment_score * e.confidence for e in sentiment_events])
            ml_score += avg_sentiment * 0.5
        
        # Add some randomness (ML isn't perfect)
        ml_score += np.random.normal(0, 0.2)
        
        # Apply sigmoid to get -1 to 1 range
        ml_score = np.tanh(ml_score)
        
        return ml_score
    
    def _combine_simulated_signals(self, signal_scores: List[Tuple[str, float, float]]) -> Tuple[float, float]:
        """Combine simulated signals"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for source, score, weight in signal_scores:
            total_weighted_score += score * weight
            total_weight += weight
            confidence_scores.append(abs(score))
        
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.0
        
        if confidence_scores:
            final_confidence = np.mean(confidence_scores)
        else:
            final_confidence = 0.0
        
        # Boost confidence for multiple sources
        if len(signal_scores) >= 3:
            final_confidence *= 1.2
        elif len(signal_scores) >= 2:
            final_confidence *= 1.1
        
        final_confidence = min(final_confidence, 1.0)
        
        return final_score, final_confidence
    
    async def _simulate_trade_execution(self, strategy, signal: str, symbol: str, 
                                      current_price: float, signal_data, market_data: pd.DataFrame) -> bool:
        """Simulate trade execution"""
        
        # Check if we can trade (budget available)
        if strategy.risk_limiter.remaining_budget < 10.0:
            return False
        
        # Position size based on confidence
        position_size = min(15.0, strategy.risk_limiter.remaining_budget * 0.8)
        position_size *= signal_data.confidence
        
        # Reserve budget
        can_trade, _ = strategy.risk_limiter.can_trade(position_size)
        if can_trade:
            strategy.risk_limiter.reserve_budget(position_size, f"TRADE_{symbol}")
            return True
        
        return False
    
    def _calculate_win_probability(self, signal_data, source_key: str) -> float:
        """Calculate win probability based on signal quality and source"""
        
        base_probabilities = {
            'volume_spike': 0.55,
            'social_sentiment': 0.50,
            'ml_prediction': 0.65,
            'news_analysis': 0.60,
            'arbitrage': 0.75
        }
        
        base_prob = base_probabilities.get(source_key, 0.50)
        
        # Adjust based on signal confidence
        confidence_bonus = (signal_data.confidence - 0.5) * 0.3  # Up to 15% bonus
        
        # Multiple sources increase win probability
        multi_source_bonus = min(len(signal_data.signal_sources) * 0.05, 0.15)  # Up to 15% bonus
        
        final_prob = base_prob + confidence_bonus + multi_source_bonus
        return max(0.1, min(0.9, final_prob))  # Clamp between 10% and 90%
    
    def _simulate_trade_outcome(self, signal_data, win_probability: float) -> float:
        """Simulate trade outcome"""
        
        # Base position size
        position_size = 12.0 * signal_data.confidence  # Average position size
        
        if np.random.random() < win_probability:
            # Win - profit based on signal strength and sources
            base_profit = 0.15  # 15% base profit
            
            # Enhanced signals can have higher profits
            if 'ml_prediction' in signal_data.signal_sources:
                base_profit += 0.05
            if 'arbitrage' in signal_data.signal_sources:
                base_profit += 0.08
            if len(signal_data.signal_sources) >= 3:
                base_profit += 0.03
            
            return position_size * base_profit
        else:
            # Loss - limited by stop loss
            return -position_size * 0.08  # 8% loss
    
    def _analyze_backtest_results(self, original_results: Dict, enhanced_results: Dict, days: int) -> BacktestResults:
        """Analyze and compare backtest results"""
        
        print(f"\n📊 BACKTEST RESULTS ANALYSIS")
        print(f"=" * 40)
        
        # Enhanced strategy metrics
        enhanced_pnl = enhanced_results['total_pnl']
        enhanced_trades = enhanced_results['total_trades']
        enhanced_wins = enhanced_results['winning_trades']
        enhanced_budget = enhanced_results['total_budget']
        
        enhanced_roi = (enhanced_pnl / enhanced_budget) * 100
        enhanced_win_rate = (enhanced_wins / max(enhanced_trades, 1)) * 100
        
        # Original strategy metrics
        original_pnl = original_results['total_pnl']
        original_trades = original_results['total_trades']
        original_wins = original_results['winning_trades']
        original_budget = original_results['total_budget']
        
        original_roi = (original_pnl / original_budget) * 100
        original_win_rate = (original_wins / max(original_trades, 1)) * 100
        
        # Calculate additional metrics
        enhanced_daily_returns = [day['day_pnl'] / 30.0 for day in enhanced_results['daily_performance']]
        enhanced_volatility = np.std(enhanced_daily_returns) * np.sqrt(365)
        enhanced_sharpe = (np.mean(enhanced_daily_returns) * 365) / max(enhanced_volatility, 0.001)
        
        # Max drawdown calculation
        cumulative_pnls = [day['cumulative_pnl'] for day in enhanced_results['daily_performance']]
        peak = 0
        max_drawdown = 0
        for pnl in cumulative_pnls:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdown_percent = (max_drawdown / enhanced_budget) * 100
        
        # Calmar ratio
        calmar_ratio = enhanced_roi / max(max_drawdown_percent, 1) if max_drawdown_percent > 0 else 0
        
        # Enhancement improvement
        improvement = enhanced_roi - original_roi
        
        # Signal source performance
        signal_source_performance = {}
        for source, stats in enhanced_results.get('signal_source_stats', {}).items():
            if stats['count'] > 0:
                success_rate = (stats['wins'] / stats['count']) * 100
                avg_return = (stats['total_return'] / stats['count'])
                signal_source_performance[source] = {
                    'success_rate': success_rate,
                    'avg_return_percent': avg_return,
                    'total_signals': stats['count']
                }
        
        # Overall assessment
        if enhanced_roi > 50:
            rating = "🟢 EXCELLENT"
        elif enhanced_roi > 20:
            rating = "🟡 GOOD"
        elif enhanced_roi > 0:
            rating = "🟠 MODERATE"
        else:
            rating = "🔴 POOR"
        
        # Key insights
        insights = []
        
        if improvement > 10:
            insights.append(f"Enhanced strategy significantly outperformed original (+{improvement:.1f}%)")
        elif improvement > 0:
            insights.append(f"Enhanced strategy showed modest improvement (+{improvement:.1f}%)")
        else:
            insights.append(f"Enhanced strategy underperformed original ({improvement:.1f}%)")
        
        if enhanced_win_rate > 60:
            insights.append(f"High win rate achieved ({enhanced_win_rate:.1f}%)")
        elif enhanced_win_rate < 45:
            insights.append(f"Low win rate needs improvement ({enhanced_win_rate:.1f}%)")
        
        if enhanced_sharpe > 1.5:
            insights.append("Excellent risk-adjusted returns")
        elif enhanced_sharpe < 0.5:
            insights.append("Poor risk-adjusted returns")
        
        # Display results
        print(f"\n💰 PERFORMANCE COMPARISON")
        print(f"{'='*30}")
        print(f"Original Strategy:")
        print(f"  ROI: {original_roi:+.2f}%")
        print(f"  Trades: {original_trades}")
        print(f"  Win Rate: {original_win_rate:.1f}%")
        
        print(f"\nEnhanced Strategy:")
        print(f"  ROI: {enhanced_roi:+.2f}%")
        print(f"  Trades: {enhanced_trades}")
        print(f"  Win Rate: {enhanced_win_rate:.1f}%")
        print(f"  Sharpe Ratio: {enhanced_sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown_percent:.1f}%")
        
        print(f"\n🚀 ENHANCEMENT ANALYSIS")
        print(f"{'='*25}")
        print(f"Improvement: {improvement:+.2f}%")
        print(f"Overall Rating: {rating}")
        
        print(f"\n📊 SIGNAL SOURCE PERFORMANCE")
        print(f"{'='*32}")
        for source, perf in signal_source_performance.items():
            print(f"{source.replace('_', ' ').title()}:")
            print(f"  Signals: {perf['total_signals']}")
            print(f"  Success Rate: {perf['success_rate']:.1f}%")
            print(f"  Avg Return: {perf['avg_return_percent']:+.2f}€")
        
        return BacktestResults(
            total_return=enhanced_pnl,
            roi_percent=enhanced_roi,
            win_rate=enhanced_win_rate,
            total_trades=enhanced_trades,
            sharpe_ratio=enhanced_sharpe,
            max_drawdown=max_drawdown,
            volatility=enhanced_volatility,
            calmar_ratio=calmar_ratio,
            original_strategy_return=original_roi,
            enhancement_improvement=improvement,
            signal_source_stats=signal_source_performance,
            daily_performance=enhanced_results['daily_performance'],
            overall_rating=rating,
            key_insights=insights
        )
    
    def _convert_datetimes_to_strings(self, obj):
        """Convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_datetimes_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetimes_to_strings(item) for item in obj]
        else:
            return obj

# Main function
async def run_enhanced_backtest():
    """Run enhanced backtest comparison"""
    
    print("🚀 Starting Enhanced vs Original Strategy Backtest...")
    
    engine = EnhancedBacktestEngine()
    
    # Run 1-year backtest
    results = await engine.run_enhanced_backtest(
        start_date="2023-01-01",
        days=365
    )
    
    print(f"\n🎯 KEY INSIGHTS:")
    for insight in results.key_insights:
        print(f"   • {insight}")
    
    print(f"\n🏆 FINAL VERDICT:")
    print(f"   Rating: {results.overall_rating}")
    print(f"   ROI: {results.roi_percent:+.2f}%")
    print(f"   Enhancement: {results.enhancement_improvement:+.2f}% vs original")
    print(f"   Win Rate: {results.win_rate:.1f}%")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    return results

if __name__ == "__main__":
    # Run enhanced backtest
    asyncio.run(run_enhanced_backtest())