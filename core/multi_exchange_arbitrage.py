#!/usr/bin/env python3
"""
Multi-Exchange Arbitrage Detection System
=========================================

Real-time arbitrage opportunity detection across multiple exchanges:
- Price difference analysis between exchanges
- Liquidity assessment for arbitrage execution
- Transaction cost calculations
- Risk-adjusted arbitrage scoring
- Real-time opportunity alerts
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ExchangePrice:
    """Price data from a specific exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    mid_price: float
    volume_24h: float
    timestamp: datetime
    spread_percent: float
    liquidity_score: float
    
    def __post_init__(self):
        if self.mid_price == 0:
            self.mid_price = (self.bid + self.ask) / 2
        if self.spread_percent == 0:
            self.spread_percent = ((self.ask - self.bid) / self.mid_price) * 100

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    price_difference: float
    profit_percent: float
    estimated_profit: float
    volume_available: float
    confidence: float
    timestamp: datetime
    execution_cost: float
    net_profit: float
    risk_score: float
    expiry_time: datetime
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expiry_time

@dataclass 
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_base_url: str
    trading_fee: float  # Percentage
    withdrawal_fee: Dict[str, float]  # Per asset
    min_trade_amount: Dict[str, float]  # Per asset
    deposit_time: int  # Minutes
    withdrawal_time: int  # Minutes
    supported_symbols: List[str]
    rate_limit: int  # Requests per minute

class ExchangeConnector:
    """
    Generic exchange connector for price data
    
    Handles API communication, rate limiting, and error handling
    """
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.session = None
        self.rate_limiter = deque(maxlen=config.rate_limit)
        self.last_prices = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_ticker(self, symbol: str) -> Optional[ExchangePrice]:
        """Get ticker data for a symbol"""
        
        if not await self._check_rate_limit():
            logger.warning(f"Rate limit exceeded for {self.config.name}")
            return None
        
        try:
            # Simulate API call with realistic data
            price_data = await self._fetch_ticker_data(symbol)
            
            if price_data:
                return ExchangePrice(
                    exchange=self.config.name,
                    symbol=symbol,
                    bid=price_data['bid'],
                    ask=price_data['ask'],
                    mid_price=price_data['price'],
                    volume_24h=price_data['volume'],  
                    timestamp=datetime.now(),
                    spread_percent=0.0,  # Will be calculated in __post_init__
                    liquidity_score=self._calculate_liquidity_score(price_data)
                )
                
        except Exception as e:
            logger.error(f"Error fetching {symbol} from {self.config.name}: {e}")
        
        return None
    
    async def _fetch_ticker_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker data from exchange API (simulated)"""
        
        # Simulate different exchange behaviors and prices
        base_price = self._get_base_price(symbol)
        
        # Each exchange has slightly different prices due to market dynamics
        exchange_variance = {
            'binance': 0.0,      # Reference exchange
            'coinbase': 0.002,   # Usually slightly higher
            'kraken': -0.001,    # Usually slightly lower  
            'huobi': 0.001,      # Small premium
            'kucoin': -0.002,    # Small discount
            'gate': 0.003,       # Higher variance
            'bybit': 0.0005,     # Small premium
            'okx': -0.0005       # Small discount
        }
        
        variance = exchange_variance.get(self.config.name.lower(), 0.0)
        
        # Add some randomness to simulate real market conditions
        random_factor = np.random.normal(0, 0.001)  # 0.1% standard deviation
        total_variance = variance + random_factor
        
        adjusted_price = base_price * (1 + total_variance)
        
        # Calculate bid/ask spread (typically 0.05-0.2%)
        spread_percent = np.random.uniform(0.0005, 0.002)  # 0.05-0.2%
        spread = adjusted_price * spread_percent
        
        bid = adjusted_price - spread / 2
        ask = adjusted_price + spread / 2
        
        # Volume varies by exchange
        base_volume = 1000000
        volume_multiplier = {
            'binance': 5.0,
            'coinbase': 2.0, 
            'kraken': 1.5,
            'huobi': 1.2,
            'kucoin': 0.8,
            'gate': 0.6,
            'bybit': 1.0,
            'okx': 1.3
        }.get(self.config.name.lower(), 1.0)
        
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
        
        return {
            'bid': bid,
            'ask': ask,
            'price': adjusted_price,
            'volume': volume
        }
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol (simulated current market price)"""
        
        # Realistic current crypto prices (simulated)
        base_prices = {
            'BTC/USDT': 67500.0,
            'ETH/USDT': 3800.0,
            'SOL/USDT': 180.0,
            'AVAX/USDT': 42.0,
            'MATIC/USDT': 0.85,
            'ADA/USDT': 0.48,
            'DOT/USDT': 7.2,
            'ATOM/USDT': 11.5,
            'NEAR/USDT': 5.8,
            'FTM/USDT': 0.72
        }
        
        return base_prices.get(symbol, 1000.0)
    
    def _calculate_liquidity_score(self, price_data: Dict[str, Any]) -> float:
        """Calculate liquidity score (0-1)"""
        
        volume = price_data['volume']
        price = price_data['price']
        
        # Higher volume and lower spread = higher liquidity
        volume_score = min(volume / 10000000, 1.0)  # Normalize to 10M volume
        spread = (price_data['ask'] - price_data['bid']) / price
        spread_score = max(0, 1 - (spread / 0.01))  # Penalty for spreads > 1%
        
        return (volume_score * 0.7) + (spread_score * 0.3)
    
    async def _check_rate_limit(self) -> bool:
        """Check if we can make an API call"""
        
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        while self.rate_limiter and self.rate_limiter[0] < current_time - 60:
            self.rate_limiter.popleft()
        
        # Check if under limit
        if len(self.rate_limiter) >= self.config.rate_limit:
            return False
        
        # Add current request
        self.rate_limiter.append(current_time)
        return True

class MultiExchangeArbitrageDetector:
    """
    Multi-exchange arbitrage detection system
    
    Monitors price differences across exchanges and identifies profitable opportunities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize exchange configurations
        self.exchanges = self._initialize_exchanges()
        self.connectors = {}
        
        # Arbitrage detection parameters
        self.min_profit_percent = self.config.get('min_profit_percent', 0.5)  # 0.5%
        self.min_profit_amount = self.config.get('min_profit_amount', 10.0)   # $10
        self.max_execution_time = self.config.get('max_execution_time', 300)  # 5 minutes
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Data storage
        self.price_data = defaultdict(dict)  # {symbol: {exchange: ExchangePrice}}
        self.arbitrage_history = deque(maxlen=1000)
        self.active_opportunities = {}
        
        # Performance tracking
        self.detection_stats = {
            'opportunities_found': 0,
            'total_checks': 0,
            'avg_profit_percent': 0.0,
            'best_opportunity': None
        }
        
        logger.info("🔄 Multi-Exchange Arbitrage Detector initialized")
    
    def _initialize_exchanges(self) -> Dict[str, ExchangeConfig]:
        """Initialize exchange configurations"""
        
        exchanges = {
            'binance': ExchangeConfig(
                name='binance',
                api_base_url='https://api.binance.com',
                trading_fee=0.1,  # 0.1%
                withdrawal_fee={'BTC': 0.0005, 'ETH': 0.005, 'USDT': 1.0},
                min_trade_amount={'BTC': 0.00001, 'ETH': 0.001, 'USDT': 10.0},
                deposit_time=10,
                withdrawal_time=30,
                supported_symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT'],
                rate_limit=1200  # per minute
            ),
            'coinbase': ExchangeConfig(
                name='coinbase',
                api_base_url='https://api.exchange.coinbase.com',
                trading_fee=0.5,  # 0.5%
                withdrawal_fee={'BTC': 0.0004, 'ETH': 0.01, 'USDT': 0.0},
                min_trade_amount={'BTC': 0.001, 'ETH': 0.01, 'USDT': 1.0},
                deposit_time=15,
                withdrawal_time=60,
                supported_symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                rate_limit=10  # per second = 600 per minute
            ),
            'kraken': ExchangeConfig(
                name='kraken',
                api_base_url='https://api.kraken.com',
                trading_fee=0.26,  # 0.26%
                withdrawal_fee={'BTC': 0.00015, 'ETH': 0.0025, 'USDT': 5.0},
                min_trade_amount={'BTC': 0.0001, 'ETH': 0.005, 'USDT': 5.0},
                deposit_time=20,
                withdrawal_time=45,
                supported_symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT'],
                rate_limit=60  # per minute
            ),
            'kucoin': ExchangeConfig(
                name='kucoin',
                api_base_url='https://api.kucoin.com',
                trading_fee=0.1,  # 0.1%
                withdrawal_fee={'BTC': 0.0005, 'ETH': 0.01, 'USDT': 2.0},
                min_trade_amount={'BTC': 0.00001, 'ETH': 0.0001, 'USDT': 0.1},
                deposit_time=25,
                withdrawal_time=35,
                supported_symbols=['BTC/USDT', 'ETH/USDT', 'MATIC/USDT', 'ATOM/USDT'],
                rate_limit=200  # per minute
            )
        }
        
        # Filter exchanges based on config
        enabled_exchanges = self.config.get('enabled_exchanges', list(exchanges.keys()))
        return {name: config for name, config in exchanges.items() if name in enabled_exchanges}
    
    async def initialize_connectors(self):
        """Initialize exchange connectors"""
        
        for exchange_name, exchange_config in self.exchanges.items():
            connector = ExchangeConnector(exchange_config)
            self.connectors[exchange_name] = connector
        
        logger.info(f"✅ Initialized {len(self.connectors)} exchange connectors")
    
    async def detect_arbitrage_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across all exchanges"""
        
        if not self.connectors:
            await self.initialize_connectors()
        
        opportunities = []
        
        # Collect price data from all exchanges
        await self._collect_price_data(symbols)
        
        # Analyze each symbol for arbitrage opportunities
        for symbol in symbols:
            symbol_opportunities = await self._analyze_symbol_arbitrage(symbol)
            opportunities.extend(symbol_opportunities)
        
        # Filter and rank opportunities
        filtered_opportunities = self._filter_opportunities(opportunities)
        ranked_opportunities = self._rank_opportunities(filtered_opportunities)
        
        # Update statistics
        self.detection_stats['total_checks'] += 1
        self.detection_stats['opportunities_found'] += len(ranked_opportunities)
        
        if ranked_opportunities:
            avg_profit = np.mean([opp.profit_percent for opp in ranked_opportunities])
            self.detection_stats['avg_profit_percent'] = avg_profit
            
            best_opp = max(ranked_opportunities, key=lambda x: x.net_profit)
            if (not self.detection_stats['best_opportunity'] or 
                best_opp.net_profit > self.detection_stats['best_opportunity'].net_profit):
                self.detection_stats['best_opportunity'] = best_opp
        
        # Store opportunities
        for opp in ranked_opportunities:
            self.active_opportunities[f"{opp.symbol}_{opp.buy_exchange}_{opp.sell_exchange}"] = opp
            self.arbitrage_history.append(opp)
        
        logger.info(f"🔍 Detected {len(ranked_opportunities)} arbitrage opportunities")
        
        return ranked_opportunities
    
    async def _collect_price_data(self, symbols: List[str]):
        """Collect price data from all exchanges"""
        
        tasks = []
        
        for symbol in symbols:
            for exchange_name, connector in self.connectors.items():
                if symbol in self.exchanges[exchange_name].supported_symbols:
                    task = self._fetch_exchange_price(connector, symbol)
                    tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, ExchangePrice):
                self.price_data[result.symbol][result.exchange] = result
            elif isinstance(result, Exception):
                logger.warning(f"Price fetch failed: {result}")
    
    async def _fetch_exchange_price(self, connector: ExchangeConnector, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from a single exchange"""
        
        try:
            async with connector:
                price_data = await connector.get_ticker(symbol)
                return price_data
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from {connector.config.name}: {e}")
            return None
    
    async def _analyze_symbol_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Analyze arbitrage opportunities for a single symbol"""
        
        opportunities = []
        
        if symbol not in self.price_data or len(self.price_data[symbol]) < 2:
            return opportunities
        
        exchanges_data = self.price_data[symbol]
        exchange_names = list(exchanges_data.keys())
        
        # Compare all exchange pairs
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                
                exchange1 = exchange_names[i]
                exchange2 = exchange_names[j]
                
                price1 = exchanges_data[exchange1]
                price2 = exchanges_data[exchange2]
                
                # Check both directions
                opportunities.extend([
                    self._calculate_arbitrage(symbol, price1, price2),  # Buy on 1, sell on 2
                    self._calculate_arbitrage(symbol, price2, price1)   # Buy on 2, sell on 1
                ])
        
        # Filter out None results
        return [opp for opp in opportunities if opp is not None]
    
    def _calculate_arbitrage(self, symbol: str, buy_price: ExchangePrice, 
                           sell_price: ExchangePrice) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity between two exchanges"""
        
        # Basic price difference
        price_diff = sell_price.bid - buy_price.ask
        if price_diff <= 0:
            return None
        
        profit_percent = (price_diff / buy_price.ask) * 100
        
        # Calculate execution costs
        buy_exchange_config = self.exchanges[buy_price.exchange]
        sell_exchange_config = self.exchanges[sell_price.exchange]
        
        # Trading fees
        buy_fee_percent = buy_exchange_config.trading_fee
        sell_fee_percent = sell_exchange_config.trading_fee
        
        # Withdrawal fees (simplified - assuming USDT)
        withdrawal_fee = buy_exchange_config.withdrawal_fee.get('USDT', 0)
        
        # Calculate available volume (limited by liquidity)
        available_volume = min(
            buy_price.volume_24h * 0.01,  # Max 1% of daily volume
            sell_price.volume_24h * 0.01,
            100000  # Max $100k position
        )
        
        # Calculate profits and costs
        trade_amount = min(available_volume, 50000)  # Max $50k for high-risk strategy
        
        buy_cost = trade_amount * (1 + buy_fee_percent / 100)
        sell_revenue = trade_amount * (sell_price.bid / buy_price.ask) * (1 - sell_fee_percent / 100)
        
        gross_profit = sell_revenue - buy_cost
        execution_cost = withdrawal_fee + (trade_amount * 0.001)  # Est. other costs
        net_profit = gross_profit - execution_cost
        
        # Risk assessment
        risk_score = self._calculate_risk_score(buy_price, sell_price, trade_amount)
        
        # Confidence based on liquidity and spread stability
        confidence = self._calculate_confidence(buy_price, sell_price, profit_percent)
        
        # Only return if profitable and meets minimum thresholds
        if (net_profit > self.min_profit_amount and 
            profit_percent > self.min_profit_percent and
            confidence >= self.min_confidence):
            
            return ArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=buy_price.exchange,
                sell_exchange=sell_price.exchange,
                buy_price=buy_price.ask,
                sell_price=sell_price.bid,
                price_difference=price_diff,
                profit_percent=profit_percent,
                estimated_profit=gross_profit,
                volume_available=available_volume,
                confidence=confidence,
                timestamp=datetime.now(),
                execution_cost=execution_cost,
                net_profit=net_profit,
                risk_score=risk_score,
                expiry_time=datetime.now() + timedelta(seconds=self.max_execution_time)
            )
        
        return None
    
    def _calculate_risk_score(self, buy_price: ExchangePrice, sell_price: ExchangePrice, 
                             trade_amount: float) -> float:
        """Calculate risk score for arbitrage opportunity (0=low risk, 1=high risk)"""
        
        risk_score = 0.0
        
        # Exchange reliability risk
        exchange_reliability = {
            'binance': 0.1, 'coinbase': 0.2, 'kraken': 0.3, 'kucoin': 0.4,
            'huobi': 0.5, 'gate': 0.6, 'bybit': 0.3, 'okx': 0.4
        }
        
        buy_risk = exchange_reliability.get(buy_price.exchange, 0.5)
        sell_risk = exchange_reliability.get(sell_price.exchange, 0.5)
        risk_score += (buy_risk + sell_risk) / 2 * 0.3
        
        # Liquidity risk
        if buy_price.liquidity_score < 0.5 or sell_price.liquidity_score < 0.5:
            risk_score += 0.2
        
        # Size risk
        if trade_amount > 25000:  # Large trades
            risk_score += 0.2
        
        # Time risk (based on exchange deposit/withdrawal times)
        buy_config = self.exchanges[buy_price.exchange]
        sell_config = self.exchanges[sell_price.exchange]
        total_time = buy_config.withdrawal_time + sell_config.deposit_time
        
        if total_time > 60:  # More than 1 hour
            risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    def _calculate_confidence(self, buy_price: ExchangePrice, sell_price: ExchangePrice, 
                             profit_percent: float) -> float:
        """Calculate confidence in arbitrage opportunity"""
        
        confidence = 0.5  # Base confidence
        
        # Liquidity confidence
        liquidity_conf = (buy_price.liquidity_score + sell_price.liquidity_score) / 2
        confidence += liquidity_conf * 0.3
        
        # Profit margin confidence
        if profit_percent > 2.0:  # High profit opportunities
            confidence += 0.2
        elif profit_percent > 1.0:
            confidence += 0.1
        
        # Spread stability (lower spreads = higher confidence)
        avg_spread = (buy_price.spread_percent + sell_price.spread_percent) / 2
        if avg_spread < 0.1:  # Very tight spreads
            confidence += 0.2
        elif avg_spread < 0.3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria"""
        
        filtered = []
        
        for opp in opportunities:
            # Remove expired opportunities
            if opp.is_expired():
                continue
            
            # Minimum profit requirements
            if opp.net_profit < self.min_profit_amount:
                continue
            
            if opp.profit_percent < self.min_profit_percent:
                continue
            
            # Minimum confidence
            if opp.confidence < self.min_confidence:
                continue
            
            # Risk limits
            if opp.risk_score > 0.8:  # Very high risk
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by attractiveness"""
        
        # Composite scoring
        for opp in opportunities:
            # Score combines profit, confidence, and inverse risk
            opp_score = (
                opp.net_profit * 0.4 +           # Absolute profit weight
                opp.profit_percent * 10 * 0.3 +  # Percentage profit weight  
                opp.confidence * 100 * 0.2 +     # Confidence weight
                (1 - opp.risk_score) * 50 * 0.1  # Inverse risk weight
            )
            opp.score = opp_score
        
        # Sort by score descending
        return sorted(opportunities, key=lambda x: getattr(x, 'score', 0), reverse=True)
    
    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get currently active arbitrage opportunities"""
        
        # Remove expired opportunities
        current_time = datetime.now()
        expired_keys = [
            key for key, opp in self.active_opportunities.items()
            if opp.expiry_time < current_time
        ]
        
        for key in expired_keys:
            del self.active_opportunities[key]
        
        return list(self.active_opportunities.values())
    
    def get_best_opportunities(self, limit: int = 5) -> List[ArbitrageOpportunity]:
        """Get best arbitrage opportunities"""
        
        active = self.get_active_opportunities()
        ranked = self._rank_opportunities(active)
        
        return ranked[:limit]
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get arbitrage detection summary"""
        
        active_opportunities = self.get_active_opportunities()
        
        summary = {
            'total_checks': self.detection_stats['total_checks'],
            'opportunities_found': self.detection_stats['opportunities_found'],
            'active_opportunities': len(active_opportunities),
            'avg_profit_percent': self.detection_stats['avg_profit_percent'],
            'best_current_opportunity': None,
            'top_exchanges': self._get_top_exchanges(),
            'top_symbols': self._get_top_symbols()
        }
        
        if active_opportunities:
            best = max(active_opportunities, key=lambda x: x.net_profit)
            summary['best_current_opportunity'] = {
                'symbol': best.symbol,
                'profit': best.net_profit,
                'profit_percent': best.profit_percent,
                'buy_exchange': best.buy_exchange,
                'sell_exchange': best.sell_exchange,
                'confidence': best.confidence
            }
        
        return summary
    
    def _get_top_exchanges(self) -> List[Tuple[str, int]]:
        """Get exchanges with most arbitrage opportunities"""
        
        exchange_counts = defaultdict(int)
        
        for opp in self.arbitrage_history:
            exchange_counts[opp.buy_exchange] += 1
            exchange_counts[opp.sell_exchange] += 1
        
        return sorted(exchange_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_top_symbols(self) -> List[Tuple[str, int]]:
        """Get symbols with most arbitrage opportunities"""
        
        symbol_counts = defaultdict(int)
        
        for opp in self.arbitrage_history:
            symbol_counts[opp.symbol] += 1
        
        return sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]

# Factory function
def create_arbitrage_detector(config: Dict[str, Any] = None) -> MultiExchangeArbitrageDetector:
    """Create multi-exchange arbitrage detector"""
    
    default_config = {
        'enabled_exchanges': ['binance', 'coinbase', 'kraken', 'kucoin'],
        'min_profit_percent': 0.8,  # 0.8% minimum profit
        'min_profit_amount': 15.0,  # $15 minimum profit
        'max_execution_time': 180,  # 3 minutes max
        'min_confidence': 0.75      # 75% minimum confidence
    }
    
    if config:
        default_config.update(config)
    
    return MultiExchangeArbitrageDetector(default_config)

# Test function
async def test_arbitrage_detection():
    """Test arbitrage detection system"""
    
    print("🔄 Testing Multi-Exchange Arbitrage Detection...")
    
    # Create detector
    config = {
        'enabled_exchanges': ['binance', 'coinbase', 'kraken'],
        'min_profit_percent': 0.3,  # Lower threshold for testing
        'min_profit_amount': 5.0,
        'min_confidence': 0.6
    }
    
    detector = create_arbitrage_detector(config)
    
    # Test symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    try:
        # Detect opportunities
        opportunities = await detector.detect_arbitrage_opportunities(test_symbols)
        
        print(f"📊 Found {len(opportunities)} arbitrage opportunities")
        
        # Show top opportunities
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n🏆 Opportunity #{i}:")
            print(f"   Symbol: {opp.symbol}")
            print(f"   Buy: {opp.buy_exchange} @ ${opp.buy_price:.2f}")
            print(f"   Sell: {opp.sell_exchange} @ ${opp.sell_price:.2f}")
            print(f"   Profit: ${opp.net_profit:.2f} ({opp.profit_percent:.2f}%)")
            print(f"   Confidence: {opp.confidence:.2f}")
            print(f"   Risk Score: {opp.risk_score:.2f}")
            print(f"   Volume: ${opp.volume_available:,.0f}")
        
        # Show detection summary
        summary = detector.get_detection_summary()
        print(f"\n📈 Detection Summary:")
        print(f"   Total checks: {summary['total_checks']}")
        print(f"   Opportunities found: {summary['opportunities_found']}")
        print(f"   Active opportunities: {summary['active_opportunities']}")
        print(f"   Avg profit: {summary['avg_profit_percent']:.2f}%")
        
        if summary['top_exchanges']:
            print(f"   Top exchanges: {summary['top_exchanges'][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test
    asyncio.run(test_arbitrage_detection())