# Multi-Exchange Strategy Adaptation Guide

## Übersicht

Diese Anleitung zeigt, wie bestehende Strategien für Multi-Exchange Trading angepasst werden.

## 1. Basis-Anpassungen für bestehende Strategien

### Schritt 1: Strategy Base Class erweitern

```python
# Bestehende Strategie
class MomentumStrategy(Strategy):
    def calculate_signal(self, symbol: str, data: pd.DataFrame, current_price: float):
        # Ursprüngliche Logik
        return signal, signal_data

# Multi-Exchange Version
class MultiExchangeMomentumStrategy(MultiExchangeStrategy):
    async def calculate_signal(self, symbol: str, data=None, current_price: float = None):
        # 1. Multi-Exchange Marktdaten holen
        market_data = await self.get_multi_exchange_market_data(symbol)
        
        # 2. Arbitrage-Möglichkeiten prüfen
        if self.enable_arbitrage:
            arbitrage_opportunities = await self._check_arbitrage(symbol)
            if arbitrage_opportunities:
                return 'ARBITRAGE', arbitrage_opportunities
        
        # 3. Ursprüngliche Strategielogik mit erweiterten Daten
        signal, signal_data = self._calculate_momentum_signal(market_data)
        
        # 4. Exchange-spezifische Anpassungen
        signal_data['preferred_exchange'] = await self._select_best_exchange(symbol, signal)
        
        return signal, signal_data
```

### Schritt 2: Order Execution anpassen

```python
# Alt: Direkte Exchange-Calls
def execute_trade(self, symbol, side, amount):
    order = self.exchange.create_order(symbol, side, amount)
    return order

# Neu: Multi-Exchange Logic
async def execute_trade(self, symbol, side, amount):
    # Smart routing für beste Execution
    result = await self.execute_order_multi_exchange(
        symbol=symbol,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        amount=amount,
        order_type=OrderType.MARKET
    )
    return result
```

## 2. Spezifische Strategy-Anpassungen

### Momentum Strategy

```python
class EnhancedMomentumStrategy(MultiExchangeStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.momentum_threshold = params.get('momentum_threshold', 0.02)
        self.cross_exchange_validation = params.get('cross_exchange_validation', True)
    
    async def calculate_signal(self, symbol: str, data=None, current_price=None):
        # Multi-Exchange Preisdaten
        market_data = await self.get_multi_exchange_market_data(symbol)
        
        if not market_data:
            return 'HOLD', {'reason': 'No market data'}
        
        # Momentum basierend auf Price Spread
        price_spread = market_data['price_spread_percent']
        avg_price = market_data['average_price']
        
        # Cross-Exchange Validation
        if self.cross_exchange_validation:
            if price_spread > 2.0:  # 2% Spread indicates momentum
                return 'BUY', {
                    'confidence': min(0.9, price_spread / 5),
                    'reason': f'Strong momentum (spread: {price_spread:.2f}%)',
                    'target_exchange': market_data['exchanges'][0]
                }
        
        return 'HOLD', {'reason': 'No momentum signal'}
```

### Mean Reversion Strategy

```python
class MultiExchangeMeanReversionStrategy(MultiExchangeStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.reversion_threshold = params.get('reversion_threshold', 0.015)
        self.enable_cross_exchange_arbitrage = True
    
    async def calculate_signal(self, symbol: str, data=None, current_price=None):
        market_data = await self.get_multi_exchange_market_data(symbol)
        
        if not market_data:
            return 'HOLD', {'reason': 'No market data'}
        
        # Mean Reversion basierend auf Exchange-Preisabweichungen
        avg_price = market_data['average_price']
        min_price = market_data['min_price']
        max_price = market_data['max_price']
        
        # Finde Exchanges mit extremen Preisen
        individual_tickers = market_data['individual_tickers']
        
        for exchange, ticker in individual_tickers.items():
            price = ticker['last']
            deviation = abs(price - avg_price) / avg_price
            
            if deviation > self.reversion_threshold:
                if price < avg_price:  # Unterbewertet
                    return 'BUY', {
                        'confidence': min(0.8, deviation * 10),
                        'reason': f'Mean reversion opportunity on {exchange}',
                        'target_exchange': exchange,
                        'expected_reversion_price': avg_price
                    }
                elif price > avg_price:  # Überbewertet
                    return 'SELL', {
                        'confidence': min(0.8, deviation * 10),
                        'reason': f'Mean reversion opportunity on {exchange}',
                        'target_exchange': exchange,
                        'expected_reversion_price': avg_price
                    }
        
        return 'HOLD', {'reason': 'No mean reversion opportunity'}
```

### Scalping Strategy

```python
class MultiExchangeScalpingStrategy(MultiExchangeStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.execution_priority = 'speed'  # Kritisch für Scalping
        self.preferred_exchanges = ['binance']  # Niedrigste Latenz
        self.min_profit_bps = params.get('min_profit_bps', 5)  # 5 basis points
    
    async def calculate_signal(self, symbol: str, data=None, current_price=None):
        # Prüfe Latenz aller Exchanges
        if self.exchange_monitor:
            latencies = {}
            for exchange in self.preferred_exchanges:
                health = self.exchange_monitor.get_exchange_health(exchange)
                if health:
                    latencies[exchange] = health.latency_ms
        
        # Nur auf schnellsten Exchanges handeln
        fast_exchanges = [ex for ex, latency in latencies.items() if latency < 100]
        
        if not fast_exchanges:
            return 'HOLD', {'reason': 'No fast exchanges available'}
        
        # Micro-Arbitrage zwischen schnellen Exchanges
        market_data = await self.get_multi_exchange_market_data(symbol)
        
        if market_data['price_spread_percent'] * 10000 > self.min_profit_bps:  # Convert to bps
            return 'SCALP', {
                'confidence': 0.7,
                'reason': f'Scalping opportunity: {market_data["price_spread_percent"]:.4f}%',
                'fast_exchanges': fast_exchanges
            }
        
        return 'HOLD', {'reason': 'No scalping opportunity'}
```

## 3. Arbitrage-Integration

### Bestehende Strategie erweitern

```python
class ArbitrageAwareStrategy(MultiExchangeStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.enable_arbitrage = True
        self.min_arbitrage_profit = 0.5  # 0.5%
    
    async def calculate_signal(self, symbol: str, data=None, current_price=None):
        # 1. Prüfe Arbitrage-Möglichkeiten zuerst
        if self.arbitrage_engine:
            opportunities = self.arbitrage_engine.get_active_opportunities()
            symbol_opps = [opp for opp in opportunities if opp.symbol == symbol]
            
            if symbol_opps:
                best_opp = max(symbol_opps, key=lambda x: x.profit_percent)
                if best_opp.profit_percent >= self.min_arbitrage_profit:
                    return 'ARBITRAGE', {
                        'confidence': 0.95,
                        'arbitrage_data': best_opp.to_dict(),
                        'reason': f'Arbitrage: {best_opp.profit_percent:.2f}% profit'
                    }
        
        # 2. Fallback zur normalen Strategielogik
        return await super().calculate_signal(symbol, data, current_price)
```

## 4. Risk Management Anpassungen

### Portfolio Risk über mehrere Exchanges

```python
class MultiExchangeRiskManager:
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
    
    async def calculate_total_exposure(self, symbol: str) -> float:
        """Berechne Gesamtexposure über alle Exchanges"""
        total_exposure = 0
        
        # Aggregiere Balances von allen Exchanges
        aggregated_balances = await self.exchange_manager.get_aggregated_balance()
        
        if symbol in aggregated_balances:
            balance = aggregated_balances[symbol]
            total_exposure = balance.total_balance
        
        return total_exposure
    
    async def check_concentration_risk(self) -> Dict[str, float]:
        """Prüfe Konzentrationsrisiko pro Exchange"""
        balances = await self.exchange_manager.get_aggregated_balance()
        
        concentration_risk = {}
        total_value = sum(balance.total_balance for balance in balances.values())
        
        for currency, balance in balances.items():
            for exchange_name, exchange_balance in balance.exchanges.items():
                risk_ratio = exchange_balance.total / total_value
                concentration_risk[f"{exchange_name}_{currency}"] = risk_ratio
        
        return concentration_risk
```

## 5. Configuration Template

### config/strategy_multi_exchange.json

```json
{
  "strategies": {
    "enhanced_momentum": {
      "class": "EnhancedMomentumStrategy",
      "params": {
        "preferred_exchanges": ["binance", "kucoin"],
        "fallback_exchanges": ["bybit"],
        "execution_priority": "price",
        "enable_arbitrage": true,
        "cross_exchange_hedging": false,
        "momentum_threshold": 0.02,
        "cross_exchange_validation": true
      }
    },
    "multi_mean_reversion": {
      "class": "MultiExchangeMeanReversionStrategy",
      "params": {
        "preferred_exchanges": ["kucoin", "bybit"],
        "execution_priority": "liquidity",
        "reversion_threshold": 0.015,
        "enable_cross_exchange_arbitrage": true
      }
    },
    "scalping": {
      "class": "MultiExchangeScalpingStrategy",
      "params": {
        "preferred_exchanges": ["binance"],
        "execution_priority": "speed",
        "min_profit_bps": 5,
        "max_latency_ms": 100
      }
    }
  },
  "risk_management": {
    "max_exposure_per_exchange": 0.5,
    "concentration_limit": 0.3,
    "cross_exchange_hedging": true
  }
}
```

## 6. Testing und Deployment

### Schritt 1: Paper Trading Test

```python
async def test_multi_exchange_strategy():
    # Verwende Testnet-Konfiguration
    config = load_config('config/multi_exchange_testnet.json')
    
    bot = MultiExchangeTradingBot(config)
    await bot.initialize()
    
    # Simuliere Trading-Zyklen
    for i in range(10):
        await bot.run_trading_cycle()
        await asyncio.sleep(30)
    
    # Analysiere Ergebnisse
    status = bot.get_status()
    print(f"Test results: {status}")
```

### Schritt 2: Live Trading mit kleinen Positionen

```python
# Beginne mit minimalen Positionen
INITIAL_POSITION_SIZE = 0.001  # Sehr kleine Beträge
MAX_DAILY_TRADES = 10

# Graduell erhöhen basierend auf Performance
def adjust_position_size(performance_metrics):
    if performance_metrics['success_rate'] > 0.8:
        return min(INITIAL_POSITION_SIZE * 2, 0.01)
    return INITIAL_POSITION_SIZE
```

## 7. Monitoring und Alerts

### Performance Tracking

```python
class MultiExchangePerformanceTracker:
    def __init__(self):
        self.trades_per_exchange = {}
        self.profit_per_exchange = {}
        self.arbitrage_opportunities = []
    
    def track_trade(self, exchange: str, symbol: str, profit: float):
        if exchange not in self.trades_per_exchange:
            self.trades_per_exchange[exchange] = 0
            self.profit_per_exchange[exchange] = 0
        
        self.trades_per_exchange[exchange] += 1
        self.profit_per_exchange[exchange] += profit
    
    def get_performance_report(self):
        return {
            'trades_per_exchange': self.trades_per_exchange,
            'profit_per_exchange': self.profit_per_exchange,
            'total_arbitrage_opportunities': len(self.arbitrage_opportunities)
        }
```

## Zusammenfassung

Die Multi-Exchange Integration ermöglicht:

1. **Bessere Preisfindung** durch Vergleich mehrerer Exchanges
2. **Arbitrage-Möglichkeiten** zwischen Exchanges
3. **Ausfallsicherheit** durch automatisches Failover
4. **Optimierte Execution** durch Smart Routing
5. **Risikodiversifikation** über mehrere Plattformen

**Wichtige Punkte:**
- Beginne mit Paper Trading
- Verwende kleine Positionsgrößen
- Überwache Latenz und Performance
- Implementiere robuste Fehlerbehandlung
- Teste Failover-Szenarien