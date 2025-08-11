#!/usr/bin/env python3
"""
Multi-Exchange Integration Examples
Shows how to adapt existing strategies for multi-exchange trading
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Multi-exchange imports
from core.exchange_manager import MultiExchangeManager, OrderSide, OrderType
from core.arbitrage_engine import ArbitrageEngine
from utils.exchange_monitor import ExchangeMonitor

# Strategy imports
from strategies.strategy_base import Strategy
from strategies.momentum_strategy import MomentumStrategy

# Other imports
try:
    from utils.notifier import send_info, send_warning
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiExchangeStrategy(Strategy):
    """
    Enhanced strategy base class for multi-exchange trading
    """
    
    def __init__(self, params: Dict = None, ml_components=None, exchange_manager=None):
        super().__init__(params, ml_components)
        self.exchange_manager = exchange_manager
        self.arbitrage_engine = None
        self.exchange_monitor = None
        
        # Multi-exchange configuration
        self.preferred_exchanges = params.get('preferred_exchanges', ['binance'])
        self.fallback_exchanges = params.get('fallback_exchanges', [])
        self.execution_priority = params.get('execution_priority', 'speed')  # speed, price, liquidity
        self.enable_arbitrage = params.get('enable_arbitrage', False)
        self.cross_exchange_hedging = params.get('cross_exchange_hedging', False)
        
        logger.info(f"MultiExchangeStrategy {self.name} initialized")
    
    def set_exchange_components(self, exchange_manager, arbitrage_engine=None, exchange_monitor=None):
        """Set multi-exchange components"""
        self.exchange_manager = exchange_manager
        self.arbitrage_engine = arbitrage_engine
        self.exchange_monitor = exchange_monitor
    
    async def execute_order_multi_exchange(self, symbol: str, side: OrderSide, 
                                         amount: float, order_type: OrderType = OrderType.MARKET,
                                         price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute order with multi-exchange logic
        """
        try:
            # 1. Choose best exchange for execution
            best_exchange = await self._select_best_exchange(symbol, side, amount)
            
            if not best_exchange:
                logger.error("No suitable exchange found for order execution")
                return {'success': False, 'error': 'No suitable exchange'}
            
            # 2. Execute order
            order = await self.exchange_manager.create_order_on_exchange(
                best_exchange, symbol, side, order_type, amount, price
            )
            
            if order:
                logger.info(f"Order executed on {best_exchange}: {order.id}")
                
                # 3. Optional: Execute hedging positions on other exchanges
                if self.cross_exchange_hedging:
                    await self._execute_hedging_positions(symbol, side, amount, best_exchange)
                
                return {
                    'success': True,
                    'order': order.to_dict(),
                    'exchange': best_exchange
                }
            else:
                return {'success': False, 'error': 'Order execution failed'}
            
        except Exception as e:
            logger.error(f"Error in multi-exchange order execution: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _select_best_exchange(self, symbol: str, side: OrderSide, amount: float) -> Optional[str]:
        """
        Select best exchange for order execution based on strategy preferences
        """
        try:
            # 1. Filter available exchanges
            available_exchanges = self.exchange_manager.get_connected_exchanges()
            
            # 2. Apply preference order
            candidate_exchanges = []
            
            # Add preferred exchanges first
            for exchange in self.preferred_exchanges:
                if exchange in available_exchanges:
                    candidate_exchanges.append(exchange)
            
            # Add fallback exchanges
            for exchange in self.fallback_exchanges:
                if exchange in available_exchanges and exchange not in candidate_exchanges:
                    candidate_exchanges.append(exchange)
            
            # Add any remaining exchanges
            for exchange in available_exchanges:
                if exchange not in candidate_exchanges:
                    candidate_exchanges.append(exchange)
            
            if not candidate_exchanges:
                return None
            
            # 3. Select based on execution priority
            if self.execution_priority == 'price':
                return await self._select_by_price(symbol, side, candidate_exchanges)
            elif self.execution_priority == 'liquidity':
                return await self._select_by_liquidity(symbol, candidate_exchanges)
            else:  # speed
                return await self._select_by_speed(candidate_exchanges)
            
        except Exception as e:
            logger.error(f"Error selecting best exchange: {e}")
            return candidate_exchanges[0] if candidate_exchanges else None
    
    async def _select_by_price(self, symbol: str, side: OrderSide, exchanges: List[str]) -> Optional[str]:
        """Select exchange with best price"""
        try:
            best_exchange = await self.exchange_manager.get_best_price(symbol, side)
            if best_exchange and best_exchange[0] in exchanges:
                return best_exchange[0]
            return exchanges[0] if exchanges else None
        except:
            return exchanges[0] if exchanges else None
    
    async def _select_by_liquidity(self, symbol: str, exchanges: List[str]) -> Optional[str]:
        """Select exchange with best liquidity"""
        try:
            tickers = await self.exchange_manager.fetch_ticker_all(symbol)
            
            best_exchange = None
            best_volume = 0
            
            for exchange in exchanges:
                if exchange in tickers:
                    volume = tickers[exchange].get('quoteVolume', 0)
                    if volume > best_volume:
                        best_volume = volume
                        best_exchange = exchange
            
            return best_exchange or (exchanges[0] if exchanges else None)
        except:
            return exchanges[0] if exchanges else None
    
    async def _select_by_speed(self, exchanges: List[str]) -> Optional[str]:
        """Select exchange with best speed/latency"""
        try:
            if self.exchange_monitor:
                best_exchange = None
                best_latency = float('inf')
                
                for exchange in exchanges:
                    health = self.exchange_monitor.get_exchange_health(exchange)
                    if health and health.latency_ms and health.latency_ms < best_latency:
                        best_latency = health.latency_ms
                        best_exchange = exchange
                
                if best_exchange:
                    return best_exchange
            
            return exchanges[0] if exchanges else None
        except:
            return exchanges[0] if exchanges else None
    
    async def _execute_hedging_positions(self, symbol: str, side: OrderSide, 
                                       amount: float, primary_exchange: str):
        """
        Execute hedging positions on other exchanges
        """
        try:
            # Simplified hedging: execute opposite position on another exchange
            hedge_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            hedge_amount = amount * 0.5  # 50% hedge
            
            # Find different exchange for hedging
            available_exchanges = [
                ex for ex in self.exchange_manager.get_connected_exchanges()
                if ex != primary_exchange
            ]
            
            if available_exchanges:
                hedge_exchange = available_exchanges[0]
                hedge_order = await self.exchange_manager.create_order_on_exchange(
                    hedge_exchange, symbol, hedge_side, OrderType.MARKET, hedge_amount
                )
                
                if hedge_order:
                    logger.info(f"Hedging position executed on {hedge_exchange}: {hedge_order.id}")
                
        except Exception as e:
            logger.error(f"Error executing hedging positions: {e}")
    
    async def get_multi_exchange_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated market data from all exchanges
        """
        try:
            tickers = await self.exchange_manager.fetch_ticker_all(symbol)
            
            if not tickers:
                return {}
            
            # Calculate aggregated metrics
            prices = [ticker['last'] for ticker in tickers.values() if ticker.get('last')]
            volumes = [ticker.get('quoteVolume', 0) for ticker in tickers.values()]
            
            if not prices:
                return {}
            
            aggregated_data = {
                'symbol': symbol,
                'exchanges': list(tickers.keys()),
                'average_price': sum(prices) / len(prices),
                'min_price': min(prices),
                'max_price': max(prices),
                'price_spread_percent': ((max(prices) - min(prices)) / min(prices)) * 100,
                'total_volume': sum(volumes),
                'exchange_count': len(tickers),
                'individual_tickers': tickers
            }
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error getting multi-exchange market data: {e}")
            return {}


class EnhancedMomentumStrategy(MultiExchangeStrategy):
    """
    Enhanced Momentum Strategy with multi-exchange capabilities
    """
    
    async def calculate_signal(self, symbol: str, data=None, current_price: float = None):
        """
        Calculate momentum signal using multi-exchange data
        """
        try:
            # 1. Get multi-exchange market data
            market_data = await self.get_multi_exchange_market_data(symbol)
            
            if not market_data:
                return 'HOLD', {'reason': 'No market data available'}
            
            # 2. Check for arbitrage opportunities
            arbitrage_signal = None
            if self.enable_arbitrage and self.arbitrage_engine:
                opportunities = self.arbitrage_engine.get_active_opportunities()
                symbol_opportunities = [opp for opp in opportunities if opp.symbol == symbol]
                
                if symbol_opportunities:
                    best_opp = max(symbol_opportunities, key=lambda x: x.profit_percent)
                    if best_opp.profit_percent >= 1.0:  # 1% minimum
                        arbitrage_signal = {
                            'type': 'arbitrage',
                            'buy_exchange': best_opp.buy_exchange,
                            'sell_exchange': best_opp.sell_exchange,
                            'profit_percent': best_opp.profit_percent
                        }
            
            # 3. Calculate momentum using price spread
            price_spread = market_data.get('price_spread_percent', 0)
            avg_price = market_data.get('average_price', current_price or 0)
            
            # 4. Enhanced momentum logic
            signal = 'HOLD'
            confidence = 0.5
            reason = 'No clear signal'
            
            # High spread might indicate momentum opportunity
            if price_spread > 2.0:  # 2% spread
                signal = 'BUY'
                confidence = min(0.8, price_spread / 10)  # Higher spread = higher confidence
                reason = f'High price spread ({price_spread:.2f}%) indicates momentum'
            elif price_spread < 0.5:  # Low spread
                signal = 'HOLD'
                reason = f'Low price spread ({price_spread:.2f}%), waiting for opportunity'
            
            # Override with arbitrage if profitable
            if arbitrage_signal:
                return 'ARBITRAGE', {
                    'confidence': 0.9,
                    'reason': f'Arbitrage opportunity: {arbitrage_signal["profit_percent"]:.2f}%',
                    'arbitrage_data': arbitrage_signal,
                    'market_data': market_data
                }
            
            return signal, {
                'confidence': confidence,
                'reason': reason,
                'market_data': market_data,
                'price_spread_percent': price_spread
            }
            
        except Exception as e:
            logger.error(f"Error calculating multi-exchange signal: {e}")
            return 'HOLD', {'reason': f'Error: {e}'}


class MultiExchangeTradingBot:
    """
    Complete trading bot with multi-exchange capabilities
    """
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.exchange_manager = None
        self.arbitrage_engine = None
        self.exchange_monitor = None
        self.strategies = {}
        
        # State
        self.running = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "multi_exchange": {
                "enabled": True,
                "primary_exchange": "binance",
                "enabled_exchanges": ["binance"],
                "failover_enabled": True
            },
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "testnet": True,
                    "priority": 1
                }
            },
            "arbitrage": {
                "enabled": False,
                "min_profit_percent": 0.5,
                "symbols": ["BTC/USDT", "ETH/USDT"]
            },
            "monitoring": {
                "enabled": True
            }
        }
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Multi-Exchange Trading Bot...")
            
            # 1. Initialize Exchange Manager
            self.exchange_manager = MultiExchangeManager(self.config)
            
            # 2. Connect to exchanges
            connection_results = await self.exchange_manager.connect_all()
            logger.info(f"Exchange connections: {connection_results}")
            
            # 3. Initialize Exchange Monitor
            if self.config.get('monitoring', {}).get('enabled', True):
                monitor_config = self.config.get('failover', {})
                self.exchange_monitor = ExchangeMonitor(self.exchange_manager, monitor_config)
                await self.exchange_monitor.start_monitoring()
            
            # 4. Initialize Arbitrage Engine
            if self.config.get('arbitrage', {}).get('enabled', False):
                arbitrage_config = self.config.get('arbitrage', {})
                self.arbitrage_engine = ArbitrageEngine(self.exchange_manager, arbitrage_config)
                await self.arbitrage_engine.start_monitoring()
            
            # 5. Initialize Strategies
            await self._initialize_strategies()
            
            logger.info("Multi-Exchange Trading Bot initialized successfully")
            
            if NOTIFIER_AVAILABLE:
                send_info("🚀 Multi-Exchange Trading Bot initialized and ready!")
            
        except Exception as e:
            logger.error(f"Error initializing bot: {e}")
            raise
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # Example: Enhanced Momentum Strategy
            momentum_params = {
                'preferred_exchanges': ['binance', 'kucoin'],
                'fallback_exchanges': ['bybit'],
                'execution_priority': 'price',
                'enable_arbitrage': self.config.get('arbitrage', {}).get('enabled', False),
                'cross_exchange_hedging': False
            }
            
            momentum_strategy = EnhancedMomentumStrategy(momentum_params)
            momentum_strategy.set_exchange_components(
                self.exchange_manager, 
                self.arbitrage_engine, 
                self.exchange_monitor
            )
            
            self.strategies['enhanced_momentum'] = momentum_strategy
            
            logger.info("Strategies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
    
    async def run_trading_cycle(self):
        """Run single trading cycle"""
        try:
            # Monitor symbols
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            
            for symbol in symbols:
                for strategy_name, strategy in self.strategies.items():
                    try:
                        # Get signal
                        signal, signal_data = await strategy.calculate_signal(symbol)
                        
                        logger.info(f"{strategy_name} signal for {symbol}: {signal} "
                                  f"(confidence: {signal_data.get('confidence', 0):.2f})")
                        
                        # Execute if strong signal
                        if signal in ['BUY', 'SELL'] and signal_data.get('confidence', 0) > 0.7:
                            side = OrderSide.BUY if signal == 'BUY' else OrderSide.SELL
                            
                            result = await strategy.execute_order_multi_exchange(
                                symbol=symbol,
                                side=side,
                                amount=0.001,  # Small test amount
                                order_type=OrderType.MARKET
                            )
                            
                            if result['success']:
                                logger.info(f"Order executed: {result}")
                                if NOTIFIER_AVAILABLE:
                                    send_info(f"📈 Order executed!\n"
                                            f"Strategy: {strategy_name}\n"
                                            f"Symbol: {symbol}\n"
                                            f"Side: {signal}\n"
                                            f"Exchange: {result['exchange']}")
                            else:
                                logger.error(f"Order failed: {result['error']}")
                        
                        elif signal == 'ARBITRAGE':
                            logger.info(f"Arbitrage opportunity detected: {signal_data}")
                            # Arbitrage engine handles execution automatically
                        
                    except Exception as e:
                        logger.error(f"Error in trading cycle for {symbol} with {strategy_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def start(self):
        """Start the trading bot"""
        try:
            await self.initialize()
            self.running = True
            
            logger.info("Starting trading bot main loop...")
            
            while self.running:
                try:
                    await self.run_trading_cycle()
                    await asyncio.sleep(60)  # Wait 1 minute
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds before retry
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("Stopping trading bot...")
            self.running = False
            
            if self.arbitrage_engine:
                await self.arbitrage_engine.stop_monitoring()
            
            if self.exchange_monitor:
                await self.exchange_monitor.stop_monitoring()
            
            if self.exchange_manager:
                await self.exchange_manager.disconnect_all()
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        try:
            status = {
                'running': self.running,
                'timestamp': datetime.now().isoformat(),
                'strategies': list(self.strategies.keys()),
                'exchanges': {},
                'arbitrage': {},
                'monitoring': {}
            }
            
            if self.exchange_manager:
                status['exchanges'] = {
                    'connected': self.exchange_manager.get_connected_exchanges(),
                    'primary': self.exchange_manager.primary_exchange
                }
            
            if self.arbitrage_engine:
                status['arbitrage'] = self.arbitrage_engine.get_stats()
            
            if self.exchange_monitor:
                status['monitoring'] = {
                    'active': self.exchange_monitor.monitoring_active,
                    'primary': self.exchange_monitor.get_primary_exchange(),
                    'healthy_exchanges': self.exchange_monitor.get_healthy_exchanges()
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}


async def main():
    """Main example function"""
    try:
        # Create bot with config
        config_path = "config/multi_exchange_config.json"
        if not Path(config_path).exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            config_path = None
        
        bot = MultiExchangeTradingBot(config_path)
        
        # Start bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())