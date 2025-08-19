"""
Real Exchange Paper Trading Engine
==================================

Echte Exchange Demo APIs für realistisches Paper Trading
mit echten Marktdaten und Exchange-spezifischen Features.
"""

import asyncio
import logging
import ccxt
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RealDemoPosition:
    """Real Exchange Demo Position"""
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    strategy: str
    status: str
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    fee: float = 0.0
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None


class RealExchangePaperEngine:
    """
    Real Exchange Paper Trading Engine
    Nutzt echte Exchange Testnet/Demo APIs für realistisches Paper Trading
    """
    
    def __init__(self, exchange: str = 'binance', api_key: str = None, 
                 api_secret: str = None, testnet: bool = True):
        self.exchange_name = exchange
        self.testnet = testnet
        
        # Demo Portfolio Management
        self.demo_balance = 10000.0  # Start with $10k demo money
        self.demo_positions: Dict[str, RealDemoPosition] = {}
        self.trade_history: List[RealDemoPosition] = []
        
        # Performance Tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
        
        # Exchange Client
        self.exchange = None
        
        # Initialize Exchange
        self._initialize_exchange(api_key, api_secret)
        
        logger.info(f"🏦 Real Exchange Paper Engine initialized: {exchange} (testnet: {testnet})")
    
    def _initialize_exchange(self, api_key: str, api_secret: str):
        """Initialisiert echte Exchange API Verbindung"""
        try:
            if self.exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': self.testnet,  # Use testnet for demo
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'  # Spot trading
                    }
                })
            elif self.exchange_name.lower() == 'coinbase':
                self.exchange = ccxt.coinbasepro({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'passphrase': '',  # Add passphrase for Coinbase
                    'sandbox': self.testnet
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
            
            # Test connection
            asyncio.run(self._test_connection())
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.exchange_name}: {e}")
            self.exchange = None
            raise
    
    async def _test_connection(self):
        """Testet Exchange API Verbindung"""
        try:
            # Test mit einfacher API Call
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            
            logger.info(f"✅ Exchange connection successful")
            logger.info(f"📊 Demo account balance: {balance.get('total', {})}")
            
        except Exception as e:
            logger.error(f"❌ Exchange connection test failed: {e}")
            raise
    
    async def get_real_market_price(self, symbol: str) -> float:
        """Holt echten Marktpreis von Exchange"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"❌ Error fetching real price for {symbol}: {e}")
            # Fallback zu letztem bekannten Preis oder Dummy
            return 100.0
    
    async def execute_demo_trade(self, symbol: str, side: str, size: float, 
                                strategy: str = "manual", **kwargs) -> Optional[RealDemoPosition]:
        """
        Führt Demo-Trade mit echten Exchange APIs aus
        
        Args:
            symbol: Trading pair (z.B. 'BTC/USDT')
            side: 'LONG' oder 'SHORT'  
            size: Position size
            strategy: Strategy name
            
        Returns:
            RealDemoPosition oder None bei Fehler
        """
        try:
            if not self.exchange:
                logger.error("❌ Exchange not initialized")
                return None
            
            # Hole echten Marktpreis
            current_price = await self.get_real_market_price(symbol)
            
            # Berechne Trade-Kosten mit echten Exchange Fees
            fee_info = await self._get_trading_fees(symbol)
            fee_rate = fee_info.get('taker', 0.001)  # Default 0.1%
            
            position_value = size * current_price
            fee = position_value * fee_rate
            
            # Prüfe Demo Balance
            if position_value + fee > self.demo_balance:
                logger.warning(f"❌ Insufficient demo balance: ${self.demo_balance:.2f}")
                return None
            
            # Erstelle Demo Order (ohne echte Exchange Order)
            trade_id = f"DEMO_{self.exchange_name.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Simuliere Exchange Order ID
            exchange_order_id = f"ORDER_{uuid.uuid4().hex[:12]}"
            
            # Erstelle Demo Position
            demo_position = RealDemoPosition(
                id=trade_id,
                symbol=symbol,
                side=side,
                size=size,
                entry_price=current_price,
                current_price=current_price,
                timestamp=datetime.now(timezone.utc),
                strategy=strategy,
                status='OPEN',
                exchange_order_id=exchange_order_id,
                fee=fee
            )
            
            # Update Demo Balance
            self.demo_balance -= (position_value + fee)
            
            # Speichere Position
            self.demo_positions[trade_id] = demo_position
            
            logger.info(f"🏦 Real Demo Trade Executed: {side} {size} {symbol} @ ${current_price:.2f}")
            logger.info(f"💰 Demo Balance: ${self.demo_balance:.2f}")
            logger.info(f"🆔 Exchange Order ID: {exchange_order_id}")
            
            return demo_position
            
        except Exception as e:
            logger.error(f"❌ Demo trade execution failed: {e}")
            return None
    
    async def close_demo_trade(self, trade_id: str, exit_price: Optional[float] = None) -> Optional[RealDemoPosition]:
        """
        Schließt Demo Trade mit echten Marktpreisen
        """
        try:
            # Finde Position
            position = self.demo_positions.get(trade_id)
            if not position:
                logger.error(f"❌ Demo position {trade_id} not found")
                return None
            
            # Hole aktuellen echten Marktpreis
            if exit_price is None:
                exit_price = await self.get_real_market_price(position.symbol)
            
            # Berechne P&L
            if position.side == 'LONG':
                pnl = position.size * (exit_price - position.entry_price)
            else:
                pnl = position.size * (position.entry_price - exit_price)
            
            # Deduct exit fee
            exit_fee_info = await self._get_trading_fees(position.symbol)
            exit_fee_rate = exit_fee_info.get('taker', 0.001)
            exit_fee = position.size * exit_price * exit_fee_rate
            pnl -= (position.fee + exit_fee)
            
            # Calculate percentage P&L
            position_value = position.size * position.entry_price
            pnl_percentage = (pnl / position_value) * 100
            
            # Update position
            position.exit_price = exit_price
            position.exit_timestamp = datetime.now(timezone.utc)
            position.pnl = pnl
            position.pnl_percentage = pnl_percentage
            position.status = 'CLOSED'
            position.fee += exit_fee
            
            # Update demo balance
            if position.side == 'LONG':
                self.demo_balance += (position.size * exit_price)
            else:
                self.demo_balance += pnl
            
            # Move to history
            self.trade_history.append(position)
            del self.demo_positions[trade_id]
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            logger.info(f"📊 Real Demo Trade Closed: {position.symbol}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
            logger.info(f"💰 Demo Balance: ${self.demo_balance:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"❌ Demo trade closing failed: {e}")
            return None
    
    async def _get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Holt echte Trading Fees von Exchange"""
        try:
            if self.exchange:
                fees = await self.exchange.fetch_trading_fees()
                symbol_fees = fees.get(symbol, {})
                return {
                    'maker': symbol_fees.get('maker', 0.001),
                    'taker': symbol_fees.get('taker', 0.001)
                }
            else:
                return {'maker': 0.001, 'taker': 0.001}
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch trading fees: {e}")
            return {'maker': 0.001, 'taker': 0.001}
    
    def _update_performance_metrics(self, closed_position: RealDemoPosition):
        """Update performance metrics nach Trade-Close"""
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += closed_position.pnl
        
        if closed_position.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        # Win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            ) * 100
    
    async def update_positions_with_real_prices(self):
        """Updated alle offenen Positionen mit echten Marktpreisen"""
        for position in self.demo_positions.values():
            try:
                current_price = await self.get_real_market_price(position.symbol)
                position.current_price = current_price
            except Exception as e:
                logger.error(f"❌ Error updating price for {position.symbol}: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Liefert Demo Portfolio Status mit echten Exchange Daten"""
        # Calculate unrealized P&L with real prices
        unrealized_pnl = 0.0
        open_positions_data = []
        
        for position in self.demo_positions.values():
            # Calculate current P&L
            if position.side == 'LONG':
                position_pnl = position.size * (position.current_price - position.entry_price)
            else:
                position_pnl = position.size * (position.entry_price - position.current_price)
            
            position_pnl -= position.fee
            unrealized_pnl += position_pnl
            
            # Add position data
            open_positions_data.append({
                'id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': position_pnl,
                'pnl_percentage': (position_pnl / (position.size * position.entry_price)) * 100,
                'strategy': position.strategy,
                'exchange_order_id': position.exchange_order_id,
                'duration': int((datetime.now(timezone.utc) - position.timestamp).total_seconds() / 60)
            })
        
        # Calculate total portfolio value
        total_portfolio_value = self.demo_balance + unrealized_pnl
        
        # Calculate daily P&L (simplified)
        daily_pnl = self.performance_metrics['total_pnl']
        daily_pnl_percentage = (daily_pnl / 10000) * 100  # Assuming $10k start
        
        return {
            'mode': f'REAL PAPER TRADING ({self.exchange_name.upper()})',
            'exchange': self.exchange_name,
            'testnet': self.testnet,
            'demo_balance': self.demo_balance,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.performance_metrics['total_pnl'],
            'total_portfolio_value': total_portfolio_value,
            'daily_pnl': daily_pnl,
            'daily_pnl_percentage': daily_pnl_percentage,
            'open_positions': len(self.demo_positions),
            'open_positions_data': open_positions_data,
            'total_trades': self.performance_metrics['total_trades'],
            'winning_trades': self.performance_metrics['winning_trades'],
            'losing_trades': self.performance_metrics['losing_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'performance_metrics': self.performance_metrics,
            'exchange_status': 'connected' if self.exchange else 'disconnected'
        }
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Liefert aktive Demo Trades"""
        trades = []
        for position in self.demo_positions.values():
            # Calculate current P&L
            if position.side == 'LONG':
                current_pnl = position.size * (position.current_price - position.entry_price)
            else:
                current_pnl = position.size * (position.entry_price - position.current_price)
            
            current_pnl -= position.fee
            
            trades.append({
                'id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'pnl': current_pnl,
                'pnl_percentage': (current_pnl / (position.size * position.entry_price)) * 100,
                'strategy': position.strategy,
                'timestamp': position.timestamp.isoformat(),
                'exchange_order_id': position.exchange_order_id,
                'mode': 'real_paper'
            })
        
        return trades
    
    def get_balance(self) -> float:
        """Liefert aktuelle Demo Balance"""
        return self.demo_balance
    
    def reset_demo_account(self) -> Dict[str, Any]:
        """Reset Demo Account zu Startzustand"""
        try:
            self.demo_balance = 10000.0
            self.demo_positions.clear()
            self.trade_history.clear()
            self.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            }
            
            logger.info(f"🔄 {self.exchange_name} demo account reset to $10,000")
            
            return {
                'success': True,
                'message': f'{self.exchange_name} demo account reset successfully',
                'new_balance': self.demo_balance
            }
            
        except Exception as e:
            logger.error(f"❌ Demo account reset failed: {e}")
            return {
                'success': False,
                'message': f'Demo account reset failed: {str(e)}'
            }
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[Dict[str, Any]]:
        """Holt echte Marktdaten von Exchange"""
        try:
            if not self.exchange:
                return []
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            market_data = []
            for candle in ohlcv:
                market_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return []
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Liefert Exchange Informationen"""
        try:
            if not self.exchange:
                return {'status': 'disconnected'}
            
            markets = await self.exchange.load_markets()
            
            return {
                'status': 'connected',
                'exchange': self.exchange_name,
                'testnet': self.testnet,
                'markets_count': len(markets),
                'available_symbols': list(markets.keys())[:20],  # First 20 symbols
                'features': {
                    'spot_trading': True,
                    'real_market_data': True,
                    'demo_account': self.testnet
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting exchange info: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }