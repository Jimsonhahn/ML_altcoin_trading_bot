"""
Adaptive Auto Strategy - Vollautomatische "Sorglos" Trading Strategie
===================================================================

Diese Strategie:
- Wählt automatisch die beste Strategie basierend auf Marktbedingungen
- Limitiert Tagesrisiko auf 100€ und skaliert dann mit Portfolio-Wachstum
- Nutzt auch High-Risk Strategien, aber mit intelligentem Risk Management
- Vollständig autonome Entscheidungsfindung
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .strategy_base import Strategy

logger = logging.getLogger(__name__)


class AdaptiveAutoStrategy(Strategy):
    """Vollautomatische adaptive Strategie mit intelligentem Risk Management"""
    
    def __init__(self, exchange_manager, config: dict):
        super().__init__(exchange_manager, config)
        
        # Strategie-Konfiguration
        self.daily_risk_limit = config.get('daily_risk_limit', 100.0)  # 100€ Start-Limit
        self.portfolio_scale_factor = config.get('portfolio_scale_factor', 0.02)  # 2% des Portfolios
        self.min_daily_limit = config.get('min_daily_limit', 50.0)  # Minimum 50€
        self.max_daily_limit = config.get('max_daily_limit', 500.0)  # Maximum 500€
        
        # Verfügbare Strategien (nach Risiko sortiert)
        self.available_strategies = {
            'conservative': [
                'mean_reversion',
                'arbitrage', 
                'lazy_billionaire_strategy'
            ],
            'moderate': [
                'momentum',
                'grid_trading',
                'candle_momentum'
            ],
            'aggressive': [
                'high_risk_daily',
                'ml_strategy',
                'optimized_candle_momentum'
            ]
        }
        
        # Aktueller Status
        self.current_strategy = None
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        
        # Performance Tracking
        self.strategy_performance = {}
        self.market_conditions = {}
        
    def calculate_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Hauptlogik: Wählt automatisch beste Strategie und generiert Signal"""
        try:
            # 1. Tägliche Limits prüfen und zurücksetzen
            self._check_daily_reset()
            
            # 2. Aktuelles Tagesrisiko prüfen
            current_daily_risk = self._calculate_current_daily_risk()
            max_allowed_risk = self._calculate_dynamic_risk_limit()
            
            if current_daily_risk >= max_allowed_risk:
                logger.info(f"Tagesrisiko-Limit erreicht: {current_daily_risk:.2f}€ / {max_allowed_risk:.2f}€")
                return self._create_no_trade_signal(symbol, "Daily risk limit reached")
            
            # 3. Marktbedingungen analysieren
            market_regime = self._analyze_market_conditions(symbol, timeframe)
            
            # 4. Beste Strategie für aktuelle Marktlage wählen
            selected_strategy = self._select_optimal_strategy(market_regime, current_daily_risk, max_allowed_risk)
            
            # 5. Position Size basierend auf verbleibendem Tagesrisiko berechnen
            remaining_risk = max_allowed_risk - current_daily_risk
            position_size = self._calculate_position_size(symbol, remaining_risk, selected_strategy)
            
            # 6. Signal von gewählter Strategie generieren
            signal = self._generate_strategy_signal(selected_strategy, symbol, timeframe, position_size)
            
            # 7. Final Risk Check
            if signal['action'] != 'HOLD':
                signal = self._apply_final_risk_check(signal, remaining_risk)
            
            # Logging
            logger.info(f"Auto-Strategy: {selected_strategy} | Market: {market_regime['regime']} | "
                       f"Risk: {current_daily_risk:.2f}€/{max_allowed_risk:.2f}€ | Signal: {signal['action']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Fehler in AdaptiveAutoStrategy: {str(e)}")
            return self._create_no_trade_signal(symbol, f"Strategy error: {str(e)}")
    
    def _check_daily_reset(self):
        """Prüft ob neuer Tag und resettet Daily-Stats"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(f"Neuer Tag - Daily Stats Reset. Vorheriger P&L: {self.daily_pnl:.2f}€")
            self.daily_pnl = 0.0
            self.daily_trades = []
            self.last_reset_date = today
    
    def _calculate_current_daily_risk(self) -> float:
        """Berechnet aktuelles Tagesrisiko basierend auf offenen Positionen"""
        # Hier würde man echte Positionen abfragen
        # Für Demo: simuliere basierend auf Daily Trades
        total_risk = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        
        # Add unrealized risk from open positions
        # total_risk += self._get_unrealized_risk()
        
        return total_risk
    
    def _calculate_dynamic_risk_limit(self) -> float:
        """Berechnet dynamisches Risiko-Limit basierend auf Portfolio-Größe"""
        try:
            # Portfolio-Wert holen (vereinfacht)
            portfolio_value = self._get_portfolio_value()
            
            # Dynamisches Limit: Minimum von Fixed + Portfolio-basiert
            portfolio_based_limit = portfolio_value * self.portfolio_scale_factor
            dynamic_limit = max(self.daily_risk_limit, portfolio_based_limit)
            
            # Caps anwenden
            dynamic_limit = max(self.min_daily_limit, min(dynamic_limit, self.max_daily_limit))
            
            return dynamic_limit
            
        except Exception as e:
            logger.error(f"Fehler bei dynamischem Risk Limit: {str(e)}")
            return self.daily_risk_limit
    
    def _analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analysiert Marktbedingungen um optimale Strategie zu wählen"""
        try:
            # Marktdaten holen
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=100)
            
            if df is None or len(df) < 20:
                return {'regime': 'unknown', 'volatility': 'medium', 'trend': 'sideways'}
            
            # Volatilität berechnen
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].rolling(20).std().iloc[-1] * 100
            
            # Trend berechnen (SMA Crossover)
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            # Trend bestimmen
            if sma_20 > sma_50 and current_price > sma_20:
                trend = 'uptrend'
            elif sma_20 < sma_50 and current_price < sma_20:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Volatilitäts-Regime
            if volatility > 3:
                vol_regime = 'high'
            elif volatility < 1:
                vol_regime = 'low'
            else:
                vol_regime = 'medium'
            
            # Market Regime kombinieren
            if trend == 'uptrend' and vol_regime == 'medium':
                regime = 'trending_bullish'
            elif trend == 'downtrend' and vol_regime == 'medium':
                regime = 'trending_bearish'  
            elif vol_regime == 'high':
                regime = 'volatile'
            elif vol_regime == 'low':
                regime = 'consolidation'
            else:
                regime = 'mixed'
            
            return {
                'regime': regime,
                'volatility': vol_regime,
                'trend': trend,
                'vol_value': volatility,
                'price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Market Analysis: {str(e)}")
            return {'regime': 'unknown', 'volatility': 'medium', 'trend': 'sideways'}
    
    def _select_optimal_strategy(self, market_regime: Dict, current_risk: float, max_risk: float) -> str:
        """Wählt optimale Strategie basierend auf Marktbedingungen und Risiko"""
        
        remaining_risk_ratio = (max_risk - current_risk) / max_risk
        regime = market_regime['regime']
        volatility = market_regime['volatility']
        
        # Strategy Selection Logic
        if remaining_risk_ratio > 0.7:  # Viel Risiko-Budget übrig
            if regime == 'trending_bullish':
                return 'momentum'  # Nutze Trend
            elif regime == 'volatile' and volatility == 'high':
                return 'high_risk_daily'  # High-Risk bei hoher Vola
            elif regime == 'consolidation':
                return 'grid_trading'  # Range Trading
            else:
                return 'candle_momentum'  # Default moderate
                
        elif remaining_risk_ratio > 0.3:  # Moderates Risiko-Budget
            if regime == 'trending_bullish':
                return 'candle_momentum'
            elif regime == 'consolidation':
                return 'mean_reversion'
            else:
                return 'lazy_billionaire_strategy'
                
        else:  # Wenig Risiko-Budget - Conservative
            if regime == 'consolidation':
                return 'arbitrage'
            else:
                return 'lazy_billionaire_strategy'
    
    def _calculate_position_size(self, symbol: str, remaining_risk: float, strategy: str) -> float:
        """Berechnet Positionsgröße basierend auf verbleibendem Risiko"""
        
        # Strategy Risk Multipliers
        risk_multipliers = {
            'arbitrage': 0.5,
            'lazy_billionaire_strategy': 0.6,
            'mean_reversion': 0.7,
            'momentum': 0.8,
            'grid_trading': 0.9,
            'candle_momentum': 1.0,
            'high_risk_daily': 1.2,
            'ml_strategy': 1.1,
            'optimized_candle_momentum': 1.0
        }
        
        base_risk = remaining_risk * 0.3  # Nutze max 30% des verbleibenden Risikos pro Trade
        strategy_risk = base_risk * risk_multipliers.get(strategy, 0.8)
        
        # Minimum Position Size (10€)
        min_position = 10.0
        max_position = min(strategy_risk, remaining_risk * 0.5)  # Max 50% des Verbleibs
        
        position_size = max(min_position, max_position)
        
        return position_size
    
    def _generate_strategy_signal(self, strategy_name: str, symbol: str, timeframe: str, position_size: float) -> Dict[str, Any]:
        """Generiert Signal von der gewählten Strategie"""
        
        # Hier würden wir die echten Strategie-Implementierungen aufrufen
        # Für Demo: simuliere Strategien
        
        try:
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit=50)
            if df is None or len(df) < 10:
                return self._create_no_trade_signal(symbol, "Insufficient data")
            
            # Vereinfachte Strategie-Logik (würde echte Implementierungen verwenden)
            current_price = df['close'].iloc[-1]
            
            if strategy_name == 'momentum':
                signal = self._momentum_logic(df, symbol, position_size)
            elif strategy_name == 'mean_reversion':
                signal = self._mean_reversion_logic(df, symbol, position_size)
            elif strategy_name == 'high_risk_daily':
                signal = self._high_risk_logic(df, symbol, position_size)
            elif strategy_name == 'lazy_billionaire_strategy':
                signal = self._lazy_billionaire_logic(df, symbol, position_size)
            else:
                # Default moderate signal
                signal = self._default_signal_logic(df, symbol, position_size)
            
            signal['strategy_used'] = strategy_name
            return signal
            
        except Exception as e:
            logger.error(f"Fehler bei Signal Generation für {strategy_name}: {str(e)}")
            return self._create_no_trade_signal(symbol, f"Signal generation error: {str(e)}")
    
    def _momentum_logic(self, df: pd.DataFrame, symbol: str, position_size: float) -> Dict[str, Any]:
        """Momentum Strategy Logic"""
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        current_price = df['close'].iloc[-1]
        sma_10 = df['sma_10'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        
        if sma_10 > sma_20 and current_price > sma_10:
            return {
                'action': 'BUY',
                'symbol': symbol,
                'position_size': position_size,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'confidence': 0.7,
                'reason': 'Momentum uptrend detected'
            }
        elif sma_10 < sma_20 and current_price < sma_10:
            return {
                'action': 'SELL',
                'symbol': symbol,
                'position_size': position_size,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'confidence': 0.6,
                'reason': 'Momentum downtrend detected'
            }
        else:
            return self._create_no_trade_signal(symbol, "No momentum signal")
    
    def _mean_reversion_logic(self, df: pd.DataFrame, symbol: str, position_size: float) -> Dict[str, Any]:
        """Mean Reversion Strategy Logic"""
        df['sma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['upper_bb'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_bb'] = df['sma_20'] - (df['std_20'] * 2)
        
        current_price = df['close'].iloc[-1]
        upper_bb = df['upper_bb'].iloc[-1]
        lower_bb = df['lower_bb'].iloc[-1]
        
        if current_price <= lower_bb:
            return {
                'action': 'BUY',
                'symbol': symbol,
                'position_size': position_size,
                'stop_loss_pct': 1.5,
                'take_profit_pct': 3.0,
                'confidence': 0.8,
                'reason': 'Price below lower Bollinger Band'
            }
        elif current_price >= upper_bb:
            return {
                'action': 'SELL',
                'symbol': symbol,
                'position_size': position_size,
                'stop_loss_pct': 1.5,
                'take_profit_pct': 3.0,
                'confidence': 0.8,
                'reason': 'Price above upper Bollinger Band'
            }
        else:
            return self._create_no_trade_signal(symbol, "Price within normal range")
    
    def _high_risk_logic(self, df: pd.DataFrame, symbol: str, position_size: float) -> Dict[str, Any]:
        """High Risk Strategy Logic - Aggressive but controlled"""
        # RSI + Volume Spike Strategy
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['volume_ma'] = df['volume'].rolling(10).mean()
        
        current_rsi = df['rsi'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume_ma'].iloc[-1]
        
        volume_spike = current_volume > (avg_volume * 1.5)
        
        if current_rsi < 30 and volume_spike:
            return {
                'action': 'BUY',
                'symbol': symbol,
                'position_size': position_size * 0.8,  # Reduce size for high risk
                'stop_loss_pct': 3.0,
                'take_profit_pct': 6.0,
                'confidence': 0.9,
                'reason': 'Oversold with volume spike'
            }
        elif current_rsi > 70 and volume_spike:
            return {
                'action': 'SELL',
                'symbol': symbol,
                'position_size': position_size * 0.8,
                'stop_loss_pct': 3.0,
                'take_profit_pct': 6.0,
                'confidence': 0.9,
                'reason': 'Overbought with volume spike'
            }
        else:
            return self._create_no_trade_signal(symbol, "No high-risk setup")
    
    def _lazy_billionaire_logic(self, df: pd.DataFrame, symbol: str, position_size: float) -> Dict[str, Any]:
        """Lazy Billionaire Strategy - Conservative DCA approach"""
        # Simple DCA with trend filter
        df['sma_50'] = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1] if len(df) >= 50 else current_price
        
        # Only buy on dips in uptrend
        if current_price > sma_50:
            price_change_24h = (current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]
            
            if price_change_24h < -0.02:  # 2% dip
                return {
                    'action': 'BUY',
                    'symbol': symbol,
                    'position_size': position_size * 0.7,  # Conservative size
                    'stop_loss_pct': 5.0,  # Wide stop
                    'take_profit_pct': 8.0,  # Patient target
                    'confidence': 0.6,
                    'reason': 'Buying dip in uptrend'
                }
        
        return self._create_no_trade_signal(symbol, "No lazy billionaire setup")
    
    def _default_signal_logic(self, df: pd.DataFrame, symbol: str, position_size: float) -> Dict[str, Any]:
        """Default moderate signal logic"""
        return self._momentum_logic(df, symbol, position_size)
    
    def _apply_final_risk_check(self, signal: Dict[str, Any], remaining_risk: float) -> Dict[str, Any]:
        """Finaler Risk Check vor Signal-Ausgabe"""
        
        if signal['action'] == 'HOLD':
            return signal
        
        # Position Size vs Remaining Risk
        position_value = signal['position_size']
        if position_value > remaining_risk:
            signal['position_size'] = remaining_risk * 0.8
            signal['reason'] += ' (Position size reduced for risk management)'
        
        # Minimum Position Check
        if signal['position_size'] < 10:
            return self._create_no_trade_signal(signal['symbol'], "Position size too small after risk adjustment")
        
        return signal
    
    def _get_portfolio_value(self) -> float:
        """Holt aktuellen Portfolio-Wert"""
        try:
            # Hier würde echtes Portfolio abgefragt
            # Für Demo: return fixen Wert
            return 1000.0  # 1000€ Demo Portfolio
        except:
            return 1000.0
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Berechnet RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_no_trade_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Erstellt HOLD Signal"""
        return {
            'action': 'HOLD',
            'symbol': symbol,
            'position_size': 0,
            'confidence': 0.0,
            'reason': reason,
            'strategy_used': 'adaptive_auto'
        }
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Gibt aktuelle Strategie-Konfiguration zurück"""
        return {
            'name': 'Adaptive Auto Strategy',
            'daily_risk_limit': self.daily_risk_limit,
            'current_daily_pnl': self.daily_pnl,
            'available_strategies': self.available_strategies,
            'current_strategy': self.current_strategy
        }