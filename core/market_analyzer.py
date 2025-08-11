"""
Market Analyzer - Marktphasen-Erkennung für dynamische Strategieallokation
Analysiert Marktbedingungen und bestimmt die optimale Marktphase
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum

# Try to import TA-Lib, fallback to pandas if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not available, using pandas for technical indicators")


class IndicatorType(Enum):
    """Enum für verschiedene Indikatortypen"""
    VOLATILITY = "volatility"
    TREND = "trend"
    MOMENTUM = "momentum"
    FEAR_GREED = "fear_greed"


@dataclass
class MarketMetrics:
    """Datenklasse für Marktmetriken"""
    price_change_24h: float
    volatility: float
    trend_strength: float
    fear_greed_index: int
    confidence: float
    bollinger_width: float
    atr_normalized: float
    ema_trend: float
    adx_value: float
    timestamp: datetime


class MarketAnalyzer:
    """
    Hauptklasse für die Marktanalyse und Phasen-Erkennung
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # API URLs
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.coinapi_url = "https://rest.coinapi.io/v1/"
        self.binance_url = "https://api.binance.com/api/v3/"
        
        # Konfiguration
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        self.timeframe = self.config.get('timeframe', '1h')
        self.lookback_period = self.config.get('lookback_period', 100)
        
        # Cache für Marktdaten
        self.market_data_cache = {}
        self.cache_duration = timedelta(minutes=5)
        self.last_update = datetime.min
        
        # Gewichtungen für verschiedene Indikatoren
        self.indicator_weights = {
            IndicatorType.VOLATILITY: 0.3,
            IndicatorType.TREND: 0.3,
            IndicatorType.MOMENTUM: 0.2,
            IndicatorType.FEAR_GREED: 0.2
        }
        
        self.logger.info("MarketAnalyzer initialized")
    
    async def get_market_metrics(self) -> Dict:
        """
        Sammelt und analysiert alle Marktmetriken
        """
        try:
            # Prüfe Cache
            if self._is_cache_valid():
                self.logger.debug("Using cached market metrics")
                return self.market_data_cache
            
            # Sammle Daten parallel
            tasks = [
                self._get_fear_greed_index(),
                self._get_price_data(),
                self._get_volatility_metrics()
            ]
            
            fear_greed, price_data, volatility_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verarbeite Ergebnisse
            metrics = await self._process_market_data(fear_greed, price_data, volatility_data)
            
            # Cache aktualisieren
            self.market_data_cache = metrics
            self.last_update = datetime.now()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting market metrics: {e}")
            return self._get_fallback_metrics()
    
    def _is_cache_valid(self) -> bool:
        """Prüft ob der Cache noch gültig ist"""
        return datetime.now() - self.last_update < self.cache_duration
    
    async def _get_fear_greed_index(self) -> int:
        """
        Holt den Fear & Greed Index von der API
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data') and len(data['data']) > 0:
                            return int(data['data'][0]['value'])
            
            self.logger.warning("Could not fetch Fear & Greed Index, using default")
            return 50  # Neutral fallback
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
            return 50
    
    async def _get_price_data(self) -> Dict:
        """
        Holt Preisdaten von Binance API
        """
        try:
            price_data = {}
            
            async with aiohttp.ClientSession() as session:
                # Hole 24h Ticker für alle Symbole
                for symbol in self.symbols:
                    url = f"{self.binance_url}ticker/24hr?symbol={symbol}"
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            price_data[symbol] = {
                                'price_change_percent': float(data.get('priceChangePercent', 0)),
                                'volume': float(data.get('volume', 0)),
                                'count': int(data.get('count', 0))
                            }
                
                # Hole Klines für technische Analyse
                main_symbol = self.symbols[0]  # Hauptsymbol (meist BTC)
                klines_url = f"{self.binance_url}klines?symbol={main_symbol}&interval={self.timeframe}&limit={self.lookback_period}"
                
                async with session.get(klines_url, timeout=10) as response:
                    if response.status == 200:
                        klines = await response.json()
                        price_data['klines'] = self._process_klines(klines)
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return {}
    
    def _process_klines(self, klines: List) -> pd.DataFrame:
        """
        Verarbeitet Klines zu einem DataFrame
        """
        try:
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Konvertiere zu numerischen Werten
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing klines: {e}")
            return pd.DataFrame()
    
    async def _get_volatility_metrics(self) -> Dict:
        """
        Berechnet Volatilitäts-Metriken
        """
        try:
            # Hole historische Daten für Volatilitätsberechnung
            main_symbol = self.symbols[0]
            
            # Simuliere Volatilitätsdaten (in Produktionsumgebung von echter API)
            # Hier würde normalerweise eine historische API-Abfrage stattfinden
            volatility_data = {
                'atr_5': np.random.uniform(0.02, 0.08),  # 2-8% ATR
                'atr_14': np.random.uniform(0.015, 0.06),  # 1.5-6% ATR
                'bollinger_width': np.random.uniform(0.1, 0.4),  # 10-40% Bollinger Band Width
                'volatility_rank': np.random.uniform(0, 100)  # Volatilitäts-Ranking
            }
            
            return volatility_data
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return {}
    
    async def _process_market_data(self, fear_greed: int, price_data: Dict, volatility_data: Dict) -> Dict:
        """
        Verarbeitet alle gesammelten Marktdaten zu finalen Metriken
        """
        try:
            # Berechne gewichtete Preisänderung
            if price_data:
                price_changes = [data.get('price_change_percent', 0) for data in price_data.values() if isinstance(data, dict)]
                avg_price_change = np.mean(price_changes) if price_changes else 0
            else:
                avg_price_change = 0
            
            # Berechne technische Indikatoren
            technical_metrics = self._calculate_technical_indicators(price_data)
            
            # Berechne Gesamtvolatilität
            volatility = self._calculate_overall_volatility(volatility_data)
            
            # Berechne Trend-Stärke
            trend_strength = self._calculate_trend_strength(technical_metrics)
            
            # Berechne Konfidenz
            confidence = self._calculate_confidence(fear_greed, volatility, trend_strength)
            
            # Zusammenstellung der finalen Metriken
            metrics = {
                'price_change_24h': avg_price_change / 100,  # Konvertiere zu Dezimal
                'volatility': volatility,
                'trend_strength': trend_strength,
                'fear_greed_index': fear_greed,
                'confidence': confidence,
                'bollinger_width': volatility_data.get('bollinger_width', 0.2),
                'atr_normalized': volatility_data.get('atr_14', 0.03),
                'ema_trend': technical_metrics.get('ema_trend', 0),
                'adx_value': technical_metrics.get('adx', 25),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Market metrics calculated: Phase indicators ready")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return self._get_fallback_metrics()
    
    def _calculate_technical_indicators(self, price_data: Dict) -> Dict:
        """
        Berechnet technische Indikatoren
        """
        try:
            indicators = {}
            
            # Prüfe ob Klines verfügbar sind
            if 'klines' in price_data and not price_data['klines'].empty:
                df = price_data['klines']
                
                # EMA Trend (12 vs 26)
                if HAS_TALIB:
                    ema_12 = talib.EMA(df['close'], timeperiod=12)
                    ema_26 = talib.EMA(df['close'], timeperiod=26)
                    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                else:
                    ema_12 = df['close'].ewm(span=12).mean()
                    ema_26 = df['close'].ewm(span=26).mean()
                    adx = self._calculate_adx_simple(df)
                
                # Trend-Richtung
                if len(ema_12) > 0 and len(ema_26) > 0:
                    latest_ema_12 = ema_12.iloc[-1]
                    latest_ema_26 = ema_26.iloc[-1]
                    ema_trend = (latest_ema_12 - latest_ema_26) / latest_ema_26
                    indicators['ema_trend'] = ema_trend
                
                # ADX Wert
                if len(adx) > 0 and not np.isnan(adx.iloc[-1]):
                    indicators['adx'] = adx.iloc[-1]
                else:
                    indicators['adx'] = 25  # Neutral
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {'ema_trend': 0, 'adx': 25}
    
    def _calculate_adx_simple(self, df: pd.DataFrame) -> pd.Series:
        """
        Vereinfachte ADX-Berechnung ohne TA-Lib
        """
        try:
            # Vereinfachte ADX-Berechnung
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = ranges.rolling(window=14).mean()
            
            # Vereinfachte Directional Movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            return adx
            
        except Exception as e:
            self.logger.error(f"Error calculating simple ADX: {e}")
            return pd.Series([25] * len(df))
    
    def _calculate_overall_volatility(self, volatility_data: Dict) -> float:
        """
        Berechnet die Gesamtvolatilität
        """
        try:
            if not volatility_data:
                return 0.2  # Default moderate volatility
            
            # Gewichtete Volatilität verschiedener Metriken
            atr_weight = 0.4
            bollinger_weight = 0.3
            rank_weight = 0.3
            
            atr_vol = volatility_data.get('atr_14', 0.03)
            bollinger_vol = volatility_data.get('bollinger_width', 0.2)
            rank_vol = volatility_data.get('volatility_rank', 50) / 100
            
            overall_vol = (atr_vol * atr_weight + 
                          bollinger_vol * bollinger_weight + 
                          rank_vol * rank_weight)
            
            return min(max(overall_vol, 0.0), 1.0)  # Clamp zwischen 0 und 1
            
        except Exception as e:
            self.logger.error(f"Error calculating overall volatility: {e}")
            return 0.2
    
    def _calculate_trend_strength(self, technical_metrics: Dict) -> float:
        """
        Berechnet die Trend-Stärke
        """
        try:
            ema_trend = technical_metrics.get('ema_trend', 0)
            adx_value = technical_metrics.get('adx', 25)
            
            # Normalisiere ADX (0-100 -> 0-1)
            adx_normalized = min(adx_value / 100, 1.0)
            
            # Kombiniere EMA-Trend mit ADX
            # ADX zeigt Stärke, EMA zeigt Richtung
            if adx_value > 25:  # Starker Trend
                trend_strength = abs(ema_trend) * adx_normalized
            else:  # Schwacher Trend
                trend_strength = abs(ema_trend) * 0.5
            
            return min(max(trend_strength, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.3
    
    def _calculate_confidence(self, fear_greed: int, volatility: float, trend_strength: float) -> float:
        """
        Berechnet die Konfidenz der Marktphasen-Erkennung
        """
        try:
            # Konfidenz basierend auf Datenverfügbarkeit und Eindeutigkeit
            base_confidence = 0.5
            
            # Hohe Fear/Greed Werte = höhere Konfidenz
            if fear_greed <= 20 or fear_greed >= 80:
                base_confidence += 0.3
            elif fear_greed <= 30 or fear_greed >= 70:
                base_confidence += 0.2
            else:
                base_confidence += 0.1
            
            # Hohe Volatilität = niedrigere Konfidenz
            volatility_penalty = min(volatility * 0.3, 0.2)
            base_confidence -= volatility_penalty
            
            # Starke Trends = höhere Konfidenz
            if trend_strength > 0.6:
                base_confidence += 0.2
            elif trend_strength > 0.4:
                base_confidence += 0.1
            
            return min(max(base_confidence, 0.1), 0.95)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_fallback_metrics(self) -> Dict:
        """
        Gibt Fallback-Metriken bei Fehlern zurück
        """
        return {
            'price_change_24h': 0.0,
            'volatility': 0.2,
            'trend_strength': 0.3,
            'fear_greed_index': 50,
            'confidence': 0.3,
            'bollinger_width': 0.2,
            'atr_normalized': 0.03,
            'ema_trend': 0.0,
            'adx_value': 25,
            'timestamp': datetime.now()
        }
    
    def get_market_phase_description(self, phase: str) -> str:
        """
        Gibt eine Beschreibung der Marktphase zurück
        """
        descriptions = {
            'sideways': 'Seitwärtsmarkt - Niedrige Volatilität, unklare Richtung',
            'bull': 'Bullmarkt - Starker Aufwärtstrend, hohe Kaufkraft',
            'volatile': 'Volatiler Markt - Hohe Schwankungen, unvorhersagbare Bewegungen',
            'bear': 'Bärenmarkt - Starker Abwärtstrend, Verkaufsdruck',
            'extreme_fear': 'Extreme Angst - Panik am Markt, Kapitalschutz prioritär'
        }
        return descriptions.get(phase, 'Unbekannte Marktphase')
    
    async def get_detailed_analysis(self) -> Dict:
        """
        Gibt eine detaillierte Marktanalyse zurück
        """
        try:
            metrics = await self.get_market_metrics()
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'indicators': {
                    'volatility_level': self._classify_volatility(metrics['volatility']),
                    'trend_classification': self._classify_trend(metrics['trend_strength'], metrics['ema_trend']),
                    'fear_greed_classification': self._classify_fear_greed(metrics['fear_greed_index']),
                    'market_regime': self._determine_market_regime(metrics)
                },
                'recommendations': self._generate_recommendations(metrics)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            return {'error': str(e)}
    
    def _classify_volatility(self, volatility: float) -> str:
        """Klassifiziert Volatilität"""
        if volatility < 0.15:
            return 'LOW'
        elif volatility < 0.35:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    def _classify_trend(self, trend_strength: float, ema_trend: float) -> str:
        """Klassifiziert Trend"""
        if trend_strength < 0.3:
            return 'SIDEWAYS'
        elif ema_trend > 0.02:
            return 'STRONG_UP'
        elif ema_trend > 0.005:
            return 'WEAK_UP'
        elif ema_trend < -0.02:
            return 'STRONG_DOWN'
        elif ema_trend < -0.005:
            return 'WEAK_DOWN'
        else:
            return 'NEUTRAL'
    
    def _classify_fear_greed(self, fear_greed: int) -> str:
        """Klassifiziert Fear & Greed"""
        if fear_greed <= 20:
            return 'EXTREME_FEAR'
        elif fear_greed <= 40:
            return 'FEAR'
        elif fear_greed <= 60:
            return 'NEUTRAL'
        elif fear_greed <= 80:
            return 'GREED'
        else:
            return 'EXTREME_GREED'
    
    def _determine_market_regime(self, metrics: Dict) -> str:
        """Bestimmt das Marktregime"""
        volatility = metrics['volatility']
        trend_strength = metrics['trend_strength']
        fear_greed = metrics['fear_greed_index']
        price_change = metrics['price_change_24h']
        
        if fear_greed <= 20:
            return 'EXTREME_FEAR'
        elif volatility > 0.4:
            return 'VOLATILE'
        elif trend_strength > 0.6 and price_change > 0.05:
            return 'BULL'
        elif trend_strength > 0.6 and price_change < -0.05:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generiert Empfehlungen basierend auf Metriken"""
        recommendations = []
        
        volatility = metrics['volatility']
        trend_strength = metrics['trend_strength']
        fear_greed = metrics['fear_greed_index']
        
        if fear_greed <= 20:
            recommendations.append("Extreme Angst: Kapitalschutz priorisieren")
            recommendations.append("Stablecoin-Parking in Betracht ziehen")
        elif volatility > 0.4:
            recommendations.append("Hohe Volatilität: Arbitrage-Strategien bevorzugen")
            recommendations.append("Positionsgrößen reduzieren")
        elif trend_strength > 0.6:
            recommendations.append("Starker Trend: Momentum-Strategien nutzen")
            recommendations.append("Trend-folgende Positionen eingehen")
        else:
            recommendations.append("Seitwärtsmarkt: Grid-Trading und Range-Strategien")
            recommendations.append("Mean-Reversion-Ansätze prüfen")
        
        return recommendations