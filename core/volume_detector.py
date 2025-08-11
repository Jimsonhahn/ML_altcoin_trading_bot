#!/usr/bin/env python3
"""
Volume Spike Detection System
============================

Detects significant volume anomalies for high-risk trading:
- 300%+ volume spikes identification
- Multi-timeframe volume analysis
- Breakout confirmation with volume
- Real-time volume monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import ccxt

logger = logging.getLogger(__name__)

@dataclass
class VolumeSpike:
    """Volume spike detection result"""
    symbol: str
    timestamp: datetime
    current_volume: float
    average_volume: float
    spike_ratio: float
    confidence: float
    timeframe: str
    price_change: float
    breakout_detected: bool
    metadata: Dict[str, Any]

@dataclass
class VolumeProfile:
    """Historical volume profile for a symbol"""
    symbol: str
    timeframe: str
    volume_history: deque
    average_volume: float
    std_volume: float
    last_update: datetime
    spike_threshold: float

class VolumeDetector:
    """
    Advanced volume spike detection for high-risk opportunities
    
    Features:
    - Multi-timeframe volume analysis (1m, 5m, 15m)
    - Statistical anomaly detection
    - Breakout confirmation
    - Real-time monitoring
    - Configurable spike thresholds
    """
    
    def __init__(self, 
                 exchanges: List[str] = ['binance'],
                 timeframes: List[str] = ['1m', '5m', '15m'],
                 spike_threshold: float = 3.0,  # 300% spike
                 history_periods: int = 100,
                 min_confidence: float = 0.7):
        
        self.exchanges = {}
        self.timeframes = timeframes
        self.spike_threshold = spike_threshold
        self.history_periods = history_periods
        self.min_confidence = min_confidence
        
        # Volume profiles by symbol and timeframe
        self.volume_profiles: Dict[str, Dict[str, VolumeProfile]] = defaultdict(dict)
        
        # Recent spikes tracking
        self.recent_spikes: Dict[str, List[VolumeSpike]] = defaultdict(list)
        
        # Initialize exchanges
        self._init_exchanges(exchanges)
        
        logger.info(f"📊 Volume Detector initialized")
        logger.info(f"🎯 Spike threshold: {self.spike_threshold:.1f}x ({(self.spike_threshold-1)*100:.0f}%)")
        logger.info(f"⏰ Timeframes: {', '.join(self.timeframes)}")
    
    def _init_exchanges(self, exchange_names: List[str]):
        """Initialize exchange connections"""
        for name in exchange_names:
            try:
                if name.lower() == 'binance':
                    self.exchanges[name] = ccxt.binance({
                        'apiKey': '',  # Read-only for volume data
                        'secret': '',
                        'sandbox': False,
                        'rateLimit': 1200,
                    })
                # Add other exchanges as needed
                logger.info(f"✅ Connected to {name}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to {name}: {e}")
    
    async def detect_volume_spikes(self, symbols: List[str]) -> List[VolumeSpike]:
        """
        Detect volume spikes across multiple symbols and timeframes
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of detected volume spikes
        """
        all_spikes = []
        
        for symbol in symbols:
            for timeframe in self.timeframes:
                try:
                    spike = await self._analyze_symbol_timeframe(symbol, timeframe)
                    if spike and spike.confidence >= self.min_confidence:
                        all_spikes.append(spike)
                        
                        # Store in recent spikes
                        self.recent_spikes[symbol].append(spike)
                        self._cleanup_old_spikes(symbol)
                        
                        logger.info(f"🔥 Volume spike detected: {symbol} {timeframe} "
                                  f"{spike.spike_ratio:.1f}x volume "
                                  f"(confidence: {spike.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                    continue
        
        # Sort by confidence and spike ratio
        all_spikes.sort(key=lambda x: (x.confidence, x.spike_ratio), reverse=True)
        
        return all_spikes
    
    async def _analyze_symbol_timeframe(self, symbol: str, timeframe: str) -> Optional[VolumeSpike]:
        """Analyze single symbol/timeframe combination"""
        
        # Get historical data
        exchange = list(self.exchanges.values())[0]  # Use first available exchange
        
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=self.history_periods + 1)
            
            if len(ohlcv) < 20:  # Need minimum data
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Current and historical volumes
            current_volume = df['volume'].iloc[-1]
            historical_volumes = df['volume'].iloc[:-1]
            
            # Skip if no volume
            if current_volume <= 0:
                return None
            
            # Calculate statistics
            avg_volume = historical_volumes.mean()
            std_volume = historical_volumes.std()
            
            if avg_volume <= 0:
                return None
            
            # Calculate spike ratio
            spike_ratio = current_volume / avg_volume
            
            # Check if it's a significant spike
            if spike_ratio < self.spike_threshold:
                return None
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_spike_confidence(
                df, current_volume, avg_volume, std_volume, spike_ratio
            )
            
            # Check for breakout confirmation
            breakout_detected = self._detect_breakout(df)
            
            # Calculate price change
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # Update volume profile
            self._update_volume_profile(symbol, timeframe, historical_volumes, avg_volume, std_volume)
            
            return VolumeSpike(
                symbol=symbol,
                timestamp=df['timestamp'].iloc[-1],
                current_volume=current_volume,
                average_volume=avg_volume,
                spike_ratio=spike_ratio,
                confidence=confidence,
                timeframe=timeframe,
                price_change=price_change,
                breakout_detected=breakout_detected,
                metadata={
                    'std_volume': std_volume,
                    'volume_zscore': (current_volume - avg_volume) / std_volume if std_volume > 0 else 0,
                    'recent_high': df['high'].iloc[-10:].max(),
                    'recent_low': df['low'].iloc[-10:].min(),
                    'current_price': df['close'].iloc[-1]
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_spike_confidence(self, df: pd.DataFrame, current_volume: float, 
                                  avg_volume: float, std_volume: float, spike_ratio: float) -> float:
        """Calculate confidence score for volume spike"""
        confidence = 0.0
        
        # Base confidence from spike magnitude
        if spike_ratio >= 5.0:
            confidence += 0.4
        elif spike_ratio >= 4.0:
            confidence += 0.3
        elif spike_ratio >= 3.0:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Z-score based confidence
        if std_volume > 0:
            z_score = (current_volume - avg_volume) / std_volume
            if z_score >= 3.0:  # 3+ standard deviations
                confidence += 0.3
            elif z_score >= 2.0:
                confidence += 0.2
            elif z_score >= 1.5:
                confidence += 0.1
        
        # Price movement confirmation
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        if abs(price_change) > 0.05:  # 5%+ price move
            confidence += 0.2
        elif abs(price_change) > 0.02:  # 2%+ price move
            confidence += 0.1
        
        # Volume consistency over recent periods
        recent_volumes = df['volume'].iloc[-5:]
        if len(recent_volumes) >= 3:
            recent_avg = recent_volumes.mean()
            if recent_avg > avg_volume * 1.5:  # Sustained higher volume
                confidence += 0.1
        
        # Breakout confirmation bonus
        if self._detect_breakout(df):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _detect_breakout(self, df: pd.DataFrame) -> bool:
        """Detect if volume spike coincides with price breakout"""
        if len(df) < 20:
            return False
        
        try:
            # Recent range analysis
            recent_data = df.iloc[-20:]
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            range_size = (range_high - range_low) / range_low
            
            current_price = df['close'].iloc[-1]
            
            # Check for breakout above recent high
            if current_price > range_high * 1.01:  # 1% above recent high
                return True
            
            # Check for breakdown below recent low
            if current_price < range_low * 0.99:  # 1% below recent low
                return True
            
            # Check for range compression and expansion
            if range_size < 0.05:  # Tight range (5%)
                recent_volatility = recent_data['close'].pct_change().std()
                current_move = abs((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2])
                
                if current_move > recent_volatility * 2:  # 2x normal volatility
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return False
    
    def _update_volume_profile(self, symbol: str, timeframe: str, volumes: pd.Series, 
                             avg_volume: float, std_volume: float):
        """Update volume profile for symbol/timeframe"""
        profile_key = f"{symbol}_{timeframe}"
        
        if profile_key not in self.volume_profiles[symbol]:
            self.volume_profiles[symbol][timeframe] = VolumeProfile(
                symbol=symbol,
                timeframe=timeframe,
                volume_history=deque(maxlen=self.history_periods),
                average_volume=avg_volume,
                std_volume=std_volume,
                last_update=datetime.now(),
                spike_threshold=self.spike_threshold
            )
        
        profile = self.volume_profiles[symbol][timeframe]
        profile.volume_history.extend(volumes.tolist())
        profile.average_volume = avg_volume
        profile.std_volume = std_volume
        profile.last_update = datetime.now()
    
    def _cleanup_old_spikes(self, symbol: str, max_age_hours: int = 24):
        """Remove old spikes from tracking"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        self.recent_spikes[symbol] = [
            spike for spike in self.recent_spikes[symbol]
            if spike.timestamp > cutoff_time
        ]
    
    def get_recent_spikes(self, symbol: str = None, hours: int = 6) -> List[VolumeSpike]:
        """Get recent volume spikes"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if symbol:
            return [
                spike for spike in self.recent_spikes.get(symbol, [])
                if spike.timestamp > cutoff_time
            ]
        else:
            all_spikes = []
            for spikes in self.recent_spikes.values():
                all_spikes.extend([
                    spike for spike in spikes
                    if spike.timestamp > cutoff_time
                ])
            
            return sorted(all_spikes, key=lambda x: x.timestamp, reverse=True)
    
    def get_top_volume_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top volume spike candidates for trading"""
        recent_spikes = self.get_recent_spikes(hours=2)  # Very recent
        
        # Group by symbol and get best spike per symbol
        symbol_best = {}
        for spike in recent_spikes:
            if spike.symbol not in symbol_best or spike.confidence > symbol_best[spike.symbol].confidence:
                symbol_best[spike.symbol] = spike
        
        # Convert to trading candidates
        candidates = []
        for spike in symbol_best.values():
            candidates.append({
                'symbol': spike.symbol,
                'confidence': spike.confidence,
                'spike_ratio': spike.spike_ratio,
                'price_change': spike.price_change,
                'breakout_detected': spike.breakout_detected,
                'timestamp': spike.timestamp,
                'timeframe': spike.timeframe,
                'score': spike.confidence * spike.spike_ratio,  # Combined score
                'metadata': spike.metadata
            })
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:limit]
    
    async def monitor_symbols(self, symbols: List[str], callback=None) -> None:
        """Continuously monitor symbols for volume spikes"""
        logger.info(f"🔄 Starting volume monitoring for {len(symbols)} symbols")
        
        while True:
            try:
                spikes = await self.detect_volume_spikes(symbols)
                
                for spike in spikes:
                    if callback:
                        await callback(spike)
                    else:
                        logger.info(f"🚨 Volume Alert: {spike.symbol} "
                                  f"{spike.spike_ratio:.1f}x volume spike "
                                  f"(confidence: {spike.confidence:.2f})")
                
                # Wait before next scan
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in volume monitoring: {e}")
                await asyncio.sleep(30)  # Shorter retry interval

# Utility function for high-risk strategy integration
def create_volume_detector(config: Dict[str, Any] = None) -> VolumeDetector:
    """Create volume detector with configuration"""
    if config is None:
        config = {}
    
    return VolumeDetector(
        exchanges=config.get('exchanges', ['binance']),
        timeframes=config.get('timeframes', ['1m', '5m', '15m']),
        spike_threshold=config.get('spike_threshold', 3.0),
        history_periods=config.get('history_periods', 100),
        min_confidence=config.get('min_confidence', 0.7)
    )