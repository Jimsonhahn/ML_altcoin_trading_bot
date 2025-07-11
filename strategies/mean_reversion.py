#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean Reversion Strategy - Complete Working Implementation
=========================================================

Diese Strategie nutzt Bollinger Bands und RSI, um Überkaufte/Überverkaufte
Situationen zu identifizieren und von Preiskorrekturen zu profitieren.

Funktionsweise:
- Kauft wenn der Preis unter das untere Bollinger Band fällt (überverkauft)
- Verkauft wenn der Preis über das obere Bollinger Band steigt (überkauft)
- RSI wird zur Bestätigung verwendet
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
from datetime import datetime

from .strategy_base import Strategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Trading Strategy

    Nutzt Preiskorrekturen nach extremen Bewegungen.
    Ideal für seitwärts tendierende Märkte.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert die Mean Reversion Strategy

        Args:
            config: Konfigurationsdictionary mit Strategie-Parametern
        """
        super().__init__(config)

        # Strategy name for logging
        self.name = "Mean Reversion"

        # Lookback period for calculations
        self.lookback_period = config.get('lookback_period', 20)

        # Entry/Exit thresholds in standard deviations
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)

        # Bollinger Bands parameters
        self.bb_period = config.get('bollinger_period', 20)
        self.bb_std = config.get('bollinger_std', 2.0)

        # RSI parameters for confirmation
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)

        # Additional filters
        self.use_volume_filter = config.get('use_volume_filter', True)
        self.volume_threshold = config.get('volume_threshold', 1.2)  # 20% above average

        # Risk management
        self.max_positions = config.get('max_positions', 3)
        self.position_timeout = config.get('position_timeout', 48)  # hours

        logger.info(f"Mean Reversion Strategy initialized:")
        logger.info(f"  - Bollinger Period: {self.bb_period}")
        logger.info(f"  - Bollinger Std: {self.bb_std}")
        logger.info(f"  - RSI Period: {self.rsi_period}")
        logger.info(f"  - RSI Oversold: {self.rsi_oversold}")
        logger.info(f"  - RSI Overbought: {self.rsi_overbought}")

    def calculate_signal(self, data: pd.DataFrame, symbol: str,
                         current_position: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate mean reversion trading signal based on Bollinger Bands and RSI

        Args:
            data: DataFrame mit OHLCV Daten
            symbol: Trading Symbol (z.B. 'BTC/USDT')
            current_position: Aktuelle Position falls vorhanden

        Returns:
            Tuple of (signal, signal_data)
            signal: 'BUY', 'SELL', or 'HOLD'
            signal_data: Dict mit Konfidenz und anderen Metriken
        """
        # Validate data
        if data is None or data.empty:
            logger.warning(f"No data available for {symbol}")
            return 'HOLD', {'confidence': 0.0, 'reason': 'no_data'}

        # Check if we have enough data
        min_required = max(self.bb_period, self.rsi_period) + 1
        if len(data) < min_required:
            logger.debug(f"Insufficient data for {symbol}: {len(data)} < {min_required}")
            return 'HOLD', {
                'confidence': 0.0,
                'reason': 'insufficient_data',
                'required_candles': min_required,
                'available_candles': len(data)
            }

        try:
            # Extract price and volume data
            close_prices = data['close']
            volume = data['volume'] if 'volume' in data.columns else None

            # Calculate Bollinger Bands
            sma = close_prices.rolling(window=self.bb_period).mean()
            std = close_prices.rolling(window=self.bb_period).std()
            upper_band = sma + (self.bb_std * std)
            lower_band = sma - (self.bb_std * std)

            # Get current values
            current_price = float(close_prices.iloc[-1])
            current_sma = float(sma.iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])
            current_std = float(std.iloc[-1])

            # Calculate position relative to bands
            band_width = current_upper - current_lower
            price_position = (current_price - current_lower) / band_width if band_width > 0 else 0.5

            # Calculate standard deviations from mean
            deviations_from_mean = (current_price - current_sma) / current_std if current_std > 0 else 0

            # Calculate RSI
            rsi_series = self._calculate_rsi(close_prices, self.rsi_period)
            current_rsi = float(rsi_series.iloc[-1])

            # Volume analysis
            volume_ok = True
            volume_ratio = 1.0
            if self.use_volume_filter and volume is not None and len(volume) > 20:
                avg_volume = volume.rolling(window=20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    volume_ok = volume_ratio >= self.volume_threshold

            # Initialize signal and confidence
            signal = 'HOLD'
            confidence = 0.0
            reason = 'no_signal'

            # Check if we have a position
            if current_position is None:
                # No position - look for entry signals

                # BUY Signal: Price at lower band + RSI oversold + volume confirmation
                if (current_price <= current_lower and
                        current_rsi < self.rsi_oversold and
                        volume_ok):

                    signal = 'BUY'
                    # Confidence based on how far below the band and how oversold
                    distance_factor = min(1.0, abs(deviations_from_mean) / 3)
                    rsi_factor = min(1.0, (self.rsi_oversold - current_rsi) / self.rsi_oversold)
                    confidence = 0.5 + (distance_factor * 0.25) + (rsi_factor * 0.25)
                    confidence = min(0.95, confidence)
                    reason = 'oversold_at_lower_band'

                # SELL Signal: Price at upper band + RSI overbought + volume confirmation
                elif (current_price >= current_upper and
                      current_rsi > self.rsi_overbought and
                      volume_ok):

                    signal = 'SELL'
                    # Confidence based on how far above the band and how overbought
                    distance_factor = min(1.0, abs(deviations_from_mean) / 3)
                    rsi_factor = min(1.0, (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
                    confidence = 0.5 + (distance_factor * 0.25) + (rsi_factor * 0.25)
                    confidence = min(0.95, confidence)
                    reason = 'overbought_at_upper_band'

            else:
                # Have position - look for exit signals
                position_age_hours = 0
                if hasattr(current_position, 'entry_time'):
                    position_age_hours = (datetime.now() - current_position.entry_time).total_seconds() / 3600

                if current_position.side == 'buy':
                    # Long position - exit conditions
                    if current_price >= current_sma:
                        # Price returned to mean - take profit
                        signal = 'SELL'
                        confidence = 0.8
                        reason = 'price_returned_to_mean'

                    elif current_rsi > self.rsi_overbought:
                        # RSI overbought - strong exit signal
                        signal = 'SELL'
                        confidence = 0.9
                        reason = 'rsi_overbought'

                    elif position_age_hours > self.position_timeout:
                        # Position timeout
                        signal = 'SELL'
                        confidence = 0.6
                        reason = 'position_timeout'

                else:  # short position
                    # Short position - exit conditions
                    if current_price <= current_sma:
                        # Price returned to mean - take profit
                        signal = 'BUY'
                        confidence = 0.8
                        reason = 'price_returned_to_mean'

                    elif current_rsi < self.rsi_oversold:
                        # RSI oversold - strong exit signal
                        signal = 'BUY'
                        confidence = 0.9
                        reason = 'rsi_oversold'

                    elif position_age_hours > self.position_timeout:
                        # Position timeout
                        signal = 'BUY'
                        confidence = 0.6
                        reason = 'position_timeout'

            # Prepare comprehensive signal data
            signal_data = {
                'signal': signal,
                'confidence': float(confidence),
                'strategy': 'mean_reversion',
                'reason': reason,
                'indicators': {
                    'current_price': current_price,
                    'sma': current_sma,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'band_width': float(band_width),
                    'price_position': float(price_position),  # 0 = at lower band, 1 = at upper band
                    'deviations_from_mean': float(deviations_from_mean),
                    'rsi': current_rsi,
                    'volume_ratio': float(volume_ratio)
                },
                'thresholds': {
                    'rsi_oversold': self.rsi_oversold,
                    'rsi_overbought': self.rsi_overbought,
                    'bb_std': self.bb_std
                }
            }

            # Add position info if available
            if current_position:
                signal_data['position_info'] = {
                    'side': current_position.side,
                    'entry_price': getattr(current_position, 'entry_price', 0),
                    'unrealized_pnl': self._calculate_unrealized_pnl(current_position, current_price)
                }

            return signal, signal_data

        except Exception as e:
            logger.error(f"Error calculating mean reversion signal for {symbol}: {e}")
            return 'HOLD', {
                'confidence': 0.0,
                'reason': 'calculation_error',
                'error': str(e)
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            prices: Series of prices
            period: RSI period (default 14)

        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral RSI for NaN values

        return rsi

    def _calculate_unrealized_pnl(self, position: Any, current_price: float) -> float:
        """Calculate unrealized P&L for a position"""
        if not hasattr(position, 'entry_price') or not hasattr(position, 'amount'):
            return 0.0

        if position.side == 'buy':
            return (current_price - position.entry_price) * position.amount
        else:  # short
            return (position.entry_price - current_price) * position.amount

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy"""
        return {
            'name': self.name,
            'type': 'mean_reversion',
            'description': 'Trades price reversions to the mean using Bollinger Bands and RSI',
            'parameters': {
                'bollinger_period': self.bb_period,
                'bollinger_std': self.bb_std,
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'use_volume_filter': self.use_volume_filter
            },
            'suitable_for': 'Ranging/sideways markets',
            'risk_level': 'Medium'
        }