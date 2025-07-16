#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature-Extraktion für ML-Komponenten.
Dieses Modul enthält gemeinsame Funktionen zur Feature-Extraktion für ML-Analysen.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Logging einrichten
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Kapselt Funktionen zur Feature-Extraktion für ML-Analysen.
    """

    def __init__(self, settings: Any):  # `settings` allows dynamic config for indicators if needed
        self.settings = settings
        # Potenziell hier lookback-Perioden oder andere Indikatorparameter aus den Settings laden
        # self.rsi_period = settings.get('features.rsi_period', 14)
        # self.ma_short = settings.get('features.ma_short', 20)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet gängige technische Indikatoren für einen OHLCV-DataFrame.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        df_indicators = df.copy()

        # Prüfen, ob erforderliche Spalten vorhanden sind, falls nicht, mit 0 füllen (für konsistente Berechnung)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df_indicators.columns:
                df_indicators[col] = np.nan  # Use NaN to distinguish missing from actual zero
                logger.warning(f"Fehlende Spalte für Indikatorberechnung: {col}. Mit NaN gefüllt.")

        # Stelle sicher, dass numerische Typen vorliegen, bevor gerechnet wird
        for col in required_columns:
            df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')

        # 1. Returns
        df_indicators['return'] = df_indicators['close'].pct_change()
        df_indicators['log_return'] = np.log(df_indicators['close'] / df_indicators['close'].shift(1))

        # 2. Moving Averages
        df_indicators['sma_5'] = df_indicators['close'].rolling(window=5).mean()
        df_indicators['sma_10'] = df_indicators['close'].rolling(window=10).mean()
        df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
        df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()

        # 3. Exponential Moving Averages
        df_indicators['ema_5'] = df_indicators['close'].ewm(span=5, adjust=False).mean()
        df_indicators['ema_10'] = df_indicators['close'].ewm(span=10, adjust=False).mean()
        df_indicators['ema_12'] = df_indicators['close'].ewm(span=12, adjust=False).mean()  # For MACD
        df_indicators['ema_13'] = df_indicators['close'].ewm(span=13, adjust=False).mean()  # For Elder Ray
        df_indicators['ema_20'] = df_indicators['close'].ewm(span=20, adjust=False).mean()
        df_indicators['ema_26'] = df_indicators['close'].ewm(span=26, adjust=False).mean()  # For MACD
        df_indicators['ema_50'] = df_indicators['close'].ewm(span=50, adjust=False).mean()
        df_indicators['ema_200'] = df_indicators['close'].ewm(span=200, adjust=False).mean()

        # 4. Bollinger Bands
        df_indicators['bollinger_mid'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['bollinger_std'] = df_indicators['close'].rolling(window=20).std()
        df_indicators['bollinger_upper'] = df_indicators['bollinger_mid'] + (df_indicators['bollinger_std'] * 2)
        df_indicators['bollinger_lower'] = df_indicators['bollinger_mid'] - (df_indicators['bollinger_std'] * 2)
        # Avoid division by zero
        df_indicators['bollinger_pct'] = (df_indicators['close'] - df_indicators['bollinger_lower']) / (
                df_indicators['bollinger_upper'] - df_indicators['bollinger_lower']).replace(0, np.nan)

        # 5. RSI
        delta = df_indicators['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use .mean() for correct rolling average calculation
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_indicators['rsi_14'] = 100 - (100 / (1 + rs))

        # 6. MACD (needs ema_12 and ema_26, which are calculated above)
        df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
        df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
        df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']

        # 7. Volatilität
        df_indicators['volatility_5'] = df_indicators['return'].rolling(window=5).std()
        df_indicators['volatility_10'] = df_indicators['return'].rolling(window=10).std()
        df_indicators['volatility_20'] = df_indicators['return'].rolling(window=20).std()

        # 8. Average True Range (ATR)
        high_low = df_indicators['high'] - df_indicators['low']
        high_close_prev = abs(df_indicators['high'] - df_indicators['close'].shift(1))
        low_close_prev = abs(df_indicators['low'] - df_indicators['close'].shift(1))
        df_indicators['tr'] = pd.DataFrame(
            {'high_low': high_low, 'high_close_prev': high_close_prev, 'low_close_prev': low_close_prev}).max(axis=1)
        df_indicators['atr_14'] = df_indicators['tr'].ewm(span=14,
                                                          adjust=False).mean()  # Using EMA for ATR is common, or can use rolling mean

        # 9. Volumen-Indikatoren
        df_indicators['volume_sma_5'] = df_indicators['volume'].rolling(window=5).mean()
        df_indicators['volume_sma_20'] = df_indicators['volume'].rolling(window=20).mean()
        # Avoid division by zero
        df_indicators['volume_ratio'] = (df_indicators['volume'] / df_indicators['volume_sma_20']).replace(
            [np.inf, -np.inf], np.nan)

        # 10. On-Balance Volume (OBV)
        df_indicators['obv'] = (df_indicators['volume'] * df_indicators['close'].diff().apply(np.sign)).cumsum()

        # 11. Stochastic Oscillator
        lowest_low = df_indicators['low'].rolling(window=14).min()
        highest_high = df_indicators['high'].rolling(window=14).max()
        df_indicators['stoch_k'] = 100 * ((df_indicators['close'] - lowest_low) / (highest_high - lowest_low)).replace(
            0, np.nan)  # Avoid div by zero
        df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()

        # 12. Commodity Channel Index (CCI)
        typical_price = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
        cci_ma = typical_price.rolling(window=20).mean()
        cci_std = typical_price.rolling(window=20).std()
        # Avoid division by zero
        df_indicators['cci_20'] = (typical_price - cci_ma) / (0.015 * cci_std).replace(0, np.nan)

        # 13. Rate of Change (ROC)
        # Avoid division by zero
        df_indicators['roc_10'] = ((df_indicators['close'] - df_indicators['close'].shift(10)) / df_indicators[
            'close'].shift(10)).replace(0, np.nan) * 100

        # 14. Williams %R
        period = 14
        highest_high_w = df_indicators['high'].rolling(window=period).max()
        lowest_low_w = df_indicators['low'].rolling(window=period).min()
        # Avoid division by zero
        df_indicators['williams_r'] = -100 * (
                    (highest_high_w - df_indicators['close']) / (highest_high_w - lowest_low_w)).replace(0, np.nan)

        # 15. Parabolic SAR (vereinfacht) - Requires more complex logic, simplified as placeholder for feature set consistency
        # For a true PSAR, it's an iterative calculation. Here, just a simple derived value.
        df_indicators['psar'] = df_indicators['close'].shift(1)

        return df_indicators

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet erweiterte technische Indikatoren für einen OHLCV-DataFrame.
        Diese Funktionen sind für spezifische Strategien oder erweiterte Analysen gedacht.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        df_adv = df.copy()

        # 1. Ichimoku Cloud
        # Stellen Sie sicher, dass sma_200 (für Kumo) und andere Basis-EMAs berechnet wurden
        # Tenkan-sen (Conversion Line): (9-Period High + 9-Period Low) / 2
        high_9 = df_adv['high'].rolling(window=9).max()
        low_9 = df_adv['low'].rolling(window=9).min()
        df_adv['ichimoku_tenkan'] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line): (26-Period High + 26-Period Low) / 2
        high_26 = df_adv['high'].rolling(window=26).max()
        low_26 = df_adv['low'].rolling(window=26).min()
        df_adv['ichimoku_kijun'] = (high_26 + low_26) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2 plotted 26 periods ahead
        df_adv['ichimoku_senkou_a'] = ((df_adv['ichimoku_tenkan'] + df_adv['ichimoku_kijun']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-Period High + 52-Period Low) / 2 plotted 26 periods ahead
        high_52 = df_adv['high'].rolling(window=52).max()
        low_52 = df_adv['low'].rolling(window=52).min()
        df_adv['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)

        # Chikou Span (Lagging Span): Close price plotted 26 periods behind
        df_adv['ichimoku_chikou'] = df_adv['close'].shift(-26)

        # 2. Elder Ray Index (requires ema_13 which should be calculated by calculate_technical_indicators)
        df_adv['elder_bull_power'] = df_adv['high'] - df_adv['ema_13']
        df_adv['elder_bear_power'] = df_adv['low'] - df_adv['ema_13']

        # 3. Klinger Volume Oscillator
        if 'volume' in df_adv.columns and 'high' in df_adv.columns and 'low' in df_adv.columns and 'close' in df_adv.columns:
            # Volume Force (VF)
            # CM (Trend Component) = High - Low
            # If (current_close > previous_close), CM = CM
            # Else if (current_close < previous_close), CM = -CM
            # Else (current_close == previous_close), CM = 0

            # Create a 'trend' series based on close price direction
            close_diff = df_adv['close'].diff()
            trend = np.sign(close_diff.fillna(0))  # 1 for up, -1 for down, 0 for no change

            # Calculate the value according to the definition of Volume Force (VF)
            # This is a simplification; actual KVO uses CMF-like concepts
            df_adv['klinger_sv'] = df_adv['volume'] * trend  # Simplified Volume Force

            df_adv['klinger_ema_short'] = df_adv['klinger_sv'].ewm(span=34, adjust=False).mean()
            df_adv['klinger_ema_long'] = df_adv['klinger_sv'].ewm(span=55, adjust=False).mean()
            df_adv['klinger_kvo'] = df_adv['klinger_ema_short'] - df_adv['klinger_ema_long']

        # 4. Ehlers Fisher Transform
        if 'close' in df_adv.columns:
            price = df_adv['close']
            highest_n = price.rolling(window=10).max()
            lowest_n = price.rolling(window=10).min()

            # Skalieren auf Bereich -1 bis 1
            # Avoid division by zero
            raw_value = 2 * ((price - lowest_n) / (highest_n - lowest_n)).replace(0, np.nan) - 1
            raw_value = raw_value.clip(-0.999, 0.999)  # Clip values to avoid issues with log(0) or log(negative)

            # Fisher Transform
            df_adv['fisher_transform'] = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
            df_adv['fisher_transform_signal'] = df_adv['fisher_transform'].shift(1)

        # 5. Money Flow Index (requires technical indicators to be run for typical_price)
        if all(col in df_adv.columns for col in ['high', 'low', 'close', 'volume']):
            typical_price = (df_adv['high'] + df_adv['low'] + df_adv['close']) / 3
            raw_money_flow = typical_price * df_adv['volume']

            # Positive/Negative Money Flow
            money_flow_pos = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
            money_flow_neg = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

            mf_pos_sum = pd.Series(money_flow_pos, index=df_adv.index).rolling(window=14).sum()
            mf_neg_sum = pd.Series(money_flow_neg, index=df_adv.index).rolling(window=14).sum()

            money_ratio = np.where(mf_neg_sum != 0, mf_pos_sum / mf_neg_sum, np.nan)  # Use nan instead of 0
            df_adv['mfi_14'] = 100 - (100 / (1 + money_ratio))

        # 6. Chande Momentum Oscillator
        if 'close' in df_adv.columns:
            price_diff = df_adv['close'].diff(1)

            up_sum = pd.Series(np.where(price_diff > 0, price_diff, 0), index=df_adv.index).rolling(window=14).sum()
            down_sum = pd.Series(np.where(price_diff < 0, abs(price_diff), 0), index=df_adv.index).rolling(
                window=14).sum()

            # Avoid division by zero
            df_adv['cmo_14'] = 100 * ((up_sum - down_sum) / (up_sum + down_sum).replace(0, np.nan))

        # 7. Aroon Indicator
        if 'high' in df_adv.columns and 'low' in df_adv.columns:
            period = 14

            # Using idxmax/idxmin and index differences is more robust
            high_idx = df_adv['high'].rolling(window=period).apply(lambda x: x.idxmax(), raw=False)
            low_idx = df_adv['low'].rolling(window=period).apply(lambda x: x.idxmin(), raw=False)

            # Calculate days since max/min relative to the end of the window
            df_adv['aroon_up'] = 100 * (period - (
                        df_adv.index.to_series().apply(lambda x: (x - df_adv.loc[high_idx.loc[x], :].name).days) / (
                            24 * 60 * 60 / 3600))) / period  # Needs correction for timeframes other than days
            df_adv['aroon_down'] = 100 * (period - (
                        df_adv.index.to_series().apply(lambda x: (x - df_adv.loc[low_idx.loc[x], :].name).days) / (
                            24 * 60 * 60 / 3600))) / period  # Needs correction for timeframes other than days

            # More robust Aroon calculation based on TA-Lib approach or simple pandas rolling index
            def aroon_up(s, period):
                return s.rolling(period).apply(lambda x: float(np.where(x == x.max())[0][-1]), raw=True)

            def aroon_down(s, period):
                return s.rolling(period).apply(lambda x: float(np.where(x == x.min())[0][-1]), raw=True)

            df_adv['aroon_up'] = 100 * (period - aroon_up(df_adv['high'], period)) / period
            df_adv['aroon_down'] = 100 * (period - aroon_down(df_adv['low'], period)) / period

            df_adv['aroon_oscillator'] = df_adv['aroon_up'] - df_adv['aroon_down']

        # 8. Keltner Channel (requires ATR from calculate_technical_indicators)
        if all(col in df_adv.columns for col in ['high', 'low', 'close', 'atr_14']):
            typical_price = (df_adv['high'] + df_adv['low'] + df_adv['close']) / 3

            df_adv['keltner_middle'] = typical_price.rolling(
                window=20).mean()  # Keltner middle is often EMA, but SMA is also used
            df_adv['keltner_upper'] = df_adv['keltner_middle'] + 2 * df_adv['atr_14']
            df_adv['keltner_lower'] = df_adv['keltner_middle'] - 2 * df_adv['atr_14']

        return df_adv

    def extract_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extracts a comprehensive set of features for a single symbol from its OHLCV data.
        Returns a DataFrame with a single row (latest timestamp) and feature columns.
        """
        if df.empty:
            logger.warning(f"Leerer DataFrame für Feature-Extraktion von {symbol}.")
            return pd.DataFrame()

        # Calculate all necessary indicators first
        df_with_indicators = self.calculate_technical_indicators(df)
        df_with_indicators = self.calculate_advanced_indicators(df_with_indicators)

        # Drop rows with NaN values (from rolling calculations at the beginning)
        # Only take the latest valid row for feature extraction
        df_with_indicators = df_with_indicators.dropna()

        if df_with_indicators.empty:
            logger.warning(f"Nach Indikatorberechnung und NaN-Bereinigung ist der DataFrame für {symbol} leer.")
            return pd.DataFrame()

        features_dict = {}

        # 1. Price-based features
        last_close = df_with_indicators['close'].iloc[-1]
        features_dict['close'] = last_close
        features_dict['return'] = df_with_indicators['return'].iloc[-1]
        features_dict['log_return'] = df_with_indicators['log_return'].iloc[-1]

        # 2. Technical indicator features (latest values)
        last_row = df_with_indicators.iloc[-1]

        # Define a consistent list of indicators to extract. This list should cover what's needed by ML models.
        indicator_features_to_extract = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_12', 'ema_13', 'ema_20', 'ema_26', 'ema_50', 'ema_200',
            'bollinger_mid', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'bollinger_pct',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'volatility_5', 'volatility_10', 'volatility_20',
            'atr_14', 'volume_sma_5', 'volume_sma_20', 'volume_ratio', 'obv', 'stoch_k', 'stoch_d',
            'cci_20', 'roc_10', 'williams_r', 'psar',
            # Advanced indicators
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_chikou',
            'elder_bull_power', 'elder_bear_power', 'klinger_kvo', 'fisher_transform', 'fisher_transform_signal',
            'mfi_14', 'cmo_14', 'aroon_up', 'aroon_down', 'aroon_oscillator',
            'keltner_middle', 'keltner_upper', 'keltner_lower'
        ]

        for feat in indicator_features_to_extract:
            if feat in last_row and pd.notna(last_row[feat]):
                features_dict[feat] = last_row[feat]
            else:
                features_dict[feat] = np.nan

                # 3. Add higher-level features for market regime or clustering that combine indicators
        # Price relative to MAs (for market regime)
        if 'close' in last_row and 'ema_20' in last_row and pd.notna(last_row['ema_20']) and last_row['ema_20'] != 0:
            features_dict['rel_to_ema20'] = last_row['close'] / last_row['ema_20'] - 1
        else:
            features_dict['rel_to_ema20'] = np.nan

        if 'close' in last_row and 'ema_50' in last_row and pd.notna(last_row['ema_50']) and last_row['ema_50'] != 0:
            features_dict['rel_to_ema50'] = last_row['close'] / last_row['ema_50'] - 1
        else:
            features_dict['rel_to_ema50'] = np.nan

        if 'close' in last_row and 'ema_200' in last_row and pd.notna(last_row['ema_200']) and last_row['ema_200'] != 0:
            features_dict['rel_to_ema200'] = last_row['close'] / last_row['ema_200'] - 1
        else:
            features_dict['rel_to_ema200'] = np.nan

        # Momentum over longer period (e.g., 30 days or period based on timeframe)
        if len(df_with_indicators) >= 30:  # Ensure enough data
            features_dict['momentum_30d'] = (
                        df_with_indicators['close'].iloc[-1] / df_with_indicators['close'].iloc[-30] - 1) if \
            df_with_indicators['close'].iloc[-30] != 0 else np.nan
        else:
            features_dict['momentum_30d'] = np.nan

        # Example: Volatility ratio
        if 'volatility_5' in last_row and 'volatility_20' in last_row and pd.notna(last_row['volatility_20']) and \
                last_row['volatility_20'] != 0:
            features_dict['volatility_ratio'] = last_row['volatility_5'] / last_row['volatility_20']
        else:
            features_dict['volatility_ratio'] = np.nan

        # Convert to DataFrame
        features_df = pd.DataFrame([features_dict], index=[df_with_indicators.index[-1]])

        # Add symbol prefix to all columns for uniqueness across multiple assets
        symbol_prefix = f"{symbol.replace('/', '_')}_"
        features_df.columns = [symbol_prefix + col for col in features_df.columns]

        return features_df

    def extract_features_for_latest(self, market_data: Dict[str, pd.DataFrame], symbols: List[str]) -> pd.DataFrame:
        """
        Extracts features for the latest timestamp across multiple symbols.
        Combines logic for market regime features.
        """
        combined_features = []
        for symbol in symbols:
            if symbol in market_data and not market_data[symbol].empty:
                # Ensure enough data points before extracting features
                if len(market_data[symbol]) >= self.settings.get('ml.min_data_points_for_ml', 200):
                    # Extract features for the latest row of the given symbol's DataFrame
                    # The extract_features method already returns a single-row DataFrame with prefixed columns
                    features_df_single_symbol = self.extract_features(market_data[symbol], symbol)
                    if not features_df_single_symbol.empty:
                        combined_features.append(features_df_single_symbol)
                else:
                    logger.warning(
                        f"Nicht genügend Daten für {symbol} zur Feature-Extraktion. Benötigt: {self.settings.get('ml.min_data_points_for_ml', 200)}, vorhanden: {len(market_data[symbol])}")
            else:
                logger.warning(f"Keine Marktdaten für {symbol} zur Feature-Extraktion vorhanden.")

        if not combined_features:
            return pd.DataFrame()

        # Concatenate all single-row DataFrames into one, aligning by index (timestamp)
        final_features_df = pd.concat(combined_features, axis=1)

        # Ensure only the latest timestamp is kept if concatenation created multiple rows (e.g., if data timestamps differ slightly)
        if not final_features_df.empty:
            final_features_df = pd.DataFrame(
                final_features_df.iloc[-1]).T  # Take the very last row (transpose to keep it a row)
            final_features_df.index = [final_features_df.index[-1]]  # Ensure index is correct for single row

        return final_features_df

    def get_expected_feature_names(self, symbols: List[str]) -> List[str]:
        """
        Generates a consistent list of all expected feature names across all symbols.
        This is crucial for matching features during model loading/prediction.
        It runs a dummy extraction to get all possible feature names.
        """
        # Create a dummy DataFrame to run through feature extraction and get all possible columns
        # Ensure dummy data is sufficient for all indicator calculations
        dummy_df = pd.DataFrame(np.random.rand(self.settings.get('ml.min_data_points_for_ml', 200) + 100, 5),
                                # Add buffer
                                columns=['open', 'high', 'low', 'close', 'volume'],
                                index=pd.date_range(end=datetime.now(),
                                                    periods=self.settings.get('ml.min_data_points_for_ml', 200) + 100,
                                                    freq='H'))

        all_feature_names = set()
        for symbol in symbols:
            # Run extraction for a dummy symbol to get the pattern of feature names
            features_df = self.extract_features(dummy_df, symbol)
            if not features_df.empty:
                for col in features_df.columns:
                    all_feature_names.add(col)

        return sorted(list(all_feature_names))  # Return sorted list for consistent order