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

# Logging einrichten
logger = logging.getLogger(__name__)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet gängige technische Indikatoren für einen OHLCV-DataFrame.

    Args:
        df: DataFrame mit OHLCV-Daten

    Returns:
        DataFrame mit berechneten Indikatoren
    """
    try:
        # Kopie des DataFrames erstellen
        df_indicators = df.copy()

        # Prüfen, ob erforderliche Spalten vorhanden sind
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Fehlende Spalten für Indikatorberechnung: {missing_columns}")
            for col in missing_columns:
                df_indicators[col] = 0

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
        df_indicators['ema_20'] = df_indicators['close'].ewm(span=20, adjust=False).mean()
        df_indicators['ema_50'] = df_indicators['close'].ewm(span=50, adjust=False).mean()
        df_indicators['ema_200'] = df_indicators['close'].ewm(span=200, adjust=False).mean()

        # 4. Bollinger Bands
        df_indicators['bollinger_mid'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['bollinger_std'] = df_indicators['close'].rolling(window=20).std()
        df_indicators['bollinger_upper'] = df_indicators['bollinger_mid'] + (df_indicators['bollinger_std'] * 2)
        df_indicators['bollinger_lower'] = df_indicators['bollinger_mid'] - (df_indicators['bollinger_std'] * 2)
        df_indicators['bollinger_pct'] = (df_indicators['close'] - df_indicators['bollinger_lower']) / (
                df_indicators['bollinger_upper'] - df_indicators['bollinger_lower'])

        # 5. RSI
        delta = df_indicators['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df_indicators['rsi_14'] = 100 - (100 / (1 + rs))

        # 6. MACD
        df_indicators['macd'] = df_indicators['ema_12'] - df_indicators[
            'ema_26'] if 'ema_12' in df_indicators.columns and 'ema_26' in df_indicators.columns else df_indicators[
                                                                                                          'close'].ewm(
            span=12, adjust=False).mean() - df_indicators['close'].ewm(span=26, adjust=False).mean()
        df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9, adjust=False).mean()
        df_indicators['macd_hist'] = df_indicators['macd'] - df_indicators['macd_signal']

        # 7. Volatilität
        df_indicators['volatility_5'] = df_indicators['return'].rolling(window=5).std()
        df_indicators['volatility_10'] = df_indicators['return'].rolling(window=10).std()
        df_indicators['volatility_20'] = df_indicators['return'].rolling(window=20).std()

        # 8. Average True Range (ATR)
        df_indicators['tr'] = np.maximum(
            df_indicators['high'] - df_indicators['low'],
            np.maximum(
                abs(df_indicators['high'] - df_indicators['close'].shift(1)),
                abs(df_indicators['low'] - df_indicators['close'].shift(1))
            )
        )
        df_indicators['atr_14'] = df_indicators['tr'].rolling(window=14).mean()

        # 9. Volumen-Indikatoren
        df_indicators['volume_sma_5'] = df_indicators['volume'].rolling(window=5).mean()
        df_indicators['volume_sma_20'] = df_indicators['volume'].rolling(window=20).mean()
        df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma_20']

        # 10. On-Balance Volume (OBV)
        df_indicators['obv'] = np.where(
            df_indicators['close'] > df_indicators['close'].shift(1),
            df_indicators['volume'],
            np.where(
                df_indicators['close'] < df_indicators['close'].shift(1),
                -df_indicators['volume'],
                0
            )
        ).cumsum()

        # 11. Stochastic Oscillator
        df_indicators['stoch_k'] = 100 * ((df_indicators['close'] - df_indicators['low'].rolling(window=14).min()) /
                                          (df_indicators['high'].rolling(window=14).max() - df_indicators[
                                              'low'].rolling(window=14).min()))
        df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()

        # 12. Commodity Channel Index (CCI)
        typical_price = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
        df_indicators['cci_20'] = (typical_price - typical_price.rolling(window=20).mean()) / (
                0.015 * typical_price.rolling(window=20).std())

        # 13. Rate of Change (ROC)
        df_indicators['roc_10'] = ((df_indicators['close'] - df_indicators['close'].shift(10)) / df_indicators[
            'close'].shift(10)) * 100

        # 14. Williams %R
        df_indicators['williams_r'] = -100 * (
                (df_indicators['high'].rolling(window=14).max() - df_indicators['close']) /
                (df_indicators['high'].rolling(window=14).max() - df_indicators['low'].rolling(window=14).min()))

        # 15. Parabolic SAR (vereinfacht)
        df_indicators['psar'] = df_indicators['close'].shift(1)  # Vereinfachte Version als Platzhalter

        return df_indicators

    except Exception as e:
        logger.error(f"Fehler bei der Berechnung technischer Indikatoren: {e}")
        return df


def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet erweiterte technische Indikatoren für einen OHLCV-DataFrame.

    Args:
        df: DataFrame mit OHLCV-Daten

    Returns:
        DataFrame mit berechneten Indikatoren
    """
    try:
        # Kopie des DataFrames erstellen
        df_adv = df.copy()

        # 1. Ichimoku Cloud
        high_9 = df_adv['high'].rolling(window=9).max()
        low_9 = df_adv['low'].rolling(window=9).min()
        df_adv['ichimoku_tenkan'] = (high_9 + low_9) / 2

        high_26 = df_adv['high'].rolling(window=26).max()
        low_26 = df_adv['low'].rolling(window=26).min()
        df_adv['ichimoku_kijun'] = (high_26 + low_26) / 2

        df_adv['ichimoku_senkou_a'] = ((df_adv['ichimoku_tenkan'] + df_adv['ichimoku_kijun']) / 2).shift(26)

        high_52 = df_adv['high'].rolling(window=52).max()
        low_52 = df_adv['low'].rolling(window=52).min()
        df_adv['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)

        df_adv['ichimoku_chikou'] = df_adv['close'].shift(-26)

        # 2. Elder Ray Index
        df_adv['elder_bull_power'] = df_adv['high'] - df_adv['ema_13'] if 'ema_13' in df_adv.columns else df_adv[
                                                                                                              'high'] - \
                                                                                                          df_adv[
                                                                                                              'close'].ewm(
                                                                                                              span=13,
                                                                                                              adjust=False).mean()
        df_adv['elder_bear_power'] = df_adv['low'] - df_adv['ema_13'] if 'ema_13' in df_adv.columns else df_adv[
                                                                                                             'low'] - \
                                                                                                         df_adv[
                                                                                                             'close'].ewm(
                                                                                                             span=13,
                                                                                                             adjust=False).mean()

        # 3. Klinger Volume Oscillator
        if 'volume' in df_adv.columns:
            df_adv['klinger_sv'] = df_adv['volume'] * (
                    2 * (df_adv['close'] - df_adv['close'].shift(1)) / (df_adv['high'] - df_adv['low']) - 1)
            df_adv['klinger_ema_short'] = df_adv['klinger_sv'].ewm(span=34, adjust=False).mean()
            df_adv['klinger_ema_long'] = df_adv['klinger_sv'].ewm(span=55, adjust=False).mean()
            df_adv['klinger_kvo'] = df_adv['klinger_ema_short'] - df_adv['klinger_ema_long']

        # 4. Ehlers Fisher Transform
        if 'close' in df_adv.columns:
            price = df_adv['close']
            highest_2 = price.rolling(window=10).max()
            lowest_2 = price.rolling(window=10).min()

            # Skalieren auf Bereich -1 bis 1
            raw_value = 2 * ((price - lowest_2) / (highest_2 - lowest_2) - 0.5)

            # Fisher Transform
            df_adv['fisher_transform'] = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
            df_adv['fisher_transform_signal'] = df_adv['fisher_transform'].shift(1)

        # 5. Money Flow Index
        if all(col in df_adv.columns for col in ['high', 'low', 'close', 'volume']):
            typical_price = (df_adv['high'] + df_adv['low'] + df_adv['close']) / 3
            raw_money_flow = typical_price * df_adv['volume']

            # Positive/Negative Money Flow
            money_flow_pos = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
            money_flow_neg = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

            mf_pos_sum = pd.Series(money_flow_pos).rolling(window=14).sum()
            mf_neg_sum = pd.Series(money_flow_neg).rolling(window=14).sum()

            money_ratio = np.where(mf_neg_sum != 0, mf_pos_sum / mf_neg_sum, 0)
            df_adv['mfi_14'] = 100 - (100 / (1 + money_ratio))

        # 6. Chande Momentum Oscillator
        if 'close' in df_adv.columns:
            price_diff = df_adv['close'].diff(1)

            # Summe der positiven und negativen Preisänderungen
            up_sum = pd.Series(np.where(price_diff > 0, price_diff, 0)).rolling(window=14).sum()
            down_sum = pd.Series(np.where(price_diff < 0, abs(price_diff), 0)).rolling(window=14).sum()

            # CMO berechnen
            df_adv['cmo_14'] = 100 * ((up_sum - down_sum) / (up_sum + down_sum))

        # 7. Aroon Indicator
        if 'high' in df_adv.columns and 'low' in df_adv.columns:
            # Aroon Up: 100 * (14 - Tage seit 14-Tage-Hoch) / 14
            # Aroon Down: 100 * (14 - Tage seit 14-Tage-Tief) / 14
            period = 14

            # Berechnung der Anzahl der Tage seit Hoch/Tief
            rolling_high = df_adv['high'].rolling(window=period).max()
            rolling_low = df_adv['low'].rolling(window=period).min()

            days_since_high = np.zeros(len(df_adv))
            days_since_low = np.zeros(len(df_adv))

            for i in range(period, len(df_adv)):
                high_val = rolling_high.iloc[i]
                low_val = rolling_low.iloc[i]

                # Suche nach letztem Hoch/Tief
                for j in range(period):
                    if df_adv['high'].iloc[i - j] == high_val:
                        days_since_high[i] = j
                        break

                for j in range(period):
                    if df_adv['low'].iloc[i - j] == low_val:
                        days_since_low[i] = j
                        break

            df_adv['aroon_up'] = 100 * (period - pd.Series(days_since_high)) / period
            df_adv['aroon_down'] = 100 * (period - pd.Series(days_since_low)) / period
            df_adv['aroon_oscillator'] = df_adv['aroon_up'] - df_adv['aroon_down']

        # 8. Keltner Channel
        if all(col in df_adv.columns for col in ['high', 'low', 'close']):
            typical_price = (df_adv['high'] + df_adv['low'] + df_adv['close']) / 3

            # True Range
            tr = np.zeros(len(df_adv))
            for i in range(1, len(df_adv)):
                tr1 = df_adv['high'].iloc[i] - df_adv['low'].iloc[i]
                tr2 = abs(df_adv['high'].iloc[i] - df_adv['close'].iloc[i - 1])
                tr3 = abs(df_adv['low'].iloc[i] - df_adv['close'].iloc[i - 1])
                tr[i] = max(tr1, tr2, tr3)

            atr = pd.Series(tr).rolling(window=10).mean()

            df_adv['keltner_middle'] = typical_price.rolling(window=20).mean()
            df_adv['keltner_upper'] = df_adv['keltner_middle'] + 2 * atr
            df_adv['keltner_lower'] = df_adv['keltner_middle'] - 2 * atr

        return df_adv

    except Exception as e:
        logger.error(f"Fehler bei der Berechnung erweiterter Indikatoren: {e}")
        return df


def extract_market_regime_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> Dict[str, float]:
    """
    Extrahiert Features für die Marktregime-Erkennung.

    Args:
        df: DataFrame mit OHLCV-Daten
        btc_df: Optional DataFrame mit Bitcoin-Daten als Referenz

    Returns:
        Dictionary mit Feature-Namen und -Werten
    """
    try:
        features = {}

        # Wenn kein BTC-DataFrame übergeben wurde, den übergebenen verwenden
        if btc_df is None:
            btc_df = df

        # 1. Rendite-Features
        features['mean_return'] = df['return'].mean() if 'return' in df.columns else df['close'].pct_change().mean()
        features['volatility'] = df['return'].std() if 'return' in df.columns else df['close'].pct_change().std()

        # 2. Kurse relativ zu Moving Averages
        if 'close' in df.columns:
            if 'ema_20' in df.columns:
                features['rel_to_ema20'] = df['close'].iloc[-1] / df['ema_20'].iloc[-1] - 1
            else:
                ema_20 = df['close'].ewm(span=20, adjust=False).mean()
                features['rel_to_ema20'] = df['close'].iloc[-1] / ema_20.iloc[-1] - 1

            if 'ema_50' in df.columns:
                features['rel_to_ema50'] = df['close'].iloc[-1] / df['ema_50'].iloc[-1] - 1
            else:
                ema_50 = df['close'].ewm(span=50, adjust=False).mean()
                features['rel_to_ema50'] = df['close'].iloc[-1] / ema_50.iloc[-1] - 1

            if 'ema_200' in df.columns:
                features['rel_to_ema200'] = df['close'].iloc[-1] / df['ema_200'].iloc[-1] - 1
            else:
                ema_200 = df['close'].ewm(span=200, adjust=False).mean()
                features['rel_to_ema200'] = df['close'].iloc[-1] / ema_200.iloc[-1] - 1

        # 3. Relative Stärke zu Bitcoin (wenn nicht BTC)
        if btc_df is not df and 'return' in df.columns and 'return' in btc_df.columns:
            # Gemeinsamer Index
            common_idx = df.index.intersection(btc_df.index)

            if len(common_idx) > 0:
                asset_returns = df.loc[common_idx, 'return']
                btc_returns = btc_df.loc[common_idx, 'return']

                features['rel_strength_to_btc'] = asset_returns.mean() - btc_returns.mean()

        # 4. RSI-Features
        if 'rsi_14' in df.columns:
            features['rsi'] = df['rsi_14'].iloc[-1]
            features['rsi_5d_mean'] = df['rsi_14'].iloc[-5:].mean()

        # 5. Volatilitäts-Features
        if 'volatility_20' in df.columns:
            features['volatility_20d'] = df['volatility_20'].iloc[-1]
        elif 'return' in df.columns:
            features['volatility_20d'] = df['return'].rolling(20).std().iloc[-1]

        # 6. Volumen-Features
        if 'volume' in df.columns and 'volume_sma_20' in df.columns:
            features['volume_ratio'] = df['volume'].iloc[-1] / df['volume_sma_20'].iloc[-1]
        elif 'volume' in df.columns:
            features['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

        # 7. Trend-Features
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            features['macd_hist'] = df['macd_hist'].iloc[-1]
            features['macd_hist_5d_sum'] = df['macd_hist'].iloc[-5:].sum()

        # 8. Momentum-Features
        if 'close' in df.columns:
            features['momentum_14d'] = df['close'].iloc[-1] / df['close'].iloc[-15] - 1 if len(df) >= 15 else 0

        return features

    except Exception as e:
        logger.error(f"Fehler bei der Extraktion von Marktregime-Features: {e}")
        return {}


def extract_asset_clustering_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extrahiert Features für Asset-Clustering.

    Args:
        df: DataFrame mit OHLCV-Daten

    Returns:
        Dictionary mit Feature-Namen und -Werten
    """
    try:
        features = {}

        # 1. Rendite-Statistiken
        if 'return' in df.columns:
            returns = df['return'].dropna()
        else:
            returns = df['close'].pct_change().dropna()

        features['mean_return'] = returns.mean()
        features['volatility'] = returns.std()
        features['skewness'] = returns.skew() if len(returns) > 3 else 0
        features['kurtosis'] = returns.kurt() if len(returns) > 3 else 0

        # 2. Volumen-Features
        if 'volume' in df.columns:
            volume = df['volume'].dropna()
            volume_change = volume.pct_change().dropna()

            features['volume_mean'] = volume.mean()
            features['volume_std'] = volume.std()
            features['volume_change_mean'] = volume_change.mean()

            # Volumen/Preis-Korrelation
            if 'close' in df.columns:
                features['vol_price_corr'] = volume.corr(df['close'])

        # 3. Trend-Features
        if 'close' in df.columns:
            close = df['close']

            # MA-Verhältnis
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                features['ma_ratio'] = df['sma_20'].iloc[-1] / df['sma_50'].iloc[-1]
            else:
                sma_20 = close.rolling(20).mean()
                sma_50 = close.rolling(50).mean()
                features['ma_ratio'] = sma_20.iloc[-1] / sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) and \
                                                                            sma_50.iloc[-1] != 0 else 1

            # Distanz zu MAs
            if 'sma_20' in df.columns:
                features['dist_to_ma20'] = close.iloc[-1] / df['sma_20'].iloc[-1] - 1
            else:
                sma_20 = close.rolling(20).mean()
                features['dist_to_ma20'] = close.iloc[-1] / sma_20.iloc[-1] - 1 if not pd.isna(sma_20.iloc[-1]) and \
                                                                                   sma_20.iloc[-1] != 0 else 0

            if 'sma_50' in df.columns:
                features['dist_to_ma50'] = close.iloc[-1] / df['sma_50'].iloc[-1] - 1
            else:
                sma_50 = close.rolling(50).mean()
                features['dist_to_ma50'] = close.iloc[-1] / sma_50.iloc[-1] - 1 if not pd.isna(sma_50.iloc[-1]) and \
                                                                                   sma_50.iloc[-1] != 0 else 0

        # 4. Volatilitäts-Features
        if 'volatility_20' in df.columns:
            features['volatility_20d'] = df['volatility_20'].iloc[-1]
        elif 'return' in df.columns:
            features['volatility_20d'] = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()

        # 5. RSI-Feature
        if 'rsi_14' in df.columns:
            features['rsi'] = df['rsi_14'].iloc[-1]

        # 6. Momentum-Feature
        if 'roc_10' in df.columns:
            features['momentum'] = df['roc_10'].iloc[-1]
        elif 'close' in df.columns:
            features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-11] - 1 if len(df) >= 11 else 0

        # 7. Drawdown-Feature
        if 'close' in df.columns:
            close = df['close']
            rolling_max = close.rolling(min(len(close), 30)).max()
            drawdown = (close / rolling_max - 1) * 100
            features['max_drawdown'] = drawdown.min()

        # 8. Return-to-Vol Ratio (Sharpe-ähnlich)
        if features['volatility'] > 0:
            features['return_to_vol_ratio'] = features['mean_return'] / features['volatility']
        else:
            features['return_to_vol_ratio'] = 0

        return features

    except Exception as e:
        logger.error(f"Fehler bei der Extraktion von Asset-Clustering-Features: {e}")
        return {}


def extract_coin_features(df: pd.DataFrame, min_days: int = 3) -> Dict[str, Any]:
    """
    Extrahiert Features für die Analyse neuer Coins.

    Args:
        df: DataFrame mit OHLCV-Daten
        min_days: Minimale Anzahl an Tagen für die Analyse

    Returns:
        Dictionary mit Features oder Status-Information
    """
    try:
        # Prüfen, ob genügend Daten vorhanden sind
        if len(df) < min_days:
            return {"status": "insufficient_data", "days_available": len(df)}

        # Basis-Features extrahieren
        features = extract_asset_clustering_features(df)

        # Erweiterte Coin-spezifische Features
        if 'volume' in df.columns:
            # Volumen-Trend
            vol_trend = df['volume'].pct_change().mean()
            features['volume_trend'] = vol_trend

            # Volumen-Spikes
            vol_mean = df['volume'].mean()
            vol_std = df['volume'].std()
            vol_spikes = (df['volume'] > vol_mean + 2 * vol_std).sum()
            features['volume_spikes'] = vol_spikes / len(df)

        # Preisvolatilität auf Tagesbasis
        if all(col in df.columns for col in ['high', 'low', 'close']):
            daily_range = (df['high'] - df['low']) / df['close']
            features['daily_range_mean'] = daily_range.mean()
            features['daily_range_max'] = daily_range.max()

        # Preis-Momentum
        if 'close' in df.columns:
            price_trend_3d = df['close'].iloc[-1] / df['close'].iloc[-min(4, len(df))] - 1 if len(df) >= 4 else 0
            features['price_trend_3d'] = price_trend_3d

        # Aufwärts-/Abwärts-Tage-Verhältnis
        if 'return' in df.columns:
            up_days = (df['return'] > 0).sum()
            down_days = (df['return'] < 0).sum()
            features['up_down_ratio'] = up_days / down_days if down_days > 0 else up_days

        # Ergebnis mit Status
        return {
            "status": "analyzed",
            "features": features
        }

    except Exception as e:
        logger.error(f"Fehler bei der Extraktion von Coin-Features: {e}")
        return {"status": "error", "message": str(e)}


def extract_enhanced_features(df: pd.DataFrame, symbol: str = None,
                              include_sentiment: bool = True) -> Dict[str, float]:
    """
    Extrahiert erweiterte Features mit Sentiment- und On-Chain-Daten.

    Args:
        df: DataFrame mit OHLCV-Daten
        symbol: Symbol des Assets (für Sentiment-Daten)
        include_sentiment: Ob Sentiment-Features einbezogen werden sollen

    Returns:
        Dictionary mit Feature-Namen und -Werten
    """
    try:
        # Basis-Features extrahieren
        features = extract_market_regime_features(df)

        # Erweiterte technische Indikatoren berechnen
        df_adv = calculate_advanced_indicators(df)

        # Letzte Werte der erweiterten Indikatoren hinzufügen
        last_row = df_adv.iloc[-1]

        # Ichimoku-Features
        if 'ichimoku_tenkan' in last_row:
            features['ichimoku_tenkan'] = last_row['ichimoku_tenkan']
            features['ichimoku_kijun'] = last_row['ichimoku_kijun']

            # Cloud-Positionen
            if 'ichimoku_senkou_a' in last_row and 'ichimoku_senkou_b' in last_row:
                cloud_top = max(last_row['ichimoku_senkou_a'], last_row['ichimoku_senkou_b'])
                cloud_bottom = min(last_row['ichimoku_senkou_a'], last_row['ichimoku_senkou_b'])

                # Preis relativ zur Cloud
                features['price_above_cloud'] = 1 if last_row['close'] > cloud_top else 0
                features['price_in_cloud'] = 1 if cloud_bottom <= last_row['close'] <= cloud_top else 0
                features['price_below_cloud'] = 1 if last_row['close'] < cloud_bottom else 0

                # Abstand zur Cloud
                if last_row['close'] > cloud_top:
                    features['cloud_distance'] = (last_row['close'] / cloud_top) - 1
                elif last_row['close'] < cloud_bottom:
                    features['cloud_distance'] = (last_row['close'] / cloud_bottom) - 1
                else:
                    features['cloud_distance'] = 0

        # Elder Ray Features
        if 'elder_bull_power' in last_row:
            features['elder_bull_power'] = last_row['elder_bull_power']
            features['elder_bear_power'] = last_row['elder_bear_power']

            # Kombinierte Elder-Power
            features['elder_power_ratio'] = last_row['elder_bull_power'] / abs(last_row['elder_bear_power']) if \
                last_row['elder_bear_power'] != 0 else 0

        # Klinger Volume Oscillator
        if 'klinger_kvo' in last_row:
            features['klinger_kvo'] = last_row['klinger_kvo']

            # KVO-Trend (letzte 5 Werte)
            if len(df_adv) >= 5:
                kvo_values = df_adv['klinger_kvo'].tail(5)
                features['klinger_kvo_slope'] = (kvo_values.iloc[-1] - kvo_values.iloc[0]) / 5

        # Fisher Transform
        if 'fisher_transform' in last_row:
            features['fisher_transform'] = last_row['fisher_transform']

            # Fisher-Divergenz
            if 'fisher_transform_signal' in last_row:
                features['fisher_divergence'] = last_row['fisher_transform'] - last_row[
                    'fisher_transform_signal']

        # Money Flow Index
        if 'mfi_14' in last_row:
            features['mfi'] = last_row['mfi_14']

            # MFI-Overbought/Oversold
            features['mfi_overbought'] = 1 if last_row['mfi_14'] > 80 else 0
            features['mfi_oversold'] = 1 if last_row['mfi_14'] < 20 else 0

        # Chande Momentum Oscillator
        if 'cmo_14' in last_row:
            features['cmo'] = last_row['cmo_14']

        # Aroon Indicator
        if 'aroon_oscillator' in last_row:
            features['aroon_oscillator'] = last_row['aroon_oscillator']

        # Keltner Channel
        if all(x in last_row for x in ['keltner_middle', 'keltner_upper', 'keltner_lower']):
            # Preis relativ zum Keltner Channel
            features['keltner_position'] = (last_row['close'] - last_row['keltner_lower']) / (
                    last_row['keltner_upper'] - last_row['keltner_lower']) if (last_row['keltner_upper'] -
                                                                               last_row[
                                                                                   'keltner_lower']) != 0 else 0.5

            # Channel-Weite (Volatilität)
            features['keltner_width'] = (last_row['keltner_upper'] - last_row['keltner_lower']) / last_row[
                'keltner_middle']

        # Sentiment-Features hinzufügen, falls gewünscht
        if include_sentiment and symbol:
            # Import hier, um Zirkelbezüge zu vermeiden
            try:
                from ml_components.market_sentiment import MarketSentimentAnalyzer

                # Sentiment-Analyzer initialisieren
                sentiment_analyzer = MarketSentimentAnalyzer()

                # Sentiment-Features extrahieren
                df_with_sentiment = sentiment_analyzer.extract_sentiment_features(df, symbol)

                # Letzte Sentiment-Werte hinzufügen
                last_sentiment = df_with_sentiment.iloc[-1]

                # Fear & Greed
                if 'fear_greed_value' in last_sentiment:
                    features['fear_greed_index'] = last_sentiment['fear_greed_value']

                # Social Sentiment
                if 'social_sentiment' in last_sentiment:
                    features['social_sentiment'] = last_sentiment['social_sentiment']

                    # Sentiment-Trend (letzte 7 Tage)
                    if len(df_with_sentiment) >= 7:
                        sentiment_values = df_with_sentiment['social_sentiment'].tail(7)
                        features['sentiment_trend'] = (sentiment_values.iloc[-1] - sentiment_values.iloc[0]) / 7

                # On-Chain-Metriken
                onchain_cols = [col for col in last_sentiment.index if col.startswith('onchain_')]
                for col in onchain_cols:
                    features[col] = last_sentiment[col]

                    # Trends für wichtige On-Chain-Metriken
                    if col in ['onchain_active_addresses', 'onchain_transaction_count']:
                        if len(df_with_sentiment) >= 7:
                            metric_values = df_with_sentiment[col].tail(7)
                            features[f"{col}_trend"] = (metric_values.iloc[-1] - metric_values.iloc[0]) / \
                                                       metric_values.iloc[0] if metric_values.iloc[0] != 0 else 0
            except (ImportError, ModuleNotFoundError):
                logger.warning("MarketSentimentAnalyzer nicht verfügbar, Sentiment-Features werden übersprungen")

        return features

    except Exception as e:
        logger.error(f"Fehler bei der Extraktion erweiterter Features: {e}")
        return {}