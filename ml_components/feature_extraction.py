# ml_components/feature_extraction.py (Existing file, ensure it works as expected)
import logging
import pandas as pd
import numpy as np
import ta  # Assuming you have `ta` (Technical Analysis) library installed
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts various technical indicators and features from OHLCV data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Default periods for indicators, can be overridden by config
        self.rsi_period = config.get('rsi_period', 14)
        self.ma_short = config.get('ma_short', 20)
        self.ma_long = config.get('ma_long', 50)
        self.bollinger_window = config.get('bollinger_window', 20)
        self.bollinger_std_dev = config.get('bollinger_std_dev', 2.0)
        self.atr_period = config.get('atr_period', 14)

        logger.info("FeatureExtractor initialized.")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a set of technical indicators for a given DataFrame.
        Expected DataFrame columns: 'open', 'high', 'low', 'close', 'volume'.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature extraction.")
            return pd.DataFrame()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Basic Checks
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.error("Missing required OHLCV columns for feature extraction.")
            # Attempt to proceed with available columns, or raise error
            # For robustness, we will try to calculate what we can.
            pass

        # Momentum Indicators
        if 'close' in df.columns:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
            df['sma_short'] = ta.trend.sma_indicator(df['close'], window=self.ma_short)
            df['sma_long'] = ta.trend.sma_indicator(df['close'], window=self.ma_long)
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
            df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()  # Rate of Change

        # Volatility Indicators
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                       window=self.atr_period).average_true_range()

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=self.bollinger_window,
                                                        window_dev=self.bollinger_std_dev)
            df['bb_bbm'] = bb_indicator.bollinger_mavg()
            df['bb_bbh'] = bb_indicator.bollinger_hband()
            df['bb_bbl'] = bb_indicator.bollinger_lband()
            df['bb_wband'] = bb_indicator.bollinger_wband()  # Bollinger Band Width

        # Volume Indicators
        if 'volume' in df.columns and 'close' in df.columns:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['ch_mf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'],
                                                              df['volume']).chaikin_money_flow()  # Chaikin Money Flow

        # Ensure no infinite values (replace with NaN then handle NaNs later)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # It's better to handle NaNs (e.g., imputation or dropping rows) outside this function
        # in the calling component (MarketRegimeDetector) after all features are calculated.

        logger.debug(f"Calculated {len(df.columns) - 5} technical indicators.")  # Subtract OHLCV columns
        return df

    def get_expected_feature_names(self, symbols: List[str]) -> List[str]:
        """
        Returns a list of expected feature column names based on the configured indicators.
        Useful for ensuring consistency in ML model training/prediction.
        """
        base_features = [
            'rsi', 'sma_short', 'sma_long', 'macd', 'macd_signal', 'macd_diff', 'momentum',
            'atr', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_wband', 'obv', 'ch_mf',
            # Additional features specific to regime detection
            'rolling_std',  # This one will be added in market_regime.py, not directly here
            'fear_greed_index'  # This one will be added from market_sentiment.py
        ]

        # Combine with OHLCV for completeness if needed elsewhere, but typically ML models
        # use derived features. For clustering, usually only derived features are used.

        # Prefix features for each symbol if multiple symbols are used for combined features
        prefixed_features = []
        for symbol in symbols:
            prefix = f"{symbol.replace('/', '_')}_"
            for feature in base_features:
                if feature == 'rolling_std' or feature == 'atr':  # These are directly calculated in market_regime.py
                    prefixed_features.append(f"{prefix}{feature}")
                elif feature == 'fear_greed_index':  # This is a global sentiment, not symbol-specific usually
                    if symbol == symbols[0]:  # Add only once for the first symbol or if it's a global feature
                        prefixed_features.append(feature)
                else:
                    prefixed_features.append(f"{prefix}{feature}")

        # Ensure uniqueness and order
        return sorted(list(set(prefixed_features)))