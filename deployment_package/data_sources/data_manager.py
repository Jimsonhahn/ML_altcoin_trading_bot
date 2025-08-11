# data_sources/data_manager.py (Existing file, ensure it works as expected)
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from config.settings import Settings
from data_sources.binance_source import BinanceDataSource  # Use the actual class name
from utils.error_handler import handle_errors, ErrorCategory, handle_data_error

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages fetching, caching, and serving historical OHLCV data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = settings.get('data.cache_dir', 'data/market_data')
        self.use_cache = settings.get('data.use_cache', True)
        self.min_candles = settings.get('data.min_candles', 200)

        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize data source based on settings
        data_source_name = settings.get('data.source_name', 'binance')
        if data_source_name.lower() == 'binance':
            self.data_source = BinanceDataSource(settings)  # Pass settings for API keys etc.
        else:
            raise ValueError(f"Unsupported data source: {data_source_name}")

        logger.info(f"DataManager initialized with cache: {self.use_cache} at {self.cache_dir}")

    @handle_errors(category=ErrorCategory.DATA, max_retries=2, retry_delay=1.0)
    def get_historical_data(self, symbol: str, timeframe: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given symbol and timeframe within a date range.
        Prioritizes cache, then fetches from source if not available or incomplete.
        """
        cache_file_path = os.path.join(self.cache_dir,
                                       f"{symbol.replace('/', '_')}_{timeframe}_{start_date_str}_{end_date_str}.csv")

        # Check cache first
        if self.use_cache and os.path.exists(cache_file_path):
            try:
                df = pd.read_csv(cache_file_path, index_col='timestamp', parse_dates=True)
                logger.info(f"Loaded {len(df)} candles for {symbol} ({timeframe}) from cache.")
                return df
            except Exception as e:
                logger.warning(f"Error loading cached data from {cache_file_path}: {e}. Fetching from source.")

        # Fetch from source
        logger.info(
            f"Fetching historical data for {symbol} ({timeframe}) from {start_date_str} to {end_date_str} from {self.data_source.__class__.__name__}...")
        try:
            ohlcv = self.data_source.fetch_ohlcv_range(symbol, timeframe, start_date_str, end_date_str)
            df = self.convert_ohlcv_to_dataframe(ohlcv)

            if self.use_cache and not df.empty:
                df.to_csv(cache_file_path)
                logger.info(f"Cached {len(df)} candles for {symbol} ({timeframe}).")

            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe}) from source.")
            return df
        except Exception as e:
            handle_data_error(e, symbol=symbol, timeframe=timeframe)

    def convert_ohlcv_to_dataframe(self, ohlcv_data: List[List[Any]]) -> pd.DataFrame:
        """Converts raw OHLCV list data to a pandas DataFrame."""
        if not ohlcv_data:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Ensure numerical types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @handle_errors(category=ErrorCategory.DATA, max_retries=3, retry_delay=0.5)
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches the latest OHLCV data directly from the source.
        Used for real-time updates in live trading.
        """
        logger.debug(f"Fetching latest {limit} candles for {symbol} ({timeframe}) from source.")
        try:
            ohlcv = self.data_source.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = self.convert_ohlcv_to_dataframe(ohlcv)
            return df
        except Exception as e:
            handle_data_error(e, symbol=symbol, timeframe=timeframe)