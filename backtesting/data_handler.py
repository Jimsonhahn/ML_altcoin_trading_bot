"""
Data Handler - Point-in-Time historische Marktdaten
Strikt ohne Lookahead-Bias für realistische Backtests
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
import json
import gzip
from collections import deque
from abc import ABC, abstractmethod

from .event_models import MarketEvent, create_market_event_from_candle
from .event_bus import EventBus

logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstrakte Basis-Klasse für Data Handler
    Garantiert Point-in-Time Datenlieferung ohne Future-Information
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.current_timestamp: Optional[datetime] = None
        self.symbols: List[str] = []
        self.data_exhausted = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialisiert Data Handler und lädt Daten"""
        pass
    
    @abstractmethod
    async def get_next_market_events(self) -> List[MarketEvent]:
        """Gibt nächste MarketEvents zurück (Point-in-Time)"""
        pass
    
    @abstractmethod
    def has_more_data(self) -> bool:
        """Prüft ob weitere Daten verfügbar sind"""
        pass
    
    async def stream_data(self) -> None:
        """
        Hauptmethode: Streamt Marktdaten als Events
        Läuft bis alle Daten verarbeitet sind
        """
        await self.initialize()
        
        logger.info(f"DataHandler startet Streaming für {len(self.symbols)} Symbole")
        
        while self.has_more_data():
            try:
                # Get next batch of market events
                market_events = await self.get_next_market_events()
                
                if not market_events:
                    continue
                
                # Publish events to bus
                for event in market_events:
                    await self.event_bus.publish(event)
                
                # Update current timestamp
                if market_events:
                    self.current_timestamp = market_events[0].timestamp
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in data streaming: {e}")
                await asyncio.sleep(0.1)
        
        self.data_exhausted = True
        logger.info("DataHandler: Alle Daten gestreamt")


class HistoricalDataHandler(DataHandler):
    """
    Historical Data Handler für OHLCV Daten
    Unterstützt CSV, Parquet und komprimierte Formate
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 data_directory: str,
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 timeframe: str = "1h",
                 data_format: str = "csv"):
        
        super().__init__(event_bus)
        self.data_directory = Path(data_directory)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data_format = data_format
        
        # Data storage
        self.data_frames: Dict[str, pd.DataFrame] = {}
        self.data_indices: Dict[str, int] = {}
        
        # Lookahead prevention
        self.max_lookahead_bars = 0  # Strikt keine Future-Daten
        
        # Data quality tracking
        self.missing_data_points: Dict[str, List[datetime]] = {}
        self.data_quality_scores: Dict[str, float] = {}
    
    async def initialize(self) -> None:
        """Lädt historische Daten für alle Symbole"""
        
        logger.info(f"Lade historische Daten von {self.start_date} bis {self.end_date}")
        
        for symbol in self.symbols:
            try:
                df = await self._load_symbol_data(symbol)
                
                if df is not None and not df.empty:
                    # Filter by date range
                    df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                    
                    # Sort by timestamp (wichtig für Point-in-Time)
                    df = df.sort_index()
                    
                    # Data quality check
                    quality_score = self._calculate_data_quality(df)
                    
                    self.data_frames[symbol] = df
                    self.data_indices[symbol] = 0
                    self.data_quality_scores[symbol] = quality_score
                    
                    logger.info(f"Geladen: {symbol} - {len(df)} Bars, Qualität: {quality_score:.2%}")
                else:
                    logger.warning(f"Keine Daten für {symbol} gefunden")
                    
            except Exception as e:
                logger.error(f"Fehler beim Laden von {symbol}: {e}")
        
        if not self.data_frames:
            raise ValueError("Keine Daten geladen!")
        
        # Set initial timestamp
        self.current_timestamp = min(df.index[0] for df in self.data_frames.values())
    
    async def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Lädt Daten für einzelnes Symbol"""
        
        file_patterns = {
            'csv': f"{symbol}_{self.timeframe}.csv",
            'parquet': f"{symbol}_{self.timeframe}.parquet",
            'json': f"{symbol}_{self.timeframe}.json",
            'gz': f"{symbol}_{self.timeframe}.csv.gz"
        }
        
        file_name = file_patterns.get(self.data_format, f"{symbol}.csv")
        file_path = self.data_directory / file_name
        
        if not file_path.exists():
            # Try alternative naming
            file_path = self.data_directory / f"{symbol}.{self.data_format}"
        
        if not file_path.exists():
            return None
        
        # Load based on format
        if self.data_format == 'csv':
            df = pd.read_csv(file_path, parse_dates=True, index_col='timestamp')
        elif self.data_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif self.data_format == 'json':
            df = pd.read_json(file_path, orient='records')
            df.set_index('timestamp', inplace=True)
        elif self.data_format == 'gz':
            df = pd.read_csv(file_path, compression='gzip', parse_dates=True, index_col='timestamp')
        else:
            raise ValueError(f"Unsupported format: {self.data_format}")
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing columns in {symbol} data")
            return None
        
        return df
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Berechnet Datenqualitäts-Score"""
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Check for zero volumes
        zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
        
        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        anomaly_count = 0
        
        for col in price_cols:
            # Check for negative prices
            anomaly_count += (df[col] <= 0).sum()
            
            # Check for extreme price jumps (>50% in one bar)
            if len(df) > 1:
                returns = df[col].pct_change()
                anomaly_count += (returns.abs() > 0.5).sum()
        
        anomaly_ratio = anomaly_count / (len(df) * len(price_cols))
        
        # Calculate quality score
        quality_score = 1.0 - (missing_ratio + zero_volume_ratio + anomaly_ratio) / 3
        
        return max(0.0, min(1.0, quality_score))
    
    async def get_next_market_events(self) -> List[MarketEvent]:
        """
        Gibt nächste MarketEvents zurück
        WICHTIG: Strikt Point-in-Time, keine Future-Information
        """
        
        if not self.current_timestamp:
            return []
        
        market_events = []
        
        for symbol, df in self.data_frames.items():
            idx = self.data_indices[symbol]
            
            # Check if more data available
            if idx >= len(df):
                continue
            
            # Get current row (Point-in-Time)
            current_time = df.index[idx]
            
            # Only process if timestamp matches current time
            if current_time <= self.current_timestamp:
                row = df.iloc[idx]
                
                # Create market event
                event = self._create_market_event(symbol, current_time, row)
                market_events.append(event)
                
                # Increment index
                self.data_indices[symbol] += 1
        
        # Advance timestamp if all symbols processed current time
        if market_events:
            # Find next timestamp
            next_timestamps = []
            for symbol, df in self.data_frames.items():
                idx = self.data_indices[symbol]
                if idx < len(df):
                    next_timestamps.append(df.index[idx])
            
            if next_timestamps:
                self.current_timestamp = min(next_timestamps)
        
        return market_events
    
    def _create_market_event(self, symbol: str, timestamp: datetime, row: pd.Series) -> MarketEvent:
        """Erstellt MarketEvent aus Datenzeile"""
        
        # Basic OHLCV
        event = create_market_event_from_candle(
            symbol=symbol,
            timestamp=timestamp,
            ohlcv=(row['open'], row['high'], row['low'], row['close'], row['volume'])
        )
        
        # Add additional data if available
        if 'bid' in row and 'ask' in row:
            event.bid_price = row['bid']
            event.ask_price = row['ask']
            event.spread_bps = ((row['ask'] - row['bid']) / row['close']) * 10000
        
        if 'trades' in row:
            event.trades_count = int(row['trades'])
        
        if 'vwap' in row:
            event.vwap = row['vwap']
        
        # Data quality
        event.data_quality = self.data_quality_scores.get(symbol, 1.0)
        
        return event
    
    def has_more_data(self) -> bool:
        """Prüft ob weitere Daten verfügbar sind"""
        
        for symbol, df in self.data_frames.items():
            idx = self.data_indices[symbol]
            if idx < len(df):
                return True
        
        return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """Gibt Informationen über geladene Daten zurück"""
        
        info = {
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'timeframe': self.timeframe,
            'current_timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'data_exhausted': self.data_exhausted,
            'symbol_stats': {}
        }
        
        for symbol, df in self.data_frames.items():
            info['symbol_stats'][symbol] = {
                'total_bars': len(df),
                'processed_bars': self.data_indices[symbol],
                'remaining_bars': len(df) - self.data_indices[symbol],
                'quality_score': self.data_quality_scores.get(symbol, 0.0),
                'date_range': f"{df.index[0]} - {df.index[-1]}"
            }
        
        return info


class TickDataHandler(DataHandler):
    """
    Handler für Tick-Level Daten (Trade-by-Trade)
    Höchste Granularität für präzise Backtests
    """
    
    def __init__(self,
                 event_bus: EventBus,
                 tick_data_files: Dict[str, str],
                 start_date: datetime,
                 end_date: datetime,
                 batch_size: int = 1000):
        
        super().__init__(event_bus)
        self.tick_data_files = tick_data_files
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        
        # Tick data readers
        self.tick_readers: Dict[str, Any] = {}
        self.tick_buffers: Dict[str, deque] = {}
        
        # Orderbook reconstruction
        self.orderbook_states: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialisiert Tick Data Readers"""
        
        for symbol, file_path in self.tick_data_files.items():
            try:
                # Initialize reader based on file format
                if file_path.endswith('.gz'):
                    reader = gzip.open(file_path, 'rt')
                else:
                    reader = open(file_path, 'r')
                
                self.tick_readers[symbol] = reader
                self.tick_buffers[symbol] = deque(maxlen=self.batch_size * 2)
                self.orderbook_states[symbol] = {
                    'bids': {},
                    'asks': {},
                    'last_price': None,
                    'last_size': None
                }
                
                # Pre-fill buffer
                await self._fill_buffer(symbol)
                
                logger.info(f"Tick reader initialisiert für {symbol}")
                
            except Exception as e:
                logger.error(f"Fehler bei Tick-Daten Initialisierung für {symbol}: {e}")
    
    async def _fill_buffer(self, symbol: str) -> None:
        """Füllt Tick-Buffer für Symbol"""
        
        reader = self.tick_readers.get(symbol)
        if not reader:
            return
        
        buffer = self.tick_buffers[symbol]
        
        try:
            for _ in range(self.batch_size):
                line = reader.readline()
                if not line:
                    break
                
                # Parse tick data (assumes CSV format)
                tick = self._parse_tick_line(line)
                
                if tick and self.start_date <= tick['timestamp'] <= self.end_date:
                    buffer.append(tick)
                    
        except Exception as e:
            logger.error(f"Error filling buffer for {symbol}: {e}")
    
    def _parse_tick_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parst einzelne Tick-Zeile"""
        
        try:
            # Example format: timestamp,price,size,side,exchange
            parts = line.strip().split(',')
            
            return {
                'timestamp': pd.to_datetime(parts[0]),
                'price': float(parts[1]),
                'size': float(parts[2]),
                'side': parts[3],
                'exchange': parts[4] if len(parts) > 4 else 'unknown'
            }
        except Exception:
            return None
    
    async def get_next_market_events(self) -> List[MarketEvent]:
        """Konvertiert Ticks zu MarketEvents"""
        
        # Implementation würde Tick-Aggregation zu OHLCV oder
        # Orderbook-Snapshots beinhalten
        
        # Simplified for example
        return []
    
    def has_more_data(self) -> bool:
        """Prüft ob weitere Tick-Daten verfügbar"""
        
        for buffer in self.tick_buffers.values():
            if buffer:
                return True
        
        for reader in self.tick_readers.values():
            # Check if reader has more data
            pos = reader.tell()
            line = reader.readline()
            if line:
                reader.seek(pos)  # Reset position
                return True
        
        return False


class SimulatedDataHandler(DataHandler):
    """
    Simulierter Data Handler für Testing
    Generiert synthetische Marktdaten
    """
    
    def __init__(self,
                 event_bus: EventBus,
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 timeframe_minutes: int = 60,
                 initial_prices: Optional[Dict[str, float]] = None):
        
        super().__init__(event_bus)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe_minutes = timeframe_minutes
        
        # Price simulation parameters
        self.initial_prices = initial_prices or {symbol: 100.0 for symbol in symbols}
        self.current_prices = self.initial_prices.copy()
        
        # Volatility parameters
        self.volatilities = {symbol: 0.02 for symbol in symbols}  # 2% daily vol
        self.drift = {symbol: 0.0001 for symbol in symbols}  # Slight upward drift
        
        # Current time
        self.current_timestamp = start_date
    
    async def initialize(self) -> None:
        """Initialisierung für simulierte Daten"""
        logger.info(f"SimulatedDataHandler initialisiert für {len(self.symbols)} Symbole")
    
    async def get_next_market_events(self) -> List[MarketEvent]:
        """Generiert synthetische Marktdaten"""
        
        if self.current_timestamp > self.end_date:
            return []
        
        market_events = []
        
        for symbol in self.symbols:
            # Generate price movement
            price_data = self._generate_price_bar(symbol)
            
            # Create market event
            event = create_market_event_from_candle(
                symbol=symbol,
                timestamp=self.current_timestamp,
                ohlcv=price_data
            )
            
            market_events.append(event)
        
        # Advance timestamp
        self.current_timestamp += timedelta(minutes=self.timeframe_minutes)
        
        return market_events
    
    def _generate_price_bar(self, symbol: str) -> Tuple[float, float, float, float, float]:
        """Generiert OHLCV Bar"""
        
        current_price = self.current_prices[symbol]
        volatility = self.volatilities[symbol]
        drift = self.drift[symbol]
        
        # Generate returns
        returns = np.random.normal(drift, volatility / np.sqrt(252 * 24))  # Hourly vol
        
        # Generate OHLC
        open_price = current_price
        close_price = current_price * (1 + returns)
        
        # Intrabar volatility
        intrabar_vol = volatility / np.sqrt(252 * 24 * 2)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intrabar_vol)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intrabar_vol)))
        
        # Volume (log-normal distributed)
        avg_volume = 1000000  # 1M base volume
        volume = np.random.lognormal(np.log(avg_volume), 0.5)
        
        # Update current price
        self.current_prices[symbol] = close_price
        
        return (open_price, high_price, low_price, close_price, volume)
    
    def has_more_data(self) -> bool:
        """Prüft ob weitere simulierte Daten generiert werden sollen"""
        return self.current_timestamp <= self.end_date