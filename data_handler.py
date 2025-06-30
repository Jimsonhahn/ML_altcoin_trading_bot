# ML_altcoin_trading_bot/data_handler.py
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as pta
import time
import logging
from sklearn.preprocessing import StandardScaler

import config  # Importiert Ihre Konfigurationsdatei

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, exchange_name=config.EXCHANGE_NAME, api_key=config.EXCHANGE_API_KEY,
                 api_secret=config.EXCHANGE_API_SECRET, exchange_options=config.EXCHANGE_OPTIONS):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                **exchange_options
            })
            self.exchange.load_markets()
            logger.info(f"Successfully connected to {exchange_name}.")
        except Exception as e:
            logger.error(f"Error initializing exchange {exchange_name}: {e}")
            raise

        self.data = pd.DataFrame()
        self.scaler = StandardScaler()  # Für Feature Skalierung
        self.fitted_scaler = False

    def fetch_ohlcv(self, symbol=config.TRADING_PAIR, timeframe=config.TIMEFRAME,
                    limit_days=config.HISTORICAL_DATA_DAYS):
        logger.info(f"Fetching OHLCV data for {symbol} with timeframe {timeframe} for the last {limit_days} days.")
        try:
            # Berechne 'since' basierend auf limit_days
            # ccxt erwartet Millisekunden seit Epoche
            since = self.exchange.milliseconds() - limit_days * 24 * 60 * 60 * 1000

            all_ohlcv = []
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)  # Limit kann variieren
                if len(ohlcv) == 0:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + self.exchange.parse_timeframe(timeframe) * 1000  # Nächster Startpunkt
                if len(ohlcv) < 1000:  # Wenn weniger als das Limit zurückkommt, sind wir am Ende
                    break
                time.sleep(self.exchange.rateLimit / 1000)  # Respektiere Rate Limits

            if not all_ohlcv:
                logger.warning(f"No data fetched for {symbol}. Check symbol and timeframe.")
                return pd.DataFrame()

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Datentypen sicherstellen
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)  # Fälle entfernen, wo Umwandlung fehlschlug
            self.data = df
            logger.info(f"Fetched {len(self.data)} data points for {symbol}.")
            return self.data
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self):
        if self.data.empty:
            logger.warning("Data is empty, cannot add technical indicators.")
            return

        logger.info("Adding base technical indicators...")
        # RSI
        self.data['RSI'] = pta.rsi(self.data['close'], length=14)
        # MACD
        macd = pta.macd(self.data['close'])
        if macd is not None and not macd.empty:
            self.data['MACD_line'] = macd[f'MACD_{macd.columns[0].split("_")[1]}_{macd.columns[0].split("_")[2]}']
            self.data['MACD_hist'] = macd[f'MACDh_{macd.columns[1].split("_")[1]}_{macd.columns[1].split("_")[2]}']
            self.data['MACD_signal'] = macd[f'MACDs_{macd.columns[2].split("_")[1]}_{macd.columns[2].split("_")[2]}']
        else:
            self.data['MACD_line'] = self.data['MACD_hist'] = self.data['MACD_signal'] = np.nan

        # Bollinger Bands
        bbands = pta.bbands(self.data['close'], length=20)
        if bbands is not None and not bbands.empty:
            self.data['BB_lower'] = bbands[bbands.columns[0]]  # z.B. BBL_20_2.0
            self.data['BB_middle'] = bbands[bbands.columns[1]]  # z.B. BBM_20_2.0
            self.data['BB_upper'] = bbands[bbands.columns[2]]  # z.B. BBU_20_2.0
        else:
            self.data['BB_lower'] = self.data['BB_middle'] = self.data['BB_upper'] = np.nan

        logger.info("Base technical indicators added.")

    def add_expanded_features(self):
        if self.data.empty:
            logger.warning("Data is empty, cannot add expanded features.")
            return
        logger.info("Adding expanded features (lags, volatility, MA distance)...")

        self.data['returns'] = self.data['close'].pct_change().fillna(0)

        for lag in config.LAG_FEATURES_N:
            self.data[f'returns_lag_{lag}'] = self.data['returns'].shift(lag).fillna(0)

        self.data[f'volatility_{config.VOLATILITY_WINDOW}d'] = self.data['returns'].rolling(
            window=config.VOLATILITY_WINDOW).std().fillna(0) * np.sqrt(config.VOLATILITY_WINDOW)

        for window in config.MA_WINDOWS:
            self.data[f'MA{window}'] = pta.sma(self.data['close'], length=window)
            self.data[f'dist_MA{window}'] = (self.data['close'] - self.data[f'MA{window}']) / self.data[f'MA{window}']
            self.data[f'MA{window}'].fillna(method='bfill', inplace=True)  # Fill NaNs at beginning
            self.data[f'dist_MA{window}'].fillna(0, inplace=True)

        logger.info("Expanded features added.")

    def add_regime_features_atr_adx(self):
        if self.data.empty:
            logger.warning("Data is empty, cannot add regime features.")
            return
        logger.info("Adding ATR/ADX based regime features...")

        self.data['ATR'] = pta.atr(self.data['high'], self.data['low'], self.data['close'], length=config.ATR_LENGTH)

        adx_indicator = pta.adx(self.data['high'], self.data['low'], self.data['close'], length=config.ADX_LENGTH)
        if adx_indicator is not None and not adx_indicator.empty:
            self.data['ADX'] = adx_indicator[f'ADX_{config.ADX_LENGTH}']
            self.data['DMP'] = adx_indicator[f'DMP_{config.ADX_LENGTH}']  # DI+
            self.data['DMN'] = adx_indicator[f'DMN_{config.ADX_LENGTH}']  # DI-
        else:
            self.data['ADX'] = self.data['DMP'] = self.data['DMN'] = np.nan

        self.data.fillna(method='bfill', inplace=True)  # Fill NaNs from indicators at beginning

        conditions = [
            (self.data['ADX'] > config.ADX_TREND_THRESHOLD) & (self.data['DMP'] > self.data['DMN']) & (
                        self.data['ATR'] > self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE),
            (self.data['ADX'] > config.ADX_TREND_THRESHOLD) & (self.data['DMP'] > self.data['DMN']) & (
                        self.data['ATR'] <= self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE),
            (self.data['ADX'] > config.ADX_TREND_THRESHOLD) & (self.data['DMN'] > self.data['DMP']) & (
                        self.data['ATR'] > self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE),
            (self.data['ADX'] > config.ADX_TREND_THRESHOLD) & (self.data['DMN'] > self.data['DMP']) & (
                        self.data['ATR'] <= self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE),
            (self.data['ADX'] <= config.ADX_TREND_THRESHOLD) & (
                        self.data['ATR'] > self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE),
            (self.data['ADX'] <= config.ADX_TREND_THRESHOLD) & (
                        self.data['ATR'] <= self.data['close'] * config.VOLA_THRESHOLD_PERCENT_OF_PRICE)
        ]
        # Regimes: 0: Bull Trend High Vola, 1: Bull Trend Low Vola, 2: Bear Trend High Vola,
        #          3: Bear Trend Low Vola, 4: Ranging High Vola, 5: Ranging Low Vola
        choices = [0, 1, 2, 3, 4, 5]
        self.data['regime_atr_adx'] = np.select(conditions, choices, default=5)
        logger.info("ATR/ADX regime features added.")

    def create_labels_triple_barrier(self, look_forward=config.LOOK_FORWARD_CANDLES,
                                     tp_pct=config.TP_PERCENT, sl_pct=config.SL_PERCENT):
        if self.data.empty:
            logger.warning("Data is empty, cannot create labels.")
            return pd.Series(dtype='int')

        logger.info(
            f"Creating labels using Triple Barrier Method (TP: {tp_pct * 100}%, SL: {sl_pct * 100}%, Look Forward: {look_forward} candles)")

        out = pd.Series(np.nan, index=self.data.index)  # Initialize with NaNs
        prices = self.data['close']

        for i in range(len(prices) - look_forward):
            entry_price = prices.iloc[i]

            # Path prices
            path = prices.iloc[i + 1: i + 1 + look_forward]

            # Take profit target
            tp_target = entry_price * (1 + tp_pct)
            # Stop loss target
            sl_target = entry_price * (1 - sl_pct)

            # When was TP hit?
            tp_hit_times = path[path >= tp_target].index
            # When was SL hit?
            sl_hit_times = path[path <= sl_target].index

            if not tp_hit_times.empty and not sl_hit_times.empty:
                # Both hit, see which one first
                if tp_hit_times[0] < sl_hit_times[0]:
                    out.iloc[i] = 1  # TP hit first (Buy signal)
                else:
                    out.iloc[i] = 2  # SL hit first (Sell/Avoid signal for long)
            elif not tp_hit_times.empty:
                out.iloc[i] = 1  # Only TP hit
            elif not sl_hit_times.empty:
                out.iloc[i] = 2  # Only SL hit
            else:
                # Neither TP nor SL hit within the look_forward window
                # Could be 0 (Hold), or based on price at end of window
                # For simplicity, let's say 0 if neither hit and price is near entry,
                # or assign based on final price relative to entry
                final_price_in_window = path.iloc[-1]
                if final_price_in_window > entry_price * (1 + sl_pct / 2):  # Slight gain or small loss but not SL
                    out.iloc[i] = 1  # Bias towards positive if no strong signal
                elif final_price_in_window < entry_price * (1 - sl_pct / 2):
                    out.iloc[i] = 2
                else:
                    out.iloc[i] = 0  # Neutral or Hold

        self.data['target'] = out.fillna(0).astype(int)  # Fill any remaining NaNs (e.g. at the very end) with 0
        logger.info(f"Labels created. Class distribution: \n{self.data['target'].value_counts(normalize=True)}")
        return self.data['target']

    def get_feature_names(self):
        # Dynamically generate feature names based on config
        feature_names = list(config.BASE_FEATURES)
        feature_names.extend([f'returns_lag_{lag}' for lag in config.LAG_FEATURES_N])
        feature_names.append(f'volatility_{config.VOLATILITY_WINDOW}d')
        feature_names.extend([f'dist_MA{window}' for window in config.MA_WINDOWS])
        feature_names.append('regime_atr_adx')  # Add regime as a feature

        # Ensure all these features actually exist in self.data before returning
        # This is a basic check; robust check would iterate and confirm.
        if not self.data.empty:
            return [f for f in feature_names if f in self.data.columns]
        return feature_names  # Return intended names if data is empty

    def preprocess_data(self, fit_scaler=False):
        if self.data.empty:
            logger.warning("Data is empty, cannot preprocess.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        self.add_technical_indicators()
        self.add_expanded_features()
        self.add_regime_features_atr_adx()
        self.create_labels_triple_barrier()

        self.data.dropna(inplace=True)  # Drop rows with NaNs created by indicators/features

        if self.data.empty:
            logger.warning("Data became empty after adding features and dropping NaNs.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        feature_names = self.get_feature_names()
        X = self.data[feature_names].copy()

        if 'target' not in self.data.columns:
            logger.error("Target variable not created. Cannot proceed.")
            return pd.DataFrame(), pd.Series(dtype='float64')
        y = self.data['target'].copy()

        if fit_scaler:
            logger.info("Fitting and transforming data with StandardScaler.")
            X_scaled = self.scaler.fit_transform(X)
            self.fitted_scaler = True
        elif self.fitted_scaler:
            logger.info("Transforming data with already fitted StandardScaler.")
            X_scaled = self.scaler.transform(X)
        else:
            logger.warning("Scaler not fitted, and fit_scaler is False. Returning unscaled features.")
            X_scaled = X.values  # Return as numpy array for consistency

        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        logger.info(f"Preprocessing complete. Shape of X: {X_scaled_df.shape}, Shape of y: {y.shape}")
        return X_scaled_df, y

    def get_latest_features(self, n_candles=100 + max(config.MA_WINDOWS) + config.LSTM_SEQUENCE_LENGTH):
        """
        Fetches latest data and processes it to get features for the most recent candle.
        n_candles needs to be large enough for all indicator calculations.
        For LSTM, it needs to be sequence_length + indicator calculation needs.
        """
        logger.info(f"Fetching latest {n_candles} candles for live prediction features.")
        # Calculate how many days back n_candles approximately is for the fetch_ohlcv limit_days
        # This is a rough estimate, assumes timeframe is less than 1 day
        timeframe_minutes = self.exchange.parse_timeframe(config.TIMEFRAME) / (60 * 1000)  # timeframe in minutes
        days_needed = (n_candles * timeframe_minutes) / (24 * 60) + 2  # Add a small buffer

        live_data_df = self.fetch_ohlcv(symbol=config.TRADING_PAIR, timeframe=config.TIMEFRAME,
                                        limit_days=int(np.ceil(days_needed)))

        if live_data_df.empty or len(live_data_df) < max(config.MA_WINDOWS):  # Ensure enough data for MAs
            logger.error("Not enough live data fetched to calculate all features.")
            return None, None  # Return None or empty DataFrame

        # Temporarily assign to self.data to use existing methods
        original_data = self.data.copy() if not self.data.empty else None
        self.data = live_data_df

        self.add_technical_indicators()
        self.add_expanded_features()
        self.add_regime_features_atr_adx()
        # No labels needed for live prediction

        self.data.dropna(subset=self.get_feature_names(), inplace=True)  # Drop rows if any feature is NaN

        if self.data.empty:
            logger.error("Live data became empty after feature calculation and NaN drop.")
            if original_data is not None: self.data = original_data  # Restore
            return None, None

        feature_names = self.get_feature_names()
        X_live = self.data[feature_names].copy()

        if not self.fitted_scaler:
            logger.error("Scaler has not been fitted. Cannot scale live features. Train a model first.")
            if original_data is not None: self.data = original_data  # Restore
            return None, None  # Or handle this case by fitting scaler on live_data (not recommended for consistency)

        X_live_scaled = self.scaler.transform(X_live)
        X_live_scaled_df = pd.DataFrame(X_live_scaled, columns=X_live.columns, index=X_live.index)

        current_market_price = self.data['close'].iloc[-1]

        # Restore original data if it existed
        if original_data is not None:
            self.data = original_data
        else:
            self.data = pd.DataFrame()  # Clear it if it was only used for live features

        logger.info(f"Latest features processed. Returning features for timestamp: {X_live_scaled_df.index[-1]}")
        # Return only the latest row of features
        return X_live_scaled_df.iloc[[-1]], current_market_price


if __name__ == '__main__':
    # Example Usage:
    logging.basicConfig(level=logging.INFO)

    data_handler = DataHandler()

    # Fetch historical data
    # data_handler.fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit_days=100) # Smaller for quick test

    # Preprocess data (this will also fit the scaler)
    # X, y = data_handler.preprocess_data(fit_scaler=True)

    # if not X.empty:
    # print("Features (X) head:")
    # print(X.head())
    # print("\nTarget (y) head:")
    # print(y.head())
    # print(f"\nScaler fitted: {data_handler.fitted_scaler}")

    # print("\n--- Simulating fetching latest features (as if for live trading) ---")
    # # Make sure scaler is fitted (e.g. by running preprocess_data with fit_scaler=True first,
    # # or by loading a saved scaler if model_builder handles that)
    # if data_handler.fitted_scaler:
    #     latest_features, current_price = data_handler.get_latest_features()
    #     if latest_features is not None:
    #         print("\nLatest scaled features for prediction:")
    #         print(latest_features)
    #         print(f"Current market price: {current_price}")
    #     else:
    #         print("Could not get latest features.")
    # else:
    #     print("Scaler not fitted. Cannot get latest features. Run preprocess_data(fit_scaler=True) first.")

    # For a full run:
    raw_data = data_handler.fetch_ohlcv(symbol=config.TRADING_PAIR, timeframe=config.TIMEFRAME,
                                        limit_days=config.HISTORICAL_DATA_DAYS)
    if not raw_data.empty:
        X_processed, y_processed = data_handler.preprocess_data(fit_scaler=True)
        if not X_processed.empty:
            print("Processed X head:")
            print(X_processed.head())
            print("Processed y head:")
            print(y_processed.head())

            # Test getting latest features
            print("\nGetting latest features for prediction:")
            latest_X, latest_price = data_handler.get_latest_features()
            if latest_X is not None:
                print(latest_X)
                print(f"Latest price: {latest_price}")
        else:
            print("Data processing resulted in empty X.")
    else:
        print("Failed to fetch raw data.")