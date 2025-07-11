# ML_altcoin_trading_bot/config.py

# --- API Keys & Exchange Setup ---
EXCHANGE_API_KEY = 'YOUR_API_KEY'  # ERSETZEN SIE DIES!
EXCHANGE_API_SECRET = 'YOUR_API_SECRET'  # ERSETZEN SIE DIES!
EXCHANGE_NAME = 'binance'  # Oder 'kucoin', 'kraken' etc. (unterstützt von ccxt)
# Für Sandbox/Testnet (falls von der Börse unterstützt und ccxt es abbildet)
# EXCHANGE_OPTIONS = {
# 'options': {
# 'defaultType': 'future', # oder 'spot'
#         # 'adjustForTimeDifference': True, # Wichtig für ccxt
#     },
# 'urls': {
# 'api': {
# 'public': 'https://testnet.binancefuture.com/fapi/v1', # Beispiel Binance Futures Testnet
# 'private': 'https://testnet.binancefuture.com/fapi/v1',
#         }
#     }
# }
EXCHANGE_OPTIONS = { # Für Live Trading (Beispiel Binance Spot)
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
}


# --- Trading Parameters ---
TRADING_PAIR = 'BTC/USDT'
TIMEFRAME = '1h'  # z.B. '1m', '5m', '15m', '1h', '4h', '1d'
HISTORICAL_DATA_DAYS = 365  # Wie viele Tage an historischen Daten für das Training laden

# --- Feature Engineering ---
BASE_FEATURES = ['RSI', 'MACD_line', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower']
# Erweiterte Features (werden in data_handler hinzugefügt)
LAG_FEATURES_N = [1, 2, 3, 5] # Für gelaggte Returns
VOLATILITY_WINDOW = 20 # Für rollierende Volatilität
MA_WINDOWS = [50, 200] # Für Moving Averages

# --- Regime Detection (ADX/ATR Basiert) ---
ATR_LENGTH = 14
ADX_LENGTH = 14
VOLA_THRESHOLD_PERCENT_OF_PRICE = 0.02 # z.B. ATR > 2% des Preises = hohe Vola
ADX_TREND_THRESHOLD = 25

# --- Target Variable (Triple Barrier Method) ---
LOOK_FORWARD_CANDLES = 10 # Wie viele Kerzen in die Zukunft schauen für TP/SL
TP_PERCENT = 0.02  # Take Profit 2%
SL_PERCENT = 0.01  # Stop Loss 1%

# --- Model Training ---
MODEL_TYPE = 'RandomForest'  # 'RandomForest' oder 'LSTM'
MODEL_SAVE_PATH = 'trained_model.pkl'
SCALER_SAVE_PATH = 'scaler.pkl'
REGIME_MODEL_SAVE_PATH = 'regime_model_hmm.pkl' # Falls HMM verwendet wird

# RandomForest Hyperparameters (Beispiel - für GridSearchCV)
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced', None]
}
CV_SPLITS = 5 # Für TimeSeriesSplit

# LSTM Parameters (Beispiel, falls verwendet)
LSTM_SEQUENCE_LENGTH = 60 # Anzahl vergangener Kerzen als Input für LSTM
LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# --- Backtesting ---
BACKTEST_INITIAL_BALANCE = 10000  # USD oder Äquivalent
BACKTEST_COMMISSION_RATE = 0.001  # 0.1% Kommission pro Trade
BACKTEST_RISK_PER_TRADE = 0.02 # z.B. 2% des Kapitals pro Trade riskieren für Positionsgröße

# --- Live Trading ---
LIVE_TRADING_RISK_PER_TRADE = 0.01 # Risiko pro Trade im Live-Modus
ORDER_TYPE = 'LIMIT' # 'MARKET' oder 'LIMIT' (Limit Orders sind oft besser bzgl. Slippage)
SLIPPAGE_FACTOR_LIMIT_ORDER = 0.0005 # Für Limit Orders: Preis um diesen Faktor schlechter setzen, um Ausführung zu erhöhen

# --- Logging ---
LOG_FILE = 'trading_bot.log'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Telegram Notifications ---
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # ERSETZEN SIE DIES!
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'  # ERSETZEN SIE DIES! (kann eine Gruppen-ID oder Ihre User-ID sein)