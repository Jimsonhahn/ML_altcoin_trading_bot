
# Timestamp patch for Binance
import ccxt
import os

def get_patched_exchange(exchange_name='binance', api_key=None, api_secret=None):
    """Get exchange with timestamp fix"""

    exchange_config = {
        'apiKey': api_key or os.getenv('BINANCE_API_KEY'),
        'secret': api_secret or os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'recvWindow': 60000,  # 60 second tolerance
            'adjustForTimeDifference': True
        }
    }

    if exchange_name == 'binance':
        # Use testnet if no real API keys
        if not exchange_config['apiKey']:
            exchange_config['urls'] = {
                'api': 'https://testnet.binance.vision/api'
            }

        exchange = ccxt.binance(exchange_config)
    else:
        exchange = getattr(ccxt, exchange_name)(exchange_config)

    return exchange
