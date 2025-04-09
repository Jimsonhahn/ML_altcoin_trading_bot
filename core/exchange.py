#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exchange-Verbindungen für den Trading Bot.
Stellt Verbindungen zu verschiedenen Kryptobörsen her und bietet eine einheitliche API.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from datetime import datetime

import ccxt

from config.settings import Settings


class ExchangeException(Exception):
    """Basisklasse für alle Exchange-spezifischen Ausnahmen"""
    pass


class OrderFailedException(ExchangeException):
    """Wird ausgelöst, wenn eine Order fehlschlägt"""
    pass


class ConnectionFailedException(ExchangeException):
    """Wird ausgelöst, wenn die Verbindung zum Exchange fehlschlägt"""
    pass


class ExchangeBase:
    """Basisklasse für alle Exchange-Implementierungen"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Exchange-Basisklasse.

        Args:
            settings: Bot-Konfiguration
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.name = "base"

    def connect(self) -> bool:
        """
        Stellt eine Verbindung zum Exchange her.

        Returns:
            True bei erfolgreicher Verbindung, False sonst
        """
        raise NotImplementedError("Subklassen müssen connect() implementieren")

    def get_balance(self, currency: str = "USDT") -> float:
        """
        Ruft den Kontostand für eine bestimmte Währung ab.

        Args:
            currency: Währungscode

        Returns:
            Kontostand als Float
        """
        raise NotImplementedError("Subklassen müssen get_balance() implementieren")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft den aktuellen Ticker für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')

        Returns:
            Ticker-Informationen
        """
        raise NotImplementedError("Subklassen müssen get_ticker() implementieren")

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Ruft OHLCV-Daten für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '1h', '1d', etc.)
            limit: Anzahl der Einträge

        Returns:
            DataFrame mit OHLCV-Daten
        """
        raise NotImplementedError("Subklassen müssen get_ohlcv() implementieren")

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[
        str, Any]:
        """
        Platziert eine Order.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            order_type: Ordertyp ('market', 'limit', etc.)
            side: Orderrichtung ('buy', 'sell')
            amount: Ordermenge
            price: Orderpreis (nur für Limit-Orders)

        Returns:
            Order-Informationen
        """
        raise NotImplementedError("Subklassen müssen place_order() implementieren")

    def get_order(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Ruft Informationen zu einer Order ab.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (optional, wenn vom Exchange benötigt)

        Returns:
            Order-Informationen
        """
        raise NotImplementedError("Subklassen müssen get_order() implementieren")

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Storniert eine offene Order.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (optional, wenn vom Exchange benötigt)

        Returns:
            True bei erfolgreicher Stornierung, False sonst
        """
        raise NotImplementedError("Subklassen müssen cancel_order() implementieren")

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Ruft alle offenen Orders ab.

        Args:
            symbol: Handelssymbol (optional, filtert nach Symbol)

        Returns:
            Liste von Order-Informationen
        """
        raise NotImplementedError("Subklassen müssen get_open_orders() implementieren")


class BinanceExchange(ExchangeBase):
    """Implementierung für Binance Exchange"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Binance-Exchange-Klasse.

        Args:
            settings: Bot-Konfiguration
        """
        super().__init__(settings)
        self.name = "binance"
        self.exchange = None
        self.last_request_time = 0
        self.rate_limit = 1.0 / settings.get('exchange.api_throttle_rate', 1.0)  # Minimale Zeit zwischen Anfragen

    def _throttle(self) -> None:
        """Drosselt API-Anfragen, um Rate Limits einzuhalten"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self.last_request_time = time.time()

    def connect(self) -> bool:
        """
        Stellt eine Verbindung zu Binance her.

        Returns:
            True bei erfolgreicher Verbindung, False sonst
        """
        try:
            # API-Schlüssel aus Umgebungsvariablen laden
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')

            if not api_key or not api_secret:
                self.logger.error("Binance API credentials not found in environment variables")
                return False

            # Testnet oder Live-Modus
            testnet = self.settings.get('exchange.testnet', True)

            # Exchange-Instanz erstellen
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            # Testnet aktivieren, falls konfiguriert
            if testnet:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("Connected to Binance Testnet")
            else:
                self.logger.info("Connected to Binance Live Exchange")

            # Verbindung testen
            self.exchange.fetch_balance()
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            raise ConnectionFailedException(f"Could not connect to Binance: {e}")

    def get_balance(self, currency: str = "USDT") -> float:
        """
        Ruft den Kontostand für eine bestimmte Währung ab.

        Args:
            currency: Währungscode

        Returns:
            Kontostand als Float
        """
        self._throttle()

        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get(currency, 0.0)
        except Exception as e:
            self.logger.error(f"Failed to get balance for {currency}: {e}")
            return 0.0

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft den aktuellen Ticker für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')

        Returns:
            Ticker-Informationen
        """
        self._throttle()

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise ExchangeException(f"Could not get ticker for {symbol}: {e}")

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Ruft OHLCV-Daten für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '1h', '1d', etc.)
            limit: Anzahl der Einträge

        Returns:
            DataFrame mit OHLCV-Daten
        """
        self._throttle()

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            raise ExchangeException(f"Could not get OHLCV for {symbol}: {e}")

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[
        str, Any]:
        """
        Platziert eine Order.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            order_type: Ordertyp ('market', 'limit', etc.)
            side: Orderrichtung ('buy', 'sell')
            amount: Ordermenge
            price: Orderpreis (nur für Limit-Orders)

        Returns:
            Order-Informationen
        """
        self._throttle()

        try:
            params = {}

            # Für Limit-Orders den Preis hinzufügen
            if order_type.lower() == 'limit' and price is not None:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type.lower(),
                    side=side.lower(),
                    amount=amount,
                    price=price,
                    params=params
                )
            else:
                # Für Market-Orders
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type.lower(),
                    side=side.lower(),
                    amount=amount,
                    params=params
                )

            self.logger.info(f"Order placed: {side.upper()} {amount} {symbol} at {price if price else 'market price'}")
            return order

        except Exception as e:
            self.logger.error(f"Failed to place order for {symbol}: {e}")
            raise OrderFailedException(f"Could not place order for {symbol}: {e}")

    def get_order(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Ruft Informationen zu einer Order ab.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (erforderlich für Binance)

        Returns:
            Order-Informationen
        """
        self._throttle()

        if not symbol:
            raise ExchangeException("Symbol is required to get order information on Binance")

        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            self.logger.error(f"Failed to get order {order_id} for {symbol}: {e}")
            raise ExchangeException(f"Could not get order {order_id}: {e}")

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Storniert eine offene Order.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (erforderlich für Binance)

        Returns:
            True bei erfolgreicher Stornierung, False sonst
        """
        self._throttle()

        if not symbol:
            raise ExchangeException("Symbol is required to cancel order on Binance")

        try:
            self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            return False

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Ruft alle offenen Orders ab.

        Args:
            symbol: Handelssymbol (optional, filtert nach Symbol)

        Returns:
            Liste von Order-Informationen
        """
        self._throttle()

        try:
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            return open_orders
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []


class PaperTradeExchange(ExchangeBase):
    """Implementierung für Paper Trading (Simulation)"""

    def __init__(self, settings: Settings):
        """
        Initialisiert die Paper-Trading-Exchange-Klasse.

        Args:
            settings: Bot-Konfiguration
        """
        super().__init__(settings)
        self.name = "paper"

        # Verwenden einer echten Exchange für Daten
        self.data_exchange = BinanceExchange(settings)

        # Paper-Trading-Zustand
        self.balance = {
            "USDT": settings.get('backtest.initial_balance', 10000),
        }
        self.open_orders = []
        self.order_id_counter = 1
        self.trades = []
        self.commission_rate = settings.get('backtest.commission', 0.001)

    def connect(self) -> bool:
        """
        Stellt eine Verbindung zum Daten-Exchange her.

        Returns:
            True bei erfolgreicher Verbindung, False sonst
        """
        try:
            success = self.data_exchange.connect()
            if success:
                self.logger.info("Connected to data exchange for paper trading")
            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to data exchange: {e}")
            return False

    def get_balance(self, currency: str = "USDT") -> float:
        """
        Ruft den simulierten Kontostand für eine bestimmte Währung ab.

        Args:
            currency: Währungscode

        Returns:
            Kontostand als Float
        """
        return self.balance.get(currency, 0.0)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft den aktuellen Ticker für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')

        Returns:
            Ticker-Informationen
        """
        return self.data_exchange.get_ticker(symbol)

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        Ruft OHLCV-Daten für ein Symbol ab.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '1h', '1d', etc.)
            limit: Anzahl der Einträge

        Returns:
            DataFrame mit OHLCV-Daten
        """
        return self.data_exchange.get_ohlcv(symbol, timeframe, limit)

    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[
        str, Any]:
        """
        Platziert eine simulierte Order.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            order_type: Ordertyp ('market', 'limit', etc.)
            side: Orderrichtung ('buy', 'sell')
            amount: Ordermenge
            price: Orderpreis (nur für Limit-Orders)

        Returns:
            Order-Informationen
        """
        ticker = self.get_ticker(symbol)
        current_price = ticker['last']

        # Für Market Orders den aktuellen Preis verwenden
        if order_type.lower() == 'market':
            price = current_price

        # Prüfen, ob genug Guthaben vorhanden ist
        base_currency, quote_currency = symbol.split('/')

        order_value = amount * price
        commission = order_value * self.commission_rate

        # Überprüfen, ob das Konto die Transaktion abdecken kann
        if side.lower() == 'buy':
            if self.balance.get(quote_currency, 0) < order_value + commission:
                self.logger.warning(f"Insufficient {quote_currency} balance for order")
                raise OrderFailedException(f"Insufficient {quote_currency} balance")

        elif side.lower() == 'sell':
            if self.balance.get(base_currency, 0) < amount:
                self.logger.warning(f"Insufficient {base_currency} balance for order")
                raise OrderFailedException(f"Insufficient {base_currency} balance")

        # Order-ID generieren
        order_id = str(self.order_id_counter)
        self.order_id_counter += 1

        # Zeitstempel
        timestamp = int(time.time() * 1000)

        # Order erstellen
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type.lower(),
            'side': side.lower(),
            'amount': amount,
            'price': price,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
            'status': 'open',
            'filled': 0.0,
            'remaining': amount,
            'cost': 0.0,
            'fee': {
                'currency': quote_currency,
                'cost': 0.0,
                'rate': self.commission_rate
            }
        }

        # Für Market Orders sofort ausführen
        if order_type.lower() == 'market':
            self._execute_order(order)
        else:
            # Limit Orders zu offenen Orders hinzufügen
            self.open_orders.append(order)
            self.logger.info(
                f"Placed {order_type} {side} order: {amount} {base_currency} @ {price} {quote_currency}")

        return order

    def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine simulierte Order aus.

        Args:
            order: Order-Informationen

        Returns:
            Aktualisierte Order-Informationen
        """
        symbol = order['symbol']
        side = order['side']
        amount = order['amount']
        price = order['price']

        base_currency, quote_currency = symbol.split('/')

        # Order-Kosten berechnen
        cost = amount * price
        commission = cost * self.commission_rate

        # Kontostand aktualisieren
        if side == 'buy':
            # Quote-Währung abziehen (z.B. USDT)
            self.balance[quote_currency] = self.balance.get(quote_currency, 0) - cost - commission

            # Base-Währung hinzufügen (z.B. BTC)
            self.balance[base_currency] = self.balance.get(base_currency, 0) + amount

        elif side == 'sell':
            # Base-Währung abziehen
            self.balance[base_currency] = self.balance.get(base_currency, 0) - amount

            # Quote-Währung hinzufügen
            self.balance[quote_currency] = self.balance.get(quote_currency, 0) + cost - commission

        # Order-Status aktualisieren
        order['status'] = 'closed'
        order['filled'] = amount
        order['remaining'] = 0.0
        order['cost'] = cost
        order['fee']['cost'] = commission

        # Zu Trades hinzufügen
        self.trades.append({
            'order_id': order['id'],
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'cost': cost,
            'commission': commission,
            'timestamp': int(time.time() * 1000)
        })

        self.logger.info(f"Executed {side} order: {amount} {base_currency} @ {price} {quote_currency}")

        return order

    def get_order(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Ruft Informationen zu einer simulierten Order ab.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (nicht verwendet, aber für Kompatibilität)

        Returns:
            Order-Informationen
        """
        # Offene Orders durchsuchen
        for order in self.open_orders:
            if order['id'] == order_id:
                return order

        # Geschlossene Orders durchsuchen (aus Trades rekonstruieren)
        for trade in self.trades:
            if trade['order_id'] == order_id:
                return {
                    'id': order_id,
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'amount': trade['amount'],
                    'price': trade['price'],
                    'timestamp': trade['timestamp'],
                    'datetime': datetime.fromtimestamp(trade['timestamp'] / 1000).isoformat(),
                    'status': 'closed',
                    'filled': trade['amount'],
                    'remaining': 0.0,
                    'cost': trade['cost'],
                    'fee': {
                        'currency': trade['symbol'].split('/')[1],
                        'cost': trade['commission'],
                        'rate': self.commission_rate
                    }
                }

        # Order nicht gefunden
        return None

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Storniert eine offene simulierte Order.

        Args:
            order_id: ID der Order
            symbol: Handelssymbol (nicht verwendet, aber für Kompatibilität)

        Returns:
            True bei erfolgreicher Stornierung, False sonst
        """
        for i, order in enumerate(self.open_orders):
            if order['id'] == order_id:
                # Order aus offenen Orders entfernen
                self.open_orders.pop(i)
                self.logger.info(f"Cancelled order {order_id}")
                return True

        return False

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Ruft alle offenen simulierten Orders ab.

        Args:
            symbol: Handelssymbol (optional, filtert nach Symbol)

        Returns:
            Liste von Order-Informationen
        """
        if symbol:
            return [order for order in self.open_orders if order['symbol'] == symbol]
        else:
            return self.open_orders

    def update(self) -> None:
        """
        Aktualisiert den simulierten Exchange-Zustand.
        Prüft, ob Limit-Orders ausgeführt werden können.
        """
        for i in range(len(self.open_orders) - 1, -1, -1):
            order = self.open_orders[i]
            symbol = order['symbol']
            price = order['price']
            side = order['side']

            try:
                # Aktuellen Preis abfragen
                ticker = self.get_ticker(symbol)
                current_price = ticker['last']

                # Prüfen, ob Order ausgeführt werden kann
                can_execute = False

                if side == 'buy' and current_price <= price:
                    # Kauforder kann ausgeführt werden, wenn der Preis unter den Limit-Preis fällt
                    can_execute = True
                elif side == 'sell' and current_price >= price:
                    # Verkaufsorder kann ausgeführt werden, wenn der Preis über den Limit-Preis steigt
                    can_execute = True

                if can_execute:
                    # Order ausführen
                    self._execute_order(order)
                    # Aus offenen Orders entfernen
                    self.open_orders.pop(i)

            except Exception as e:
                self.logger.error(f"Error updating order {order['id']}: {e}")


class ExchangeFactory:
    """Factory-Klasse zum Erstellen von Exchange-Instanzen"""

    @staticmethod
    def create(settings: Settings, mode: str = "paper") -> ExchangeBase:
        """
        Erstellt eine Exchange-Instanz basierend auf dem Modus.

        Args:
            settings: Bot-Konfiguration
            mode: Trading-Modus ('live', 'paper', 'backtest')

        Returns:
            Exchange-Instanz
        """
        if mode == "live":
            exchange_name = settings.get('exchange.name', 'binance').lower()

            if exchange_name == "binance":
                return BinanceExchange(settings)
            else:
                raise ValueError(f"Unsupported exchange: {exchange_name}")

        elif mode == "paper":
            return PaperTradeExchange(settings)

        elif mode == "backtest":
            # Backtesting-Exchange wird von der Backtesting-Engine verwaltet
            return PaperTradeExchange(settings)

        else:
            raise ValueError(f"Unsupported trading mode: {mode}")