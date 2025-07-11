#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position Management für den Trading Bot.
Verwaltet offene Positionen und deren Stop-Loss/Take-Profit-Levels.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid


class Position:
    """Repräsentiert eine offene Trading-Position"""

    def __init__(self, symbol: str, entry_price: float, amount: float, side: str = "buy",
                 order_id: Optional[str] = None, entry_time: Optional[datetime] = None):
        """
        Initialisiert eine neue Position.

        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            entry_price: Einstiegspreis
            amount: Positionsgröße
            side: Positionsrichtung ('buy' oder 'sell')
            order_id: ID der ursprünglichen Order (optional)
            entry_time: Zeitpunkt des Positionseinstiegs (optional)
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.entry_price = entry_price
        self.amount = amount
        self.side = side
        self.order_id = order_id
        self.entry_time = entry_time or datetime.now()

        # Stop-Loss und Take-Profit
        self.stop_loss = None
        self.take_profit = None

        # Trailing-Stop-Parameter
        self.trailing_stop = None
        self.trailing_stop_activation = None
        self.highest_price = entry_price
        self.lowest_price = entry_price

        # Exit-Daten
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.profit_loss = None
        self.profit_loss_percent = None

        # Status
        self.is_open = True

        # Logger
        self.logger = logging.getLogger(__name__)

    def set_stop_loss(self, price: Optional[float] = None, percentage: Optional[float] = None) -> None:
        """
        Setzt den Stop-Loss für diese Position.

        Args:
            price: Absoluter Stop-Loss-Preis (optional)
            percentage: Stop-Loss als Prozentsatz vom Einstiegspreis (optional)
        """
        if price is not None:
            self.stop_loss = price
        elif percentage is not None:
            if self.side == "buy":
                # Long-Position: Stop-Loss unter dem Einstiegspreis
                self.stop_loss = self.entry_price * (1 - percentage)
            else:
                # Short-Position: Stop-Loss über dem Einstiegspreis
                self.stop_loss = self.entry_price * (1 + percentage)

        self.logger.info(f"Set stop loss for {self.symbol} at {self.stop_loss}")

    def set_take_profit(self, price: Optional[float] = None, percentage: Optional[float] = None) -> None:
        """
        Setzt das Take-Profit für diese Position.

        Args:
            price: Absoluter Take-Profit-Preis (optional)
            percentage: Take-Profit als Prozentsatz vom Einstiegspreis (optional)
        """
        if price is not None:
            self.take_profit = price
        elif percentage is not None:
            if self.side == "buy":
                # Long-Position: Take-Profit über dem Einstiegspreis
                self.take_profit = self.entry_price * (1 + percentage)
            else:
                # Short-Position: Take-Profit unter dem Einstiegspreis
                self.take_profit = self.entry_price * (1 - percentage)

        self.logger.info(f"Set take profit for {self.symbol} at {self.take_profit}")

    def set_trailing_stop(self, percentage: float, activation_percentage: Optional[float] = None) -> None:
        """
        Aktiviert einen Trailing-Stop für diese Position.

        Args:
            percentage: Trailing-Stop als Prozentsatz vom Höchst-/Tiefstwert
            activation_percentage: Aktivierungs-Prozentsatz (optional)
        """
        self.trailing_stop = percentage
        self.trailing_stop_activation = activation_percentage

        self.logger.info(f"Set trailing stop for {self.symbol} at {percentage}%")

    def update(self, current_price: float) -> bool:
        """
        Aktualisiert den Zustand der Position mit dem aktuellen Preis.
        Prüft, ob Stop-Loss oder Take-Profit erreicht wurden.

        Args:
            current_price: Aktueller Marktpreis

        Returns:
            True, wenn die Position geschlossen werden sollte, False sonst
        """
        if not self.is_open:
            return False

        # Höchst- und Tiefstwerte aktualisieren
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        # Trailing-Stop aktualisieren, falls aktiviert
        if self.trailing_stop is not None:
            # Prüfen, ob der Aktivierungsprozentsatz erreicht wurde
            activation_reached = True

            if self.trailing_stop_activation is not None:
                if self.side == "buy":
                    # Long-Position: Aktivierung, wenn Preis um x% gestiegen
                    activation_price = self.entry_price * (1 + self.trailing_stop_activation)
                    activation_reached = current_price >= activation_price
                else:
                    # Short-Position: Aktivierung, wenn Preis um x% gefallen
                    activation_price = self.entry_price * (1 - self.trailing_stop_activation)
                    activation_reached = current_price <= activation_price

            if activation_reached:
                if self.side == "buy":
                    # Long-Position: Trailing-Stop unter dem Höchstwert
                    trailing_stop_price = self.highest_price * (1 - self.trailing_stop)

                    # Nur aktualisieren, wenn der neue Stop höher ist
                    if self.stop_loss is None or trailing_stop_price > self.stop_loss:
                        self.stop_loss = trailing_stop_price
                else:
                    # Short-Position: Trailing-Stop über dem Tiefstwert
                    trailing_stop_price = self.lowest_price * (1 + self.trailing_stop)

                    # Nur aktualisieren, wenn der neue Stop niedriger ist
                    if self.stop_loss is None or trailing_stop_price < self.stop_loss:
                        self.stop_loss = trailing_stop_price

        # Stop-Loss prüfen
        if self.stop_loss is not None:
            if (self.side == "buy" and current_price <= self.stop_loss) or \
                    (self.side == "sell" and current_price >= self.stop_loss):
                self.close_position(current_price, "stop_loss")
                return True

        # Take-Profit prüfen
        if self.take_profit is not None:
            if (self.side == "buy" and current_price >= self.take_profit) or \
                    (self.side == "sell" and current_price <= self.take_profit):
                self.close_position(current_price, "take_profit")
                return True

        return False

    def close_position(self, exit_price: float, reason: str) -> None:
        """
        Schließt die Position zu einem bestimmten Preis.

        Args:
            exit_price: Ausstiegspreis
            reason: Grund für den Ausstieg
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.is_open = False

        # Gewinn/Verlust berechnen
        if self.side == "buy":
            self.profit_loss = (exit_price - self.entry_price) * self.amount
            self.profit_loss_percent = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.profit_loss = (self.entry_price - exit_price) * self.amount
            self.profit_loss_percent = (self.entry_price - exit_price) / self.entry_price * 100

        log_msg = f"Closed {self.symbol} position at {exit_price} ({reason}). "
        log_msg += f"P/L: {self.profit_loss:.2f} ({self.profit_loss_percent:.2f}%)"

        if self.profit_loss >= 0:
            self.logger.info(log_msg)
        else:
            self.logger.warning(log_msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Position in ein Dictionary.

        Returns:
            Position als Dictionary
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'amount': self.amount,
            'side': self.side,
            'order_id': self.order_id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'trailing_stop_activation': self.trailing_stop_activation,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'profit_loss': self.profit_loss,
            'profit_loss_percent': self.profit_loss_percent,
            'is_open': self.is_open
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Erstellt eine Position aus einem Dictionary.

        Args:
            data: Position als Dictionary

        Returns:
            Position-Objekt
        """
        position = cls(
            symbol=data['symbol'],
            entry_price=data['entry_price'],
            amount=data['amount'],
            side=data['side'],
            order_id=data['order_id']
        )

        # ID übernehmen
        position.id = data['id']

        # Zeiten konvertieren
        if data['entry_time']:
            position.entry_time = datetime.fromisoformat(data['entry_time'])

        if data['exit_time']:
            position.exit_time = datetime.fromisoformat(data['exit_time'])

        # Weitere Attribute setzen
        position.stop_loss = data['stop_loss']
        position.take_profit = data['take_profit']
        position.trailing_stop = data['trailing_stop']
        position.trailing_stop_activation = data['trailing_stop_activation']
        position.highest_price = data['highest_price']
        position.lowest_price = data['lowest_price']
        position.exit_price = data['exit_price']
        position.exit_reason = data['exit_reason']
        position.profit_loss = data['profit_loss']
        position.profit_loss_percent = data['profit_loss_percent']
        position.is_open = data['is_open']

        return position


class PositionManager:
    """Verwaltet mehrere Trading-Positionen"""

    def __init__(self):
        """Initialisiert den Position Manager"""
        self.positions: Dict[str, Position] = {}  # Aktuelle offene Positionen
        self.closed_positions: List[Position] = []  # Historie geschlossener Positionen
        self.logger = logging.getLogger(__name__)

    def add_position(self, position: Position) -> None:
        """
        Fügt eine neue Position hinzu.

        Args:
            position: Position-Objekt
        """
        self.positions[position.id] = position
        self.logger.info(
            f"Added {position.side} position for {position.symbol}: "
            f"{position.amount} @ {position.entry_price}"
        )

    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Ruft eine Position anhand ihrer ID ab.

        Args:
            position_id: ID der Position

        Returns:
            Position-Objekt oder None, wenn nicht gefunden
        """
        return self.positions.get(position_id)

    def get_position_by_symbol(self, symbol: str, side: Optional[str] = None) -> Optional[Position]:
        """
        Ruft eine Position anhand ihres Symbols ab.

        Args:
            symbol: Handelssymbol
            side: Optionale Filterung nach Positionsrichtung

        Returns:
            Position-Objekt oder None, wenn nicht gefunden
        """
        for position in self.positions.values():
            if position.symbol == symbol and (side is None or position.side == side):
                return position
        return None

    def get_all_positions(self) -> List[Position]:
        """
        Ruft alle offenen Positionen ab.

        Returns:
            Liste von Position-Objekten
        """
        return list(self.positions.values())

    def get_total_position_value(self) -> float:
        """
        Berechnet den Gesamtwert aller offenen Positionen.

        Returns:
            Gesamtwert der Positionen
        """
        total = 0.0
        for position in self.positions.values():
            total += position.entry_price * position.amount
        return total

    def update_positions(self, current_prices: Dict[str, float]) -> List[Position]:
        """
        Aktualisiert alle offenen Positionen mit aktuellen Preisen.

        Args:
            current_prices: Dictionary mit Symbolen als Schlüssel und Preisen als Werten

        Returns:
            Liste von geschlossenen Positionen in diesem Update
        """
        closed_positions = []

        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]

            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]

                # Position aktualisieren und prüfen, ob sie geschlossen werden soll
                if position.update(current_price):
                    # Position aus offenen Positionen entfernen
                    closed_position = self.positions.pop(position_id)

                    # Zu geschlossenen Positionen hinzufügen
                    self.closed_positions.append(closed_position)
                    closed_positions.append(closed_position)

        return closed_positions

    def close_position(self, position_id: str, exit_price: float, reason: str = "manual") -> Optional[Position]:
        """
        Schließt eine Position manuell.

        Args:
            position_id: ID der Position
            exit_price: Ausstiegspreis
            reason: Grund für den Ausstieg

        Returns:
            Geschlossene Position oder None, wenn nicht gefunden
        """
        if position_id in self.positions:
            position = self.positions[position_id]
            position.close_position(exit_price, reason)

            # Position aus offenen Positionen entfernen
            closed_position = self.positions.pop(position_id)

            # Zu geschlossenen Positionen hinzufügen
            self.closed_positions.append(closed_position)

            return closed_position

        return None

    def close_all_positions(self, current_prices: Dict[str, float], reason: str = "manual") -> List[Position]:
        """
        Schließt alle offenen Positionen.

        Args:
            current_prices: Dictionary mit Symbolen als Schlüssel und Preisen als Werten
            reason: Grund für den Ausstieg

        Returns:
            Liste von geschlossenen Positionen
        """
        closed_positions = []

        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]

            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position.close_position(current_price, reason)

                # Position aus offenen Positionen entfernen
                closed_position = self.positions.pop(position_id)

                # Zu geschlossenen Positionen hinzufügen
                self.closed_positions.append(closed_position)
                closed_positions.append(closed_position)

        return closed_positions

    def get_closed_positions(self) -> List[Position]:
        """
        Ruft alle geschlossenen Positionen ab.

        Returns:
            Liste von geschlossenen Position-Objekten
        """
        return self.closed_positions

    def get_position_stats(self) -> Dict[str, Any]:
        """
        Berechnet Statistiken zu allen Positionen.

        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit_pct': 0.0,
            'avg_loss_pct': 0.0,
            'max_profit_pct': 0.0,
            'max_loss_pct': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }

        # Keine geschlossenen Positionen
        if not self.closed_positions:
            return stats

        # Statistiken berechnen
        profits = []
        losses = []

        for position in self.closed_positions:
            if position.profit_loss_percent is None:
                continue

            stats['total_pnl'] += position.profit_loss or 0

            if position.profit_loss_percent >= 0:
                stats['winning_trades'] += 1
                profits.append(position.profit_loss_percent)
                stats['total_profit'] += position.profit_loss or 0

                if position.profit_loss_percent > stats['max_profit_pct']:
                    stats['max_profit_pct'] = position.profit_loss_percent
            else:
                stats['losing_trades'] += 1
                losses.append(position.profit_loss_percent)
                stats['total_loss'] += position.profit_loss or 0

                if position.profit_loss_percent < stats['max_loss_pct']:
                    stats['max_loss_pct'] = position.profit_loss_percent

        # Win-Rate berechnen
        total_closed = stats['winning_trades'] + stats['losing_trades']
        if total_closed > 0:
            stats['win_rate'] = stats['winning_trades'] / total_closed * 100

        # Durchschnittlichen Gewinn/Verlust berechnen
        if profits:
            stats['avg_profit_pct'] = sum(profits) / len(profits)
        if losses:
            stats['avg_loss_pct'] = sum(losses) / len(losses)

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Position Manager in ein Dictionary.

        Returns:
            Position Manager als Dictionary
        """
        return {
            'positions': [position.to_dict() for position in self.positions.values()],
            'closed_positions': [position.to_dict() for position in self.closed_positions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionManager':
        """
        Erstellt einen Position Manager aus einem Dictionary.

        Args:
            data: Position Manager als Dictionary

        Returns:
            Position Manager-Objekt
        """
        manager = cls()

        # Offene Positionen wiederherstellen
        for position_data in data.get('positions', []):
            position = Position.from_dict(position_data)
            manager.positions[position.id] = position

        # Geschlossene Positionen wiederherstellen
        for position_data in data.get('closed_positions', []):
            position = Position.from_dict(position_data)
            manager.closed_positions.append(position)

        return manager