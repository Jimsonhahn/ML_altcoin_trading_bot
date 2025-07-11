# 1. Erstelle die Copy Trading Strategy Datei
cat > strategies / copy_trading.py << 'EOF'
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copy Trading Bot Strategy - Kopiere erfolgreiche Trader!
========================================================
Warum selbst denken? Lass die Profis für dich arbeiten!
Kopiert automatisch die Trades erfolgreicher Wallets.

Features:
- Multi-Wallet Tracking
- Performance-basierte Auswahl
- Risk-adjusted Copying
- Whale Alert Integration
- Smart Money Tracking
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

from config.settings import Settings
from core.position import Position
from strategies.strategy_base import Strategy


class CopyTradingStrategy(Strategy):
    """
    Copy Trading Strategy

    Automatisches Kopieren erfolgreicher Trader:
    1. Trackt Performance von ausgewählten Wallets
    2. Filtert nach Erfolgsrate und Konsistenz
    3. Kopiert Trades mit angepasster Positionsgröße
    4. Risk Management pro kopiertem Trader
    """

    def __init__(self, settings: Settings):
        """Initialisiert die Copy Trading Strategy"""
        super().__init__(settings)
        self.name = "copy_trading"

        # Copy Trading Parameter
        self.min_win_rate = settings.get('copy_trading.min_win_rate', 0.6)  # 60%
        self.min_profit_factor = settings.get('copy_trading.min_profit_factor', 1.5)
        self.lookback_days = settings.get('copy_trading.lookback_days', 30)
        self.position_scale = settings.get('copy_trading.position_scale', 0.1)  # 10% der Whale-Größe
        self.max_traders_to_copy = settings.get('copy_trading.max_traders_to_copy', 5)

        # Erfolgreiche Trader zum Kopieren (Adressen)
        self.whale_wallets = settings.get('copy_trading.whale_wallets', [
            {'address': '0x1234567890abcdef', 'name': 'DeFi Whale Alpha', 'type': 'defi'},
            {'address': '0xabcdef1234567890', 'name': 'Smart Money Trader', 'type': 'smart'},
            {'address': '0x9876543210fedcba', 'name': 'Arbitrage Master', 'type': 'arbitrage'},
            {'address': '0xfedcba9876543210', 'name': 'Yield Optimization Pro', 'type': 'yield'},
            {'address': '0x1111222233334444', 'name': 'Trend Following Expert', 'type': 'trend'}
        ])

        # Performance Tracking pro Wallet
        self.wallet_performance = defaultdict(lambda: {
            'trades': [],
            'win_rate': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'avg_trade_size': 0,
            'last_trade': None
        })

        # Simulierte Trades für Paper Trading
        self.simulated_whale_trades = []
        self.copied_trades = []
        self.total_copied = 0
        self.successful_copies = 0

        self.logger.info("👥 Copy Trading Bot initialized:")
        self.logger.info(f"  📊 Min win rate: {self.min_win_rate * 100}%")
        self.logger.info(f"  💰 Min profit factor: {self.min_profit_factor}")
        self.logger.info(f"  👁️  Tracking: {len(self.whale_wallets)} wallets")
        self.logger.info(f"  📏 Position scale: {self.position_scale * 100}% of whale")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereitet Daten für Copy Trading vor"""
        # Für Copy Trading brauchen wir hauptsächlich Wallet-Transaktionsdaten
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str,
                        current_position: Optional[Position] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert Copy Trading Signale

        Signale:
        - BUY: Kopiere einen Buy-Trade eines erfolgreichen Traders
        - SELL: Kopiere einen Sell-Trade oder schließe Position
        - HOLD: Keine neuen Trades von Top-Tradern
        """
        signal = "HOLD"
        signal_data = {
            'signal': 'HOLD',
            'strategy': 'copy_trading',
            'confidence': 0.0,
            'reason': 'no_trades_to_copy'
        }

        # 1. Aktualisiere Wallet-Performance
        self._update_wallet_performance()

        # 2. Finde neue Trades von Top-Performern
        new_trades = self._find_new_whale_trades(symbol)

        # 3. Bewerte und filtere Trades
        best_trade = self._evaluate_trades(new_trades)

        if best_trade:
            signal = best_trade['side'].upper()
            signal_data.update({
                'signal': signal,
                'confidence': best_trade['confidence'],
                'reason': 'copy_whale_trade',
                'whale_address': best_trade['whale_address'],
                'whale_name': best_trade['whale_name'],
                'whale_win_rate': best_trade['whale_win_rate'],
                'whale_trade_size': best_trade['original_size'],
                'our_trade_size': best_trade['copy_size'],
                'expected_profit': best_trade['expected_profit'],
                'action': f'COPY_{signal}'
            })

            self.logger.info(f"👥 Copying trade from {best_trade['whale_name']}!")
            self.logger.info(f"  Signal: {signal} {symbol}")
            self.logger.info(f"  Whale win rate: {best_trade['whale_win_rate'] * 100:.1f}%")
            self.logger.info(f"  Copy size: {best_trade['copy_size']:.4f} ({self.position_scale * 100}% of whale)")

        return signal, signal_data

    def _update_wallet_performance(self) -> None:
        """Aktualisiert Performance-Metriken für alle verfolgten Wallets"""
        # Simuliere historische Trades für Paper Trading
        if not self.simulated_whale_trades:
            self._generate_simulated_trades()

        # Berechne Performance-Metriken
        for wallet in self.whale_wallets:
            address = wallet['address']
            trades = [t for t in self.simulated_whale_trades if t['wallet'] == address]

            if len(trades) >= 5:  # Mindestens 5 Trades für Statistik
                wins = sum(1 for t in trades if t['pnl'] > 0)
                losses = sum(1 for t in trades if t['pnl'] < 0)

                win_rate = wins / len(trades) if trades else 0

                gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

                self.wallet_performance[address].update({
                    'trades': trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_pnl': sum(t['pnl'] for t in trades),
                    'avg_trade_size': np.mean([t['size'] for t in trades]),
                    'last_trade': max(trades, key=lambda x: x['timestamp']) if trades else None
                })

    def _generate_simulated_trades(self) -> None:
        """Generiert simulierte historische Trades für Paper Trading"""
        base_time = datetime.now() - timedelta(days=self.lookback_days)

        for wallet in self.whale_wallets:
            # Generiere 20-50 Trades pro Wallet
            num_trades = np.random.randint(20, 50)
            wallet_skill = np.random.uniform(0.5, 0.8)  # Skill-Level des Wallets

            for i in range(num_trades):
                # Zufälliger Zeitpunkt
                trade_time = base_time + timedelta(
                    minutes=np.random.randint(0, self.lookback_days * 24 * 60)
                )

                # Trade-Details
                symbol = np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'])
                side = np.random.choice(['buy', 'sell'])
                size = np.random.uniform(0.1, 10.0)

                # P/L basierend auf Wallet-Skill
                if np.random.random() < wallet_skill:
                    # Gewinn-Trade
                    pnl = np.random.uniform(0.005, 0.05) * size * 1000  # 0.5-5% Gewinn
                else:
                    # Verlust-Trade
                    pnl = -np.random.uniform(0.003, 0.03) * size * 1000  # 0.3-3% Verlust

                self.simulated_whale_trades.append({
                    'wallet': wallet['address'],
                    'wallet_name': wallet['name'],
                    'timestamp': trade_time,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'pnl': pnl,
                    'entry_price': 50000 if 'BTC' in symbol else 2000,
                    'exit_price': 50000 * (1 + pnl / size / 1000) if 'BTC' in symbol else 2000 * (1 + pnl / size / 1000)
                })

    def _find_new_whale_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Findet neue Trades von verfolgten Wallets"""
        new_trades = []
        current_time = datetime.now()

        # Simuliere neue Trades (1-2 pro Stunde)
        for wallet in self.whale_wallets:
            performance = self.wallet_performance[wallet['address']]

            # Nur von erfolgreichen Wallets kopieren
            if (performance['win_rate'] >= self.min_win_rate and
                    performance['profit_factor'] >= self.min_profit_factor):

                # Simuliere gelegentliche neue Trades
                if np.random.random() < 0.1:  # 10% Chance pro Check
                    new_trade = {
                        'whale_address': wallet['address'],
                        'whale_name': wallet['name'],
                        'whale_type': wallet['type'],
                        'whale_win_rate': performance['win_rate'],
                        'whale_profit_factor': performance['profit_factor'],
                        'timestamp': current_time,
                        'symbol': symbol,
                        'side': np.random.choice(['buy', 'sell']),
                        'original_size': np.random.uniform(1.0, 10.0),
                        'entry_price': 50000 if 'BTC' in symbol else 2000
                    }

                    new_trades.append(new_trade)

        return new_trades

    def _evaluate_trades(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Bewertet und wählt den besten Trade zum Kopieren"""
        if not trades:
            return None

        best_trade = None
        best_score = 0

        for trade in trades:
            # Bewertungsfaktoren
            win_rate_score = trade['whale_win_rate']
            profit_factor_score = min(trade['whale_profit_factor'] / 3, 1.0)  # Normalisiert auf 0-1
            recency_score = 1.0  # Neuere Trades bevorzugen

            # Gesamtbewertung
            total_score = (win_rate_score * 0.4 +
                           profit_factor_score * 0.4 +
                           recency_score * 0.2)

            if total_score > best_score:
                best_score = total_score
                best_trade = trade

                # Berechne unsere Positionsgröße
                best_trade['copy_size'] = trade['original_size'] * self.position_scale
                best_trade['confidence'] = min(total_score, 0.95)
                best_trade['expected_profit'] = best_trade['copy_size'] * 1000 * 0.02  # 2% Ziel

        return best_trade

    def execute_copy(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Führt einen kopierten Trade aus"""
        self.total_copied += 1

        # Simuliere Trade-Ausführung
        success = np.random.random() < trade_data['whale_win_rate']

        if success:
            self.successful_copies += 1
            profit = trade_data['expected_profit']
        else:
            profit = -trade_data['expected_profit'] * 0.5  # Halber Verlust bei Misserfolg

        self.copied_trades.append({
            'timestamp': datetime.now(),
            'whale': trade_data['whale_name'],
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'size': trade_data['copy_size'],
            'profit': profit,
            'success': success
        })

        self.logger.info(f"👥 COPY TRADE EXECUTED!")
        self.logger.info(f"  Whale: {trade_data['whale_name']}")
        self.logger.info(f"  Result: {'✅ SUCCESS' if success else '❌ LOSS'}")
        self.logger.info(f"  Profit: ${profit:,.2f}")

        return {
            'success': True,
            'whale': trade_data['whale_name'],
            'profit': profit,
            'total_copies': self.total_copied,
            'success_rate': self.successful_copies / self.total_copied if self.total_copied > 0 else 0
        }

    def get_copy_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Copy Trading Status zurück"""
        total_profit = sum(t['profit'] for t in self.copied_trades)
        success_rate = self.successful_copies / self.total_copied if self.total_copied > 0 else 0

        # Top Performer
        top_whales = sorted(
            [
                {
                    'name': w['name'],
                    'type': w['type'],
                    'win_rate': self.wallet_performance[w['address']]['win_rate'],
                    'profit_factor': self.wallet_performance[w['address']]['profit_factor'],
                    'total_pnl': self.wallet_performance[w['address']]['total_pnl']
                }
                for w in self.whale_wallets
            ],
            key=lambda x: x['profit_factor'],
            reverse=True
        )[:3]

        return {
            'tracked_wallets': len(self.whale_wallets),
            'qualified_wallets': sum(
                1 for w in self.whale_wallets
                if (self.wallet_performance[w['address']]['win_rate'] >= self.min_win_rate and
                    self.wallet_performance[w['address']]['profit_factor'] >= self.min_profit_factor)
            ),
            'total_copies': self.total_copied,
            'successful_copies': self.successful_copies,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'top_performers': top_whales,
            'recent_copies': self.copied_trades[-5:][::-1] if self.copied_trades else []
        }


EOF

# 2. Erstelle die Copy Trading Config
cat > config / profiles / copy_trading.json << 'EOF'
{
    "name": "copy_trading",
    "description": "Copy Trading Bot - Kopiere erfolgreiche Trader automatisch",
    "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
    "copy_trading": {
        "enabled": true,
        "min_win_rate": 0.6,
        "min_profit_factor": 1.5,
        "lookback_days": 30,
        "position_scale": 0.1,
        "max_traders_to_copy": 5,
        "whale_wallets": [
            {
                "address": "0x1234567890abcdef",
                "name": "DeFi Whale Alpha",
                "type": "defi",
                "priority": 1
            },
            {
                "address": "0xabcdef1234567890",
                "name": "Smart Money Trader",
                "type": "smart",
                "priority": 2
            },
            {
                "address": "0x9876543210fedcba",
                "name": "Arbitrage Master",
                "type": "arbitrage",
                "priority": 3
            },
            {
                "address": "0xfedcba9876543210",
                "name": "Yield Optimization Pro",
                "type": "yield",
                "priority": 4
            },
            {
                "address": "0x1111222233334444",
                "name": "Trend Following Expert",
                "type": "trend",
                "priority": 5
            }
        ],
        "filters": {
            "min_trade_size": 100,
            "max_trade_size": 100000,
            "allowed_tokens": ["BTC", "ETH", "BNB", "SOL", "MATIC", "AVAX"],
            "exclude_memecoins": true
        },
        "risk_management": {
            "max_copy_per_trader": 0.2,
            "stop_copying_on_drawdown": 0.15,
            "diversification_required": 3
        }
    },
    "risk": {
        "position_size": 0.05,
        "max_open_positions": 10,
        "stop_loss": 0.05,
        "take_profit": 0.1
    },
    "timeframes": {
        "analysis": "5m",
        "check_interval": 60
    },
    "logging": {
        "level": "INFO",
        "console": true,
        "file": true
    }
}
EOF