#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regime-adaptive Strategie für den Altcoin Trading Bot.
Diese Strategie wechselt zwischen verschiedenen Trading-Strategien basierend auf dem erkannten Marktregime.
"""

import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

from strategies.strategy_base import Strategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from core.position import Position
from ml_components.market_regime import MarketRegimeDetector


class RegimeAdaptiveStrategy(Strategy):
    """
    Adaptive Strategie, die zwischen verschiedenen Strategien basierend auf dem Marktregime wechselt.
    """

    def __init__(self, settings):
        """
        Initialisiert die regime-adaptive Strategie.

        Args:
            settings: Einstellungen für die Strategie
        """
        super().__init__(settings)
        self.name = "regime_adaptive"
        self.description = "Adaptive Strategie basierend auf Marktregimes"
        self.logger = logging.getLogger(__name__)

        # ML-Komponenten
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = None
        self.current_regime_label = "Unbekannt"  # Immer ein Label bereithalten
        self.last_regime_update = datetime.now() - timedelta(days=1)

        # Verfügbare Strategien
        self.strategies = {
            "momentum": MomentumStrategy(settings),
            "mean_reversion": MeanReversionStrategy(settings),
            "ml": MLStrategy(settings)
        }

        # Aktuelle Strategie
        self.active_strategy = self.strategies["momentum"]  # Standard
        self.active_strategy_name = "momentum"

        # Strategie-Parameter
        self.update_interval = settings.get("adaptive.update_interval", 24)  # Stunden zwischen Regime-Updates
        self.regime_to_strategy_map = {
            # Standard-Mapping, wird basierend auf Regimeeigenschaften aktualisiert
        }

        # Modell laden, falls konfiguriert
        model_path = settings.get("ml.regime_model_path")
        if model_path:
            try:
                self.regime_detector.load_model(model_path)
                self.logger.info(f"Regime-Modell aus {model_path} geladen")
            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Regime-Modells: {e}")

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Aktualisiert die Strategie-Parameter.

        Args:
            params: Dictionary mit Parameternamen und -werten
        """
        for key, value in params.items():
            # Adaptive Parameter
            if key.startswith("adaptive."):
                param_name = key.split(".")[1]
                if hasattr(self, param_name):
                    setattr(self, param_name, value)
            # Parameter an aktuelle Strategie weiterleiten
            else:
                if hasattr(self.active_strategy, "update_parameters"):
                    self.active_strategy.update_parameters({key: value})

        self.logger.info(f"Adaptive Strategie-Parameter aktualisiert: {params}")

    def get_parameters(self) -> Dict[str, Any]:
        """
        Gibt die aktuellen Strategie-Parameter zurück.

        Returns:
            Dictionary mit Parameter-Namen und -Werten
        """
        params = {
            "adaptive.update_interval": self.update_interval,
            "adaptive.active_strategy": self.active_strategy_name,
            "adaptive.current_regime": self.current_regime,
            "adaptive.current_regime_label": self.current_regime_label
        }

        # Parameter der aktiven Strategie hinzufügen
        if hasattr(self.active_strategy, "get_parameters"):
            active_params = self.active_strategy.get_parameters()
            params.update(active_params)

        return params

    def _update_market_regime(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Aktualisiert das erkannte Marktregime und wechselt die Strategie bei Bedarf.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Symbol des Assets

        Returns:
            True, wenn das Regime aktualisiert wurde, sonst False
        """
        # Prüfen, ob ein Update nötig ist
        now = datetime.now()
        hours_since_update = (now - self.last_regime_update).total_seconds() / 3600

        if hours_since_update < self.update_interval:
            return False

        try:
            # Modell trainieren, falls noch nicht geschehen
            if not self.regime_detector.model_trained:
                # Marktdaten laden
                self.regime_detector.market_data = {f"{symbol.split('/')[0]}/{symbol.split('/')[1]}": df}

                # Features extrahieren
                features_df = self.regime_detector.extract_market_features()

                if not features_df.empty:
                    # Modell trainieren
                    success = self.regime_detector.train_regime_model(features_df)

                    if not success:
                        self.logger.error("Fehler beim Training des Regime-Modells")
                        return False

                    self.logger.info("Marktregime-Modell erfolgreich trainiert")

            # Features für die aktuelle Marktlage extrahieren
            formatted_symbol = f"{symbol.split('/')[0]}/{symbol.split('/')[1]}"
            self.regime_detector.market_data = {formatted_symbol: df}
            features_df = self.regime_detector.extract_market_features()

            if features_df.empty:
                self.logger.error("Keine Features für Regime-Erkennung extrahiert")
                return False

            # Aktuelles Regime vorhersagen
            latest_features = features_df.iloc[-1:].copy()

            # Hier wird nun ein Tupel zurückgegeben (regime_id, regime_label)
            regime_result = self.regime_detector.predict_regime(latest_features)

            # Sicherer Zugriff auf das Ergebnis
            if isinstance(regime_result, tuple) and len(regime_result) == 2:
                regime, regime_label = regime_result
            else:
                # Fallback, falls kein Tupel zurückgegeben wird
                regime = regime_result if isinstance(regime_result, int) else -1
                regime_label = self.regime_detector.regime_labels.get(regime, f"Regime {regime}")

            if regime >= 0:
                old_regime = self.current_regime
                self.current_regime = regime
                self.current_regime_label = regime_label  # Label speichern

                # Wenn sich das Regime geändert hat, Strategie anpassen
                if old_regime != regime:
                    self.logger.info(f"Marktregime hat sich geändert: {old_regime} -> {regime} ({regime_label})")

                    # Bestimmen der besten Strategie für dieses Regime
                    new_strategy = self._select_strategy_for_regime(regime)

                    if new_strategy != self.active_strategy_name:
                        self._switch_strategy(new_strategy)

                # Aktualisierungszeit merken
                self.last_regime_update = now
                return True

            return False

        except Exception as e:
            self.logger.error(f"Fehler bei der Marktregime-Erkennung: {e}")
            return False

    def _select_strategy_for_regime(self, regime: int) -> str:
        """
        Wählt die beste Strategie für das gegebene Regime aus.

        Args:
            regime: ID des Marktregimes

        Returns:
            Name der besten Strategie
        """
        # Zuerst prüfen, ob ein Mapping bereits definiert ist
        if regime in self.regime_to_strategy_map:
            return self.regime_to_strategy_map[regime]

        # Trading-Regeln für dieses Regime abrufen
        try:
            trading_rules = self.regime_detector.extract_trading_rules()

            if regime not in trading_rules:
                # Standard-Strategie, wenn keine Regeln verfügbar
                return "momentum"

            rule = trading_rules[regime]
            regime_label = rule.get('label', "").lower()

            # Strategie basierend auf Regime-Eigenschaften auswählen
            if "bullish" in regime_label and "hohe-volatilität" not in regime_label:
                # Bullischer Markt mit niedriger Volatilität -> Momentum
                strategy = "momentum"
            elif "bearish" in regime_label:
                # Bärischer Markt -> ML (konservativer)
                strategy = "ml"
            elif "hohe-volatilität" in regime_label:
                # Hohe Volatilität -> Mean Reversion
                strategy = "mean_reversion"
            elif "altcoin-stärke" in regime_label:
                # Altcoin-Boom -> Momentum
                strategy = "momentum"
            elif "niedrige-volatilität" in regime_label:
                # Geringe Volatilität -> ML (kann weiterhin gute Setups finden)
                strategy = "ml"
            else:
                # Standard für unklare Situationen -> ML
                strategy = "ml"

            # Mapping speichern für zukünftige Nutzung
            self.regime_to_strategy_map[regime] = strategy

            return strategy
        except Exception as e:
            self.logger.error(f"Fehler bei der Strategieauswahl: {e}")
            return "momentum"  # Standardstrategie im Fehlerfall

    def _switch_strategy(self, strategy_name: str) -> None:
        """
        Wechselt zur angegebenen Strategie.

        Args:
            strategy_name: Name der zu aktivierenden Strategie
        """
        if strategy_name not in self.strategies:
            self.logger.error(f"Strategie {strategy_name} nicht verfügbar")
            return

        old_strategy = self.active_strategy_name
        self.active_strategy = self.strategies[strategy_name]
        self.active_strategy_name = strategy_name

        self.logger.info(f"Strategie gewechselt: {old_strategy} -> {strategy_name}")

        # Bei ML-Strategie Regime-Informationen teilen, falls verfügbar
        if strategy_name == "ml" and self.current_regime is not None:
            ml_strategy = self.strategies["ml"]
            if hasattr(ml_strategy, "market_regime_detector") and ml_strategy.market_regime_detector:
                ml_strategy.current_regime = self.current_regime
                self.logger.info(f"Regime-Information an ML-Strategie übergeben: {self.current_regime}")

    def generate_signal(self, df: pd.DataFrame, symbol: str, current_position: Optional[Position] = None) -> Tuple[
        str, Dict[str, Any]]:
        """
        Generiert ein Trading-Signal basierend auf dem aktuellen Marktregime und der aktiven Strategie.

        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Symbol des Assets
            current_position: Aktuelle Position (oder None)

        Returns:
            Tuple aus Signal (BUY, SELL, HOLD) und Signal-Daten
        """
        try:
            # Marktregime aktualisieren
            self._update_market_regime(df, symbol)

            # Signal von aktiver Strategie generieren
            signal, signal_data = self.active_strategy.generate_signal(df, symbol, current_position)

            # Regime-Informationen hinzufügen (mit sicheren Fallback-Werten)
            signal_data["current_regime"] = self.current_regime if self.current_regime is not None else -1
            signal_data["active_strategy"] = self.active_strategy_name
            signal_data["regime_label"] = self.current_regime_label  # Verwenden des gespeicherten Labels

            # Log für besseres Debugging
            self.logger.info(
                f"Regime-adaptives Signal für {symbol}: {signal} "
                f"(Strategie: {self.active_strategy_name}, "
                f"Regime: {self.current_regime}, "
                f"Label: {self.current_regime_label})"
            )

            return signal, signal_data

        except Exception as e:
            self.logger.error(f"Fehler bei der Signal-Generierung: {e}")
            # Fallback-Signal im Fehlerfall
            return "HOLD", {
                "signal": "HOLD",
                "reason": f"Error: {str(e)}",
                "confidence": 0.0,
                "current_regime": self.current_regime if self.current_regime is not None else -1,
                "active_strategy": self.active_strategy_name,
                "regime_label": self.current_regime_label
            }