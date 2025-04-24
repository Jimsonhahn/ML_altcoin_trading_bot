#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML-Komponenten für den Trading Bot und Backtest-Module.
Dieses Paket enthält gemeinsame ML-Komponenten, die von beiden Modulen genutzt werden.
"""

import logging
import os
from typing import Dict, Any, Optional, List

# Importiere die ML-Komponenten
from ml_components.market_regime import MarketRegimeDetector
from ml_components.asset_clusters import AssetClusterAnalyzer
from ml_components.coin_monitor import NewCoinMonitor
from ml_components.model_monitor import ModelPerformanceMonitor

# Logging einrichten
logger = logging.getLogger(__name__)


class MLComponents:
    """
    Zentrale Klasse zur Verwaltung aller ML-Komponenten im Trading-System.
    """

    def __init__(self, settings=None, data_dir: str = "data/market_data",
                 models_dir: str = "data/ml_models", output_dir: str = "data/ml_analysis"):
        """
        Initialisiert alle ML-Komponenten.

        Args:
            settings: Einstellungen (optional)
            data_dir: Verzeichnis mit Marktdaten
            models_dir: Verzeichnis für ML-Modelle
            output_dir: Verzeichnis für Analyseergebnisse
        """
        self.settings = settings
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir

        # Verzeichnisse sicherstellen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # ML-Komponenten initialisieren
        self.market_regime_detector = MarketRegimeDetector(data_dir=data_dir)
        self.asset_cluster_analyzer = AssetClusterAnalyzer(data_dir=data_dir)
        self.coin_monitor = NewCoinMonitor(data_dir=data_dir, output_dir=os.path.join(output_dir, "new_coins"))

        # Model-Monitor initialisieren
        self.model_monitor = ModelPerformanceMonitor(output_dir=os.path.join(output_dir, "model_monitor"))

        # Status
        self.initialized = False
        self.current_regime = None
        self.current_clusters = None

        # Modelle laden, falls vorhanden
        self.load_models()

        logger.info("ML-Komponenten initialisiert")

    def load_models(self) -> bool:
        """
        Lädt gespeicherte ML-Modelle, falls vorhanden.

        Returns:
            True, wenn mindestens ein Modell geladen wurde, sonst False
        """
        loaded = False

        # 1. MarketRegimeDetector-Modell laden
        regime_model_path = os.path.join(self.models_dir, "regime_model.pkl")
        if os.path.exists(regime_model_path):
            if self.market_regime_detector.load_model(regime_model_path):
                logger.info(f"Regime-Modell aus {regime_model_path} geladen")
                loaded = True

        self.initialized = loaded
        return loaded

    def save_models(self) -> bool:
        """
        Speichert die ML-Modelle.

        Returns:
            True, wenn alle Modelle gespeichert wurden, sonst False
        """
        saved = False

        # 1. MarketRegimeDetector-Modell speichern
        if self.market_regime_detector.model_trained:
            regime_model_path = os.path.join(self.models_dir, "regime_model.pkl")
            if self.market_regime_detector.save_model(regime_model_path):
                logger.info(f"Regime-Modell unter {regime_model_path} gespeichert")
                saved = True

        return saved

    def get_current_regime_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum aktuellen Marktregime zurück.

        Returns:
            Dictionary mit Regime-Informationen
        """
        if not self.market_regime_detector.model_trained:
            return {"status": "not_available", "message": "Kein Regime-Modell trainiert"}

        if self.current_regime is None:
            return {"status": "unknown", "message": "Aktuelles Regime nicht bestimmt"}

        return self.market_regime_detector.get_current_regime_info()

    def get_portfolio_recommendation(self, n_assets: int = 10) -> Dict[str, Any]:
        """
        Gibt eine Portfolio-Empfehlung basierend auf Clustering und aktuellem Regime zurück.

        Args:
            n_assets: Anzahl der zu empfehlenden Assets

        Returns:
            Dictionary mit Portfolio-Empfehlung
        """
        if not self.asset_cluster_analyzer.clusters is not None:
            return {"status": "not_available", "message": "Keine Cluster-Informationen verfügbar"}

        # Empfehlung mit Regime-Anpassung, falls verfügbar
        if self.market_regime_detector.model_trained and self.current_regime is not None:
            return self.asset_cluster_analyzer.recommend_portfolio(
                n_assets=n_assets,
                regime_id=self.current_regime,
                regime_detector=self.market_regime_detector
            )
        else:
            # Ohne Regime-Anpassung
            return self.asset_cluster_analyzer.recommend_portfolio(n_assets=n_assets)

    def get_interesting_new_coins(self) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste interessanter neuer Coins zurück.

        Returns:
            Liste mit interessanten neuen Coins
        """
        return self.coin_monitor.get_interesting_coins()

    def record_regime_prediction(self, predicted_regime: int, actual_regime: int = None) -> None:
        """
        Zeichnet eine Regime-Vorhersage und den tatsächlichen Wert auf.

        Args:
            predicted_regime: Vorhergesagtes Regime
            actual_regime: Tatsächliches Regime (kann später aktualisiert werden)
        """
        if self.market_regime_detector and self.market_regime_detector.model_trained:
            self.model_monitor.record_prediction(
                model_id="market_regime",
                model_type="regime",
                prediction=predicted_regime,
                actual=actual_regime
            )

    def update_actual_regime(self, actual_regime: int) -> None:
        """
        Aktualisiert das tatsächliche Regime für die letzte Vorhersage.

        Args:
            actual_regime: Tatsächliches Regime
        """
        # Vereinfachung: Aktualisiert nur die letzte Vorhersage
        # In der Praxis würden Sie nach Timestamp, ID, etc. suchen
        model_key = "regime_market_regime"

        if model_key in self.model_monitor.performance_data:
            predictions = self.model_monitor.performance_data[model_key]["predictions"]
            if predictions:
                predictions[-1]["actual"] = actual_regime
                self.model_monitor._save_performance_data()

    def update_all_components(self, data_manager=None, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Aktualisiert alle ML-Komponenten.

        Args:
            data_manager: Optionaler DataManager für Datenzugriff
            symbols: Liste der zu verwendenden Symbole (optional)

        Returns:
            Dictionary mit Aktualisierungsstatus
        """
        results = {
            "regime_updated": False,
            "clusters_updated": False,
            "new_coins_checked": False,
            "new_coins_analyzed": []
        }

        try:
            # Standardsymbole, falls keine angegeben
            if not symbols:
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

            # 1. Marktregime aktualisieren
            if self.market_regime_detector.load_market_data(symbols=symbols, data_manager=data_manager):
                features_df = self.market_regime_detector.extract_market_features()

                if not features_df.empty:
                    # Modell trainieren, falls noch nicht geschehen
                    if not self.market_regime_detector.model_trained:
                        self.market_regime_detector.train_regime_model(features_df)
                        results["regime_model_trained"] = True
                        self.save_models()

                    # Aktuelles Regime vorhersagen
                    latest_features = features_df.iloc[-1:].copy()
                    previous_regime = self.current_regime
                    self.current_regime = self.market_regime_detector.predict_regime(latest_features)

                    # Vorhersage aufzeichnen
                    self.record_regime_prediction(predicted_regime=self.current_regime)

                    results["regime_updated"] = True
                    results["current_regime"] = self.current_regime

                    if self.current_regime is not None:
                        regime_label = self.market_regime_detector.regime_labels.get(
                            self.current_regime, f"Regime {self.current_regime}"
                        )
                        results["regime_label"] = regime_label

                        # Wenn Regime sich geändert hat, Event loggen
                        if previous_regime is not None and previous_regime != self.current_regime:
                            logger.info(
                                f"Marktregime hat sich geändert: {previous_regime} -> {self.current_regime} ({regime_label})")

            # 2. Asset-Cluster aktualisieren
            if self.asset_cluster_analyzer.load_market_data(symbols=symbols, data_manager=data_manager):
                # Korrelationsmatrix berechnen
                self.asset_cluster_analyzer.calculate_correlation_matrix()

                # Features extrahieren
                self.asset_cluster_analyzer.extract_asset_features()

                # Clustering durchführen
                clusters = self.asset_cluster_analyzer.run_clustering()

                if not clusters.empty:
                    self.current_clusters = clusters
                    results["clusters_updated"] = True
                    results["cluster_count"] = len(clusters['cluster'].unique())

            # 3. Nach neuen Coins suchen
            if self.settings and self.settings.get('ml.monitor_new_coins', True):
                new_coins = self.coin_monitor.check_for_new_coins()
                results["new_coins_checked"] = True
                results["new_coins_found"] = new_coins

                # Neue Coins analysieren
                coins_to_analyze = self.coin_monitor.get_coins_for_analysis()

                for coin in coins_to_analyze:
                    # Daten laden
                    df = self.coin_monitor.get_coin_data(coin)

                    if not df.empty:
                        # Coin analysieren
                        analysis_result = self.asset_cluster_analyzer.analyze_new_coin(coin, df)

                        if analysis_result and analysis_result.get('status') == 'analyzed':
                            # Status aktualisieren
                            self.coin_monitor.update_coin_status(coin, analysis_result)
                            results["new_coins_analyzed"].append(coin)

            return results

        except Exception as e:
            logger.error(f"Fehler bei der Aktualisierung der ML-Komponenten: {e}")
            results["error"] = str(e)
            return results


# Globale Instanz für einfachen Zugriff
ml_components = None


def initialize_ml(settings=None, data_dir: str = "data/market_data",
                  models_dir: str = "data/ml_models", output_dir: str = "data/ml_analysis") -> MLComponents:
    """
    Initialisiert die ML-Komponenten und gibt eine Instanz zurück.

    Args:
        settings: Einstellungen (optional)
        data_dir: Verzeichnis mit Marktdaten
        models_dir: Verzeichnis für ML-Modelle
        output_dir: Verzeichnis für Analyseergebnisse

    Returns:
        MLComponents-Instanz
    """
    global ml_components
    ml_components = MLComponents(settings, data_dir, models_dir, output_dir)
    return ml_components


def get_ml_components() -> Optional[MLComponents]:
    """
    Gibt die aktuellen ML-Komponenten zurück.

    Returns:
        MLComponents-Instanz oder None, falls nicht initialisiert
    """
    global ml_components
    return ml_components