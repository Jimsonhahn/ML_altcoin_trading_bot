#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul für das Monitoring von ML-Modellen.
Überwacht die Performance der ML-Modelle und erstellt Berichte.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt


class ModelPerformanceMonitor:
    """
    Überwacht die Performance von ML-Modellen im Laufe der Zeit.
    """

    def __init__(self, output_dir: str = "data/ml_monitor"):
        """
        Initialisiert den Model-Monitor.

        Args:
            output_dir: Verzeichnis für Monitoring-Daten
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Monitoring-Daten
        self.performance_data = {}

        # Verzeichnis sicherstellen
        os.makedirs(output_dir, exist_ok=True)

        # Vorhandene Daten laden
        self._load_performance_data()

    def _load_performance_data(self) -> None:
        """
        Lädt bereits vorhandene Performance-Daten.
        """
        data_file = os.path.join(self.output_dir, "model_performance.json")

        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    self.performance_data = json.load(f)
                self.logger.info(f"Performance-Daten geladen: {len(self.performance_data)} Modelle")
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der Performance-Daten: {e}")

    def _save_performance_data(self) -> None:
        """
        Speichert aktuelle Performance-Daten.
        """
        data_file = os.path.join(self.output_dir, "model_performance.json")

        try:
            with open(data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
            self.logger.info("Performance-Daten gespeichert")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Performance-Daten: {e}")

    def record_prediction(self, model_id: str, model_type: str,
                          prediction: Any, actual: Any,
                          timestamp: datetime = None) -> None:
        """
        Zeichnet eine Modellvorhersage und den tatsächlichen Wert auf.

        Args:
            model_id: ID oder Name des Modells
            model_type: Typ des Modells ('regime', 'cluster', etc.)
            prediction: Vorhergesagter Wert
            actual: Tatsächlicher Wert
            timestamp: Zeitstempel (optional, Standard: aktuelle Zeit)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Modell-ID erstellen (Typ_Name)
        full_model_id = f"{model_type}_{model_id}"

        # Performance-Eintrag initialisieren, falls noch nicht vorhanden
        if full_model_id not in self.performance_data:
            self.performance_data[full_model_id] = {
                "model_type": model_type,
                "model_id": model_id,
                "predictions": []
            }

        # Vorhersage aufzeichnen
        prediction_entry = {
            "timestamp": timestamp.isoformat(),
            "prediction": prediction,
            "actual": actual
        }

        self.performance_data[full_model_id]["predictions"].append(prediction_entry)

        # Daten regelmäßig speichern (z.B. nach jeder 10. Aufzeichnung)
        if len(self.performance_data[full_model_id]["predictions"]) % 10 == 0:
            self._save_performance_data()

    def calculate_model_metrics(self, model_id: str = None, model_type: str = None,
                                days: int = 30) -> Dict[str, Any]:
        """
        Berechnet Performance-Metriken für ein Modell.

        Args:
            model_id: ID oder Name des Modells (optional)
            model_type: Typ des Modells (optional)
            days: Anzahl der Tage für die Analyse

        Returns:
            Dictionary mit Performance-Metriken
        """
        results = {}

        # Filter für Modell-ID und Typ
        models_to_analyze = self.performance_data.keys()

        if model_id and model_type:
            full_model_id = f"{model_type}_{model_id}"
            if full_model_id in self.performance_data:
                models_to_analyze = [full_model_id]
            else:
                return {"error": f"Modell {full_model_id} nicht gefunden"}
        elif model_type:
            models_to_analyze = [m for m in models_to_analyze if m.startswith(f"{model_type}_")]

        # Zeitrahmen festlegen
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Metriken für jedes Modell berechnen
        for model in models_to_analyze:
            model_data = self.performance_data[model]
            predictions = model_data["predictions"]

            # Nach Datum filtern
            recent_preds = [p for p in predictions if p["timestamp"] >= cutoff_date]

            if not recent_preds:
                results[model] = {"status": "no_recent_data"}
                continue

            # Metriken berechnen (je nach Modelltyp unterschiedlich)
            if model_data["model_type"] == "regime":
                metrics = self._calculate_regime_metrics(recent_preds)
            elif model_data["model_type"] == "cluster":
                metrics = self._calculate_cluster_metrics(recent_preds)
            else:
                metrics = self._calculate_generic_metrics(recent_preds)

            results[model] = metrics

        return results

    def _calculate_regime_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Berechnet spezifische Metriken für Regime-Modelle.
        """
        metrics = {}

        # Korrekte Regime-Klassifikationen zählen
        correct = sum(1 for p in predictions if p["prediction"] == p["actual"])
        total = len(predictions)

        metrics["accuracy"] = correct / total if total > 0 else 0

        # Verteilung der vorhergesagten Regimes
        regime_distribution = {}
        for p in predictions:
            regime = p["prediction"]
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1

        metrics["regime_distribution"] = {
            k: v / total for k, v in regime_distribution.items()
        }

        # Regime-Wechsel analysieren
        regime_changes = 0
        for i in range(1, len(predictions)):
            if predictions[i]["prediction"] != predictions[i - 1]["prediction"]:
                regime_changes += 1

        metrics["regime_changes"] = regime_changes
        metrics["avg_regime_duration"] = total / (regime_changes + 1)

        return metrics

    def _calculate_cluster_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Berechnet spezifische Metriken für Cluster-Modelle.
        """
        metrics = {}

        # Ähnlich wie Regime-Metriken
        correct = sum(1 for p in predictions if p["prediction"] == p["actual"])
        total = len(predictions)

        metrics["accuracy"] = correct / total if total > 0 else 0

        # Cluster-Verteilung
        cluster_distribution = {}
        for p in predictions:
            cluster = p["prediction"]
            cluster_distribution[cluster] = cluster_distribution.get(cluster, 0) + 1

        metrics["cluster_distribution"] = {
            k: v / total for k, v in cluster_distribution.items()
        }

        return metrics

    def _calculate_generic_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Berechnet allgemeine Metriken für andere Modelltypen.
        """
        metrics = {}

        # Abhängig vom Datentyp der Vorhersagen
        try:
            # Für numerische Vorhersagen
            pred_values = [float(p["prediction"]) for p in predictions]
            actual_values = [float(p["actual"]) for p in predictions]

            # Fehlermetriken
            errors = [abs(p - a) for p, a in zip(pred_values, actual_values)]

            metrics["mean_absolute_error"] = sum(errors) / len(errors)
            metrics["root_mean_squared_error"] = (sum(e ** 2 for e in errors) / len(errors)) ** 0.5

            # Richtungsgenauigkeit (für Vorhersagen von Veränderungen)
            direction_correct = 0
            for i in range(1, len(predictions)):
                pred_direction = pred_values[i] > pred_values[i - 1]
                actual_direction = actual_values[i] > actual_values[i - 1]

                if pred_direction == actual_direction:
                    direction_correct += 1

            if len(predictions) > 1:
                metrics["direction_accuracy"] = direction_correct / (len(predictions) - 1)
        except:
            # Für nicht-numerische Vorhersagen
            metrics["error"] = "Nicht-numerische Vorhersagen, kann keine Standard-Metriken berechnen"

        return metrics

    def generate_performance_report(self, model_id: str = None,
                                    model_type: str = None,
                                    days: int = 30) -> str:
        """
        Generiert einen Performance-Bericht mit Visualisierungen.

        Args:
            model_id: ID oder Name des Modells (optional)
            model_type: Typ des Modells (optional)
            days: Anzahl der Tage für die Analyse

        Returns:
            Pfad zum generierten Bericht
        """
        # Metriken berechnen
        metrics = self.calculate_model_metrics(model_id, model_type, days)

        # Berichtsverzeichnis
        report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        # Zeitstempel für Berichtsname
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"model_report_{timestamp}"

        if model_id and model_type:
            report_name = f"{model_type}_{model_id}_{timestamp}"
        elif model_type:
            report_name = f"{model_type}_all_{timestamp}"

        report_path = os.path.join(report_dir, f"{report_name}.html")

        # HTML-Bericht erstellen
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Model Performance Report</title>")
            f.write("<style>body { font-family: Arial; margin: 20px; }")
            f.write("table { border-collapse: collapse; width: 100%; }")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("</style></head><body>")

            f.write(f"<h1>Model Performance Report</h1>")
            f.write(f"<p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            f.write(f"<p>Data from past {days} days</p>")

            # Für jedes Modell
            for model_key, model_metrics in metrics.items():
                model_info = self.performance_data.get(model_key, {})

                f.write(f"<h2>Model: {model_key}</h2>")

                # Allgemeine Informationen
                f.write("<h3>General Information</h3>")
                f.write("<table>")
                f.write(f"<tr><th>Type</th><td>{model_info.get('model_type', 'Unknown')}</td></tr>")
                f.write(f"<tr><th>ID</th><td>{model_info.get('model_id', 'Unknown')}</td></tr>")

                predictions = model_info.get('predictions', [])
                if predictions:
                    f.write(f"<tr><th>Data points</th><td>{len(predictions)}</td></tr>")

                    first_date = datetime.fromisoformat(predictions[0]['timestamp']) if predictions else None
                    last_date = datetime.fromisoformat(predictions[-1]['timestamp']) if predictions else None

                    if first_date and last_date:
                        f.write(
                            f"<tr><th>Date range</th><td>{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}</td></tr>")

                f.write("</table>")

                # Performance-Metriken
                f.write("<h3>Performance Metrics</h3>")
                f.write("<table><tr><th>Metric</th><th>Value</th></tr>")

                for metric, value in model_metrics.items():
                    if isinstance(value, dict):
                        f.write(f"<tr><th colspan='2'>{metric}</th></tr>")
                        for sub_metric, sub_value in value.items():
                            f.write(
                                f"<tr><td>{sub_metric}</td><td>{sub_value:.4f if isinstance(sub_value, float) else sub_value}</td></tr>")
                    else:
                        f.write(
                            f"<tr><td>{metric}</td><td>{value:.4f if isinstance(value, float) else value}</td></tr>")

                f.write("</table>")

                # Visualisierungen generieren und einbinden
                if model_info.get('model_type') == 'regime':
                    graph_path = self._generate_regime_visualization(model_key, days, report_dir)
                    if graph_path:
                        rel_path = os.path.basename(graph_path)
                        f.write(f"<h3>Regime Predictions vs Actual</h3>")
                        f.write(f"<img src='{rel_path}' style='max-width: 100%;'>")

            f.write("</body></html>")

        self.logger.info(f"Performance-Bericht erstellt: {report_path}")
        return report_path

    def _generate_regime_visualization(self, model_key: str, days: int, output_dir: str) -> Optional[str]:
        """
        Generiert eine Visualisierung für Regime-Vorhersagen.

        Returns:
            Pfad zur Grafik oder None bei Fehler
        """
        try:
            model_data = self.performance_data.get(model_key, {})
            predictions = model_data.get("predictions", [])

            if not predictions:
                return None

            # Nach Datum filtern
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            recent_preds = [p for p in predictions if p["timestamp"] >= cutoff_date]

            if not recent_preds:
                return None

            # Daten für Grafik aufbereiten
            dates = [datetime.fromisoformat(p["timestamp"]) for p in recent_preds]
            pred_regimes = [p["prediction"] for p in recent_preds]
            actual_regimes = [p["actual"] for p in recent_preds]

            # Grafik erstellen
            plt.figure(figsize=(12, 6))

            plt.plot(dates, pred_regimes, 'o-', label='Predicted Regime', alpha=0.7)
            plt.plot(dates, actual_regimes, 'x--', label='Actual Regime', alpha=0.7)

            plt.title(f"Regime Predictions vs Actual - {model_key}")
            plt.xlabel("Date")
            plt.ylabel("Regime ID")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Speichern
            graph_name = f"{model_key}_regime_viz.png"
            graph_path = os.path.join(output_dir, graph_name)
            plt.savefig(graph_path)
            plt.close()

            return graph_path

        except Exception as e:
            self.logger.error(f"Fehler bei der Regime-Visualisierung: {e}")
            return None