#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AssetClusterAnalyzer für den Trading Bot und Backtest-Modul.
Identifiziert Gruppen von Assets mit ähnlichem Verhalten und analysiert ihre Performance.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


class AssetClusterAnalyzer:
    """
    Clustering-Analyse für Kryptowährungen.
    Identifiziert Gruppen von Assets mit ähnlichem Verhalten.
    """

    def __init__(self, data_dir: str = "data/market_data"):
        """
        Initialisiert den Asset-Cluster-Analyzer.

        Args:
            data_dir: Verzeichnis mit historischen Marktdaten
        """
        self.data_dir = data_dir
        self.market_data = {}
        self.correlation_matrix = None
        self.clusters = None
        self.cluster_model = None
        self.feature_data = None
        self.scaler = StandardScaler()
        self.cluster_performances = None
        self.logger = logging.getLogger(__name__)

    def load_market_data(self, symbols: List[str] = None,
                         data_manager=None,  # DataManager-Objekt
                         timeframe: str = "1d",
                         start_date: str = None,
                         end_date: str = None) -> bool:
        """
        Lädt Marktdaten für die angegebenen Symbole.

        Args:
            symbols: Liste der zu ladenden Symbole (oder None für alle)
            data_manager: Optionaler DataManager zur Datenabfrage
            timeframe: Zeitrahmen der Daten
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'

        Returns:
            True, wenn Daten erfolgreich geladen wurden
        """
        try:
            # Wenn kein Startdatum angegeben, letzten Monat verwenden
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            # Wenn kein Enddatum angegeben, aktuelles Datum verwenden
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Wenn DataManager bereitgestellt wurde, diesen für das Laden verwenden
            if data_manager:
                for symbol in symbols:
                    df = data_manager.get_historical_data(
                        symbol=symbol,
                        source="binance",  # Könnte aus Einstellungen kommen
                        timeframe=timeframe,
                        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                        use_cache=True
                    )

                    if not df.empty:
                        self.market_data[symbol] = df
                        self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

                if not self.market_data:
                    self.logger.error("Keine Marktdaten geladen")
                    return False

                self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
                return True

            # Alternativ: Direkt aus dem Verzeichnis laden
            binance_dir = os.path.join(self.data_dir, "binance")

            # Wenn keine Symbole angegeben, alle verfügbaren laden
            if not symbols:
                symbols = []
                for filename in os.listdir(binance_dir):
                    if filename.endswith(f"_{timeframe}.csv") or filename.endswith(f"_{timeframe}_20230101.csv"):
                        symbol = filename.split("_")[0]
                        quote = filename.split("_")[1]
                        symbols.append(f"{symbol}/{quote}")

            # Daten für jedes Symbol laden
            for symbol in symbols:
                base, quote = symbol.split("/")
                filename_pattern = f"{base}_{quote}_{timeframe}"

                # Datei suchen
                csv_path = None
                for f in os.listdir(binance_dir):
                    if f.startswith(filename_pattern):
                        csv_path = os.path.join(binance_dir, f)
                        break

                if not csv_path:
                    self.logger.warning(f"Keine Daten gefunden für {symbol} mit Timeframe {timeframe}")
                    continue

                # Daten laden
                df = pd.read_csv(csv_path)

                # Datum als Index setzen, falls vorhanden
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('date', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'time' in df.columns:
                    df['date'] = pd.to_datetime(df['time'])
                    df.set_index('date', inplace=True)

                # Datum filtern, falls angegeben
                if start_date and end_date:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                # Daten speichern
                self.market_data[symbol] = df
                self.logger.info(f"Daten für {symbol} geladen: {len(df)} Einträge")

            if not self.market_data:
                self.logger.error("Keine Marktdaten geladen")
                return False

            self.logger.info(f"Marktdaten für {len(self.market_data)} Symbole geladen")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return False

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Berechnet die Korrelationsmatrix zwischen allen Assets.

        Returns:
            Korrelationsmatrix als DataFrame
        """
        if not self.market_data:
            self.logger.error("Keine Marktdaten vorhanden")
            return pd.DataFrame()

        try:
            # DataFrame für Returns
            returns_df = pd.DataFrame()

            # Returns für jedes Symbol extrahieren
            for symbol, df in self.market_data.items():
                short_name = symbol.split('/')[0]
                returns_df[short_name] = df['close'].pct_change()

            # NaN-Werte entfernen
            returns_df = returns_df.dropna()

            # Korrelationsmatrix berechnen
            correlation_matrix = returns_df.corr()

            self.correlation_matrix = correlation_matrix

            self.logger.info(f"Korrelationsmatrix berechnet für {len(correlation_matrix)} Assets")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der Korrelationsmatrix: {e}")
            return pd.DataFrame()

    def extract_asset_features(self) -> pd.DataFrame:
        """
        Extrahiert Features für das Asset-Clustering.

        Returns:
            DataFrame mit Features für jedes Asset
        """
        if not self.market_data:
            self.logger.error("Keine Marktdaten vorhanden")
            return pd.DataFrame()

        try:
            # Features für jedes Asset
            asset_features = []

            for symbol, df in self.market_data.items():
                # Basiswährung
                base = symbol.split('/')[0]

                # Preis- und Return-Daten
                price_data = df['close']
                returns = price_data.pct_change().dropna()

                if len(returns) < 20:
                    self.logger.warning(f"Nicht genügend Daten für {symbol}, überspringe")
                    continue

                # Return-Features
                mean_return = returns.mean()
                std_return = returns.std()
                skew_return = returns.skew()
                kurt_return = returns.kurt()

                # Volatilität
                volatility_20d = returns.rolling(20).std().mean()

                # Trend-Features
                if len(price_data) >= 50:
                    ma_20 = price_data.rolling(20).mean()
                    ma_50 = price_data.rolling(50).mean()

                    # Average Ratio zum MA
                    avg_dist_ma20 = ((price_data / ma_20) - 1).mean()
                    avg_dist_ma50 = ((price_data / ma_50) - 1).mean()

                    # Verhältnis MA20 zu MA50
                    ma_ratio = (ma_20 / ma_50).mean()
                else:
                    avg_dist_ma20 = 0
                    avg_dist_ma50 = 0
                    ma_ratio = 1

                # Volumen-Features (falls vorhanden)
                if 'volume' in df.columns:
                    vol_data = df['volume']
                    vol_change = vol_data.pct_change().dropna()
                    mean_vol_change = vol_change.mean()
                    std_vol_change = vol_change.std()

                    # Volumen/Preis-Korrelation
                    vol_price_corr = vol_data.corr(price_data)
                else:
                    mean_vol_change = 0
                    std_vol_change = 0
                    vol_price_corr = 0

                # Features für dieses Asset
                asset_feature = {
                    'symbol': base,
                    'mean_return': mean_return,
                    'volatility': std_return,
                    'skewness': skew_return,
                    'kurtosis': kurt_return,
                    'volatility_20d': volatility_20d,
                    'avg_dist_ma20': avg_dist_ma20,
                    'avg_dist_ma50': avg_dist_ma50,
                    'ma_ratio': ma_ratio,
                    'mean_vol_change': mean_vol_change,
                    'std_vol_change': std_vol_change,
                    'vol_price_corr': vol_price_corr
                }

                asset_features.append(asset_feature)

            # In DataFrame umwandeln
            feature_df = pd.DataFrame(asset_features)

            if feature_df.empty:
                self.logger.error("Keine Features extrahiert")
                return pd.DataFrame()

            # Symbol als Index
            feature_df.set_index('symbol', inplace=True)

            # NaN-Werte durch 0 ersetzen
            feature_df = feature_df.fillna(0)

            self.feature_data = feature_df

            self.logger.info(f"Features für {len(feature_df)} Assets extrahiert")
            return feature_df

        except Exception as e:
            self.logger.error(f"Fehler bei der Extraktion der Asset-Features: {e}")
            return pd.DataFrame()

    def run_clustering(self, n_clusters: int = None, method: str = 'kmeans',
                       correlation_based: bool = True) -> pd.DataFrame:
        """
        Führt das Clustering der Assets durch.

        Args:
            n_clusters: Anzahl der Cluster (oder None für automatische Bestimmung)
            method: Clustering-Methode ('kmeans' oder 'dbscan')
            correlation_based: Ob auf Korrelationen oder Features basiert werden soll

        Returns:
            DataFrame mit Cluster-Zuordnungen
        """
        try:
            if correlation_based:
                # Falls noch nicht berechnet
                if self.correlation_matrix is None:
                    self.calculate_correlation_matrix()

                if self.correlation_matrix.empty:
                    self.logger.error("Korrelationsmatrix ist leer")
                    return pd.DataFrame()

                # 1 - Korrelation als Distanzmaß
                distance_matrix = 1 - self.correlation_matrix

                # In Feature-Array umwandeln mit MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                features = mds.fit_transform(distance_matrix)

                # DataFrame erstellen
                asset_features = pd.DataFrame(
                    features,
                    columns=['dim1', 'dim2'],
                    index=self.correlation_matrix.index
                )
            else:
                # Basierend auf statistischen Features
                if self.feature_data is None:
                    self.extract_asset_features()

                if self.feature_data.empty:
                    self.logger.error("Keine Feature-Daten vorhanden")
                    return pd.DataFrame()

                asset_features = self.feature_data.copy()

            # Daten für Clustering vorbereiten
            X = self.scaler.fit_transform(asset_features)

            # Optimale Anzahl an Clustern bestimmen (wenn nicht angegeben)
            if method == 'kmeans' and n_clusters is None:
                max_clusters = min(10, len(X) // 2)  # Maximal 10 Cluster oder Hälfte der Daten
                scores = []

                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)

                    if len(set(labels)) > 1:  # Mindestens 2 unterschiedliche Cluster
                        score = silhouette_score(X, labels)
                        scores.append((k, score))

                # Beste Anzahl Cluster auswählen
                n_clusters = max(scores, key=lambda x: x[1])[0]
                self.logger.info(f"Optimale Anzahl an Clustern: {n_clusters}")

            # Clustering durchführen
            if method == 'kmeans':
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.cluster_model.fit_predict(X)
                self.cluster_centers = self.cluster_model.cluster_centers_
            elif method == 'dbscan':
                from sklearn.neighbors import NearestNeighbors

                # Optimalen eps-Parameter bestimmen
                neigh = NearestNeighbors(n_neighbors=2)
                nbrs = neigh.fit(X)
                distances, indices = nbrs.kneighbors(X)
                distances = np.sort(distances[:, 1])

                # Knick in der Kurve finden
                from scipy.signal import argrelextrema
                n = len(distances)
                indices = argrelextrema(np.diff(distances), np.greater)[0]
                if len(indices) > 0:
                    eps = distances[indices[0]]
                else:
                    eps = np.percentile(distances, 90)

                self.cluster_model = DBSCAN(eps=eps, min_samples=2)
                cluster_labels = self.cluster_model.fit_predict(X)
            else:
                self.logger.error(f"Unbekannte Clustering-Methode: {method}")
                return pd.DataFrame()

            # Ergebnisse in DataFrame
            if correlation_based:
                result_df = pd.DataFrame(
                    {'cluster': cluster_labels},
                    index=self.correlation_matrix.index
                )
            else:
                result_df = pd.DataFrame(
                    {'cluster': cluster_labels},
                    index=self.feature_data.index
                )

            # Cluster-Zentren speichern (für KMeans)
            if method == 'kmeans':
                self.cluster_centers = self.cluster_model.cluster_centers_

            self.clusters = result_df

            # Cluster-Performance berechnen
            self.analyze_cluster_performance(result_df)

            self.logger.info(
                f"Clustering abgeschlossen: {len(result_df)} Assets in {len(set(cluster_labels))} Clustern")
            return result_df

        except Exception as e:
            self.logger.error(f"Fehler beim Clustering: {e}")
            return pd.DataFrame()

    def analyze_cluster_performance(self, cluster_assignments: pd.DataFrame) -> Dict[int, Dict]:
        """
        Analysiert die Performance der verschiedenen Asset-Cluster.

        Args:
            cluster_assignments: DataFrame mit Cluster-Zuordnungen

        Returns:
            Dictionary mit Performance-Statistiken pro Cluster
        """
        if cluster_assignments.empty or not self.market_data:
            self.logger.error("Keine Cluster-Zuweisungen oder Marktdaten vorhanden")
            return {}

        try:
            # Cluster-Performance
            cluster_stats = {}

            # Für jeden Cluster
            for cluster in sorted(cluster_assignments['cluster'].unique()):
                # Assets in diesem Cluster
                cluster_assets = cluster_assignments[cluster_assignments['cluster'] == cluster].index

                # Performance-Daten sammeln
                returns_data = []

                for asset in cluster_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Returns berechnen
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    returns_data.append((asset, returns))

                if not returns_data:
                    continue

                # Durchschnittliche tägliche Performance des Clusters
                all_returns = pd.DataFrame({asset: returns for asset, returns in returns_data})
                avg_returns = all_returns.mean(axis=1)

                # Performance-Statistiken
                mean_return = avg_returns.mean()
                volatility = avg_returns.std()
                sharpe = mean_return / volatility if volatility > 0 else 0
                max_drawdown = (avg_returns.cumsum() - avg_returns.cumsum().cummax()).min()
                win_rate = (avg_returns > 0).mean()

                # Korrelationen innerhalb des Clusters
                intra_corr = all_returns.corr().values.mean()

                # Repräsentatives Asset (am nächsten zum Clusterzentrum)
                if hasattr(self, 'cluster_centers') and cluster >= 0:
                    # Skalierte Feature-Werte
                    if self.feature_data is not None and not self.feature_data.empty:
                        X = self.scaler.transform(self.feature_data)

                        # Indizes der Assets in diesem Cluster
                        cluster_indices = []
                        for idx, (asset, row) in enumerate(self.feature_data.iterrows()):
                            if asset in cluster_assets:
                                cluster_indices.append(idx)

                        if cluster_indices:
                            # Distanzen zum Clusterzentrum
                            center = self.cluster_centers[cluster]
                            distances = np.linalg.norm(X[cluster_indices] - center, axis=1)

                            # Asset mit minimaler Distanz
                            rep_idx = cluster_indices[np.argmin(distances)]
                            representative = self.feature_data.index[rep_idx]
                        else:
                            representative = cluster_assets[0] if cluster_assets else "Unknown"
                    else:
                        representative = cluster_assets[0] if cluster_assets else "Unknown"
                else:
                    representative = cluster_assets[0] if cluster_assets else "Unknown"

                # Cluster-Statistik speichern
                cluster_stats[cluster] = {
                    "assets": list(cluster_assets),
                    "count": len(cluster_assets),
                    "mean_return": mean_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "intra_correlation": intra_corr,
                    "representative_asset": representative
                }

            self.cluster_performances = cluster_stats

            self.logger.info(f"Cluster-Performance-Analyse abgeschlossen für {len(cluster_stats)} Cluster")
            return cluster_stats

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse der Cluster-Performance: {e}")
            return {}

    def recommend_portfolio(self, n_assets: int = 5, regime_id: int = None,
                            regime_detector=None) -> Dict[str, Any]:
        """
        Empfiehlt ein diversifiziertes Portfolio basierend auf dem Clustering.
        Optional kann ein Marktregime angegeben werden, um regime-spezifische Empfehlungen zu erhalten.

        Args:
            n_assets: Anzahl der zu empfehlenden Assets
            regime_id: Optionale ID des aktuellen Marktregimes
            regime_detector: Optionaler MarketRegimeDetector für regime-spezifische Empfehlungen

        Returns:
            Dictionary mit Portfolioempfehlungen
        """
        if self.clusters is None or self.cluster_performances is None:
            self.logger.error("Keine Cluster-Daten oder Performance-Analyse vorhanden")
            return {}

        try:
            recommendations = {}

            # Cluster nach Sharpe Ratio sortieren
            sorted_clusters = sorted(
                self.cluster_performances.items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )

            # Regime-spezifische Anpassung, falls verfügbar
            regime_strategy = "Diversifizierte Allokation"
            regime_bias = {}

            if regime_id is not None and regime_detector is not None and regime_detector.model_trained:
                # Trading-Regeln für dieses Regime abrufen
                trading_rules = regime_detector.extract_trading_rules()

                if regime_id in trading_rules:
                    rule = trading_rules[regime_id]
                    regime_label = rule.get('label', f"Regime {regime_id}")

                    # Strategie-Empfehlung
                    regime_strategy = rule.get('recommended_strategy', "Diversifizierte Allokation")

                    # Top-Performer in diesem Regime
                    top_performers = rule.get('top_performers', {})

                    # Bias für Assets, die in diesem Regime gut performen
                    for asset, perf in top_performers.items():
                        # Kurznamen extrahieren
                        short_name = asset.split('/')[0]
                        regime_bias[short_name] = max(1.0, 1.0 + perf * 100)  # Bonus basierend auf Performance

                    self.logger.info(f"Berücksichtige Regime {regime_id} ({regime_label}) für Portfolio-Empfehlung")

            # Positiv performende Cluster auswählen
            positive_clusters = [c for c in sorted_clusters if c[1]['mean_return'] > 0]

            if not positive_clusters:
                self.logger.warning("Keine positiv performenden Cluster gefunden")
                recommendations['strategy'] = "Defensive Strategie - nur Stablecoins empfohlen"
                recommendations['assets'] = []
                return recommendations

            # Maximale Anzahl Assets pro Cluster
            assets_per_cluster = max(1, n_assets // len(positive_clusters))

            # Empfohlene Assets sammeln
            recommended_assets = []
            allocation = {}

            for cluster_id, stats in positive_clusters:
                # Assets in diesem Cluster nach Performance sortieren
                cluster_assets = stats['assets']

                # Performance-Daten für Assets in diesem Cluster
                asset_performance = {}

                for asset in cluster_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Performance-Metrik: Sharpe Ratio
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    mean_return = returns.mean()
                    volatility = returns.std()
                    sharpe = mean_return / volatility if volatility > 0 else 0

                    # Regime-Bias anwenden
                    bias = regime_bias.get(asset, 1.0)
                    asset_performance[asset] = sharpe * bias

                # Top-Assets aus diesem Cluster auswählen
                top_assets = sorted(
                    asset_performance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:assets_per_cluster]

                if top_assets:
                    for asset, _ in top_assets:
                        recommended_assets.append(asset)
                else:
                    # Falls keine Performance-Daten, repräsentatives Asset verwenden
                    rep_asset = stats.get('representative_asset')
                    if rep_asset and rep_asset != "Unknown":
                        recommended_assets.append(rep_asset)

            # Auf die gewünschte Anzahl begrenzen
            recommended_assets = recommended_assets[:n_assets]

            # Einfache Allokation basierend auf Sharpe Ratio
            if recommended_assets:
                asset_sharpes = {}

                for asset in recommended_assets:
                    # Vollständiges Symbol rekonstruieren
                    asset_symbols = [s for s in self.market_data.keys() if s.startswith(asset)]

                    if not asset_symbols:
                        continue

                    asset_symbol = asset_symbols[0]

                    # Sharpe berechnen
                    df = self.market_data[asset_symbol]
                    returns = df['close'].pct_change().dropna()

                    mean_return = returns.mean()
                    volatility = returns.std()
                    sharpe = mean_return / volatility if volatility > 0 else 0

                    # Regime-Bias anwenden
                    bias = regime_bias.get(asset, 1.0)
                    asset_sharpes[asset] = max(0.01, sharpe * bias)

                # Proportional zum Sharpe Ratio allokieren
                total_sharpe = sum(asset_sharpes.values())

                for asset, sharpe in asset_sharpes.items():
                    allocation[asset] = sharpe / total_sharpe

            # Portfolio-Empfehlung
            recommendations = {
                'assets': recommended_assets,
                'allocation': allocation,
                'strategy': regime_strategy + " über performante Cluster",
                'regime_adjusted': regime_id is not None
            }

            self.logger.info(f"Portfolio mit {len(recommended_assets)} Assets empfohlen")
            return recommendations

        except Exception as e:
            self.logger.error(f"Fehler bei der Portfolio-Empfehlung: {e}")
            return {}

    def analyze_new_coin(self, coin_symbol: str, coin_data: pd.DataFrame, min_days: int = 3) -> Dict[str, Any]:
        """
        Analysiert einen neuen Coin und weist ihn einem Cluster zu.

        Args:
            coin_symbol: Symbol des Coins (z.B. 'BTC/USDT')
            coin_data: DataFrame mit Kursdaten des Coins
            min_days: Minimale Anzahl an Tagen für die Analyse

        Returns:
            Analyse-Ergebnisse oder leeres Dictionary bei Fehler
        """
        try:
            # Prüfen, ob genügend Daten vorhanden sind
            if len(coin_data) < min_days:
                self.logger.warning(f"Nicht genügend Daten für {coin_symbol}: {len(coin_data)} Tage")
                return {"status": "insufficient_data", "days_available": len(coin_data)}

            # Features für diesen Coin extrahieren
            features = self._extract_coin_features(coin_data, coin_symbol)

            # Ähnliche Coins finden
            similar_coins = []

            if self.clusters is not None:
                # Vorheriges Clustering laden
                if self.feature_data is not None:
                    # Feature-Matrix
                    X = self.feature_data.copy()

                    if not X.empty:
                        # Neue Features normalisieren
                        X_scaled = self.scaler.transform(X)

                        # Feature-Vektor für den neuen Coin
                        if all(f in X.columns for f in features.keys()):
                            coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                            coin_features_scaled = self.scaler.transform(coin_features)

                            # Ähnlichkeit berechnen (euklidische Distanz)
                            distances = np.sqrt(((X_scaled - coin_features_scaled) ** 2).sum(axis=1))

                            # Top-5 ähnlichste Coins
                            top_indices = np.argsort(distances)[:5]
                            similar_coins = [X.index[i] for i in top_indices]

                # Cluster vorhersagen
                predicted_cluster = -1  # Ausreißer als Standard

                if hasattr(self, 'cluster_model') and self.cluster_model is not None:
                    # Feature-Vektor für den neuen Coin
                    if all(f in self.feature_data.columns for f in features.keys()):
                        coin_features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                        coin_features_scaled = self.scaler.transform(coin_features)

                        # Cluster vorhersagen
                        predicted_cluster = self.cluster_model.predict(coin_features_scaled)[0]

            # Empfohlene Strategie für diesen Coin basierend auf Cluster
            if self.clusters is not None and predicted_cluster >= 0:
                # Assets im gleichen Cluster
                cluster_assets = self.clusters[self.clusters['cluster'] == predicted_cluster].index.tolist()

                # Performance-Daten aus dem Cluster-Analyzer
                cluster_performance = self.cluster_performances.get(predicted_cluster, {})

                recommended_strategy = "Neutral - Beobachten"

                # Basierend auf Cluster-Performance
                if cluster_performance:
                    mean_return = cluster_performance.get('mean_return', 0)
                    sharpe_ratio = cluster_performance.get('sharpe_ratio', 0)

                    if mean_return > 0.01 and sharpe_ratio > 0.5:
                        recommended_strategy = "Opportunistisches Long - Positive Cluster-Performance"
                    elif mean_return > 0:
                        recommended_strategy = "Vorsichtiges Long - Leicht positive Cluster-Performance"
                    elif mean_return < -0.01:
                        recommended_strategy = "Vermeiden - Negative Cluster-Performance"
            else:
                cluster_assets = []
                recommended_strategy = "Neutral - Neue Coin ohne Clusterzuordnung"
                predicted_cluster = -1

            # Analyseergebnis
            analysis_result = {
                "status": "analyzed",
                "coin": coin_symbol,
                "predicted_cluster": predicted_cluster,
                "similar_coins": similar_coins,
                "recommended_strategy": recommended_strategy,
                "cluster_assets": cluster_assets,
                "features": features
            }

            self.logger.info(
                f"Analyse für {coin_symbol} abgeschlossen: Cluster {predicted_cluster}, Strategie: {recommended_strategy}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse des neuen Coins {coin_symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def _extract_coin_features(self, df: pd.DataFrame, coin_symbol: str) -> Dict[str, float]:
        """
        Extrahiert Features für einen Coin aus dessen Marktdaten.

        Args:
            df: DataFrame mit historischen Daten
            coin_symbol: Symbol des Coins

        Returns:
            Dictionary mit Feature-Namen und -Werten
        """
        try:
            # Sicherstellen, dass wichtige Spalten vorhanden sind
            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.logger.warning(f"Fehlende Spalten für {coin_symbol}: {missing_columns}")
                for col in missing_columns:
                    df[col] = 0

            # Returns berechnen
            df['return'] = df['close'].pct_change()

            # Basisfeatures
            features = {}

            # 1. Rendite-Statistiken
            features['mean_return'] = df['return'].mean()
            features['volatility'] = df['return'].std()
            features['skewness'] = df['return'].skew() if len(df) > 3 else 0
            features['kurtosis'] = df['return'].kurtosis() if len(df) > 3 else 0

            # 2. Volatilität
            features['volatility_20d'] = df['return'].rolling(min(10, len(df))).std().mean()

            # 3. True Range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            features['avg_true_range'] = df['true_range'].mean() / df['close'].mean()

            # 4. Volumen-Statistiken (wenn verfügbar)
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change()
                features['volume_mean'] = df['volume'].mean()
                features['volume_std'] = df['volume'].std()
                features['volume_change_mean'] = df['volume_change'].mean()
            else:
                features['volume_mean'] = 0
                features['volume_std'] = 0
                features['volume_change_mean'] = 0

            # 5. Trendstärke
            if len(df) >= 5:
                df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
                features['trend_strength'] = (df['close'].iloc[-1] / df['ema_5'].iloc[-1] - 1)
            else:
                features['trend_strength'] = 0

            if len(df) >= 10:
                df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
                features['ema_ratio'] = df['ema_5'].iloc[-1] / df['ema_10'].iloc[-1]
            else:
                features['ema_ratio'] = 1

            # 6. RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                features['rsi'] = df['rsi'].iloc[-1]
            else:
                features['rsi'] = 50  # Neutral

            # 7. Drawdown
            if len(df) >= 5:
                rolling_max = df['close'].rolling(min(len(df), 30)).max()
                drawdown = (df['close'] / rolling_max - 1) * 100
                features['max_drawdown'] = drawdown.min()
            else:
                features['max_drawdown'] = 0

            # 8. Return-to-Volatility Ratio (Sharpe-ähnlich)
            if features['volatility'] > 0:
                features['return_to_vol_ratio'] = features['mean_return'] / features['volatility']
            else:
                features['return_to_vol_ratio'] = 0

            return features

        except Exception as e:
            self.logger.error(f"Fehler bei der Feature-Extraktion für {coin_symbol}: {e}")
            return {}