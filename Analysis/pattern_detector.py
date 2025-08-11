#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Pattern Detector
ML-basierte Mustererkennung für Trading-Daten

Diese Komponente:
- Findet wiederkehrende Erfolgsmuster
- Erkennt gefährliche Marktbedingungen  
- Identifiziert Strategie-Synergien
- Nutzt scikit-learn für Clustering
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
import json

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class SuccessPattern:
    """Erfolgsmuster Definition"""
    pattern_id: str
    pattern_name: str
    description: str
    confidence: float
    frequency: int
    avg_return: float
    success_rate: float
    conditions: Dict[str, Any]
    strategies_involved: List[str]
    market_conditions: Dict[str, Any]
    feature_importance: Dict[str, float]
    cluster_id: Optional[int] = None

@dataclass
class DangerousCondition:
    """Gefährliche Marktbedingung"""
    condition_id: str
    condition_name: str
    description: str
    danger_score: float
    frequency: int
    avg_loss: float
    affected_strategies: List[str]
    warning_indicators: List[str]
    market_features: Dict[str, Any]
    prevention_actions: List[str]

@dataclass
class StrategySynergy:
    """Strategie-Synergie"""
    synergy_id: str
    strategies: List[str]
    synergy_type: str  # 'complementary', 'reinforcing', 'diversifying'
    synergy_strength: float
    optimal_timing: str
    market_conditions: List[str]
    performance_boost: float
    correlation_features: Dict[str, float]

class PatternDetector:
    """ML-basierte Mustererkennung für Trading-Daten"""
    
    def __init__(self, lookback_days: int = 90, min_pattern_frequency: int = 5):
        """
        Initialize Pattern Detector
        
        Args:
            lookback_days: Anzahl Tage für Analyse
            min_pattern_frequency: Minimale Häufigkeit für Muster
        """
        self.lookback_days = lookback_days
        self.min_pattern_frequency = min_pattern_frequency
        
        # ML Models
        self.success_clusterer = None
        self.danger_detector = None
        self.synergy_classifier = None
        self.scaler = StandardScaler()
        
        # Results
        self.success_patterns: List[SuccessPattern] = []
        self.dangerous_conditions: List[DangerousCondition] = []
        self.strategy_synergies: List[StrategySynergy] = []
        
        # Model storage
        self.models_dir = Path("analysis/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path("analysis/pattern_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze_patterns(self, trades_df: pd.DataFrame, market_df: pd.DataFrame, 
                        decisions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Hauptanalyse-Methode für Mustererkennung
        
        Args:
            trades_df: Trading-Daten
            market_df: Marktdaten  
            decisions_df: Orchestrator-Entscheidungen
            
        Returns:
            Analyse-Ergebnisse
        """
        logger.info("🔍 Starting ML pattern detection...")
        
        start_time = datetime.utcnow()
        
        try:
            # Daten vorbereiten
            features_df = self._prepare_features(trades_df, market_df, decisions_df)
            
            if features_df.empty:
                logger.warning("No data available for pattern analysis")
                return {}
            
            # 1. Erfolgsmuster finden
            self._detect_success_patterns(features_df, trades_df)
            
            # 2. Gefährliche Bedingungen erkennen
            self._detect_dangerous_conditions(features_df, trades_df)
            
            # 3. Strategie-Synergien identifizieren
            self._detect_strategy_synergies(features_df, trades_df)
            
            # 4. Modelle speichern
            self._save_models()
            
            # 5. Ergebnisse speichern
            results = self._save_results()
            
            # 6. Visualisierungen erstellen
            self._create_visualizations(features_df)
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Pattern detection completed in {analysis_time:.1f}s")
            
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'analysis_duration_seconds': analysis_time,
                'patterns_found': {
                    'success_patterns': len(self.success_patterns),
                    'dangerous_conditions': len(self.dangerous_conditions),
                    'strategy_synergies': len(self.strategy_synergies)
                },
                'data_analyzed': len(features_df),
                'model_performance': self._get_model_performance(),
                'key_insights': self._get_key_insights(),
                'actionable_recommendations': self._get_recommendations(),
                'results_saved_to': str(self.results_dir)
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            raise

    def _prepare_features(self, trades_df: pd.DataFrame, market_df: pd.DataFrame, 
                         decisions_df: pd.DataFrame) -> pd.DataFrame:
        """Feature-Engineering für ML-Modelle"""
        logger.info("🔧 Preparing features for ML analysis...")
        
        if trades_df.empty:
            return pd.DataFrame()
            
        # Basis-Features aus Trades
        features = []
        
        for idx, trade in trades_df.iterrows():
            if pd.isna(trade.get('pnl_percentage')):
                continue
                
            feature_row = {
                # Trade-Features
                'trade_id': trade.get('trade_id', idx),
                'strategy_name': trade.get('strategy_name', 'unknown'),
                'symbol': trade.get('symbol', 'BTC_USDT'),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'quantity': trade.get('quantity', 0),
                'pnl_percentage': trade.get('pnl_percentage', 0),
                'pnl_absolute': trade.get('pnl_absolute', 0),
                'duration_minutes': trade.get('duration_minutes', 0),
                'trade_status': trade.get('trade_status', 'unknown'),
                
                # Zeit-Features
                'entry_timestamp': trade.get('entry_timestamp', datetime.utcnow()),
                'hour_of_day': pd.to_datetime(trade.get('entry_timestamp', datetime.utcnow())).hour,
                'day_of_week': pd.to_datetime(trade.get('entry_timestamp', datetime.utcnow())).weekday(),
                'is_weekend': pd.to_datetime(trade.get('entry_timestamp', datetime.utcnow())).weekday() >= 5,
                
                # Performance-Features
                'is_profitable': trade.get('pnl_percentage', 0) > 0,
                'is_large_profit': trade.get('pnl_percentage', 0) > 5,
                'is_large_loss': trade.get('pnl_percentage', 0) < -5,
                'return_magnitude': abs(trade.get('pnl_percentage', 0)),
            }
            
            # Markt-Features hinzufügen (falls verfügbar)
            if not market_df.empty:
                market_features = self._extract_market_features(trade, market_df)
                feature_row.update(market_features)
            else:
                # Default-Markt-Features
                feature_row.update({
                    'market_regime': 'unknown',
                    'volatility': 0,
                    'volume_ratio': 1,
                    'trend_strength': 0,
                    'rsi': 50,
                    'bb_position': 0.5
                })
            
            # Orchestrator-Features hinzufügen (falls verfügbar)
            if not decisions_df.empty:
                orchestrator_features = self._extract_orchestrator_features(trade, decisions_df)
                feature_row.update(orchestrator_features)
            else:
                # Default-Orchestrator-Features
                feature_row.update({
                    'total_allocation': 1.0,
                    'strategy_weight': 0.5,
                    'risk_level': 'medium',
                    'concurrent_strategies': 1
                })
            
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        
        if features_df.empty:
            return features_df
            
        # Kategorische Features kodieren
        features_df = self._encode_categorical_features(features_df)
        
        # Missing values behandeln
        features_df = features_df.fillna(0)
        
        logger.info(f"Prepared {len(features_df)} feature rows with {len(features_df.columns)} features")
        
        return features_df

    def _extract_market_features(self, trade: pd.Series, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Markt-Features für einen Trade extrahieren"""
        features = {}
        
        try:
            trade_time = pd.to_datetime(trade.get('entry_timestamp'))
            symbol = trade.get('symbol', 'BTC_USDT')
            
            # Nächste Marktdaten finden
            market_data = market_df[
                (market_df['symbol'] == symbol) &
                (pd.to_datetime(market_df['timestamp']) <= trade_time)
            ].tail(1)
            
            if not market_data.empty:
                market_row = market_data.iloc[0]
                features.update({
                    'market_regime': market_row.get('regime', 'unknown'),
                    'volatility': market_row.get('volatility', 0),
                    'volume_ratio': market_row.get('volume_ratio', 1),
                    'trend_strength': market_row.get('trend_strength', 0),
                    'rsi': market_row.get('rsi', 50),
                    'bb_position': market_row.get('bb_position', 0.5),
                    'price_change_1h': market_row.get('price_change_1h', 0),
                    'price_change_24h': market_row.get('price_change_24h', 0),
                })
            else:
                # Default-Werte wenn keine Marktdaten verfügbar
                features.update({
                    'market_regime': 'unknown',
                    'volatility': 0,
                    'volume_ratio': 1,
                    'trend_strength': 0,
                    'rsi': 50,
                    'bb_position': 0.5,
                    'price_change_1h': 0,
                    'price_change_24h': 0,
                })
                
        except Exception as e:
            logger.error(f"Failed to extract market features: {e}")
            features = {
                'market_regime': 'unknown',
                'volatility': 0,
                'volume_ratio': 1,
                'trend_strength': 0,
                'rsi': 50,
                'bb_position': 0.5,
                'price_change_1h': 0,
                'price_change_24h': 0,
            }
        
        return features

    def _extract_orchestrator_features(self, trade: pd.Series, decisions_df: pd.DataFrame) -> Dict[str, Any]:
        """Orchestrator-Features für einen Trade extrahieren"""
        features = {}
        
        try:
            trade_time = pd.to_datetime(trade.get('entry_timestamp'))
            
            # Nächste Orchestrator-Entscheidung finden
            decision_data = decisions_df[
                pd.to_datetime(decisions_df['timestamp']) <= trade_time
            ].tail(1)
            
            if not decision_data.empty:
                decision_row = decision_data.iloc[0]
                
                # Parse allocation data
                allocations = decision_row.get('allocations', {})
                if isinstance(allocations, str):
                    try:
                        allocations = json.loads(allocations)
                    except:
                        allocations = {}
                
                strategy_name = trade.get('strategy_name', 'unknown')
                
                features.update({
                    'total_allocation': sum(allocations.values()) if allocations else 1.0,
                    'strategy_weight': allocations.get(strategy_name, 0.5),
                    'risk_level': decision_row.get('risk_level', 'medium'),
                    'concurrent_strategies': len(allocations) if allocations else 1,
                    'market_confidence': decision_row.get('market_confidence', 0.5),
                    'portfolio_heat': decision_row.get('portfolio_heat', 0.5)
                })
            else:
                features.update({
                    'total_allocation': 1.0,
                    'strategy_weight': 0.5,
                    'risk_level': 'medium',
                    'concurrent_strategies': 1,
                    'market_confidence': 0.5,
                    'portfolio_heat': 0.5
                })
                
        except Exception as e:
            logger.error(f"Failed to extract orchestrator features: {e}")
            features = {
                'total_allocation': 1.0,
                'strategy_weight': 0.5,
                'risk_level': 'medium',
                'concurrent_strategies': 1,
                'market_confidence': 0.5,
                'portfolio_heat': 0.5
            }
        
        return features

    def _encode_categorical_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Kategorische Features kodieren"""
        categorical_columns = [
            'strategy_name', 'symbol', 'trade_status', 'market_regime', 'risk_level'
        ]
        
        for col in categorical_columns:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
        
        return features_df

    def _detect_success_patterns(self, features_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Erfolgsmuster mittels Clustering erkennen"""
        logger.info("🎯 Detecting success patterns...")
        
        # Nur profitable Trades für Erfolgsmuster
        profitable_trades = features_df[features_df['is_profitable'] == True].copy()
        
        if len(profitable_trades) < self.min_pattern_frequency * 2:
            logger.warning("Not enough profitable trades for pattern detection")
            return
        
        # Features für Clustering auswählen
        clustering_features = [
            'hour_of_day', 'day_of_week', 'volatility', 'trend_strength', 
            'rsi', 'bb_position', 'strategy_weight', 'concurrent_strategies',
            'market_confidence', 'duration_minutes', 'return_magnitude'
        ]
        
        # Verfügbare Features filtern
        available_features = [f for f in clustering_features if f in profitable_trades.columns]
        
        if not available_features:
            logger.warning("No suitable features available for clustering")
            return
        
        X = profitable_trades[available_features].fillna(0)
        
        # Features skalieren
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimale Anzahl Cluster finden
        optimal_clusters = self._find_optimal_clusters(X_scaled, max_clusters=min(8, len(X_scaled)//3))
        
        # K-Means Clustering
        self.success_clusterer = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = self.success_clusterer.fit_predict(X_scaled)
        
        profitable_trades['cluster'] = cluster_labels
        
        # Cluster analysieren und Muster extrahieren
        for cluster_id in range(optimal_clusters):
            cluster_trades = profitable_trades[profitable_trades['cluster'] == cluster_id]
            
            if len(cluster_trades) >= self.min_pattern_frequency:
                pattern = self._analyze_success_cluster(cluster_id, cluster_trades, available_features)
                if pattern:
                    self.success_patterns.append(pattern)
        
        logger.info(f"Found {len(self.success_patterns)} success patterns")

    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Optimale Anzahl Cluster mittels Silhouette-Score finden"""
        if len(X) < 4:
            return 2
            
        best_score = -1
        best_clusters = 2
        
        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            try:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(X)
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_clusters = n_clusters
            except:
                continue
        
        return best_clusters

    def _analyze_success_cluster(self, cluster_id: int, cluster_trades: pd.DataFrame, 
                                features: List[str]) -> Optional[SuccessPattern]:
        """Erfolgs-Cluster analysieren und Muster extrahieren"""
        try:
            # Cluster-Statistiken
            avg_return = cluster_trades['pnl_percentage'].mean()
            success_rate = (cluster_trades['pnl_percentage'] > 0).mean()
            frequency = len(cluster_trades)
            
            # Cluster-Zentrum analysieren
            cluster_center = cluster_trades[features].mean()
            
            # Charakteristische Bedingungen identifizieren
            conditions = {}
            for feature in features:
                value = cluster_center[feature]
                conditions[feature] = {
                    'mean': float(value),
                    'std': float(cluster_trades[feature].std()),
                    'range': [float(cluster_trades[feature].min()), float(cluster_trades[feature].max())]
                }
            
            # Feature-Wichtigkeit berechnen (vereinfacht)
            feature_importance = {}
            for feature in features:
                # Korrelation mit Return als Wichtigkeits-Proxy
                corr = cluster_trades[feature].corr(cluster_trades['pnl_percentage'])
                feature_importance[feature] = abs(corr) if not pd.isna(corr) else 0
            
            # Beteiligte Strategien
            strategies = cluster_trades['strategy_name'].unique().tolist()
            
            # Marktbedingungen
            market_conditions = {}
            if 'market_regime' in cluster_trades.columns:
                regime_dist = cluster_trades['market_regime'].value_counts(normalize=True)
                market_conditions['regime_distribution'] = regime_dist.to_dict()
            
            # Muster-Beschreibung generieren
            description = self._generate_pattern_description(cluster_center, conditions, features)
            
            return SuccessPattern(
                pattern_id=f"SUCCESS_{cluster_id:03d}",
                pattern_name=f"Success Pattern {cluster_id + 1}",
                description=description,
                confidence=min(0.95, success_rate),
                frequency=frequency,
                avg_return=avg_return,
                success_rate=success_rate,
                conditions=conditions,
                strategies_involved=strategies,
                market_conditions=market_conditions,
                feature_importance=feature_importance,
                cluster_id=cluster_id
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze success cluster {cluster_id}: {e}")
            return None

    def _generate_pattern_description(self, center: pd.Series, conditions: Dict, 
                                    features: List[str]) -> str:
        """Muster-Beschreibung aus Cluster-Zentrum generieren"""
        descriptions = []
        
        # Zeit-basierte Muster
        if 'hour_of_day' in center:
            hour = int(center['hour_of_day'])
            descriptions.append(f"Trades around {hour}:00")
        
        # Markt-basierte Muster
        if 'rsi' in center:
            rsi = center['rsi']
            if rsi < 30:
                descriptions.append("in oversold conditions")
            elif rsi > 70:
                descriptions.append("in overbought conditions")
        
        if 'volatility' in center:
            vol = center['volatility']
            if vol > 0.05:
                descriptions.append("during high volatility")
            elif vol < 0.02:
                descriptions.append("during low volatility")
        
        # Strategie-basierte Muster
        if 'concurrent_strategies' in center:
            concurrent = int(center['concurrent_strategies'])
            if concurrent > 3:
                descriptions.append("with multiple active strategies")
            elif concurrent == 1:
                descriptions.append("with single strategy focus")
        
        if descriptions:
            return "Profitable trades typically occur " + ", ".join(descriptions)
        else:
            return "General profitable trading pattern identified"

    def _detect_dangerous_conditions(self, features_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Gefährliche Marktbedingungen mittels Anomalieerkennung"""
        logger.info("⚠️ Detecting dangerous market conditions...")
        
        # Verlust-Trades für Anomalieerkennung
        losing_trades = features_df[features_df['is_large_loss'] == True].copy()
        
        if len(losing_trades) < self.min_pattern_frequency:
            logger.warning("Not enough loss data for danger detection")
            return
        
        # Features für Anomalieerkennung
        danger_features = [
            'volatility', 'trend_strength', 'rsi', 'bb_position',
            'volume_ratio', 'price_change_1h', 'price_change_24h',
            'portfolio_heat', 'concurrent_strategies'
        ]
        
        available_features = [f for f in danger_features if f in losing_trades.columns]
        
        if not available_features:
            logger.warning("No suitable features for danger detection")
            return
        
        X = losing_trades[available_features].fillna(0)
        
        # Isolation Forest für Anomalieerkennung
        self.danger_detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = self.danger_detector.fit_predict(X)
        
        # Anomalien als gefährliche Bedingungen interpretieren
        dangerous_indices = np.where(anomaly_labels == -1)[0]
        
        if len(dangerous_indices) > 0:
            dangerous_trades = losing_trades.iloc[dangerous_indices]
            
            # Cluster gefährliche Bedingungen
            if len(dangerous_trades) >= 3:
                dangerous_conditions = self._cluster_dangerous_conditions(dangerous_trades, available_features)
                self.dangerous_conditions.extend(dangerous_conditions)
        
        logger.info(f"Found {len(self.dangerous_conditions)} dangerous conditions")

    def _cluster_dangerous_conditions(self, dangerous_trades: pd.DataFrame, 
                                    features: List[str]) -> List[DangerousCondition]:
        """Gefährliche Bedingungen clustern"""
        conditions = []
        
        try:
            X = dangerous_trades[features].fillna(0)
            
            # DBSCAN für dichte Regionen gefährlicher Bedingungen
            dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(X)//4))
            cluster_labels = dbscan.fit_predict(X)
            
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Noise entfernen
            
            for cluster_id in unique_labels:
                cluster_mask = cluster_labels == cluster_id
                cluster_trades = dangerous_trades[cluster_mask]
                
                if len(cluster_trades) >= 2:
                    condition = self._analyze_dangerous_cluster(cluster_id, cluster_trades, features)
                    if condition:
                        conditions.append(condition)
                        
        except Exception as e:
            logger.error(f"Failed to cluster dangerous conditions: {e}")
        
        return conditions

    def _analyze_dangerous_cluster(self, cluster_id: int, cluster_trades: pd.DataFrame,
                                 features: List[str]) -> Optional[DangerousCondition]:
        """Gefährlichen Cluster analysieren"""
        try:
            # Danger-Score berechnen
            avg_loss = cluster_trades['pnl_percentage'].mean()
            loss_magnitude = abs(avg_loss)
            frequency = len(cluster_trades)
            
            danger_score = loss_magnitude * np.log(frequency + 1)
            
            # Charakteristische Features
            cluster_center = cluster_trades[features].mean()
            
            # Warning-Indikatoren identifizieren
            warning_indicators = []
            market_features = {}
            
            for feature in features:
                value = cluster_center[feature]
                market_features[feature] = float(value)
                
                # Spezifische Warnindikatoren
                if feature == 'volatility' and value > 0.1:
                    warning_indicators.append("Extreme volatility detected")
                elif feature == 'rsi' and (value < 20 or value > 80):
                    warning_indicators.append("Extreme RSI levels")
                elif feature == 'portfolio_heat' and value > 0.8:
                    warning_indicators.append("High portfolio heat")
                elif feature == 'concurrent_strategies' and value > 5:
                    warning_indicators.append("Too many concurrent strategies")
            
            # Betroffene Strategien
            affected_strategies = cluster_trades['strategy_name'].unique().tolist()
            
            # Präventionsmaßnahmen
            prevention_actions = [
                "Reduce position sizes immediately",
                "Halt new positions for affected strategies",
                "Implement emergency stop-loss",
                "Monitor market conditions closely",
                "Consider defensive positioning"
            ]
            
            # Beschreibung generieren
            condition_name = f"Dangerous Condition {cluster_id + 1}"
            description = f"Market conditions leading to average loss of {avg_loss:.2f}%"
            
            return DangerousCondition(
                condition_id=f"DANGER_{cluster_id:03d}",
                condition_name=condition_name,
                description=description,
                danger_score=danger_score,
                frequency=frequency,
                avg_loss=avg_loss,
                affected_strategies=affected_strategies,
                warning_indicators=warning_indicators,
                market_features=market_features,
                prevention_actions=prevention_actions
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze dangerous cluster {cluster_id}: {e}")
            return None

    def _detect_strategy_synergies(self, features_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Strategie-Synergien mittels Korrelationsanalyse erkennen"""
        logger.info("🤝 Detecting strategy synergies...")
        
        # Strategien mit genügend Daten
        strategy_counts = features_df['strategy_name'].value_counts()
        viable_strategies = strategy_counts[strategy_counts >= self.min_pattern_frequency].index.tolist()
        
        if len(viable_strategies) < 2:
            logger.warning("Not enough strategies for synergy analysis")
            return
        
        # Zeitbasierte Analyse für Synergien
        self._analyze_temporal_synergies(features_df, viable_strategies)
        
        # Performance-basierte Synergien
        self._analyze_performance_synergies(features_df, viable_strategies)
        
        logger.info(f"Found {len(self.strategy_synergies)} strategy synergies")

    def _analyze_temporal_synergies(self, features_df: pd.DataFrame, strategies: List[str]):
        """Zeitbasierte Strategie-Synergien analysieren"""
        try:
            # Tägliche Performance pro Strategie
            features_df['date'] = pd.to_datetime(features_df['entry_timestamp']).dt.date
            daily_performance = features_df.groupby(['date', 'strategy_name'])['pnl_percentage'].sum().unstack(fill_value=0)
            
            # Korrelationen zwischen Strategien
            correlations = daily_performance.corr()
            
            # Synergie-Paare finden
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    if strategy1 in correlations.columns and strategy2 in correlations.columns:
                        corr = correlations.loc[strategy1, strategy2]
                        
                        synergy = self._analyze_strategy_pair_synergy(
                            strategy1, strategy2, corr, daily_performance
                        )
                        
                        if synergy:
                            self.strategy_synergies.append(synergy)
                            
        except Exception as e:
            logger.error(f"Failed to analyze temporal synergies: {e}")

    def _analyze_performance_synergies(self, features_df: pd.DataFrame, strategies: List[str]):
        """Performance-basierte Synergien analysieren"""
        try:
            # Marktbedingungen für jede Strategie
            for strategy in strategies:
                strategy_trades = features_df[features_df['strategy_name'] == strategy]
                
                # Beste Marktbedingungen für diese Strategie
                if 'market_regime' in strategy_trades.columns:
                    regime_performance = strategy_trades.groupby('market_regime')['pnl_percentage'].mean()
                    best_regimes = regime_performance[regime_performance > 0].index.tolist()
                    
                    # Komplementäre Strategien in anderen Regimen finden
                    for other_strategy in strategies:
                        if other_strategy != strategy:
                            other_trades = features_df[features_df['strategy_name'] == other_strategy]
                            other_regime_perf = other_trades.groupby('market_regime')['pnl_percentage'].mean()
                            
                            # Finde Regimes wo andere Strategie gut ist, aber current nicht
                            complementary_regimes = []
                            for regime in other_regime_perf.index:
                                if (regime not in best_regimes and 
                                    other_regime_perf[regime] > 0):
                                    complementary_regimes.append(regime)
                            
                            if complementary_regimes:
                                synergy = StrategySynergy(
                                    synergy_id=f"COMP_{strategy}_{other_strategy}",
                                    strategies=[strategy, other_strategy],
                                    synergy_type="complementary",
                                    synergy_strength=len(complementary_regimes) / len(other_regime_perf),
                                    optimal_timing="regime_based",
                                    market_conditions=complementary_regimes,
                                    performance_boost=other_regime_perf[complementary_regimes].mean(),
                                    correlation_features={'regime_complementarity': len(complementary_regimes)}
                                )
                                self.strategy_synergies.append(synergy)
                                
        except Exception as e:
            logger.error(f"Failed to analyze performance synergies: {e}")

    def _analyze_strategy_pair_synergy(self, strategy1: str, strategy2: str, 
                                     correlation: float, daily_perf: pd.DataFrame) -> Optional[StrategySynergy]:
        """Analyse eines Strategie-Paars für Synergien"""
        try:
            if abs(correlation) < 0.3:  # Zu schwache Korrelation
                return None
            
            # Synergie-Typ bestimmen
            if correlation > 0.7:
                synergy_type = "reinforcing"
                synergy_strength = correlation
            elif correlation < -0.3:
                synergy_type = "diversifying"  
                synergy_strength = abs(correlation)
            else:
                synergy_type = "complementary"
                synergy_strength = abs(correlation)
            
            # Performance-Boost berechnen
            combined_performance = daily_perf[strategy1] + daily_perf[strategy2]
            individual_performance = (daily_perf[strategy1].mean() + daily_perf[strategy2].mean()) / 2
            performance_boost = combined_performance.mean() - individual_performance
            
            # Optimales Timing
            optimal_timing = "simultaneous" if correlation > 0 else "alternating"
            
            return StrategySynergy(
                synergy_id=f"SYN_{strategy1}_{strategy2}",
                strategies=[strategy1, strategy2],
                synergy_type=synergy_type,
                synergy_strength=synergy_strength,
                optimal_timing=optimal_timing,
                market_conditions=["any"],  # Vereinfacht
                performance_boost=performance_boost,
                correlation_features={
                    'correlation': correlation,
                    'combined_volatility': combined_performance.std()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze synergy for {strategy1}-{strategy2}: {e}")
            return None

    def _save_models(self):
        """ML-Modelle speichern"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if self.success_clusterer:
                joblib.dump(self.success_clusterer, 
                           self.models_dir / f"success_clusterer_{timestamp}.pkl")
            
            if self.danger_detector:
                joblib.dump(self.danger_detector,
                           self.models_dir / f"danger_detector_{timestamp}.pkl")
            
            if self.scaler:
                joblib.dump(self.scaler,
                           self.models_dir / f"feature_scaler_{timestamp}.pkl")
                           
            logger.info(f"Models saved to {self.models_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _save_results(self) -> Dict[str, str]:
        """Analyseergebnisse speichern"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        files = {}
        
        # Success patterns
        success_file = self.results_dir / f"success_patterns_{timestamp}.json"
        with open(success_file, 'w') as f:
            json.dump([asdict(pattern) for pattern in self.success_patterns], 
                     f, indent=2, default=str)
        files['success_patterns'] = str(success_file)
        
        # Dangerous conditions
        danger_file = self.results_dir / f"dangerous_conditions_{timestamp}.json"
        with open(danger_file, 'w') as f:
            json.dump([asdict(condition) for condition in self.dangerous_conditions],
                     f, indent=2, default=str)
        files['dangerous_conditions'] = str(danger_file)
        
        # Strategy synergies
        synergy_file = self.results_dir / f"strategy_synergies_{timestamp}.json"
        with open(synergy_file, 'w') as f:
            json.dump([asdict(synergy) for synergy in self.strategy_synergies],
                     f, indent=2, default=str)
        files['strategy_synergies'] = str(synergy_file)
        
        return files

    def _create_visualizations(self, features_df: pd.DataFrame):
        """Visualisierungen erstellen"""
        logger.info("📊 Creating pattern visualizations...")
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Success patterns visualization
            if self.success_patterns:
                self._create_success_patterns_chart(timestamp)
            
            # Dangerous conditions visualization
            if self.dangerous_conditions:
                self._create_danger_conditions_chart(timestamp)
            
            # Strategy synergies visualization
            if self.strategy_synergies:
                self._create_synergies_network_chart(timestamp)
            
            # Feature importance heatmap
            self._create_feature_importance_heatmap(features_df, timestamp)
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

    def _create_success_patterns_chart(self, timestamp: str):
        """Success patterns Visualisierung"""
        patterns_data = []
        
        for pattern in self.success_patterns:
            patterns_data.append({
                'Pattern': pattern.pattern_name,
                'Success Rate': pattern.success_rate * 100,
                'Avg Return': pattern.avg_return,
                'Frequency': pattern.frequency,
                'Confidence': pattern.confidence * 100
            })
        
        df = pd.DataFrame(patterns_data)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Success Rate', 'Average Return', 
                                         'Frequency', 'Confidence'))
        
        # Success Rate
        fig.add_trace(go.Bar(x=df['Pattern'], y=df['Success Rate'], 
                            name='Success Rate (%)', marker_color='green'),
                     row=1, col=1)
        
        # Average Return
        fig.add_trace(go.Bar(x=df['Pattern'], y=df['Avg Return'],
                            name='Avg Return (%)', marker_color='blue'),
                     row=1, col=2)
        
        # Frequency
        fig.add_trace(go.Bar(x=df['Pattern'], y=df['Frequency'],
                            name='Frequency', marker_color='orange'),
                     row=2, col=1)
        
        # Confidence
        fig.add_trace(go.Bar(x=df['Pattern'], y=df['Confidence'],
                            name='Confidence (%)', marker_color='purple'),
                     row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Success Patterns Analysis")
        
        fig.write_html(self.results_dir / f"success_patterns_{timestamp}.html")

    def _create_danger_conditions_chart(self, timestamp: str):
        """Dangerous conditions Visualisierung"""
        if not self.dangerous_conditions:
            return
            
        danger_data = []
        
        for condition in self.dangerous_conditions:
            danger_data.append({
                'Condition': condition.condition_name,
                'Danger Score': condition.danger_score,
                'Avg Loss': abs(condition.avg_loss),
                'Frequency': condition.frequency
            })
        
        df = pd.DataFrame(danger_data)
        
        fig = go.Figure(data=go.Scatter(
            x=df['Frequency'],
            y=df['Avg Loss'],
            mode='markers+text',
            text=df['Condition'],
            textposition="top center",
            marker=dict(
                size=df['Danger Score'] * 10,
                color=df['Danger Score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Danger Score")
            )
        ))
        
        fig.update_layout(
            title='Dangerous Market Conditions',
            xaxis_title='Frequency',
            yaxis_title='Average Loss (%)',
            template='plotly_white'
        )
        
        fig.write_html(self.results_dir / f"dangerous_conditions_{timestamp}.html")

    def _create_synergies_network_chart(self, timestamp: str):
        """Strategy synergies Netzwerk-Visualisierung"""
        if not self.strategy_synergies:
            return
        
        # Alle Strategien sammeln
        all_strategies = set()
        for synergy in self.strategy_synergies:
            all_strategies.update(synergy.strategies)
        
        strategies = list(all_strategies)
        
        # Adjacency matrix für Synergien
        n = len(strategies)
        synergy_matrix = np.zeros((n, n))
        
        for synergy in self.strategy_synergies:
            if len(synergy.strategies) == 2:
                i = strategies.index(synergy.strategies[0])
                j = strategies.index(synergy.strategies[1])
                synergy_matrix[i, j] = synergy.synergy_strength
                synergy_matrix[j, i] = synergy.synergy_strength
        
        fig = go.Figure(data=go.Heatmap(
            z=synergy_matrix,
            x=strategies,
            y=strategies,
            colorscale='Viridis',
            text=synergy_matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Strategy Synergies Matrix',
            template='plotly_white'
        )
        
        fig.write_html(self.results_dir / f"strategy_synergies_{timestamp}.html")

    def _create_feature_importance_heatmap(self, features_df: pd.DataFrame, timestamp: str):
        """Feature importance Heatmap"""
        if self.success_patterns:
            # Feature importance aus Success patterns sammeln
            all_features = set()
            for pattern in self.success_patterns:
                all_features.update(pattern.feature_importance.keys())
            
            features = sorted(list(all_features))
            importance_matrix = []
            pattern_names = []
            
            for pattern in self.success_patterns:
                pattern_names.append(pattern.pattern_name)
                importance_row = []
                for feature in features:
                    importance_row.append(pattern.feature_importance.get(feature, 0))
                importance_matrix.append(importance_row)
            
            if importance_matrix:
                fig = go.Figure(data=go.Heatmap(
                    z=importance_matrix,
                    x=features,
                    y=pattern_names,
                    colorscale='Blues',
                    text=np.array(importance_matrix),
                    texttemplate="%{text:.2f}",
                    textfont={"size": 8}
                ))
                
                fig.update_layout(
                    title='Feature Importance by Success Pattern',
                    template='plotly_white',
                    height=max(400, len(pattern_names) * 30)
                )
                
                fig.write_html(self.results_dir / f"feature_importance_{timestamp}.html")

    def _get_model_performance(self) -> Dict[str, Any]:
        """Model-Performance-Metriken"""
        performance = {}
        
        if self.success_clusterer:
            performance['success_clustering'] = {
                'n_clusters': self.success_clusterer.n_clusters,
                'inertia': float(self.success_clusterer.inertia_) if hasattr(self.success_clusterer, 'inertia_') else None
            }
        
        if self.danger_detector:
            performance['danger_detection'] = {
                'contamination': 0.1,  # Fixed parameter
                'anomaly_detection': 'isolation_forest'
            }
        
        return performance

    def _get_key_insights(self) -> List[str]:
        """Key insights aus Pattern-Analyse"""
        insights = []
        
        # Top success pattern
        if self.success_patterns:
            best_pattern = max(self.success_patterns, key=lambda x: x.avg_return)
            insights.append(f"Best success pattern: {best_pattern.pattern_name} with {best_pattern.avg_return:.2f}% avg return")
        
        # Most dangerous condition
        if self.dangerous_conditions:
            worst_condition = max(self.dangerous_conditions, key=lambda x: x.danger_score)
            insights.append(f"Most dangerous condition: {worst_condition.condition_name} with {worst_condition.danger_score:.2f} danger score")
        
        # Strongest synergy
        if self.strategy_synergies:
            best_synergy = max(self.strategy_synergies, key=lambda x: x.synergy_strength)
            insights.append(f"Strongest synergy: {'+'.join(best_synergy.strategies)} ({best_synergy.synergy_type})")
        
        return insights

    def _get_recommendations(self) -> List[str]:
        """Handlungsempfehlungen aus Pattern-Analyse"""
        recommendations = []
        
        # Aus Success patterns
        for pattern in self.success_patterns[:2]:  # Top 2
            if pattern.confidence > 0.8:
                recommendations.append(f"Replicate conditions of {pattern.pattern_name} more frequently")
        
        # Aus dangerous conditions
        for condition in self.dangerous_conditions[:2]:  # Top 2
            recommendations.extend(condition.prevention_actions[:1])  # Erste Aktion
        
        # Aus synergies
        for synergy in self.strategy_synergies[:2]:  # Top 2
            if synergy.synergy_type == "complementary":
                recommendations.append(f"Use {'+'.join(synergy.strategies)} in complementary fashion")
            elif synergy.synergy_type == "reinforcing":
                recommendations.append(f"Run {'+'.join(synergy.strategies)} simultaneously for reinforcement")
        
        return recommendations[:5]  # Top 5 recommendations

# Example usage
if __name__ == "__main__":
    # Beispiel-Daten erstellen (normalerweise aus Datenbank)
    sample_trades = pd.DataFrame({
        'trade_id': range(100),
        'strategy_name': np.random.choice(['momentum', 'mean_reversion', 'arbitrage'], 100),
        'symbol': np.random.choice(['BTC_USDT', 'ETH_USDT'], 100),
        'pnl_percentage': np.random.normal(0.5, 2, 100),
        'entry_timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'duration_minutes': np.random.randint(30, 480, 100),
        'trade_status': 'closed'
    })
    
    sample_market = pd.DataFrame({
        'symbol': np.repeat(['BTC_USDT', 'ETH_USDT'], 50),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'volatility': np.random.uniform(0.01, 0.1, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'regime': np.random.choice(['bull', 'bear', 'sideways'], 100)
    })
    
    sample_decisions = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'allocations': ['{"momentum": 0.4, "mean_reversion": 0.6}'] * 100,
        'risk_level': np.random.choice(['low', 'medium', 'high'], 100)
    })
    
    # Pattern Detector testen
    detector = PatternDetector(lookback_days=30)
    results = detector.analyze_patterns(sample_trades, sample_market, sample_decisions)
    
    print("Pattern Detection Results:")
    print(json.dumps(results, indent=2, default=str))