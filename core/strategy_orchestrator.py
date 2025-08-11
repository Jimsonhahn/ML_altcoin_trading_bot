#!/usr/bin/env python3
"""
Self-Discovering Strategy Orchestrator
======================================

Ein intelligenter Meta-Trader, der:
- Automatisch ALLE Strategien im Projekt findet
- Dynamisch ihre Fähigkeiten analysiert
- Intelligente Orchestrierung durchführt
- Kontinuierlich lernt und sich verbessert
- Mit beliebig vielen Strategien skaliert

Dieser Orchestrator weiß NICHTS über Ihre Strategien im Voraus - 
er entdeckt alles selbst und wird mit Ihrem Arsenal mitwachsen.
"""

import os
import sys
import ast
import inspect
import importlib
import importlib.util
import logging
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
import hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import statistics
from scipy import stats
import warnings

# Import learning components
from .decision_logger import DecisionLogger, OrchestratorDecision, MarketState
from analysis.learning_pipeline import LearningPipeline
from analysis.pattern_detector import PatternDetector

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

@dataclass
class StrategyDNA:
    """Genetisches Profil einer Strategie - das 'Erbgut' jeder Strategie"""
    
    # Basis-Identifikation
    name: str
    file_path: str
    class_name: str
    discovery_timestamp: datetime
    
    # Code-Analyse
    lines_of_code: int
    complexity_score: float
    dependencies: List[str]
    methods: List[str]
    async_methods: List[str]
    
    # Funktionale Eigenschaften
    risk_level: str  # 'conservative', 'moderate', 'aggressive', 'extreme'
    timeframe: str   # 'scalping', 'intraday', 'swing', 'position'
    market_conditions: List[str]  # 'trending', 'ranging', 'volatile', 'stable'
    signal_sources: List[str]     # 'technical', 'fundamental', 'sentiment', 'arbitrage'
    
    # Performance-Charakteristik
    expected_win_rate: float
    expected_return_per_trade: float
    expected_trades_per_day: float
    max_drawdown_tolerance: float
    
    # Ressourcen-Verbrauch
    cpu_intensity: float    # 0-1 (geschätzt basierend auf Code-Komplexität)
    memory_usage: float     # 0-1 (geschätzt)
    api_calls_per_hour: int # Geschätzte API-Nutzung
    
    # Lern-Eigenschaften
    adaptability_score: float    # Wie gut lernt die Strategie
    data_requirements: List[str] # Welche Daten braucht sie
    training_time: float         # Wie lange dauert Training/Setup
    
    # Markt-Kompatibilität
    supported_symbols: List[str]   # Welche Assets/Märkte
    minimum_volatility: float      # Mindest-Volatilität für Funktion
    optimal_volume_range: Tuple[float, float]  # Optimaler Volumen-Bereich
    
    # Meta-Eigenschaften (durch Beobachtung gelernt)
    cooperation_score: float       # Wie gut mit anderen Strategien
    conflict_strategies: List[str] # Strategien, die sich stören
    synergy_strategies: List[str]  # Strategien, die sich ergänzen
    
    # Performance-History (wird zur Laufzeit gefüllt)
    historical_performance: Dict[str, Any] = None
    last_performance_update: datetime = None
    confidence_level: float = 0.0  # Vertrauen in die DNA-Daten
    
    def __post_init__(self):
        if self.historical_performance is None:
            self.historical_performance = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'best_period': None,
                'worst_period': None,
                'avg_trade_duration': 0.0,
                'last_30_days_performance': []
            }

@dataclass
class StrategyHealthMetrics:
    """Gesundheits-Metriken einer Strategie"""
    
    strategy_name: str
    timestamp: datetime
    
    # Performance-Gesundheit
    recent_win_rate: float
    recent_roi: float
    performance_trend: str  # 'improving', 'stable', 'declining', 'critical'
    
    # System-Gesundheit
    execution_success_rate: float
    avg_execution_time: float
    error_rate: float
    memory_health: float
    
    # Markt-Kompatibilität
    market_adaptation_score: float
    signal_quality_score: float
    risk_compliance_score: float
    
    # Gesamt-Gesundheit
    overall_health_score: float  # 0-100
    health_status: str  # 'excellent', 'good', 'warning', 'critical', 'disabled'
    
    # Empfehlungen
    recommendations: List[str]
    required_actions: List[str]

class StrategyCodeAnalyzer:
    """Analysiert Strategie-Code und extrahiert DNA-Informationen"""
    
    def __init__(self):
        self.risk_keywords = {
            'conservative': ['conservative', 'safe', 'low_risk', 'stable', 'defensive'],
            'moderate': ['moderate', 'balanced', 'medium_risk', 'standard'],
            'aggressive': ['aggressive', 'high_risk', 'active', 'fast'],
            'extreme': ['extreme', 'maximum', 'ultra', 'turbo', 'insane', 'yolo']
        }
        
        self.timeframe_keywords = {
            'scalping': ['scalp', 'second', 'tick', 'micro', 'ultra_short'],
            'intraday': ['minute', 'hour', 'intraday', 'day_trading', 'short_term'],
            'swing': ['swing', 'daily', 'weekly', 'medium_term'],
            'position': ['position', 'monthly', 'long_term', 'hodl']
        }
        
        self.signal_keywords = {
            'technical': ['rsi', 'macd', 'bollinger', 'moving_average', 'ema', 'sma', 'stochastic'],
            'fundamental': ['news', 'earnings', 'economic', 'fundamental'],
            'sentiment': ['sentiment', 'social', 'twitter', 'reddit', 'fear_greed'],
            'arbitrage': ['arbitrage', 'spread', 'exchange', 'price_difference']
        }
        
        logger.info("🔬 Strategy Code Analyzer initialized")
    
    def analyze_strategy_file(self, file_path: Path) -> Optional[StrategyDNA]:
        """Analysiert eine Strategie-Datei und erstellt DNA-Profil"""
        
        try:
            # Code einlesen
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # AST-Analyse
            tree = ast.parse(source_code)
            
            # Basis-Informationen extrahieren
            class_info = self._extract_class_info(tree, source_code)
            if not class_info:
                return None
            
            # DNA erstellen
            dna = StrategyDNA(
                name=class_info['name'],
                file_path=str(file_path),
                class_name=class_info['class_name'],
                discovery_timestamp=datetime.now(),
                
                # Code-Analyse
                lines_of_code=len(source_code.splitlines()),
                complexity_score=self._calculate_complexity(tree),
                dependencies=self._extract_dependencies(tree),
                methods=class_info['methods'],
                async_methods=class_info['async_methods'],
                
                # Funktionale Eigenschaften (aus Code ableiten)
                risk_level=self._infer_risk_level(source_code),
                timeframe=self._infer_timeframe(source_code),
                market_conditions=self._infer_market_conditions(source_code),
                signal_sources=self._infer_signal_sources(source_code),
                
                # Performance-Schätzungen (basierend auf Code-Analyse)
                expected_win_rate=self._estimate_win_rate(source_code, class_info),
                expected_return_per_trade=self._estimate_return_per_trade(source_code),
                expected_trades_per_day=self._estimate_trade_frequency(source_code),
                max_drawdown_tolerance=self._estimate_drawdown_tolerance(source_code),
                
                # Ressourcen-Verbrauch
                cpu_intensity=self._estimate_cpu_intensity(tree, source_code),
                memory_usage=self._estimate_memory_usage(tree, source_code),
                api_calls_per_hour=self._estimate_api_usage(source_code),
                
                # Lern-Eigenschaften
                adaptability_score=self._assess_adaptability(source_code, class_info),
                data_requirements=self._extract_data_requirements(source_code),
                training_time=self._estimate_training_time(source_code),
                
                # Markt-Kompatibilität
                supported_symbols=self._extract_supported_symbols(source_code),
                minimum_volatility=self._estimate_min_volatility(source_code),
                optimal_volume_range=self._estimate_volume_range(source_code),
                
                # Meta-Eigenschaften (initial)
                cooperation_score=0.5,  # Wird durch Beobachtung gelernt
                conflict_strategies=[],
                synergy_strategies=[],
                confidence_level=0.3    # Initial niedrig, steigt mit Erfahrung
            )
            
            logger.info(f"🧬 DNA analysiert für: {dna.name} ({dna.risk_level}, {dna.timeframe})")
            return dna
            
        except Exception as e:
            logger.error(f"❌ Fehler bei DNA-Analyse von {file_path}: {e}")
            return None
    
    def _extract_class_info(self, tree: ast.AST, source_code: str) -> Optional[Dict[str, Any]]:
        """Extrahiert Klassen-Informationen aus AST"""
        
        strategy_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Prüfe auf Strategy-Vererbung
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                
                # Suche nach Strategy-ähnlichen Basisklassen
                strategy_indicators = ['strategy', 'base', 'trading', 'bot']
                is_strategy = any(
                    any(indicator in base.lower() for indicator in strategy_indicators)
                    for base in base_names
                )
                
                if is_strategy or self._has_strategy_methods(node):
                    methods = []
                    async_methods = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                        elif isinstance(item, ast.AsyncFunctionDef):
                            async_methods.append(item.name)
                    
                    strategy_classes.append({
                        'name': self._extract_strategy_name(node.name, source_code),
                        'class_name': node.name,
                        'methods': methods,
                        'async_methods': async_methods,
                        'line_number': node.lineno
                    })
        
        # Returniere die beste/erste Strategy-Klasse
        return strategy_classes[0] if strategy_classes else None
    
    def _has_strategy_methods(self, class_node: ast.ClassDef) -> bool:
        """Prüft ob Klasse Strategy-typische Methoden hat"""
        
        method_names = []
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_names.append(item.name.lower())
        
        strategy_methods = [
            'calculate_signal', 'execute_trade', 'manage_positions',
            'get_signal', 'trade', 'analyze', 'backtest'
        ]
        
        return any(method in method_names for method in strategy_methods)
    
    def _extract_strategy_name(self, class_name: str, source_code: str) -> str:
        """Extrahiert einen sauberen Strategy-Namen"""
        
        # Entferne 'Strategy' Suffix
        name = class_name.replace('Strategy', '').replace('Bot', '')
        
        # Konvertiere CamelCase zu snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name.strip('_') or class_name.lower()
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Berechnet Code-Komplexitäts-Score"""
        
        complexity = 0
        
        for node in ast.walk(tree):
            # Zyklomatische Komplexität
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                complexity += 1
        
        # Normalisiere auf 0-1 Skala
        return min(complexity / 100.0, 1.0)
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extrahiert Code-Abhängigkeiten"""
        
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        return sorted(list(dependencies))
    
    def _infer_risk_level(self, source_code: str) -> str:
        """Leitet Risk-Level aus Code ab"""
        
        code_lower = source_code.lower()
        scores = {}
        
        for risk_level, keywords in self.risk_keywords.items():
            score = sum(code_lower.count(keyword) for keyword in keywords)
            
            # Zusätzliche Heuristiken
            if 'daily_budget' in code_lower and '30' in source_code:
                score += 5 if risk_level == 'extreme' else 0
            if 'stop_loss' in code_lower:
                score += 3 if risk_level in ['conservative', 'moderate'] else 0
            if 'max_position' in code_lower:
                score += 2 if risk_level in ['conservative', 'moderate'] else 0
            
            scores[risk_level] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'moderate'
    
    def _infer_timeframe(self, source_code: str) -> str:
        """Leitet Timeframe aus Code ab"""
        
        code_lower = source_code.lower()
        scores = {}
        
        for timeframe, keywords in self.timeframe_keywords.items():
            score = sum(code_lower.count(keyword) for keyword in keywords)
            scores[timeframe] = score
        
        # Zusätzliche Heuristiken
        if 'timeout_hours' in code_lower:
            if '1' in source_code or '2' in source_code:
                scores['scalping'] += 3
            elif '6' in source_code or '8' in source_code:
                scores['intraday'] += 3
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'intraday'
    
    def _infer_market_conditions(self, source_code: str) -> List[str]:
        """Leitet optimale Marktbedingungen ab"""
        
        conditions = []
        code_lower = source_code.lower()
        
        if any(word in code_lower for word in ['trend', 'momentum', 'breakout']):
            conditions.append('trending')
        if any(word in code_lower for word in ['range', 'sideways', 'mean_reversion']):
            conditions.append('ranging')
        if any(word in code_lower for word in ['volatility', 'spike', 'volume']):
            conditions.append('volatile')
        if any(word in code_lower for word in ['stable', 'low_vol', 'calm']):
            conditions.append('stable')
        
        return conditions if conditions else ['trending', 'ranging']  # Default
    
    def _infer_signal_sources(self, source_code: str) -> List[str]:
        """Erkennt Signal-Quellen der Strategie"""
        
        sources = []
        code_lower = source_code.lower()
        
        for source_type, keywords in self.signal_keywords.items():
            if any(keyword in code_lower for keyword in keywords):
                sources.append(source_type)
        
        # Zusätzliche Erkennung
        if 'ml' in code_lower or 'machine_learning' in code_lower or 'predict' in code_lower:
            sources.append('ml')
        if 'news' in code_lower or 'sentiment' in code_lower:
            sources.append('sentiment')
        
        return sources if sources else ['technical']  # Default
    
    def _estimate_win_rate(self, source_code: str, class_info: Dict) -> float:
        """Schätzt erwartete Gewinnrate"""
        
        base_rate = 0.55  # Basis-Annahme
        
        # Adjustierungen basierend auf Code-Eigenschaften
        if 'ml' in source_code.lower():
            base_rate += 0.1  # ML oft besser
        if 'arbitrage' in source_code.lower():
            base_rate += 0.15  # Arbitrage sehr gut
        if len(class_info.get('methods', [])) > 10:
            base_rate += 0.05  # Komplexere Strategien oft besser
        if 'risk_management' in source_code.lower():
            base_rate += 0.05  # Risk Management hilft
        
        return min(base_rate, 0.85)  # Cap bei 85%
    
    def _estimate_return_per_trade(self, source_code: str) -> float:
        """Schätzt Return pro Trade"""
        
        base_return = 0.02  # 2% Basis
        
        # Heuristiken aus Code
        if 'high_risk' in source_code.lower():
            base_return *= 2
        if 'arbitrage' in source_code.lower():
            base_return *= 0.5  # Niedriger aber konsistenter
        if 'scalping' in source_code.lower():
            base_return *= 0.3  # Sehr kleine Gewinne
        
        return base_return
    
    def _estimate_trade_frequency(self, source_code: str) -> float:
        """Schätzt Trades pro Tag"""
        
        base_frequency = 1.0  # 1 Trade/Tag Basis
        
        if 'scalping' in source_code.lower():
            base_frequency *= 10
        elif 'intraday' in source_code.lower():
            base_frequency *= 3
        elif 'swing' in source_code.lower():
            base_frequency *= 0.3
        
        if 'high_risk' in source_code.lower():
            base_frequency *= 2
        
        return base_frequency
    
    def _estimate_drawdown_tolerance(self, source_code: str) -> float:
        """Schätzt maximale Drawdown-Toleranz"""
        
        if 'conservative' in source_code.lower():
            return 0.1  # 10%
        elif 'aggressive' in source_code.lower():
            return 0.3  # 30%
        elif 'extreme' in source_code.lower():
            return 0.5  # 50%
        else:
            return 0.2  # 20% Default
    
    def _estimate_cpu_intensity(self, tree: ast.AST, source_code: str) -> float:
        """Schätzt CPU-Intensität"""
        
        intensity = 0.0
        
        # Zähle rechenintensive Operationen
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                intensity += 0.1
            elif isinstance(node, ast.While):
                intensity += 0.15
            elif isinstance(node, ast.ListComp):
                intensity += 0.05
        
        # ML = sehr CPU-intensiv
        if 'ml' in source_code.lower() or 'machine_learning' in source_code.lower():
            intensity += 0.4
        
        # Viele API-Calls = CPU-Last
        if source_code.lower().count('api') > 5:
            intensity += 0.2
        
        return min(intensity, 1.0)
    
    def _estimate_memory_usage(self, tree: ast.AST, source_code: str) -> float:
        """Schätzt Speicher-Verbrauch"""
        
        usage = 0.1  # Basis-Verbrauch
        
        # DataFrame-Operationen
        if 'dataframe' in source_code.lower() or 'pd.' in source_code:
            usage += 0.3
        
        # Große Listen/Dicts
        list_count = source_code.count('[')
        dict_count = source_code.count('{')
        usage += (list_count + dict_count) * 0.01
        
        # ML-Modelle
        if 'model' in source_code.lower():
            usage += 0.2
        
        return min(usage, 1.0)
    
    def _estimate_api_usage(self, source_code: str) -> int:
        """Schätzt API-Calls pro Stunde"""
        
        api_calls = 0
        
        # Zähle potentielle API-Calls
        api_indicators = ['api', 'request', 'fetch', 'get_', 'post_']
        for indicator in api_indicators:
            api_calls += source_code.lower().count(indicator) * 10
        
        # Social Media APIs sind sehr häufig
        if 'twitter' in source_code.lower() or 'reddit' in source_code.lower():
            api_calls += 100
        
        # News APIs
        if 'news' in source_code.lower():
            api_calls += 50
        
        return min(api_calls, 1000)  # Max 1000/Stunde
    
    def _assess_adaptability(self, source_code: str, class_info: Dict) -> float:
        """Bewertet Lernfähigkeit der Strategie"""
        
        adaptability = 0.3  # Basis
        
        # ML-Strategien sind adaptiv
        if 'ml' in source_code.lower() or 'learn' in source_code.lower():
            adaptability += 0.4
        
        # Viele Parameter = anpassbar
        param_count = source_code.count('self.') + source_code.count('config')
        adaptability += min(param_count * 0.01, 0.3)
        
        # Feedback-Mechanismen
        if 'feedback' in source_code.lower() or 'update' in source_code.lower():
            adaptability += 0.2
        
        return min(adaptability, 1.0)
    
    def _extract_data_requirements(self, source_code: str) -> List[str]:
        """Extrahiert Daten-Anforderungen"""
        
        requirements = []
        
        if 'ohlcv' in source_code.lower() or 'candlestick' in source_code.lower():
            requirements.append('ohlcv')
        if 'order_book' in source_code.lower():
            requirements.append('orderbook')
        if 'news' in source_code.lower():
            requirements.append('news')
        if 'social' in source_code.lower() or 'sentiment' in source_code.lower():
            requirements.append('sentiment')
        if 'fundamental' in source_code.lower():
            requirements.append('fundamental')
        
        return requirements if requirements else ['ohlcv']
    
    def _estimate_training_time(self, source_code: str) -> float:
        """Schätzt Setup/Training-Zeit in Minuten"""
        
        if 'ml' in source_code.lower():
            return 30.0  # ML braucht Training
        elif len(source_code) > 10000:  # Große Strategien
            return 10.0
        else:
            return 2.0  # Quick setup
    
    def _extract_supported_symbols(self, source_code: str) -> List[str]:
        """Extrahiert unterstützte Trading-Paare"""
        
        symbols = []
        
        # Suche nach Symbol-Patterns
        import re
        
        # Crypto-Pairs
        crypto_patterns = [
            r'BTC[/\-_]USDT?', r'ETH[/\-_]USDT?', r'SOL[/\-_]USDT?',
            r'ADA[/\-_]USDT?', r'DOT[/\-_]USDT?', r'AVAX[/\-_]USDT?'
        ]
        
        for pattern in crypto_patterns:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            symbols.extend(matches)
        
        # Default falls nichts gefunden
        return symbols if symbols else ['BTC/USDT', 'ETH/USDT']
    
    def _estimate_min_volatility(self, source_code: str) -> float:
        """Schätzt minimale Volatilität für Strategie"""
        
        if 'scalping' in source_code.lower():
            return 0.005  # 0.5% minimal
        elif 'arbitrage' in source_code.lower():
            return 0.001  # Sehr niedrig
        elif 'swing' in source_code.lower():
            return 0.02   # 2% minimal
        else:
            return 0.01   # 1% default
    
    def _estimate_volume_range(self, source_code: str) -> Tuple[float, float]:
        """Schätzt optimalen Volumen-Bereich"""
        
        if 'arbitrage' in source_code.lower():
            return (1000000, 10000000)  # Hohe Liquidität nötig
        elif 'scalping' in source_code.lower():
            return (500000, 5000000)    # Mittlere Liquidität
        else:
            return (100000, 2000000)    # Standard

class StrategyDiscoveryEngine:
    """Haupt-Discovery-Engine: Findet und analysiert alle Strategien"""
    
    def __init__(self, strategies_dir: str = "strategies"):
        self.strategies_dir = Path(strategies_dir)
        self.code_analyzer = StrategyCodeAnalyzer()
        
        # Discovery-Datenbank
        self.discovered_strategies: Dict[str, StrategyDNA] = {}
        self.strategy_instances: Dict[str, Any] = {}
        self.discovery_history: List[Dict[str, Any]] = []
        
        # Discovery-State
        self.last_discovery_run = None
        self.discovery_running = False
        self.auto_discovery_enabled = True
        
        # Performance-Tracking
        self.strategy_performance: Dict[str, StrategyHealthMetrics] = {}
        
        logger.info(f"🔍 Strategy Discovery Engine initialized (scanning: {self.strategies_dir})")
    
    async def discover_all_strategies(self, force_refresh: bool = False) -> Dict[str, StrategyDNA]:
        """Entdeckt ALLE Strategien im Projekt - Das Herzstück!"""
        
        if self.discovery_running and not force_refresh:
            logger.info("🔄 Discovery bereits aktiv, überspringe...")
            return self.discovered_strategies
        
        self.discovery_running = True
        discovery_start = datetime.now()
        
        logger.info("🔍 STARTE VOLLSTÄNDIGE STRATEGY-DISCOVERY")
        logger.info("=" * 50)
        
        try:
            # 1. Scan filesystem für Python-Dateien
            strategy_files = await self._scan_strategy_files()
            logger.info(f"📁 Gefunden: {len(strategy_files)} Python-Dateien")
            
            # 2. Analysiere jede Datei parallel
            new_strategies = await self._analyze_strategy_files_parallel(strategy_files)
            logger.info(f"🧬 Analysiert: {len(new_strategies)} Strategien")
            
            # 3. Update Discovery-Datenbank
            for strategy_name, dna in new_strategies.items():
                old_dna = self.discovered_strategies.get(strategy_name)
                
                if old_dna is None:
                    logger.info(f"✨ NEUE STRATEGIE ENTDECKT: {strategy_name}")
                    logger.info(f"   📊 Risk: {dna.risk_level}, Timeframe: {dna.timeframe}")
                    logger.info(f"   🎯 Signals: {', '.join(dna.signal_sources)}")
                    logger.info(f"   ⚡ Complexity: {dna.complexity_score:.2f}")
                elif old_dna.file_path != dna.file_path or force_refresh:
                    logger.info(f"🔄 STRATEGIE AKTUALISIERT: {strategy_name}")
                    # Bewahre Performance-Historie
                    dna.historical_performance = old_dna.historical_performance
                    dna.confidence_level = old_dna.confidence_level
                
                self.discovered_strategies[strategy_name] = dna
            
            # 4. Entferne nicht mehr existierende Strategien
            removed_strategies = []
            for strategy_name in list(self.discovered_strategies.keys()):
                dna = self.discovered_strategies[strategy_name]
                if not Path(dna.file_path).exists():
                    removed_strategies.append(strategy_name)
                    del self.discovered_strategies[strategy_name]
                    logger.info(f"🗑️ STRATEGIE ENTFERNT: {strategy_name}")
            
            # 5. Lerne Strategie-Beziehungen
            await self._analyze_strategy_relationships()
            
            # 6. Discovery-History aktualisieren
            discovery_end = datetime.now()
            discovery_duration = (discovery_end - discovery_start).total_seconds()
            
            discovery_record = {
                'timestamp': discovery_end,
                'duration_seconds': discovery_duration,
                'strategies_found': len(self.discovered_strategies),
                'new_strategies': len(new_strategies),
                'removed_strategies': len(removed_strategies),
                'files_scanned': len(strategy_files)
            }
            
            self.discovery_history.append(discovery_record)
            self.last_discovery_run = discovery_end
            
            # 7. Zeige Discovery-Zusammenfassung
            await self._display_discovery_summary(discovery_record)
            
            return self.discovered_strategies
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategy Discovery: {e}")
            return self.discovered_strategies
            
        finally:
            self.discovery_running = False
    
    async def _scan_strategy_files(self) -> List[Path]:
        """Scannt Filesystem nach Strategie-Dateien"""
        
        strategy_files = []
        
        if not self.strategies_dir.exists():
            logger.warning(f"⚠️ Strategies-Verzeichnis nicht gefunden: {self.strategies_dir}")
            return strategy_files
        
        # Rekursiver Scan nach .py Dateien
        for file_path in self.strategies_dir.rglob("*.py"):
            # Skip __init__.py und andere System-Dateien
            if file_path.name.startswith('__'):
                continue
            if file_path.name.startswith('.'):
                continue
            if 'test_' in file_path.name:
                continue
            
            strategy_files.append(file_path)
        
        return sorted(strategy_files)
    
    async def _analyze_strategy_files_parallel(self, strategy_files: List[Path]) -> Dict[str, StrategyDNA]:
        """Analysiert Strategie-Dateien parallel für Performance"""
        
        strategies = {}
        
        # Parallel-Analyse mit ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Starte alle Analysen
            future_to_file = {
                executor.submit(self.code_analyzer.analyze_strategy_file, file_path): file_path
                for file_path in strategy_files
            }
            
            # Sammle Ergebnisse
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    dna = future.result()
                    if dna:
                        strategies[dna.name] = dna
                except Exception as e:
                    logger.error(f"❌ Fehler bei Analyse von {file_path}: {e}")
        
        return strategies
    
    async def _analyze_strategy_relationships(self):
        """Analysiert Beziehungen zwischen Strategien"""
        
        logger.info("🔗 Analysiere Strategie-Beziehungen...")
        
        strategy_names = list(self.discovered_strategies.keys())
        
        for strategy_name in strategy_names:
            dna = self.discovered_strategies[strategy_name]
            
            # Finde potentielle Konflikte
            conflicts = []
            synergies = []
            
            for other_name in strategy_names:
                if other_name == strategy_name:
                    continue
                
                other_dna = self.discovered_strategies[other_name]
                
                # Konflikt-Erkennung
                if (dna.risk_level == other_dna.risk_level == 'extreme' and
                    dna.timeframe == other_dna.timeframe):
                    conflicts.append(other_name)
                
                # Synergie-Erkennung
                if (set(dna.signal_sources) & set(other_dna.signal_sources) and
                    dna.timeframe != other_dna.timeframe):
                    synergies.append(other_name)
            
            # Update DNA
            dna.conflict_strategies = conflicts
            dna.synergy_strategies = synergies
            
            # Cooperation Score basierend auf Synergien
            dna.cooperation_score = min(len(synergies) / max(len(strategy_names) - 1, 1), 1.0)
    
    async def _display_discovery_summary(self, discovery_record: Dict[str, Any]):
        """Zeigt Discovery-Zusammenfassung"""
        
        print(f"\n🎯 STRATEGY DISCOVERY ABGESCHLOSSEN")
        print(f"=" * 40)
        print(f"⏱️  Dauer: {discovery_record['duration_seconds']:.1f}s")
        print(f"📊 Strategien gefunden: {discovery_record['strategies_found']}")
        print(f"✨ Neue Strategien: {discovery_record['new_strategies']}")
        print(f"🗑️ Entfernte Strategien: {discovery_record['removed_strategies']}")
        print(f"📁 Dateien gescannt: {discovery_record['files_scanned']}")
        
        if self.discovered_strategies:
            print(f"\n🧬 ENTDECKTE STRATEGIEN:")
            print(f"-" * 25)
            
            # Gruppiere nach Risk Level
            risk_groups = defaultdict(list)
            for name, dna in self.discovered_strategies.items():
                risk_groups[dna.risk_level].append((name, dna))
            
            for risk_level in ['conservative', 'moderate', 'aggressive', 'extreme']:
                if risk_level in risk_groups:
                    strategies = risk_groups[risk_level]
                    print(f"\n{risk_level.upper()} ({len(strategies)}):")
                    
                    for name, dna in strategies:
                        signals_str = ', '.join(dna.signal_sources[:3])
                        if len(dna.signal_sources) > 3:
                            signals_str += f" (+{len(dna.signal_sources) - 3} more)"
                        
                        print(f"  📈 {name}")
                        print(f"     ⏰ {dna.timeframe} | 🎯 {signals_str}")
                        print(f"     ⚡ Complexity: {dna.complexity_score:.2f} | 🤝 Cooperation: {dna.cooperation_score:.2f}")
                        
                        if dna.synergy_strategies:
                            synergies = ', '.join(dna.synergy_strategies[:2])
                            if len(dna.synergy_strategies) > 2:
                                synergies += f" (+{len(dna.synergy_strategies) - 2})"
                            print(f"     🤝 Synergies: {synergies}")
    
    def get_strategies_by_criteria(self, risk_level: str = None, timeframe: str = None,
                                  signal_sources: List[str] = None, 
                                  min_cooperation_score: float = None) -> List[StrategyDNA]:
        """Filtert Strategien nach Kriterien"""
        
        filtered = []
        
        for dna in self.discovered_strategies.values():
            # Risk Level Filter
            if risk_level and dna.risk_level != risk_level:
                continue
            
            # Timeframe Filter
            if timeframe and dna.timeframe != timeframe:
                continue
            
            # Signal Sources Filter
            if signal_sources and not any(source in dna.signal_sources for source in signal_sources):
                continue
            
            # Cooperation Score Filter
            if min_cooperation_score and dna.cooperation_score < min_cooperation_score:
                continue
            
            filtered.append(dna)
        
        return sorted(filtered, key=lambda x: x.confidence_level, reverse=True)
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Gibt Discovery-Statistiken zurück"""
        
        if not self.discovered_strategies:
            return {'total_strategies': 0}
        
        # Basis-Statistiken
        total_strategies = len(self.discovered_strategies)
        
        # Gruppierungen
        by_risk = defaultdict(int)
        by_timeframe = defaultdict(int)
        by_complexity = {'low': 0, 'medium': 0, 'high': 0}
        
        total_cooperation = 0
        
        for dna in self.discovered_strategies.values():
            by_risk[dna.risk_level] += 1
            by_timeframe[dna.timeframe] += 1
            
            if dna.complexity_score < 0.3:
                by_complexity['low'] += 1
            elif dna.complexity_score < 0.7:
                by_complexity['medium'] += 1
            else:
                by_complexity['high'] += 1
            
            total_cooperation += dna.cooperation_score
        
        avg_cooperation = total_cooperation / total_strategies if total_strategies > 0 else 0
        
        return {
            'total_strategies': total_strategies,
            'last_discovery': self.last_discovery_run.isoformat() if self.last_discovery_run else None,
            'by_risk_level': dict(by_risk),
            'by_timeframe': dict(by_timeframe),
            'by_complexity': dict(by_complexity),
            'average_cooperation_score': avg_cooperation,
            'discovery_runs': len(self.discovery_history)
        }
    
    async def emergency_discovery(self, file_path: str) -> Optional[StrategyDNA]:
        """Notfall-Discovery: Lädt einzelne Strategie zur Laufzeit"""
        
        logger.info(f"🚨 Emergency Discovery für: {file_path}")
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error(f"❌ Datei nicht gefunden: {file_path}")
                return None
            
            # Analysiere einzelne Datei
            dna = self.code_analyzer.analyze_strategy_file(file_path_obj)
            
            if dna:
                # Zur Discovery-Datenbank hinzufügen
                self.discovered_strategies[dna.name] = dna
                logger.info(f"✅ Emergency Discovery erfolgreich: {dna.name}")
                
                # Beziehungen neu analysieren
                await self._analyze_strategy_relationships()
                
                return dna
            else:
                logger.error(f"❌ Konnte keine gültige Strategie in {file_path} finden")
                return None
                
        except Exception as e:
            logger.error(f"❌ Emergency Discovery fehlgeschlagen: {e}")
            return None
    
    def save_discovery_database(self, file_path: str = "strategy_discovery_db.json"):
        """Speichert Discovery-Datenbank"""
        
        try:
            db_data = {
                'timestamp': datetime.now().isoformat(),
                'strategies': {},
                'discovery_history': self.discovery_history
            }
            
            # Strategien serialisieren
            for name, dna in self.discovered_strategies.items():
                dna_dict = asdict(dna)
                # DateTime zu String konvertieren
                dna_dict['discovery_timestamp'] = dna.discovery_timestamp.isoformat()
                if dna.last_performance_update:
                    dna_dict['last_performance_update'] = dna.last_performance_update.isoformat()
                
                db_data['strategies'][name] = dna_dict
            
            with open(file_path, 'w') as f:
                json.dump(db_data, f, indent=2)
            
            logger.info(f"💾 Discovery-Datenbank gespeichert: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Discovery-Datenbank: {e}")
    
    def load_discovery_database(self, file_path: str = "strategy_discovery_db.json") -> bool:
        """Lädt Discovery-Datenbank"""
        
        try:
            if not Path(file_path).exists():
                logger.info(f"📄 Discovery-Datenbank nicht gefunden: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                db_data = json.load(f)
            
            # Strategien deserialisieren
            self.discovered_strategies = {}
            
            for name, dna_dict in db_data.get('strategies', {}).items():
                # DateTime von String konvertieren
                dna_dict['discovery_timestamp'] = datetime.fromisoformat(dna_dict['discovery_timestamp'])
                if dna_dict.get('last_performance_update'):
                    dna_dict['last_performance_update'] = datetime.fromisoformat(dna_dict['last_performance_update'])
                
                dna = StrategyDNA(**dna_dict)
                self.discovered_strategies[name] = dna
            
            # History laden
            self.discovery_history = db_data.get('discovery_history', [])
            
            logger.info(f"📂 Discovery-Datenbank geladen: {len(self.discovered_strategies)} Strategien")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Discovery-Datenbank: {e}")
            return False

@dataclass 
class DecisionContext:
    """Kontext für eine Orchestrator-Entscheidung"""
    timestamp: datetime
    market_regime: str
    volatility_level: float
    portfolio_value: float
    risk_score: float
    active_strategies: List[str]
    market_indicators: Dict[str, float]
    ml_insights: Dict[str, Any]
    emergency_patterns: List[str]
    confidence_factors: Dict[str, float]

@dataclass
class EmergencyPattern:
    """Notfall-Muster Definition"""
    pattern_id: str
    pattern_name: str
    trigger_conditions: Dict[str, Any]
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    affected_strategies: List[str]
    emergency_actions: List[str]
    confidence_threshold: float
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class LearningOrchestrator:
    """
    Intelligenter Strategy Orchestrator mit Machine Learning Fähigkeiten
    
    Features:
    - DecisionLogger Integration für vollständige Rückverfolgbarkeit
    - Adaptive Learning basierend auf historischer Performance
    - Emergency Pattern Detection für Risikomanagement
    - Explain-Funktionalität für Transparenz
    - Integration mit Learning Pipeline
    """
    
    def __init__(self, 
                 db_pool,
                 discovery_engine: StrategyDiscoveryEngine,
                 decision_logger: DecisionLogger,
                 learning_pipeline: Optional[LearningPipeline] = None,
                 pattern_detector: Optional[PatternDetector] = None):
        """
        Initialize Learning Orchestrator
        
        Args:
            db_pool: Database connection pool
            discovery_engine: Strategy discovery engine
            decision_logger: Decision logging system
            learning_pipeline: Learning pipeline for insights
            pattern_detector: Pattern detection system
        """
        self.db_pool = db_pool
        self.discovery_engine = discovery_engine
        self.decision_logger = decision_logger
        self.learning_pipeline = learning_pipeline
        self.pattern_detector = pattern_detector
        
        # Learning state
        self.strategy_weights: Dict[str, float] = {}
        self.historical_performance: Dict[str, List[float]] = {}
        self.ml_insights: Dict[str, Any] = {}
        self.emergency_patterns: Dict[str, EmergencyPattern] = {}
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.confidence_decay = 0.95
        self.performance_lookback_days = 30
        self.emergency_threshold = 0.8
        
        # Decision history
        self.decision_history: deque = deque(maxlen=1000)
        self.last_performance_update = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("🧠 Learning Orchestrator initialized")

    async def initialize(self):
        """Initialize orchestrator with historical data"""
        logger.info("🚀 Initializing Learning Orchestrator...")
        
        try:
            # Load historical performance
            await self._load_historical_performance()
            
            # Initialize strategy weights
            await self._initialize_strategy_weights()
            
            # Load ML insights
            await self._load_ml_insights()
            
            # Setup emergency patterns
            await self._setup_emergency_patterns()
            
            logger.info("✅ Learning Orchestrator ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Learning Orchestrator: {e}")
            raise

    async def make_allocation_decision(self, 
                                     market_data: Dict[str, Any],
                                     portfolio_state: Dict[str, Any],
                                     strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make intelligent allocation decision with full logging and learning
        
        Args:
            market_data: Current market conditions
            portfolio_state: Current portfolio state
            strategy_signals: Signals from all strategies
            
        Returns:
            Allocation decision with explanations
        """
        start_time = datetime.utcnow()
        
        try:
            # Create decision context
            context = DecisionContext(
                timestamp=start_time,
                market_regime=market_data.get('regime', 'unknown'),
                volatility_level=market_data.get('volatility', 0.0),
                portfolio_value=portfolio_state.get('total_value', 0.0),
                risk_score=portfolio_state.get('risk_score', 0.0),
                active_strategies=list(strategy_signals.keys()),
                market_indicators=market_data.get('indicators', {}),
                ml_insights=self.ml_insights,
                emergency_patterns=[],
                confidence_factors={}
            )
            
            # 1. Check for emergency patterns FIRST
            emergency_detected = await self._check_emergency_patterns(context, strategy_signals)
            if emergency_detected:
                logger.warning("🚨 Emergency pattern detected - applying emergency allocations")
                return await self._handle_emergency_allocation(context, emergency_detected)
            
            # 2. Adaptive learning adjustment
            adapted_weights = await self._apply_adaptive_learning(context, strategy_signals)
            
            # 3. Apply ML insights
            ml_adjusted_weights = await self._apply_ml_insights(adapted_weights, context)
            
            # 4. Final allocation decision
            final_allocation = await self._make_final_allocation(ml_adjusted_weights, context)
            
            # 5. Log decision
            await self._log_allocation_decision(context, final_allocation)
            
            # 6. Store for learning
            with self._lock:
                self.decision_history.append({
                    'timestamp': start_time,
                    'context': asdict(context),
                    'allocation': final_allocation,
                    'reasoning': final_allocation.get('reasoning', [])
                })
            
            return final_allocation
            
        except Exception as e:
            logger.error(f"❌ Allocation decision failed: {e}")
            # Emergency fallback
            return await self._emergency_fallback_allocation(strategy_signals)

    async def _load_historical_performance(self):
        """Load historical performance data for learning"""
        logger.info("📊 Loading historical performance data...")
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.performance_lookback_days)
            
            async with self.db_pool.acquire() as conn:
                # Load strategy performance
                performance_query = """
                    SELECT strategy_name, pnl_percentage, timestamp, trade_status
                    FROM strategy_performance 
                    WHERE timestamp >= $1 AND trade_status = 'closed'
                    ORDER BY timestamp DESC
                """
                
                rows = await conn.fetch(performance_query, cutoff_date)
                
                # Group by strategy
                performance_by_strategy = defaultdict(list)
                for row in rows:
                    strategy_name = row['strategy_name']
                    pnl = float(row['pnl_percentage']) if row['pnl_percentage'] else 0.0
                    performance_by_strategy[strategy_name].append(pnl)
                
                # Store historical performance
                with self._lock:
                    self.historical_performance = dict(performance_by_strategy)
                
                logger.info(f"Loaded performance for {len(self.historical_performance)} strategies")
                
        except Exception as e:
            logger.error(f"Failed to load historical performance: {e}")
            self.historical_performance = {}

    async def _initialize_strategy_weights(self):
        """Initialize strategy weights based on historical performance and DNA"""
        logger.info("⚖️ Initializing strategy weights...")
        
        try:
            discovered_strategies = self.discovery_engine.discovered_strategies
            
            if not discovered_strategies:
                logger.warning("No strategies discovered yet")
                return
            
            weights = {}
            total_strategies = len(discovered_strategies)
            
            for strategy_name, dna in discovered_strategies.items():
                # Base weight from equal allocation
                base_weight = 1.0 / total_strategies
                
                # Adjust based on historical performance
                historical_returns = self.historical_performance.get(strategy_name, [])
                if historical_returns:
                    avg_return = statistics.mean(historical_returns)
                    return_volatility = statistics.stdev(historical_returns) if len(historical_returns) > 1 else 1.0
                    
                    # Sharpe-like adjustment
                    risk_adjusted_return = avg_return / max(return_volatility, 0.1)
                    performance_multiplier = 1 + (risk_adjusted_return * 0.1)  # Conservative adjustment
                else:
                    # Use DNA expectations for new strategies
                    expected_return = dna.expected_return_per_trade * dna.expected_trades_per_day
                    performance_multiplier = 1 + (expected_return * 0.05)
                
                # Adjust based on cooperation score
                cooperation_multiplier = 0.8 + (dna.cooperation_score * 0.4)  # 0.8 to 1.2 range
                
                # Final weight
                final_weight = base_weight * performance_multiplier * cooperation_multiplier
                weights[strategy_name] = max(0.01, min(final_weight, 0.5))  # Constrain between 1% and 50%
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            with self._lock:
                self.strategy_weights = weights
            
            logger.info(f"Initialized weights for {len(weights)} strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy weights: {e}")
            # Fallback to equal weights
            strategies = list(self.discovery_engine.discovered_strategies.keys())
            if strategies:
                equal_weight = 1.0 / len(strategies)
                self.strategy_weights = {s: equal_weight for s in strategies}

    async def _load_ml_insights(self):
        """Load latest ML insights from learning pipeline"""
        logger.info("🤖 Loading ML insights...")
        
        try:
            # Try to load latest insights from file
            insights_dir = Path("analysis/results")
            if insights_dir.exists():
                # Find latest insights files
                insight_files = list(insights_dir.glob("strategy_insights_*.json"))
                combination_files = list(insights_dir.glob("combination_analyses_*.json"))
                pattern_files = list(insights_dir.glob("success_patterns_*.json"))
                
                insights = {}
                
                # Load strategy insights
                if insight_files:
                    latest_insights = max(insight_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_insights) as f:
                        insights['strategy_insights'] = json.load(f)
                
                # Load combination analyses
                if combination_files:
                    latest_combinations = max(combination_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_combinations) as f:
                        insights['combinations'] = json.load(f)
                
                # Load success patterns
                if pattern_files:
                    latest_patterns = max(pattern_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_patterns) as f:
                        insights['success_patterns'] = json.load(f)
                
                with self._lock:
                    self.ml_insights = insights
                
                logger.info(f"Loaded ML insights: {list(insights.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load ML insights: {e}")
            self.ml_insights = {}

    async def _setup_emergency_patterns(self):
        """Setup emergency pattern detection"""
        logger.info("🚨 Setting up emergency patterns...")
        
        patterns = {
            'consecutive_losses': EmergencyPattern(
                pattern_id='consecutive_losses',
                pattern_name='Consecutive Strategy Losses',
                trigger_conditions={
                    'consecutive_negative_returns': 3,
                    'total_loss_threshold': -10.0  # 10% total loss
                },
                severity_level='high',
                affected_strategies=[],
                emergency_actions=['reduce_allocation', 'pause_strategy', 'increase_monitoring'],
                confidence_threshold=0.8
            ),
            
            'market_crash': EmergencyPattern(
                pattern_id='market_crash',
                pattern_name='Market Crash Detection',
                trigger_conditions={
                    'volatility_spike': 3.0,  # 3x normal volatility
                    'price_drop_threshold': -15.0,  # 15% drop
                    'volume_spike': 2.0  # 2x normal volume
                },
                severity_level='critical',
                affected_strategies=[],
                emergency_actions=['emergency_stop', 'liquidate_positions', 'cash_mode'],
                confidence_threshold=0.9
            ),
            
            'strategy_malfunction': EmergencyPattern(
                pattern_id='strategy_malfunction',
                pattern_name='Strategy Malfunction',
                trigger_conditions={
                    'error_rate_threshold': 0.5,  # 50% error rate
                    'execution_time_spike': 10.0,  # 10x normal execution time
                    'signal_anomaly': True
                },
                severity_level='medium',
                affected_strategies=[],
                emergency_actions=['disable_strategy', 'diagnostic_mode', 'alert_admin'],
                confidence_threshold=0.7
            ),
            
            'correlation_breakdown': EmergencyPattern(
                pattern_id='correlation_breakdown',
                pattern_name='Strategy Correlation Breakdown',
                trigger_conditions={
                    'correlation_drop': -0.8,  # Sudden correlation drop
                    'simultaneous_losses': True,
                    'diversification_failure': 0.9  # 90% strategies losing
                },
                severity_level='high',
                affected_strategies=[],
                emergency_actions=['rebalance_portfolio', 'increase_diversification', 'reduce_risk'],
                confidence_threshold=0.8
            )
        }
        
        with self._lock:
            self.emergency_patterns = patterns
        
        logger.info(f"Setup {len(patterns)} emergency patterns")

    async def _check_emergency_patterns(self, context: DecisionContext, 
                                       strategy_signals: Dict[str, Any]) -> Optional[EmergencyPattern]:
        """Check for emergency patterns in current context"""
        
        try:
            for pattern_id, pattern in self.emergency_patterns.items():
                
                # Check consecutive losses pattern
                if pattern_id == 'consecutive_losses':
                    if await self._check_consecutive_losses_pattern(context, pattern):
                        pattern.last_triggered = datetime.utcnow()
                        pattern.trigger_count += 1
                        context.emergency_patterns.append(pattern_id)
                        return pattern
                
                # Check market crash pattern
                elif pattern_id == 'market_crash':
                    if await self._check_market_crash_pattern(context, pattern):
                        pattern.last_triggered = datetime.utcnow()
                        pattern.trigger_count += 1
                        context.emergency_patterns.append(pattern_id)
                        return pattern
                
                # Check strategy malfunction pattern
                elif pattern_id == 'strategy_malfunction':
                    if await self._check_strategy_malfunction_pattern(context, strategy_signals, pattern):
                        pattern.last_triggered = datetime.utcnow()
                        pattern.trigger_count += 1
                        context.emergency_patterns.append(pattern_id)
                        return pattern
                
                # Check correlation breakdown pattern
                elif pattern_id == 'correlation_breakdown':
                    if await self._check_correlation_breakdown_pattern(context, pattern):
                        pattern.last_triggered = datetime.utcnow()
                        pattern.trigger_count += 1
                        context.emergency_patterns.append(pattern_id)
                        return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check emergency patterns: {e}")
            return None

    async def _check_consecutive_losses_pattern(self, context: DecisionContext, 
                                              pattern: EmergencyPattern) -> bool:
        """Check for consecutive losses pattern"""
        try:
            # Get recent performance for active strategies
            for strategy_name in context.active_strategies:
                recent_returns = self.historical_performance.get(strategy_name, [])
                
                if len(recent_returns) >= 3:
                    # Check last 3 trades
                    last_3_returns = recent_returns[:3]
                    
                    # All negative?
                    all_negative = all(r < 0 for r in last_3_returns)
                    total_loss = sum(last_3_returns)
                    
                    if all_negative and total_loss < pattern.trigger_conditions['total_loss_threshold']:
                        pattern.affected_strategies.append(strategy_name)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check consecutive losses: {e}")
            return False

    async def _check_market_crash_pattern(self, context: DecisionContext, 
                                         pattern: EmergencyPattern) -> bool:
        """Check for market crash pattern"""
        try:
            conditions = pattern.trigger_conditions
            
            # Check volatility spike
            normal_volatility = 0.02  # Assume 2% normal volatility
            volatility_spike = context.volatility_level / normal_volatility
            
            if volatility_spike > conditions['volatility_spike']:
                # Additional checks for price drop and volume
                indicators = context.market_indicators
                
                price_change = indicators.get('price_change_1h', 0)
                volume_ratio = indicators.get('volume_ratio', 1.0)
                
                if (price_change < conditions['price_drop_threshold'] and 
                    volume_ratio > conditions['volume_spike']):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check market crash: {e}")
            return False

    async def _check_strategy_malfunction_pattern(self, context: DecisionContext,
                                                 strategy_signals: Dict[str, Any],
                                                 pattern: EmergencyPattern) -> bool:
        """Check for strategy malfunction pattern"""
        try:
            for strategy_name, signal_data in strategy_signals.items():
                
                # Check error rate
                error_rate = signal_data.get('error_rate', 0.0)
                execution_time = signal_data.get('execution_time', 0.0)
                
                if (error_rate > pattern.trigger_conditions['error_rate_threshold'] or
                    execution_time > pattern.trigger_conditions['execution_time_spike']):
                    
                    pattern.affected_strategies.append(strategy_name)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check strategy malfunction: {e}")
            return False

    async def _check_correlation_breakdown_pattern(self, context: DecisionContext,
                                                  pattern: EmergencyPattern) -> bool:
        """Check for correlation breakdown pattern"""
        try:
            # Check if most strategies are losing simultaneously
            total_strategies = len(context.active_strategies)
            if total_strategies < 2:
                return False
            
            losing_strategies = 0
            
            for strategy_name in context.active_strategies:
                recent_returns = self.historical_performance.get(strategy_name, [])
                if recent_returns and recent_returns[0] < 0:  # Most recent return is negative
                    losing_strategies += 1
            
            diversification_failure = losing_strategies / total_strategies
            
            if diversification_failure > pattern.trigger_conditions['diversification_failure']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check correlation breakdown: {e}")
            return False

    async def _handle_emergency_allocation(self, context: DecisionContext, 
                                          pattern: EmergencyPattern) -> Dict[str, Any]:
        """Handle emergency allocation based on pattern"""
        logger.warning(f"🚨 Handling emergency: {pattern.pattern_name}")
        
        try:
            emergency_allocation = {
                'timestamp': context.timestamp.isoformat(),
                'decision_type': 'emergency_allocation',
                'emergency_pattern': pattern.pattern_id,
                'severity': pattern.severity_level,
                'allocations': {},
                'actions_taken': pattern.emergency_actions,
                'reasoning': [
                    f"Emergency pattern detected: {pattern.pattern_name}",
                    f"Severity: {pattern.severity_level}",
                    f"Affected strategies: {pattern.affected_strategies}",
                    f"Emergency actions: {pattern.emergency_actions}"
                ],
                'confidence': pattern.confidence_threshold
            }
            
            # Apply emergency actions
            if 'emergency_stop' in pattern.emergency_actions:
                # Stop all trading
                emergency_allocation['allocations'] = {s: 0.0 for s in context.active_strategies}
                
            elif 'reduce_allocation' in pattern.emergency_actions:
                # Reduce allocations by 50%
                current_weights = self.strategy_weights.copy()
                emergency_allocation['allocations'] = {
                    s: w * 0.5 for s, w in current_weights.items()
                }
                
            elif 'pause_strategy' in pattern.emergency_actions and pattern.affected_strategies:
                # Pause only affected strategies
                current_weights = self.strategy_weights.copy()
                for strategy in pattern.affected_strategies:
                    current_weights[strategy] = 0.0
                
                # Renormalize remaining strategies
                total_remaining = sum(w for s, w in current_weights.items() if s not in pattern.affected_strategies)
                if total_remaining > 0:
                    for strategy in current_weights:
                        if strategy not in pattern.affected_strategies:
                            current_weights[strategy] = current_weights[strategy] / total_remaining
                
                emergency_allocation['allocations'] = current_weights
            
            else:
                # Conservative allocation
                num_strategies = len(context.active_strategies)
                conservative_weight = 0.1 / num_strategies if num_strategies > 0 else 0.0
                emergency_allocation['allocations'] = {
                    s: conservative_weight for s in context.active_strategies
                }
            
            return emergency_allocation
            
        except Exception as e:
            logger.error(f"Failed to handle emergency allocation: {e}")
            # Ultra-conservative fallback
            return {
                'timestamp': context.timestamp.isoformat(),
                'decision_type': 'emergency_fallback',
                'allocations': {s: 0.01 for s in context.active_strategies},
                'reasoning': ['Emergency handling failed - ultra-conservative fallback'],
                'confidence': 0.1
            }

    async def _apply_adaptive_learning(self, context: DecisionContext, 
                                      strategy_signals: Dict[str, Any]) -> Dict[str, float]:
        """Apply adaptive learning to adjust strategy weights"""
        
        try:
            adapted_weights = self.strategy_weights.copy()
            
            for strategy_name in context.active_strategies:
                if strategy_name not in adapted_weights:
                    continue
                
                recent_returns = self.historical_performance.get(strategy_name, [])
                
                if len(recent_returns) >= 5:  # Need at least 5 trades for learning
                    
                    # Calculate recent performance trend
                    recent_5 = recent_returns[:5]
                    older_5 = recent_returns[5:10] if len(recent_returns) >= 10 else recent_5
                    
                    recent_avg = statistics.mean(recent_5)
                    older_avg = statistics.mean(older_5)
                    
                    # Performance trend
                    trend = recent_avg - older_avg
                    
                    # Adaptive adjustment
                    if trend > 0:  # Improving performance
                        multiplier = 1 + (trend * self.learning_rate)
                    else:  # Declining performance
                        multiplier = 1 + (trend * self.learning_rate * 2)  # More aggressive on downside
                    
                    # Apply bounds
                    multiplier = max(0.5, min(multiplier, 2.0))
                    
                    adapted_weights[strategy_name] *= multiplier
                    
                    context.confidence_factors[strategy_name] = {
                        'trend': trend,
                        'multiplier': multiplier,
                        'sample_size': len(recent_returns)
                    }
            
            # Renormalize weights
            total_weight = sum(adapted_weights.values())
            if total_weight > 0:
                adapted_weights = {k: v / total_weight for k, v in adapted_weights.items()}
            
            return adapted_weights
            
        except Exception as e:
            logger.error(f"Failed to apply adaptive learning: {e}")
            return self.strategy_weights.copy()

    async def _apply_ml_insights(self, weights: Dict[str, float], 
                                context: DecisionContext) -> Dict[str, float]:
        """Apply ML insights to weights"""
        
        try:
            if not self.ml_insights:
                return weights
            
            ml_adjusted = weights.copy()
            
            # Apply strategy insights
            strategy_insights = self.ml_insights.get('strategy_insights', [])
            for insight in strategy_insights:
                strategy_name = insight.get('strategy_name')
                insight_type = insight.get('insight_type')
                impact_score = insight.get('impact_score', 0)
                confidence = insight.get('confidence', 0)
                
                if strategy_name in ml_adjusted and confidence > 0.7:
                    
                    if insight_type == 'strength':
                        # Increase allocation for strong strategies
                        multiplier = 1 + (impact_score * 0.1)
                        ml_adjusted[strategy_name] *= min(multiplier, 1.5)
                        
                    elif insight_type == 'weakness':
                        # Decrease allocation for weak strategies
                        multiplier = 1 - (abs(impact_score) * 0.1)
                        ml_adjusted[strategy_name] *= max(multiplier, 0.5)
            
            # Apply combination insights
            combinations = self.ml_insights.get('combinations', [])
            for combo in combinations:
                strategies = combo.get('strategies', [])
                synergy_score = combo.get('synergy_score', 0)
                confidence = combo.get('confidence', 0)
                
                if confidence > 0.7 and synergy_score > 0.2:
                    # Boost synergistic strategy combinations
                    boost_factor = 1 + (synergy_score * 0.05)
                    for strategy in strategies:
                        if strategy in ml_adjusted:
                            ml_adjusted[strategy] *= boost_factor
            
            # Apply success patterns
            success_patterns = self.ml_insights.get('success_patterns', [])
            for pattern in success_patterns:
                strategies_involved = pattern.get('strategies_involved', [])
                avg_return = pattern.get('avg_return', 0)
                confidence = pattern.get('confidence', 0)
                
                # Check if current market conditions match pattern
                market_conditions = pattern.get('market_conditions', {})
                regime_match = market_conditions.get('regime_distribution', {}).get(context.market_regime, 0)
                
                if confidence > 0.8 and regime_match > 0.3 and avg_return > 2.0:
                    # Boost strategies that work well in current conditions
                    for strategy in strategies_involved:
                        if strategy in ml_adjusted:
                            ml_adjusted[strategy] *= (1 + regime_match * 0.1)
            
            # Renormalize
            total_weight = sum(ml_adjusted.values())
            if total_weight > 0:
                ml_adjusted = {k: v / total_weight for k, v in ml_adjusted.items()}
            
            return ml_adjusted
            
        except Exception as e:
            logger.error(f"Failed to apply ML insights: {e}")
            return weights

    async def _make_final_allocation(self, weights: Dict[str, float], 
                                    context: DecisionContext) -> Dict[str, Any]:
        """Make final allocation decision"""
        
        try:
            # Apply risk constraints
            risk_adjusted_weights = await self._apply_risk_constraints(weights, context)
            
            # Generate reasoning
            reasoning = []
            reasoning.append(f"Market regime: {context.market_regime}")
            reasoning.append(f"Portfolio value: ${context.portfolio_value:,.2f}")
            reasoning.append(f"Risk score: {context.risk_score:.2f}")
            reasoning.append(f"Active strategies: {len(context.active_strategies)}")
            
            if context.confidence_factors:
                reasoning.append("Adaptive learning adjustments applied")
            
            if self.ml_insights:
                reasoning.append("ML insights integrated")
            
            if context.emergency_patterns:
                reasoning.append(f"Emergency patterns checked: none detected")
            
            # Calculate overall confidence
            confidence_score = self._calculate_decision_confidence(risk_adjusted_weights, context)
            
            return {
                'timestamp': context.timestamp.isoformat(),
                'decision_type': 'normal_allocation',
                'allocations': risk_adjusted_weights,
                'market_regime': context.market_regime,
                'portfolio_value': context.portfolio_value,
                'risk_score': context.risk_score,
                'confidence': confidence_score,
                'reasoning': reasoning,
                'factors_considered': {
                    'adaptive_learning': bool(context.confidence_factors),
                    'ml_insights': bool(self.ml_insights),
                    'emergency_check': True,
                    'risk_constraints': True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to make final allocation: {e}")
            # Safe fallback
            equal_weights = {s: 1.0/len(context.active_strategies) for s in context.active_strategies}
            return {
                'timestamp': context.timestamp.isoformat(),
                'decision_type': 'fallback_allocation',
                'allocations': equal_weights,
                'confidence': 0.3,
                'reasoning': ['Error in final allocation - using equal weights fallback']
            }

    async def _apply_risk_constraints(self, weights: Dict[str, float], 
                                     context: DecisionContext) -> Dict[str, float]:
        """Apply risk management constraints to weights"""
        
        try:
            constrained_weights = weights.copy()
            
            # Max single strategy allocation based on risk level
            for strategy_name, weight in constrained_weights.items():
                dna = self.discovery_engine.discovered_strategies.get(strategy_name)
                if dna:
                    # Risk-based max allocation
                    if dna.risk_level == 'conservative':
                        max_allocation = 0.4
                    elif dna.risk_level == 'moderate':  
                        max_allocation = 0.3
                    elif dna.risk_level == 'aggressive':
                        max_allocation = 0.2
                    else:  # extreme
                        max_allocation = 0.1
                    
                    # Apply constraint
                    constrained_weights[strategy_name] = min(weight, max_allocation)
            
            # Portfolio heat constraint
            if context.risk_score > 0.8:  # High risk
                # Reduce all allocations
                reduction_factor = 0.7
                constrained_weights = {k: v * reduction_factor for k, v in constrained_weights.items()}
            
            # Volatility constraint
            if context.volatility_level > 0.05:  # High volatility (5%+)
                # Favor conservative strategies
                for strategy_name, weight in constrained_weights.items():
                    dna = self.discovery_engine.discovered_strategies.get(strategy_name)
                    if dna and dna.risk_level in ['aggressive', 'extreme']:
                        constrained_weights[strategy_name] *= 0.8
                    elif dna and dna.risk_level == 'conservative':
                        constrained_weights[strategy_name] *= 1.2
            
            # Renormalize
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {k: v / total_weight for k, v in constrained_weights.items()}
            
            return constrained_weights
            
        except Exception as e:
            logger.error(f"Failed to apply risk constraints: {e}")  
            return weights

    def _calculate_decision_confidence(self, weights: Dict[str, float],
                                      context: DecisionContext) -> float:
        """Calculate confidence score for decision"""
        
        try:
            confidence_factors = []
            
            # Historical data quality
            total_trades = sum(len(self.historical_performance.get(s, [])) for s in weights.keys())
            data_quality = min(total_trades / 100, 1.0)  # More data = higher confidence
            confidence_factors.append(data_quality * 0.3)
            
            # ML insights availability
            ml_confidence = 0.2 if self.ml_insights else 0.0
            confidence_factors.append(ml_confidence)
            
            # Market regime confidence
            regime_confidence = context.market_indicators.get('regime_confidence', 0.5)
            confidence_factors.append(regime_confidence * 0.2)
            
            # Strategy diversity
            active_strategies = len([w for w in weights.values() if w > 0.01])
            diversity_score = min(active_strategies / 5, 1.0)  # Up to 5 strategies
            confidence_factors.append(diversity_score * 0.2)
            
            # Emergency pattern absence
            emergency_confidence = 0.1 if not context.emergency_patterns else 0.0
            confidence_factors.append(emergency_confidence)
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"Failed to calculate decision confidence: {e}")
            return 0.5

    async def _log_allocation_decision(self, context: DecisionContext, allocation: Dict[str, Any]):
        """Log allocation decision for learning"""
        
        try:
            # Create orchestrator decision log
            decision = OrchestratorDecision(
                decision_type='strategy_allocation',
                strategy_name=None,  # Multiple strategies
                old_allocation=None,  # We could track previous allocations
                new_allocation=None,  # Could be JSON of all allocations
                market_regime=context.market_regime,
                volatility_level=context.volatility_level,
                confidence_score=allocation.get('confidence', 0.0),
                trigger_source='orchestrator_learning',
                trigger_data={
                    'active_strategies': context.active_strategies,
                    'ml_insights_used': bool(self.ml_insights),
                    'emergency_patterns_checked': len(self.emergency_patterns),
                    'adaptive_learning_applied': bool(context.confidence_factors)
                },
                decision_reasoning='; '.join(allocation.get('reasoning', [])),
                expected_impact=None,  # Could calculate expected portfolio impact
                portfolio_value_before=context.portfolio_value,
                risk_score_before=context.risk_score
            )
            
            await self.decision_logger.log_orchestrator_decision(decision)
            
            # Log market state
            market_state = MarketState(
                data_source='orchestrator',
                detected_regime=context.market_regime,
                regime_confidence=context.market_indicators.get('regime_confidence'),
                realized_volatility_24h=context.volatility_level,
                rsi_composite=context.market_indicators.get('rsi'),
                total_volume_24h=context.market_indicators.get('volume_24h'),
                systemic_risk_score=context.risk_score
            )
            
            await self.decision_logger.log_market_state(market_state)
            
        except Exception as e:
            logger.error(f"Failed to log allocation decision: {e}")

    async def _emergency_fallback_allocation(self, strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency fallback allocation when main logic fails"""
        
        strategies = list(strategy_signals.keys())
        if not strategies:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'decision_type': 'emergency_fallback',
                'allocations': {},
                'confidence': 0.0,
                'reasoning': ['No strategies available']
            }
        
        # Ultra-conservative equal allocation
        safe_weight = 0.05 / len(strategies)  # Only 5% total allocation
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'decision_type': 'emergency_fallback',
            'allocations': {s: safe_weight for s in strategies},
            'confidence': 0.1,
            'reasoning': ['Emergency fallback due to orchestrator failure']
        }

    def explain(self, decision_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain orchestrator decision in detail
        
        Args:
            decision_id: Specific decision to explain (None for latest)
            
        Returns:
            Detailed explanation of decision
        """
        try:
            # Get decision to explain
            if decision_id is None:
                if not self.decision_history:
                    return {'error': 'No decisions made yet'}
                decision_data = self.decision_history[-1]
            else:
                # Find specific decision
                decision_data = None
                for d in self.decision_history:
                    if d.get('decision_id') == decision_id:
                        decision_data = d
                        break
                
                if not decision_data:
                    return {'error': f'Decision {decision_id} not found'}
            
            # Build comprehensive explanation
            explanation = {
                'decision_summary': {
                    'timestamp': decision_data['timestamp'],
                    'decision_type': decision_data.get('allocation', {}).get('decision_type', 'unknown'),
                    'confidence': decision_data.get('allocation', {}).get('confidence', 0.0),
                    'total_strategies': len(decision_data.get('allocation', {}).get('allocations', {}))
                },
                
                'market_context': {
                    'regime': decision_data['context']['market_regime'],
                    'volatility': decision_data['context']['volatility_level'],
                    'portfolio_value': decision_data['context']['portfolio_value'],
                    'risk_score': decision_data['context']['risk_score']
                },
                
                'strategy_analysis': {},
                'decision_factors': {
                    'adaptive_learning': bool(decision_data['context'].get('confidence_factors')),
                    'ml_insights': bool(self.ml_insights),
                    'emergency_patterns': decision_data['context'].get('emergency_patterns', []),
                    'risk_constraints': True
                },
                
                'allocations_explained': {},
                'reasoning': decision_data.get('reasoning', []),
                
                'alternative_scenarios': {},
                'risk_assessment': {},
                'learning_insights': {}
            }
            
            # Explain each strategy allocation
            allocations = decision_data.get('allocation', {}).get('allocations', {})
            for strategy_name, allocation in allocations.items():
                
                # Get strategy DNA
                dna = self.discovery_engine.discovered_strategies.get(strategy_name)
                
                # Historical performance
                historical = self.historical_performance.get(strategy_name, [])
                
                strategy_explanation = {
                    'allocation_percentage': allocation * 100,
                    'strategy_profile': {
                        'risk_level': dna.risk_level if dna else 'unknown',
                        'timeframe': dna.timeframe if dna else 'unknown',
                        'signal_sources': dna.signal_sources if dna else [],
                        'cooperation_score': dna.cooperation_score if dna else 0
                    },
                    'historical_performance': {
                        'trade_count': len(historical),
                        'average_return': statistics.mean(historical) if historical else 0,
                        'recent_trend': 'improving' if len(historical) >= 2 and historical[0] > historical[1] else 'declining'
                    },
                    'confidence_factors': decision_data['context'].get('confidence_factors', {}).get(strategy_name, {}),
                    'allocation_reasoning': []
                }
                
                # Generate specific reasoning for this allocation
                if allocation > 0.15:  # High allocation
                    strategy_explanation['allocation_reasoning'].append("High allocation due to strong recent performance")
                elif allocation < 0.05:  # Low allocation
                    strategy_explanation['allocation_reasoning'].append("Low allocation due to risk management or poor performance")
                else:
                    strategy_explanation['allocation_reasoning'].append("Standard allocation based on strategy profile")
                
                explanation['allocations_explained'][strategy_name] = strategy_explanation
            
            # Risk assessment
            explanation['risk_assessment'] = {
                'portfolio_risk_level': 'low' if decision_data['context']['risk_score'] < 0.3 else 
                                       'medium' if decision_data['context']['risk_score'] < 0.7 else 'high',
                'diversification_score': len([a for a in allocations.values() if a > 0.01]) / len(allocations) if allocations else 0,
                'volatility_exposure': decision_data['context']['volatility_level'],
                'max_single_allocation': max(allocations.values()) if allocations else 0
            }
            
            # Learning insights
            explanation['learning_insights'] = {
                'strategies_learned_from': len(self.historical_performance),
                'total_historical_trades': sum(len(trades) for trades in self.historical_performance.values()),
                'ml_insights_available': len(self.ml_insights),
                'emergency_patterns_monitored': len(self.emergency_patterns),
                'last_performance_update': self.last_performance_update.isoformat() if self.last_performance_update else None
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain decision: {e}")
            return {
                'error': f'Failed to generate explanation: {str(e)}',
                'fallback_info': {
                    'total_strategies': len(self.strategy_weights),
                    'learning_active': bool(self.historical_performance),
                    'ml_insights': bool(self.ml_insights)
                }
            }

    async def update_performance_feedback(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Update performance feedback for continuous learning"""
        
        try:
            pnl_percentage = trade_result.get('pnl_percentage', 0.0)
            
            with self._lock:
                if strategy_name not in self.historical_performance:
                    self.historical_performance[strategy_name] = []
                
                # Add new result at the beginning (most recent first)
                self.historical_performance[strategy_name].insert(0, pnl_percentage)
                
                # Keep only last 100 trades per strategy
                self.historical_performance[strategy_name] = self.historical_performance[strategy_name][:100]
                
                self.last_performance_update = datetime.utcnow()
            
            # Trigger weight readjustment if significant change
            if abs(pnl_percentage) > 5.0:  # Significant win/loss
                await self._trigger_adaptive_readjustment(strategy_name, pnl_percentage)
            
        except Exception as e:
            logger.error(f"Failed to update performance feedback: {e}")

    async def _trigger_adaptive_readjustment(self, strategy_name: str, pnl_percentage: float):
        """Trigger adaptive readjustment of strategy weights"""
        
        try:
            current_weight = self.strategy_weights.get(strategy_name, 0.0)
            
            # Calculate adjustment
            if pnl_percentage > 0:  # Profit
                adjustment = min(pnl_percentage * 0.01, 0.05)  # Max 5% increase
            else:  # Loss
                adjustment = max(pnl_percentage * 0.01, -0.05)  # Max 5% decrease
            
            new_weight = max(0.01, min(current_weight + adjustment, 0.5))  # Bounds: 1% to 50%
            
            with self._lock:
                self.strategy_weights[strategy_name] = new_weight
            
            # Renormalize all weights
            await self._renormalize_weights()
            
            logger.info(f"🔄 Adapted {strategy_name} weight: {current_weight:.3f} -> {new_weight:.3f} (PnL: {pnl_percentage:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to trigger adaptive readjustment: {e}")

    async def _renormalize_weights(self):
        """Renormalize all strategy weights to sum to 1.0"""
        with self._lock:
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                self.strategy_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        with self._lock:
            return {
                'strategies_tracked': len(self.strategy_weights),
                'total_historical_trades': sum(len(trades) for trades in self.historical_performance.values()),
                'ml_insights_loaded': bool(self.ml_insights),
                'emergency_patterns': len(self.emergency_patterns),
                'decisions_made': len(self.decision_history),
                'last_performance_update': self.last_performance_update.isoformat() if self.last_performance_update else None,
                'current_weights': self.strategy_weights.copy(),
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'confidence_decay': self.confidence_decay,
                    'lookback_days': self.performance_lookback_days
                }
            }

# Factory-Funktion
def create_strategy_discovery_engine(strategies_dir: str = "strategies") -> StrategyDiscoveryEngine:
    """Erstellt Strategy Discovery Engine"""
    return StrategyDiscoveryEngine(strategies_dir)

def create_learning_orchestrator(db_pool, discovery_engine: StrategyDiscoveryEngine,
                                decision_logger: DecisionLogger,
                                learning_pipeline: Optional[LearningPipeline] = None,
                                pattern_detector: Optional[PatternDetector] = None) -> LearningOrchestrator:
    """Create Learning Orchestrator with all components"""
    return LearningOrchestrator(db_pool, discovery_engine, decision_logger, learning_pipeline, pattern_detector)

# Test-Funktion
async def test_strategy_discovery():
    """Testet das Strategy Discovery System"""
    
    print("🔍 TESTE STRATEGY DISCOVERY SYSTEM")
    print("=" * 40)
    
    try:
        # Discovery Engine erstellen
        discovery_engine = create_strategy_discovery_engine()
        
        # Vollständige Discovery durchführen
        strategies = await discovery_engine.discover_all_strategies()
        
        print(f"\n📊 DISCOVERY ERGEBNISSE:")
        print(f"   Strategien gefunden: {len(strategies)}")
        
        # Test verschiedene Filter
        print(f"\n🔬 TESTE FILTER:")
        
        aggressive_strategies = discovery_engine.get_strategies_by_criteria(risk_level='aggressive')
        print(f"   Aggressive Strategien: {len(aggressive_strategies)}")
        
        ml_strategies = discovery_engine.get_strategies_by_criteria(signal_sources=['ml'])
        print(f"   ML-Strategien: {len(ml_strategies)}")
        
        intraday_strategies = discovery_engine.get_strategies_by_criteria(timeframe='intraday')
        print(f"   Intraday-Strategien: {len(intraday_strategies)}")
        
        # Discovery Stats
        stats = discovery_engine.get_discovery_stats()
        print(f"\n📈 DISCOVERY STATISTIKEN:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Datenbank speichern/laden testen
        discovery_engine.save_discovery_database("test_discovery_db.json")
        
        # Neue Engine erstellen und DB laden
        test_engine = create_strategy_discovery_engine()
        loaded = test_engine.load_discovery_database("test_discovery_db.json")
        print(f"\n💾 Datenbank Test: {'✅' if loaded else '❌'}")
        
        if loaded:
            print(f"   Geladene Strategien: {len(test_engine.discovered_strategies)}")
        
        # Cleanup
        Path("test_discovery_db.json").unlink(missing_ok=True)
        
        print(f"\n🎉 STRATEGY DISCOVERY TEST ERFOLGREICH!")
        return True
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Discovery System testen
    asyncio.run(test_strategy_discovery())