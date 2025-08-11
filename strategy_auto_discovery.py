#!/usr/bin/env python3
"""
🔍 Strategy Auto-Discovery Engine
Automatically scan, classify and register all available trading strategies
"""

import os
import sys
import inspect
import importlib
import logging
from typing import Dict, List, Type, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import ast
import json

# Import your strategy base class
from strategies.strategy_base import Strategy as StrategyBase

class RiskLevel(Enum):
    """Strategy risk level classification"""
    CONSERVATIVE = "LOW_RISK"
    MODERATE = "MEDIUM_RISK" 
    AGGRESSIVE = "HIGH_RISK"
    SPECULATIVE = "EXTREME_RISK"

@dataclass
class StrategyMetadata:
    """Metadata for discovered strategy"""
    name: str
    class_name: str
    module_path: str
    risk_level: RiskLevel
    expected_return_range: Tuple[float, float]  # Min, Max daily return %
    typical_holding_period: str  # e.g., "1-60 minutes", "1-3 days"
    max_position_size: float  # Maximum position size as % of portfolio
    description: str
    features: List[str]
    requires_ml: bool
    supports_symbols: List[str]
    preferred_timeframes: List[str]
    performance_score: float  # 0-100 based on backtesting/live performance
    is_active: bool

class StrategyAutoDiscovery:
    """
    🔍 Automatic Strategy Discovery and Classification System
    
    Features:
    - Scans strategy directories for all strategy classes
    - Analyzes code to determine risk characteristics
    - Classifies strategies by risk level and features
    - Creates strategy registry with metadata
    - Validates strategy implementations
    - Performance-based classification updates
    """
    
    def __init__(self, 
                 strategies_dir: str = "strategies",
                 config_path: str = "config/strategy_registry.json"):
        
        self.strategies_dir = Path(strategies_dir)
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Strategy registry
        self.discovered_strategies: Dict[str, StrategyMetadata] = {}
        self.strategy_classes: Dict[str, Type] = {}
        
        # Risk classification rules
        self.risk_classification_rules = self._initialize_risk_rules()
        
        # Load existing registry
        self._load_existing_registry()
        
        self.logger.info("🔍 Strategy Auto-Discovery initialized")
    
    def _initialize_risk_rules(self) -> Dict[str, Dict]:
        """Initialize rules for automatic risk classification"""
        return {
            'CONSERVATIVE': {
                'keywords': ['defensive', 'conservative', 'mean_reversion', 'arbitrage', 
                           'rebalancing', 'dca', 'stable', 'safe', 'hedge'],
                'max_position_indicators': ['0.05', '0.08', '5%', '8%', 'conservative'],
                'holding_period_indicators': ['days', 'weeks', 'long', 'hold'],
                'risk_indicators': ['low_risk', 'defensive', 'protection', 'safe'],
                'default_max_position': 8.0,
                'expected_return': (0.3, 2.0)  # 0.3% - 2% daily
            },
            
            'MODERATE': {
                'keywords': ['momentum', 'swing', 'trend', 'pattern', 'technical',
                           'balanced', 'moderate', 'candle', 'signal'],
                'max_position_indicators': ['0.1', '0.15', '10%', '15%', 'moderate'],
                'holding_period_indicators': ['hours', 'days', 'swing', 'medium'],
                'risk_indicators': ['medium_risk', 'balanced', 'moderate'],
                'default_max_position': 5.0,
                'expected_return': (0.8, 5.0)  # 0.8% - 5% daily
            },
            
            'AGGRESSIVE': {
                'keywords': ['ml', 'ai', 'high_risk', 'aggressive', 'ultimate', 
                           'enhanced', 'profit', 'advanced', 'alpha'],
                'max_position_indicators': ['0.02', '0.05', '2%', '5%', 'aggressive'],
                'holding_period_indicators': ['minutes', 'hours', 'short', 'quick'],
                'risk_indicators': ['high_risk', 'aggressive', 'advanced'],
                'default_max_position': 3.0,
                'expected_return': (1.0, 10.0)  # 1% - 10% daily
            },
            
            'SPECULATIVE': {
                'keywords': ['scalping', 'news', 'volatility', 'breakout', 'liquidation',
                           'extreme', 'speculative', 'yolo', 'gamble'],
                'max_position_indicators': ['0.01', '0.02', '1%', '2%', 'tiny'],
                'holding_period_indicators': ['seconds', 'minutes', 'instant', 'scalp'],
                'risk_indicators': ['extreme', 'speculative', 'dangerous'],
                'default_max_position': 1.5,
                'expected_return': (2.0, 20.0)  # 2% - 20% daily
            }
        }
    
    async def discover_all_strategies(self) -> Dict[str, StrategyMetadata]:
        """
        🔍 Discover and classify all available strategies
        
        Returns:
            Dictionary of strategy name -> StrategyMetadata
        """
        self.logger.info(f"🔍 Starting strategy discovery in {self.strategies_dir}")
        
        # Scan for Python files
        strategy_files = self._scan_strategy_files()
        
        discovered_count = 0
        
        for file_path in strategy_files:
            try:
                strategies_in_file = await self._analyze_strategy_file(file_path)
                
                for strategy_metadata in strategies_in_file:
                    self.discovered_strategies[strategy_metadata.name] = strategy_metadata
                    discovered_count += 1
                    
                    self.logger.info(f"✅ Discovered: {strategy_metadata.name} "
                                   f"({strategy_metadata.risk_level.value})")
                    
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {file_path}: {e}")
                continue
        
        self.logger.info(f"🎯 Strategy discovery completed: {discovered_count} strategies found")
        
        # Save updated registry
        self._save_registry()
        
        return self.discovered_strategies
    
    def _scan_strategy_files(self) -> List[Path]:
        """Scan strategies directory for Python files"""
        strategy_files = []
        
        if not self.strategies_dir.exists():
            self.logger.error(f"❌ Strategies directory not found: {self.strategies_dir}")
            return strategy_files
        
        # Scan for .py files
        for file_path in self.strategies_dir.rglob("*.py"):
            # Skip __init__.py and base files
            if file_path.name in ['__init__.py', 'strategy_base.py']:
                continue
                
            strategy_files.append(file_path)
            
        self.logger.info(f"📁 Found {len(strategy_files)} strategy files")
        return strategy_files
    
    async def _analyze_strategy_file(self, file_path: Path) -> List[StrategyMetadata]:
        """Analyze a single strategy file for strategy classes"""
        strategies = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find classes
            tree = ast.parse(content)
            
            # Dynamic import of the module
            module_name = self._path_to_module_name(file_path)
            
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                self.logger.warning(f"⚠️ Could not import {module_name}: {e}")
                return strategies
            
            # Find strategy classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Get the actual class from the module
                    if hasattr(module, class_name):
                        strategy_class = getattr(module, class_name)
                        
                        # Check if it's a strategy class
                        if (inspect.isclass(strategy_class) and 
                            issubclass(strategy_class, StrategyBase) and 
                            strategy_class != StrategyBase):
                            
                            # Analyze the strategy
                            metadata = await self._analyze_strategy_class(
                                strategy_class, file_path, content, node
                            )
                            
                            if metadata:
                                strategies.append(metadata)
                                # Store the class for later use
                                self.strategy_classes[metadata.name] = strategy_class
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing file {file_path}: {e}")
            
        return strategies
    
    async def _analyze_strategy_class(self, 
                                    strategy_class: Type,
                                    file_path: Path,
                                    file_content: str,
                                    class_node: ast.ClassDef) -> Optional[StrategyMetadata]:
        """Analyze individual strategy class"""
        
        class_name = strategy_class.__name__
        strategy_name = self._class_name_to_strategy_name(class_name)
        
        # Extract docstring
        docstring = inspect.getdoc(strategy_class) or ""
        
        # Analyze code content
        risk_level = self._classify_risk_level(class_name, file_content, docstring)
        features = self._extract_features(file_content, docstring)
        
        # Extract configuration from code
        config = self._extract_strategy_config(file_content, class_node)
        
        # Build metadata
        metadata = StrategyMetadata(
            name=strategy_name,
            class_name=class_name,
            module_path=str(file_path),
            risk_level=risk_level,
            expected_return_range=self._estimate_return_range(risk_level, file_content),
            typical_holding_period=self._estimate_holding_period(file_content),
            max_position_size=self._estimate_max_position_size(risk_level, file_content),
            description=self._extract_description(docstring),
            features=features,
            requires_ml=self._check_ml_requirement(file_content),
            supports_symbols=self._extract_supported_symbols(file_content),
            preferred_timeframes=self._extract_timeframes(file_content),
            performance_score=self._get_performance_score(strategy_name),
            is_active=True
        )
        
        return metadata
    
    def _classify_risk_level(self, class_name: str, content: str, docstring: str) -> RiskLevel:
        """Classify strategy risk level based on code analysis"""
        
        # Combine all text for analysis
        analysis_text = f"{class_name.lower()} {content.lower()} {docstring.lower()}"
        
        # Score each risk category
        risk_scores = {}
        
        for risk_level, rules in self.risk_classification_rules.items():
            score = 0
            
            # Check keywords
            for keyword in rules['keywords']:
                if keyword in analysis_text:
                    score += 2
            
            # Check position size indicators
            for indicator in rules['max_position_indicators']:
                if indicator in analysis_text:
                    score += 1
            
            # Check holding period indicators  
            for indicator in rules['holding_period_indicators']:
                if indicator in analysis_text:
                    score += 1
                    
            # Check risk indicators
            for indicator in rules['risk_indicators']:
                if indicator in analysis_text:
                    score += 3
            
            risk_scores[risk_level] = score
        
        # Return highest scoring risk level
        best_risk_level = max(risk_scores.items(), key=lambda x: x[1])[0]
        
        # Map to enum
        risk_mapping = {
            'CONSERVATIVE': RiskLevel.CONSERVATIVE,
            'MODERATE': RiskLevel.MODERATE,
            'AGGRESSIVE': RiskLevel.AGGRESSIVE,
            'SPECULATIVE': RiskLevel.SPECULATIVE
        }
        
        return risk_mapping.get(best_risk_level, RiskLevel.MODERATE)
    
    def _extract_features(self, content: str, docstring: str) -> List[str]:
        """Extract strategy features from code analysis"""
        features = []
        
        feature_indicators = {
            'ML/AI': ['lightgbm', 'sklearn', 'tensorflow', 'pytorch', 'machine_learning', 'predictor'],
            'Technical Analysis': ['rsi', 'macd', 'bollinger', 'moving_average', 'sma', 'ema'],
            'Volume Analysis': ['volume', 'vwap', 'obv', 'volume_profile'],
            'Pattern Recognition': ['pattern', 'candlestick', 'chart_pattern', 'formation'],
            'Momentum': ['momentum', 'breakout', 'trend', 'strength'],
            'Mean Reversion': ['mean_reversion', 'revert', 'oversold', 'overbought'],
            'Arbitrage': ['arbitrage', 'spread', 'cross_exchange', 'price_difference'],
            'Risk Management': ['stop_loss', 'position_sizing', 'risk_management', 'drawdown'],
            'Multi-Timeframe': ['timeframe', 'multi_tf', 'higher_tf', 'lower_tf'],
            'News/Sentiment': ['news', 'sentiment', 'social', 'twitter', 'reddit'],
            'Options/Derivatives': ['options', 'futures', 'derivatives', 'leverage'],
            'DeFi': ['defi', 'yield', 'liquidity', 'farming', 'staking']
        }
        
        analysis_text = f"{content.lower()} {docstring.lower()}"
        
        for feature_name, indicators in feature_indicators.items():
            for indicator in indicators:
                if indicator in analysis_text:
                    features.append(feature_name)
                    break
        
        return features
    
    def _estimate_return_range(self, risk_level: RiskLevel, content: str) -> Tuple[float, float]:
        """Estimate expected return range based on risk level and code analysis"""
        
        # Base returns by risk level
        base_returns = {
            RiskLevel.CONSERVATIVE: (0.2, 1.5),
            RiskLevel.MODERATE: (0.5, 3.0),
            RiskLevel.AGGRESSIVE: (1.0, 8.0),
            RiskLevel.SPECULATIVE: (2.0, 15.0)
        }
        
        base_min, base_max = base_returns[risk_level]
        
        # Look for performance indicators in code
        performance_indicators = {
            'high_frequency': 1.5,     # Higher frequency = higher potential returns
            'scalping': 2.0,
            'breakout': 1.3,
            'momentum': 1.2,
            'mean_reversion': 0.8,     # Lower but more consistent
            'conservative': 0.7,
            'defensive': 0.6
        }
        
        multiplier = 1.0
        content_lower = content.lower()
        
        for indicator, mult in performance_indicators.items():
            if indicator in content_lower:
                multiplier = max(multiplier, mult)
        
        return (base_min * multiplier, base_max * multiplier)
    
    def _estimate_holding_period(self, content: str) -> str:
        """Estimate typical holding period from code analysis"""
        
        holding_indicators = {
            'scalping': "1-5 minutes",
            'high_frequency': "1-15 minutes",
            'intraday': "1-8 hours", 
            'swing': "1-5 days",
            'position': "1-4 weeks",
            'long_term': "1-6 months",
            'momentum': "2-24 hours",
            'mean_reversion': "4-48 hours",
            'arbitrage': "1-30 minutes"
        }
        
        content_lower = content.lower()
        
        for indicator, period in holding_indicators.items():
            if indicator in content_lower:
                return period
        
        # Default based on common patterns
        if any(word in content_lower for word in ['minutes', 'second', 'quick']):
            return "5-60 minutes"
        elif any(word in content_lower for word in ['hour', 'intraday']):
            return "1-24 hours"
        elif any(word in content_lower for word in ['day', 'daily']):
            return "1-7 days"
        else:
            return "1-24 hours"  # Default
    
    def _estimate_max_position_size(self, risk_level: RiskLevel, content: str) -> float:
        """Estimate maximum position size from risk level and code analysis"""
        
        # Default by risk level
        default_sizes = {
            RiskLevel.CONSERVATIVE: 8.0,
            RiskLevel.MODERATE: 5.0, 
            RiskLevel.AGGRESSIVE: 3.0,
            RiskLevel.SPECULATIVE: 1.5
        }
        
        base_size = default_sizes[risk_level]
        
        # Look for explicit position size mentions in code
        import re
        
        # Look for percentage patterns
        percent_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'0\.(\d+)',  # Decimal like 0.05
            r'position_size.*?(\d+(?:\.\d+)?)',
            r'max_position.*?(\d+(?:\.\d+)?)'
        ]
        
        found_sizes = []
        
        for pattern in percent_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    size = float(match)
                    if 0.01 <= size <= 50:  # Reasonable range
                        found_sizes.append(size)
                except:
                    continue
        
        if found_sizes:
            # Use the most conservative (smallest) found size
            return min(found_sizes)
        
        return base_size
    
    def _extract_description(self, docstring: str) -> str:
        """Extract clean description from docstring"""
        if not docstring:
            return "Trading strategy"
        
        # Get first meaningful line
        lines = [line.strip() for line in docstring.split('\n') if line.strip()]
        
        if lines:
            description = lines[0]
            # Clean up common docstring artifacts
            description = description.replace('"""', '').replace("'''", '')
            description = description.strip()
            return description[:200]  # Limit length
        
        return "Trading strategy"
    
    def _check_ml_requirement(self, content: str) -> bool:
        """Check if strategy requires ML components"""
        ml_indicators = [
            'lightgbm', 'sklearn', 'tensorflow', 'pytorch', 'xgboost',
            'machine_learning', 'predictor', 'model', 'train', 'predict',
            'ml_', 'ai_', 'neural', 'deep_learning'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in ml_indicators)
    
    def _extract_supported_symbols(self, content: str) -> List[str]:
        """Extract supported trading symbols from code"""
        
        # Look for common crypto symbols
        symbol_patterns = [
            r'["\']([A-Z]{3,4}USDT?)["\']',
            r'["\']([A-Z]{3,4}/USDT?)["\']',
            r'symbol.*?["\']([A-Z]+)["\']'
        ]
        
        found_symbols = set()
        
        import re
        for pattern in symbol_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                found_symbols.add(match)
        
        # Common defaults if none found
        if not found_symbols:
            found_symbols = {'BTCUSDT', 'ETHUSDT'}
        
        return list(found_symbols)
    
    def _extract_timeframes(self, content: str) -> List[str]:
        """Extract preferred timeframes from code"""
        
        timeframe_patterns = [
            r'["\'](\d+[mhd])["\']',  # 1m, 5m, 1h, 1d
            r'timeframe.*?["\']([^"\']+)["\']',
            r'interval.*?["\']([^"\']+)["\']'
        ]
        
        found_timeframes = set()
        
        import re
        for pattern in timeframe_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    found_timeframes.add(match)
        
        # Defaults based on content analysis
        if not found_timeframes:
            content_lower = content.lower()
            if any(word in content_lower for word in ['scalp', 'minute', 'quick']):
                found_timeframes = {'1m', '5m'}
            elif any(word in content_lower for word in ['hour', 'intraday']):
                found_timeframes = {'15m', '1h'}
            elif any(word in content_lower for word in ['day', 'swing']):
                found_timeframes = {'4h', '1d'}
            else:
                found_timeframes = {'15m', '1h'}
        
        return list(found_timeframes)
    
    def _extract_strategy_config(self, content: str, class_node: ast.ClassDef) -> Dict:
        """Extract strategy configuration from code"""
        config = {}
        
        # Look for common configuration patterns
        config_patterns = {
            'stop_loss': r'stop_loss.*?(\d+(?:\.\d+)?)',
            'take_profit': r'take_profit.*?(\d+(?:\.\d+)?)',
            'risk_percent': r'risk_percent.*?(\d+(?:\.\d+)?)',
            'max_positions': r'max_positions.*?(\d+)',
            'timeframe': r'timeframe.*?["\']([^"\']+)["\']'
        }
        
        import re
        for key, pattern in config_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    if key in ['stop_loss', 'take_profit', 'risk_percent']:
                        config[key] = float(match.group(1))
                    elif key == 'max_positions':
                        config[key] = int(match.group(1))
                    else:
                        config[key] = match.group(1)
                except:
                    pass
        
        return config
    
    def _get_performance_score(self, strategy_name: str) -> float:
        """Get performance score from existing data or default"""
        # This would integrate with your performance tracking system
        # For now, return a default score
        
        # Check if we have historical data
        if strategy_name in self.discovered_strategies:
            return self.discovered_strategies[strategy_name].performance_score
        
        # Default scores based on strategy type
        if 'high_risk' in strategy_name.lower():
            return 75.0  # Higher potential but riskier
        elif 'defensive' in strategy_name.lower():
            return 85.0  # More reliable
        else:
            return 70.0  # Default
    
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name for import"""
        relative_path = file_path.relative_to(Path.cwd())
        module_parts = relative_path.parts[:-1] + (relative_path.stem,)
        return '.'.join(module_parts)
    
    def _class_name_to_strategy_name(self, class_name: str) -> str:
        """Convert class name to strategy name"""
        # Remove 'Strategy' suffix if present
        if class_name.endswith('Strategy'):
            name = class_name[:-8]
        else:
            name = class_name
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        
        return name
    
    def get_strategies_by_risk_level(self, risk_level: RiskLevel) -> List[StrategyMetadata]:
        """Get all strategies for a specific risk level"""
        return [
            strategy for strategy in self.discovered_strategies.values()
            if strategy.risk_level == risk_level and strategy.is_active
        ]
    
    def get_strategies_by_feature(self, feature: str) -> List[StrategyMetadata]:
        """Get all strategies that have a specific feature"""
        return [
            strategy for strategy in self.discovered_strategies.values()
            if feature in strategy.features and strategy.is_active
        ]
    
    def get_high_performance_strategies(self, min_score: float = 80.0) -> List[StrategyMetadata]:
        """Get strategies with performance score above threshold"""
        return [
            strategy for strategy in self.discovered_strategies.values()
            if strategy.performance_score >= min_score and strategy.is_active
        ]
    
    def update_performance_score(self, strategy_name: str, new_score: float):
        """Update performance score for a strategy"""
        if strategy_name in self.discovered_strategies:
            self.discovered_strategies[strategy_name].performance_score = new_score
            self._save_registry()
            self.logger.info(f"📊 Updated performance score for {strategy_name}: {new_score:.1f}")
    
    def activate_strategy(self, strategy_name: str, active: bool = True):
        """Activate or deactivate a strategy"""
        if strategy_name in self.discovered_strategies:
            self.discovered_strategies[strategy_name].is_active = active
            self._save_registry()
            status = "activated" if active else "deactivated"
            self.logger.info(f"🔄 Strategy {strategy_name} {status}")
    
    def _load_existing_registry(self):
        """Load existing strategy registry from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                for strategy_data in data.get('strategies', []):
                    metadata = StrategyMetadata(
                        name=strategy_data['name'],
                        class_name=strategy_data['class_name'],
                        module_path=strategy_data['module_path'],
                        risk_level=RiskLevel(strategy_data['risk_level']),
                        expected_return_range=tuple(strategy_data['expected_return_range']),
                        typical_holding_period=strategy_data['typical_holding_period'],
                        max_position_size=strategy_data['max_position_size'],
                        description=strategy_data['description'],
                        features=strategy_data['features'],
                        requires_ml=strategy_data['requires_ml'],
                        supports_symbols=strategy_data['supports_symbols'],
                        preferred_timeframes=strategy_data['preferred_timeframes'],
                        performance_score=strategy_data['performance_score'],
                        is_active=strategy_data['is_active']
                    )
                    
                    self.discovered_strategies[metadata.name] = metadata
                
                self.logger.info(f"✅ Loaded {len(self.discovered_strategies)} strategies from registry")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load existing registry: {e}")
    
    def _save_registry(self):
        """Save strategy registry to file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            registry_data = {
                'timestamp': str(datetime.now()),
                'total_strategies': len(self.discovered_strategies),
                'strategies': []
            }
            
            for metadata in self.discovered_strategies.values():
                strategy_data = {
                    'name': metadata.name,
                    'class_name': metadata.class_name,
                    'module_path': metadata.module_path,
                    'risk_level': metadata.risk_level.value,
                    'expected_return_range': list(metadata.expected_return_range),
                    'typical_holding_period': metadata.typical_holding_period,
                    'max_position_size': metadata.max_position_size,
                    'description': metadata.description,
                    'features': metadata.features,
                    'requires_ml': metadata.requires_ml,
                    'supports_symbols': metadata.supports_symbols,
                    'preferred_timeframes': metadata.preferred_timeframes,
                    'performance_score': metadata.performance_score,
                    'is_active': metadata.is_active
                }
                
                registry_data['strategies'].append(strategy_data)
            
            with open(self.config_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            self.logger.info(f"✅ Strategy registry saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save strategy registry: {e}")
    
    def print_strategy_summary(self):
        """Print comprehensive strategy discovery summary"""
        print("\n🔍 STRATEGY DISCOVERY SUMMARY")
        print("=" * 60)
        
        # Group by risk level
        risk_groups = {}
        for strategy in self.discovered_strategies.values():
            risk_level = strategy.risk_level.value
            if risk_level not in risk_groups:
                risk_groups[risk_level] = []
            risk_groups[risk_level].append(strategy)
        
        for risk_level, strategies in risk_groups.items():
            print(f"\n🎯 {risk_level} ({len(strategies)} strategies):")
            
            for strategy in sorted(strategies, key=lambda x: x.performance_score, reverse=True):
                status = "🟢" if strategy.is_active else "🔴"
                ml_badge = "🤖" if strategy.requires_ml else "📊"
                
                print(f"  {status} {ml_badge} {strategy.name}")
                print(f"      Return: {strategy.expected_return_range[0]:.1f}%-{strategy.expected_return_range[1]:.1f}% daily")
                print(f"      Period: {strategy.typical_holding_period}")
                print(f"      Max Pos: {strategy.max_position_size:.1f}%")
                print(f"      Score: {strategy.performance_score:.1f}/100")
                print(f"      Features: {', '.join(strategy.features[:3])}{'...' if len(strategy.features) > 3 else ''}")
        
        total_active = sum(1 for s in self.discovered_strategies.values() if s.is_active)
        total_ml = sum(1 for s in self.discovered_strategies.values() if s.requires_ml)
        
        print(f"\n📊 TOTALS:")
        print(f"   Total Strategies: {len(self.discovered_strategies)}")
        print(f"   Active Strategies: {total_active}")
        print(f"   ML-Enhanced: {total_ml}")
        print("=" * 60)

# Example integration
async def main():
    """Example usage"""
    discovery = StrategyAutoDiscovery()
    
    # Discover all strategies
    strategies = await discovery.discover_all_strategies()
    
    # Print summary
    discovery.print_strategy_summary()
    
    # Get specific strategy groups
    high_risk_strategies = discovery.get_strategies_by_risk_level(RiskLevel.AGGRESSIVE)
    ml_strategies = discovery.get_strategies_by_feature('ML/AI')
    top_performers = discovery.get_high_performance_strategies(85.0)
    
    print(f"\n🔥 High Risk Strategies: {len(high_risk_strategies)}")
    print(f"🤖 ML Strategies: {len(ml_strategies)}")
    print(f"🏆 Top Performers (85+): {len(top_performers)}")

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    asyncio.run(main())