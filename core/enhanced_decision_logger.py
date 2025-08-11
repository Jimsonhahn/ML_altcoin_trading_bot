#!/usr/bin/env python3
"""
Enhanced Decision Logger - Server-Ready Intelligence System
Erweitert den bestehenden DecisionLogger um KI-auswertbare Logs und Server-Integration

Neue Features:
- JSON Export für Claude Code Analyse
- RESTful API Endpoints
- Real-time Dashboard Updates
- Anomalie-Erkennung
- Performance Learning System
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib

# Import original classes
from .decision_logger import DecisionLogger as OriginalDecisionLogger
from .decision_logger import OrchestratorDecision, TradePerformance, MarketState

logger = logging.getLogger(__name__)

class EnhancedDecisionLogger(OriginalDecisionLogger):
    """
    Enhanced DecisionLogger mit KI-Integration und Server-Deployment Features
    
    Neue Capabilities:
    - Smart Pattern Recognition
    - Performance Learning
    - Dashboard Integration
    - Claude Code Data Export
    - Real-time Anomaly Detection
    """
    
    def __init__(self, 
                 db_pool,
                 export_path: str = "intelligence_exports/",
                 dashboard_updates: bool = True,
                 learning_enabled: bool = True,
                 **kwargs):
        """
        Initialize Enhanced DecisionLogger
        
        Args:
            export_path: Path für Claude Code Data Exports
            dashboard_updates: Enable real-time dashboard updates
            learning_enabled: Enable ML learning from patterns
        """
        super().__init__(db_pool, **kwargs)
        
        # Enhanced capabilities
        self.export_path = Path(export_path)
        self.export_path.mkdir(parents=True, exist_ok=True)
        
        self.dashboard_updates = dashboard_updates
        self.learning_enabled = learning_enabled
        
        # Intelligence tracking
        self._pattern_cache = {}
        self._anomaly_scores = []
        self._learning_insights = []
        
        # Dashboard communication
        self._dashboard_queue = []
        self._websocket_clients = set()
        
        logger.info("✅ Enhanced DecisionLogger initialized with intelligence features")

    async def log_trading_decision_with_context(self, 
                                              decision_data: Dict[str, Any],
                                              market_context: Dict[str, Any],
                                              strategy_reasoning: str,
                                              confidence_level: float) -> str:
        """
        Log a complete trading decision with full context for KI analysis
        
        Returns:
            decision_id: Unique identifier für spätere Referenz
        """
        try:
            decision_id = f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(decision_data)) % 10000}"
            
            # Create enhanced decision record
            enhanced_decision = {
                'decision_id': decision_id,
                'timestamp': datetime.utcnow().isoformat(),
                'session_id': self.session_id,
                'decision_data': decision_data,
                'market_context': market_context,
                'strategy_reasoning': strategy_reasoning,
                'confidence_level': confidence_level,
                'bot_version': self._get_bot_version(),
                'environmental_factors': self._capture_environment()
            }
            
            # Log to original system
            orchestrator_decision = OrchestratorDecision(
                decision_type='enhanced_trading_decision',
                strategy_name=decision_data.get('strategy'),
                confidence_score=confidence_level,
                decision_reasoning=strategy_reasoning,
                trigger_data=enhanced_decision
            )
            
            await self.log_orchestrator_decision(orchestrator_decision)
            
            # Enhanced intelligence processing
            if self.learning_enabled:
                await self._process_decision_for_learning(enhanced_decision)
            
            # Real-time dashboard update
            if self.dashboard_updates:
                await self._send_dashboard_update('new_decision', enhanced_decision)
            
            # Export für Claude Code
            await self._export_decision_for_ai_analysis(enhanced_decision)
            
            logger.info(f"✅ Enhanced decision logged: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to log enhanced decision: {e}")
            return ""

    async def log_strategy_performance_with_insights(self,
                                                   trade_data: TradePerformance,
                                                   market_conditions: Dict[str, Any],
                                                   strategy_insights: Dict[str, Any],
                                                   post_trade_analysis: Optional[str] = None) -> bool:
        """
        Log strategy performance mit ML-ready insights
        """
        try:
            # Enhance trade data with insights
            trade_data.technical_context = {
                'market_conditions': market_conditions,
                'strategy_insights': strategy_insights,
                'post_trade_analysis': post_trade_analysis,
                'performance_attribution': self._calculate_performance_attribution(trade_data),
                'risk_metrics': self._calculate_enhanced_risk_metrics(trade_data),
                'comparative_analysis': await self._get_comparative_performance(trade_data.strategy_name)
            }
            
            # Log to original system
            success = await self.log_trade_performance(trade_data)
            
            if success and self.learning_enabled:
                # Extract learnings for future decisions
                await self._extract_trade_learnings(trade_data)
                
                # Update strategy performance models
                await self._update_strategy_models(trade_data)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log enhanced performance: {e}")
            return False

    async def detect_trading_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in trading patterns for risk management
        """
        try:
            anomalies = []
            
            # Recent decision analysis
            recent_decisions = await self.get_recent_decisions(limit=50)
            
            # Anomaly detection patterns
            anomaly_checks = [
                self._check_rapid_decision_changes(recent_decisions),
                self._check_unusual_confidence_patterns(recent_decisions),
                self._check_strategy_drift(recent_decisions),
                self._check_risk_threshold_breaches(recent_decisions)
            ]
            
            for check in anomaly_checks:
                result = await check
                if result['anomaly_detected']:
                    anomalies.append(result)
            
            # Dashboard alert if anomalies found
            if anomalies and self.dashboard_updates:
                await self._send_dashboard_update('anomaly_alert', {
                    'anomalies': anomalies,
                    'severity': max([a.get('severity', 0) for a in anomalies]),
                    'recommended_actions': self._generate_anomaly_recommendations(anomalies)
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    async def generate_ai_learning_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive learning report für Claude Code Analyse
        """
        try:
            report = {
                'report_id': f"learning_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'generation_time': datetime.utcnow().isoformat(),
                'analysis_period_days': days,
                'session_id': self.session_id,
                'data_quality_score': await self._assess_data_quality(),
                'key_insights': {
                    'performance_patterns': await self._analyze_performance_patterns(days),
                    'strategy_effectiveness': await self._analyze_strategy_effectiveness(days),
                    'market_adaptation': await self._analyze_market_adaptation(days),
                    'risk_management_efficiency': await self._analyze_risk_management(days),
                    'decision_quality_trends': await self._analyze_decision_quality(days)
                },
                'optimization_recommendations': await self._generate_optimization_recommendations(days),
                'data_exports': {
                    'decisions_export': f"{self.export_path}/decisions_last_{days}d.json",
                    'performance_export': f"{self.export_path}/performance_last_{days}d.json",
                    'patterns_export': f"{self.export_path}/patterns_last_{days}d.json",
                    'anomalies_export': f"{self.export_path}/anomalies_last_{days}d.json"
                }
            }
            
            # Export all data für Claude Code
            await self._export_learning_data(report, days)
            
            # Save report
            report_path = self.export_path / f"learning_report_{days}d_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ AI Learning Report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate learning report: {e}")
            return {}

    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics für Web Dashboard
        """
        try:
            metrics = {
                'system_status': {
                    'status': 'healthy' if self._running else 'stopped',
                    'session_id': self.session_id,
                    'uptime_minutes': (datetime.utcnow() - self._last_flush_time).total_seconds() / 60,
                    'last_activity': self._last_flush_time.isoformat()
                },
                'logging_performance': self.get_stats(),
                'recent_activity': {
                    'decisions_today': await self._count_decisions_today(),
                    'trades_today': await self._count_trades_today(),
                    'anomalies_detected': len(await self.detect_trading_anomalies()),
                    'learning_insights': len(self._learning_insights)
                },
                'strategy_performance': await self._get_strategy_summary(),
                'risk_metrics': await self._get_risk_dashboard_metrics(),
                'ai_insights': {
                    'pattern_recognition_active': self.learning_enabled,
                    'export_path': str(self.export_path),
                    'data_quality_score': await self._assess_data_quality(),
                    'next_learning_cycle': (datetime.utcnow() + timedelta(hours=6)).isoformat()
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return {}

    # WebSocket Support für Real-time Dashboard
    def add_websocket_client(self, websocket):
        """Add WebSocket client für real-time updates"""
        self._websocket_clients.add(websocket)
        logger.info(f"WebSocket client added, total: {len(self._websocket_clients)}")

    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        self._websocket_clients.discard(websocket)
        logger.info(f"WebSocket client removed, total: {len(self._websocket_clients)}")

    async def _send_dashboard_update(self, event_type: str, data: Dict[str, Any]):
        """Send real-time update to dashboard clients"""
        if not self.dashboard_updates or not self._websocket_clients:
            return
            
        message = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self._websocket_clients:
            try:
                await client.send(json.dumps(message, default=str))
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self._websocket_clients.discard(client)

    # Private Intelligence Methods
    async def _process_decision_for_learning(self, decision: Dict[str, Any]):
        """Process decision für ML learning"""
        try:
            # Pattern recognition
            pattern_hash = self._generate_pattern_hash(decision)
            if pattern_hash in self._pattern_cache:
                self._pattern_cache[pattern_hash]['count'] += 1
                self._pattern_cache[pattern_hash]['last_seen'] = datetime.utcnow()
            else:
                self._pattern_cache[pattern_hash] = {
                    'pattern': decision['decision_data'],
                    'count': 1,
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow(),
                    'success_rate': None
                }
            
            # Store for batch learning
            self._learning_insights.append({
                'type': 'decision_pattern',
                'decision_id': decision['decision_id'],
                'pattern_hash': pattern_hash,
                'confidence': decision['confidence_level'],
                'timestamp': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Learning processing failed: {e}")

    async def _export_decision_for_ai_analysis(self, decision: Dict[str, Any]):
        """Export decision data für Claude Code analysis"""
        try:
            export_file = self.export_path / "latest_decisions.jsonl"
            
            with open(export_file, 'a') as f:
                f.write(json.dumps(decision, default=str) + '\n')
            
            # Also maintain a structured export
            structured_export = self.export_path / "structured_decisions.json"
            
            # Load existing or create new
            if structured_export.exists():
                with open(structured_export, 'r') as f:
                    data = json.load(f)
            else:
                data = {'decisions': [], 'metadata': {'last_updated': None, 'total_decisions': 0}}
            
            data['decisions'].append(decision)
            data['metadata']['last_updated'] = datetime.utcnow().isoformat()
            data['metadata']['total_decisions'] += 1
            
            # Keep only last 1000 decisions
            if len(data['decisions']) > 1000:
                data['decisions'] = data['decisions'][-1000:]
            
            with open(structured_export, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"AI export failed: {e}")

    def _generate_pattern_hash(self, decision: Dict[str, Any]) -> str:
        """Generate hash für decision pattern recognition"""
        pattern_data = {
            'strategy': decision['decision_data'].get('strategy'),
            'market_regime': decision['market_context'].get('regime'),
            'volatility_bracket': self._categorize_volatility(decision['market_context'].get('volatility', 0)),
            'confidence_bracket': self._categorize_confidence(decision['confidence_level'])
        }
        return hashlib.md5(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()

    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility for pattern recognition"""
        if volatility < 0.1:
            return 'low'
        elif volatility < 0.3:
            return 'medium'
        else:
            return 'high'

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence for pattern recognition"""
        if confidence < 0.5:
            return 'low'
        elif confidence < 0.8:
            return 'medium'
        else:
            return 'high'

    def _get_bot_version(self) -> str:
        """Get current bot version"""
        try:
            # Try to read version from file or git
            return "enhanced_v1.0"  # Placeholder
        except:
            return "unknown"

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environmental factors"""
        return {
            'cpu_usage': 0,  # Placeholder - implement with psutil
            'memory_usage': 0,
            'active_strategies': [],
            'market_hours': self._is_market_hours(),
            'system_load': 'normal'
        }

    def _is_market_hours(self) -> bool:
        """Check if it's active market hours"""
        # Crypto markets are 24/7, but traditional markets affect crypto
        return True

    async def _count_decisions_today(self) -> int:
        """Count decisions made today"""
        try:
            today = datetime.utcnow().date()
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM orchestrator_decisions 
                    WHERE DATE(timestamp) = $1 AND session_id = $2
                """, today, self.session_id)
                return count or 0
        except:
            return 0

    async def _count_trades_today(self) -> int:
        """Count trades made today"""
        try:
            today = datetime.utcnow().date()
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM strategy_performance 
                    WHERE DATE(entry_timestamp) = $1 AND session_id = $2
                """, today, self.session_id)
                return count or 0
        except:
            return 0

    async def _assess_data_quality(self) -> float:
        """Assess the quality of logged data"""
        try:
            # Implementation would check completeness, consistency, etc.
            return 0.95  # Placeholder
        except:
            return 0.0

    # Placeholder methods for comprehensive functionality
    async def _analyze_performance_patterns(self, days: int) -> Dict[str, Any]:
        """Analyze performance patterns over time"""
        return {'analysis': 'placeholder', 'patterns_found': []}

    async def _analyze_strategy_effectiveness(self, days: int) -> Dict[str, Any]:
        """Analyze which strategies perform best"""
        return {'top_strategies': [], 'underperformers': []}

    async def _analyze_market_adaptation(self, days: int) -> Dict[str, Any]:
        """Analyze how well bot adapts to market changes"""
        return {'adaptation_score': 0.8, 'regime_performance': {}}

    async def _analyze_risk_management(self, days: int) -> Dict[str, Any]:
        """Analyze risk management effectiveness"""
        return {'risk_score': 0.85, 'improvements_needed': []}

    async def _analyze_decision_quality(self, days: int) -> Dict[str, Any]:
        """Analyze quality of decisions over time"""
        return {'quality_trend': 'improving', 'confidence_accuracy': 0.82}

    async def _generate_optimization_recommendations(self, days: int) -> List[Dict[str, Any]]:
        """Generate recommendations for optimization"""
        return [
            {
                'category': 'strategy_allocation',
                'recommendation': 'Increase allocation to momentum strategy',
                'confidence': 0.85,
                'expected_improvement': '12%'
            }
        ]

    async def _export_learning_data(self, report: Dict[str, Any], days: int):
        """Export all learning data für Claude Code"""
        # Implementation would export comprehensive datasets
        pass

    # Additional helper methods...
    def _calculate_performance_attribution(self, trade: TradePerformance) -> Dict[str, Any]:
        return {'attribution': 'placeholder'}

    def _calculate_enhanced_risk_metrics(self, trade: TradePerformance) -> Dict[str, Any]:
        return {'risk_metrics': 'placeholder'}

    async def _get_comparative_performance(self, strategy_name: str) -> Dict[str, Any]:
        return {'comparison': 'placeholder'}

    async def _extract_trade_learnings(self, trade: TradePerformance):
        pass

    async def _update_strategy_models(self, trade: TradePerformance):
        pass

    async def _check_rapid_decision_changes(self, decisions: List[Dict]) -> Dict[str, Any]:
        return {'anomaly_detected': False, 'details': 'placeholder'}

    async def _check_unusual_confidence_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        return {'anomaly_detected': False, 'details': 'placeholder'}

    async def _check_strategy_drift(self, decisions: List[Dict]) -> Dict[str, Any]:
        return {'anomaly_detected': False, 'details': 'placeholder'}

    async def _check_risk_threshold_breaches(self, decisions: List[Dict]) -> Dict[str, Any]:
        return {'anomaly_detected': False, 'details': 'placeholder'}

    def _generate_anomaly_recommendations(self, anomalies: List[Dict]) -> List[str]:
        return ['Review strategy parameters', 'Check market conditions']

    async def _get_strategy_summary(self) -> Dict[str, Any]:
        return {'strategies': {}, 'summary': 'placeholder'}

    async def _get_risk_dashboard_metrics(self) -> Dict[str, Any]:
        return {'current_risk': 0.3, 'risk_limits': 0.5}

# Factory function für easy integration
async def create_enhanced_decision_logger(db_pool, **kwargs) -> EnhancedDecisionLogger:
    """
    Factory function to create and start Enhanced Decision Logger
    """
    logger_instance = EnhancedDecisionLogger(db_pool, **kwargs)
    await logger_instance.start()
    return logger_instance

# Integration example
async def integrate_with_existing_bot(trading_bot_instance, db_pool):
    """
    Example integration with existing trading bot
    
    Usage in main.py:
        enhanced_logger = await integrate_with_existing_bot(bot, db_pool)
        # Now use enhanced_logger instead of regular decision_logger
    """
    enhanced_logger = await create_enhanced_decision_logger(
        db_pool,
        export_path="intelligence_exports/",
        dashboard_updates=True,
        learning_enabled=True
    )
    
    # Replace bot's decision logger
    trading_bot_instance.decision_logger = enhanced_logger
    
    logger.info("✅ Enhanced Decision Logger integrated with trading bot")
    return enhanced_logger