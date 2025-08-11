#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Daily Analysis Runner
Täglich laufendes Script für komplette Analyse-Pipeline

Dieses Script:
- Führt täglich alle Analysen aus
- Koordiniert Learning Pipeline, Pattern Detection und Backtest Improvements
- Generiert Berichte und Benachrichtigungen
- Integriert Ergebnisse in das Trading System
"""

import asyncio
import sys
import os
import logging
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import asyncpg
from analysis.learning_pipeline import LearningPipeline
from analysis.pattern_detector import PatternDetector
from analysis.backtest_improvements import BacktestImprovements
from utils.notifier import NotificationManager
from utils.logger import setup_logging
from config.settings import DATABASE_CONFIG

logger = logging.getLogger(__name__)

class DailyAnalysisRunner:
    """Hauptklasse für tägliche Analyse-Ausführung"""
    
    def __init__(self, config_path: Optional[str] = None, dry_run: bool = False):
        """
        Initialize Daily Analysis Runner
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            dry_run: Wenn True, keine echten Änderungen am System
        """
        self.config_path = config_path
        self.dry_run = dry_run
        self.db_pool = None
        
        # Analysis components
        self.learning_pipeline = None
        self.pattern_detector = None
        self.backtest_improvements = None
        self.notification_manager = None
        
        # Results storage
        self.results_dir = Path("analysis/daily_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = Path("analysis/daily_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Konfiguration laden"""
        default_config = {
            'analysis': {
                'lookback_days': 30,
                'min_trades_for_analysis': 10,
                'confidence_threshold': 0.7,
                'enable_learning_pipeline': True,
                'enable_pattern_detection': True,
                'enable_backtest_improvements': True
            },
            'database': DATABASE_CONFIG,
            'notifications': {
                'enabled': True,
                'telegram_enabled': False,
                'email_enabled': False,
                'min_importance_level': 'medium'
            },
            'integration': {
                'auto_apply_improvements': False,
                'max_auto_improvements_per_day': 3,
                'min_confidence_for_auto_apply': 0.9
            },
            'scheduling': {
                'run_time': '06:00',  # UTC time
                'timezone': 'UTC',
                'retry_attempts': 3,
                'retry_delay_minutes': 30
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config

    async def initialize(self):
        """System initialisieren"""
        logger.info("🚀 Initializing Daily Analysis Runner...")
        
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(**self.config['database'])
            
            # Initialize analysis components
            if self.config['analysis']['enable_learning_pipeline']:
                self.learning_pipeline = LearningPipeline(
                    db_pool=self.db_pool,
                    lookback_days=self.config['analysis']['lookback_days'],
                    min_trades_for_analysis=self.config['analysis']['min_trades_for_analysis'],
                    confidence_threshold=self.config['analysis']['confidence_threshold']
                )
            
            if self.config['analysis']['enable_pattern_detection']:
                self.pattern_detector = PatternDetector(
                    lookback_days=self.config['analysis']['lookback_days'],
                    min_pattern_frequency=self.config['analysis']['min_trades_for_analysis']
                )
            
            if self.config['analysis']['enable_backtest_improvements']:
                # Mock data handlers (ersetzen durch echte Implementierung)
                self.backtest_improvements = BacktestImprovements(
                    data_handler=None,  # Würde echten Data Handler verwenden
                    backtest_engine=None  # Würde echte Backtest Engine verwenden
                )
            
            # Notification system
            if self.config['notifications']['enabled']:
                self.notification_manager = NotificationManager(
                    telegram_enabled=self.config['notifications']['telegram_enabled'],
                    email_enabled=self.config['notifications']['email_enabled']
                )
            
            logger.info("✅ Daily Analysis Runner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Daily Analysis Runner: {e}")
            raise

    async def run_daily_analysis(self) -> Dict[str, Any]:
        """Komplette tägliche Analyse ausführen"""
        logger.info("📊 Starting daily analysis pipeline...")
        
        start_time = datetime.utcnow()
        analysis_results = {
            'timestamp': start_time.isoformat(),
            'success': False,
            'error': None,
            'duration_seconds': 0,
            'components_run': [],
            'summary': {}
        }
        
        try:
            # 1. Learning Pipeline ausführen
            learning_results = None
            if self.learning_pipeline:
                logger.info("🧠 Running Learning Pipeline...")
                learning_results = await self.learning_pipeline.run_full_analysis()
                analysis_results['components_run'].append('learning_pipeline')
                analysis_results['summary']['learning_pipeline'] = {
                    'insights_generated': learning_results.get('insights_generated', {}),
                    'key_findings': learning_results.get('key_findings', [])
                }
                logger.info("✅ Learning Pipeline completed")
            
            # 2. Pattern Detection ausführen
            pattern_results = None
            if self.pattern_detector:
                logger.info("🔍 Running Pattern Detection...")
                
                # Daten für Pattern Detection laden
                trades_df, market_df, decisions_df = await self._load_pattern_detection_data()
                
                pattern_results = self.pattern_detector.analyze_patterns(
                    trades_df, market_df, decisions_df
                )
                analysis_results['components_run'].append('pattern_detection')
                analysis_results['summary']['pattern_detection'] = {
                    'patterns_found': pattern_results.get('patterns_found', {}),
                    'key_insights': pattern_results.get('key_insights', [])
                }
                logger.info("✅ Pattern Detection completed")
            
            # 3. Backtest Improvements ausführen
            improvement_results = None
            if self.backtest_improvements and learning_results and pattern_results:
                logger.info("🧪 Running Backtest Improvements...")
                
                validation_report = await self.backtest_improvements.validate_learning_insights(
                    learning_results, pattern_results
                )
                
                improvement_results = {
                    'validation_report': validation_report,
                    'improvements_validated': validation_report.improvements_validated,
                    'risk_level': validation_report.risk_level
                }
                
                analysis_results['components_run'].append('backtest_improvements')
                analysis_results['summary']['backtest_improvements'] = {
                    'improvements_tested': validation_report.improvements_tested,
                    'improvements_validated': validation_report.improvements_validated,
                    'risk_level': validation_report.risk_level
                }
                logger.info("✅ Backtest Improvements completed")
            
            # 4. Ergebnisse konsolidieren
            consolidated_results = await self._consolidate_results(
                learning_results, pattern_results, improvement_results
            )
            
            # 5. Berichte generieren
            daily_report = await self._generate_daily_report(consolidated_results)
            
            # 6. Benachrichtigungen senden
            if self.notification_manager:
                await self._send_notifications(daily_report)
            
            # 7. Integration in Trading System (falls aktiviert)
            if not self.dry_run and self.config['integration']['auto_apply_improvements']:
                await self._apply_validated_improvements(improvement_results)
            
            # Erfolg markieren
            analysis_results['success'] = True
            analysis_results['consolidated_results'] = consolidated_results
            analysis_results['daily_report'] = daily_report
            
            end_time = datetime.utcnow()
            analysis_results['duration_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Daily analysis completed successfully in {analysis_results['duration_seconds']:.1f}s")
            
            return analysis_results
            
        except Exception as e:
            error_msg = f"Daily analysis failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            analysis_results['success'] = False
            analysis_results['error'] = error_msg
            
            # Fehler-Benachrichtigung senden
            if self.notification_manager:
                await self.notification_manager.send_error_notification(
                    "Daily Analysis Failed", error_msg
                )
            
            return analysis_results

    async def _load_pattern_detection_data(self) -> tuple:
        """Daten für Pattern Detection laden"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config['analysis']['lookback_days'])
            
            async with self.db_pool.acquire() as conn:
                # Trades laden
                trades_query = """
                    SELECT * FROM strategy_performance 
                    WHERE timestamp >= $1 
                    ORDER BY timestamp DESC
                """
                trades_rows = await conn.fetch(trades_query, cutoff_date)
                trades_df = pd.DataFrame([dict(row) for row in trades_rows])
                
                # Market data laden (falls verfügbar)
                try:
                    market_query = """
                        SELECT * FROM market_states 
                        WHERE timestamp >= $1 
                        ORDER BY timestamp DESC
                    """
                    market_rows = await conn.fetch(market_query, cutoff_date)
                    market_df = pd.DataFrame([dict(row) for row in market_rows])
                except:
                    market_df = pd.DataFrame()
                
                # Decisions laden (falls verfügbar)
                try:
                    decisions_query = """
                        SELECT * FROM orchestrator_decisions 
                        WHERE timestamp >= $1 
                        ORDER BY timestamp DESC
                    """
                    decisions_rows = await conn.fetch(decisions_query, cutoff_date)
                    decisions_df = pd.DataFrame([dict(row) for row in decisions_rows])
                except:
                    decisions_df = pd.DataFrame()
            
            logger.info(f"Loaded {len(trades_df)} trades, {len(market_df)} market states, {len(decisions_df)} decisions")
            
            return trades_df, market_df, decisions_df
            
        except Exception as e:
            logger.error(f"Failed to load pattern detection data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    async def _consolidate_results(self, learning_results: Optional[Dict], 
                                 pattern_results: Optional[Dict],
                                 improvement_results: Optional[Dict]) -> Dict[str, Any]:
        """Alle Analyseergebnisse konsolidieren"""
        logger.info("📋 Consolidating analysis results...")
        
        consolidated = {
            'timestamp': datetime.utcnow().isoformat(),
            'analysis_summary': {
                'components_run': [],
                'total_insights': 0,
                'high_priority_items': 0,
                'actionable_recommendations': []
            },
            'key_findings': [],
            'priority_actions': [],
            'risk_alerts': [],
            'performance_metrics': {}
        }
        
        # Learning Pipeline Ergebnisse
        if learning_results:
            consolidated['analysis_summary']['components_run'].append('learning_pipeline')
            consolidated['key_findings'].extend(learning_results.get('key_findings', []))
            consolidated['analysis_summary']['actionable_recommendations'].extend(
                learning_results.get('recommendations', [])
            )
            
            # Insights zählen
            insights = learning_results.get('insights_generated', {})
            total_insights = sum(insights.values()) if isinstance(insights, dict) else 0
            consolidated['analysis_summary']['total_insights'] += total_insights
        
        # Pattern Detection Ergebnisse
        if pattern_results:
            consolidated['analysis_summary']['components_run'].append('pattern_detection')
            consolidated['key_findings'].extend(pattern_results.get('key_insights', []))
            consolidated['analysis_summary']['actionable_recommendations'].extend(
                pattern_results.get('actionable_recommendations', [])
            )
            
            # Pattern counts
            patterns = pattern_results.get('patterns_found', {})
            pattern_count = sum(patterns.values()) if isinstance(patterns, dict) else 0
            consolidated['analysis_summary']['total_insights'] += pattern_count
            
            # Risk alerts aus dangerous conditions
            if 'dangerous_conditions' in pattern_results:
                consolidated['risk_alerts'].append({
                    'type': 'dangerous_market_conditions',
                    'count': pattern_results['patterns_found'].get('dangerous_conditions', 0),
                    'severity': 'high'
                })
        
        # Backtest Improvements Ergebnisse
        if improvement_results:
            consolidated['analysis_summary']['components_run'].append('backtest_improvements')
            
            validation_report = improvement_results.get('validation_report')
            if validation_report:
                consolidated['priority_actions'].extend(validation_report.recommendations)
                
                # High priority wenn viele Verbesserungen validiert
                if validation_report.improvements_validated > 2:
                    consolidated['analysis_summary']['high_priority_items'] += 1
                
                # Risk alert wenn hohes Risiko
                if validation_report.risk_level == 'HIGH':
                    consolidated['risk_alerts'].append({
                        'type': 'high_implementation_risk',
                        'details': f"{validation_report.improvements_validated} improvements with high risk",
                        'severity': 'medium'
                    })
        
        # Performance metrics aggregieren
        if learning_results and 'optimal_weights' in learning_results:
            consolidated['performance_metrics']['optimal_weights'] = learning_results['optimal_weights']
        
        # Top priority actions identifizieren
        all_recommendations = consolidated['analysis_summary']['actionable_recommendations']
        consolidated['priority_actions'].extend(all_recommendations[:5])  # Top 5
        
        logger.info(f"Consolidated results: {consolidated['analysis_summary']['total_insights']} insights, "
                   f"{len(consolidated['priority_actions'])} priority actions")
        
        return consolidated

    async def _generate_daily_report(self, consolidated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Täglichen Bericht generieren"""
        logger.info("📊 Generating daily report...")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'report_id': f"DAILY_REPORT_{timestamp}",
            'timestamp': datetime.utcnow().isoformat(),
            'executive_summary': self._generate_executive_summary(consolidated_results),
            'detailed_findings': consolidated_results,
            'recommendations': {
                'immediate_actions': consolidated_results.get('priority_actions', [])[:3],
                'short_term_improvements': consolidated_results.get('priority_actions', [])[3:6],
                'monitoring_items': consolidated_results.get('risk_alerts', [])
            },
            'performance_overview': consolidated_results.get('performance_metrics', {}),
            'next_analysis_scheduled': (datetime.utcnow() + timedelta(days=1)).isoformat()
        }
        
        # Bericht speichern
        report_file = self.reports_dir / f"daily_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # HTML-Report generieren (optional)
        await self._generate_html_report(report, timestamp)
        
        logger.info(f"Daily report generated: {report_file}")
        
        return report

    def _generate_executive_summary(self, consolidated_results: Dict[str, Any]) -> str:
        """Executive Summary generieren"""
        summary_parts = []
        
        # Komponenten
        components = consolidated_results['analysis_summary']['components_run']
        summary_parts.append(f"Ran {len(components)} analysis components: {', '.join(components)}")
        
        # Insights
        total_insights = consolidated_results['analysis_summary']['total_insights']
        summary_parts.append(f"Generated {total_insights} actionable insights")
        
        # Priority items
        priority_count = len(consolidated_results.get('priority_actions', []))
        if priority_count > 0:
            summary_parts.append(f"Identified {priority_count} priority actions requiring attention")
        
        # Risk alerts
        risk_count = len(consolidated_results.get('risk_alerts', []))
        if risk_count > 0:
            summary_parts.append(f"⚠️ {risk_count} risk alerts detected")
        
        # Overall assessment
        if risk_count == 0 and priority_count > 0:
            summary_parts.append("System is performing well with optimization opportunities identified")
        elif risk_count > 0:
            summary_parts.append("Attention required for risk management")
        else:
            summary_parts.append("System is stable with no immediate action required")
        
        return ". ".join(summary_parts) + "."

    async def _generate_html_report(self, report: Dict[str, Any], timestamp: str):
        """HTML-Version des Berichts generieren"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Daily Trading Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; }}
                    .summary {{ background-color: #e8f5e8; padding: 15px; margin: 20px 0; }}
                    .alert {{ background-color: #ffe8e8; padding: 15px; margin: 10px 0; }}
                    .recommendation {{ background-color: #e8f0ff; padding: 10px; margin: 5px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Daily Trading Analysis Report</h1>
                    <p>Generated: {report['timestamp']}</p>
                    <p>Report ID: {report['report_id']}</p>
                </div>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>{report['executive_summary']}</p>
                </div>
                
                <h2>Immediate Actions Required</h2>
                {self._format_recommendations_html(report['recommendations']['immediate_actions'])}
                
                <h2>Risk Alerts</h2>
                {self._format_alerts_html(report['detailed_findings'].get('risk_alerts', []))}
                
                <h2>Performance Overview</h2>
                {self._format_performance_html(report['performance_overview'])}
                
                <h2>Next Steps</h2>
                <p>Next analysis scheduled for: {report['next_analysis_scheduled']}</p>
            </body>
            </html>
            """
            
            html_file = self.reports_dir / f"daily_report_{timestamp}.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
                
            logger.info(f"HTML report generated: {html_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")

    def _format_recommendations_html(self, recommendations: list) -> str:
        """Recommendations als HTML formatieren"""
        if not recommendations:
            return "<p>No immediate actions required.</p>"
        
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation">{i}. {rec}</div>'
        return html

    def _format_alerts_html(self, alerts: list) -> str:
        """Alerts als HTML formatieren"""
        if not alerts:
            return "<p>No risk alerts.</p>"
        
        html = ""
        for alert in alerts:
            html += f'<div class="alert">⚠️ {alert.get("type", "Unknown")}: {alert.get("details", "No details")}</div>'
        return html

    def _format_performance_html(self, performance: dict) -> str:
        """Performance metrics als HTML formatieren"""
        if not performance:
            return "<p>No performance metrics available.</p>"
        
        html = ""
        for key, value in performance.items():
            html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        return html

    async def _send_notifications(self, daily_report: Dict[str, Any]):
        """Benachrichtigungen senden"""
        if not self.notification_manager:
            return
        
        logger.info("📱 Sending notifications...")
        
        try:
            # Bestimme Wichtigkeitslevel
            risk_alerts = daily_report['detailed_findings'].get('risk_alerts', [])
            priority_actions = daily_report['recommendations'].get('immediate_actions', [])
            
            if risk_alerts:
                importance = 'high'
            elif priority_actions:
                importance = 'medium'
            else:
                importance = 'low'
            
            # Sende nur wenn Wichtigkeit über Threshold
            min_importance = self.config['notifications']['min_importance_level']
            importance_levels = {'low': 1, 'medium': 2, 'high': 3}
            
            if importance_levels[importance] >= importance_levels[min_importance]:
                
                message = f"""
📊 Daily Trading Analysis Complete

{daily_report['executive_summary']}

🎯 Immediate Actions: {len(priority_actions)}
⚠️ Risk Alerts: {len(risk_alerts)}

Report ID: {daily_report['report_id']}
                """.strip()
                
                await self.notification_manager.send_notification(
                    title="Daily Analysis Report",
                    message=message,
                    importance=importance
                )
                
                logger.info("✅ Notifications sent successfully")
            else:
                logger.info(f"Skipped notifications (importance {importance} < threshold {min_importance})")
                
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    async def _apply_validated_improvements(self, improvement_results: Optional[Dict]):
        """Validierte Verbesserungen automatisch anwenden"""
        if not improvement_results or self.dry_run:
            return
        
        logger.info("🔧 Applying validated improvements...")
        
        try:
            validation_report = improvement_results.get('validation_report')
            if not validation_report:
                return
            
            max_auto_improvements = self.config['integration']['max_auto_improvements_per_day']
            min_confidence = self.config['integration']['min_confidence_for_auto_apply']
            
            # Hier würde die echte Integration in das Trading System erfolgen
            # Für jetzt nur logging
            
            applied_count = 0
            logger.info(f"Would apply up to {max_auto_improvements} improvements with confidence >= {min_confidence}")
            
            logger.info(f"✅ Applied {applied_count} validated improvements")
            
        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")

    async def cleanup(self):
        """Aufräumen nach Analyse"""
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("🧹 Cleanup completed")

def parse_arguments():
    """Command line arguments parsen"""
    parser = argparse.ArgumentParser(description="Run daily trading analysis")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis without applying changes"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    parser.add_argument(
        "--components",
        nargs='+',
        choices=['learning', 'patterns', 'backtests'],
        default=['learning', 'patterns', 'backtests'],
        help="Analysis components to run"
    )
    
    return parser.parse_args()

async def main():
    """Hauptfunktion"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    logger.info("🚀 Starting Daily Analysis Runner")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Components: {args.components}")
    
    runner = None
    
    try:
        # Initialize runner
        runner = DailyAnalysisRunner(
            config_path=args.config,
            dry_run=args.dry_run
        )
        
        await runner.initialize()
        
        # Run analysis
        results = await runner.run_daily_analysis()
        
        if results['success']:
            logger.info("🎉 Daily analysis completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("DAILY ANALYSIS SUMMARY")
            print("="*50)
            print(f"Duration: {results['duration_seconds']:.1f} seconds")
            print(f"Components run: {', '.join(results['components_run'])}")
            
            if 'consolidated_results' in results:
                summary = results['consolidated_results']['analysis_summary']
                print(f"Total insights: {summary['total_insights']}")
                print(f"High priority items: {summary['high_priority_items']}")
                print(f"Actionable recommendations: {len(summary['actionable_recommendations'])}")
            
            print("="*50)
            
            # Exit code 0 for success
            return 0
        else:
            logger.error("❌ Daily analysis failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        if runner:
            await runner.cleanup()

if __name__ == "__main__":
    # Import pandas here to avoid import at module level
    import pandas as pd
    
    # Run main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)