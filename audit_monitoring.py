#!/usr/bin/env python3
"""
Monitoring & Alerting Systems Audit
Prüft Überwachung, Benachrichtigungen und Live-Dashboards
"""
import os
import json
from pathlib import Path

def check_monitoring_system():
    """Prüfe Monitoring-System"""
    print("🔍 Checking Monitoring System...")
    print("=" * 50)
    
    monitoring_score = 0
    features = []
    monitoring_components = {
        'has_alerting_system': False,
        'has_advanced_logging': False,
        'has_web_dashboard': False,
        'has_health_monitoring': False,
        'has_performance_tracking': False,
        'has_real_time_metrics': False
    }
    
    # Check für Alerting System
    alerting_files = [
        'utils/alerting.py',
        'core/notifications.py', 
        'utils/notifier.py',
        'utils/notifier_clean.py',
        'utils/notifier_final.py'
    ]
    
    for file_path in alerting_files:
        if os.path.exists(file_path):
            monitoring_components['has_alerting_system'] = True
            features.append(f"Alerting System: {file_path}")
            print(f"✅ Alerting system found: {file_path}")
            break
    
    if not monitoring_components['has_alerting_system']:
        print("❌ No alerting system found")
    
    # Check für Advanced Logging
    if os.path.exists('utils/logger.py'):
        with open('utils/logger.py', 'r') as f:
            logger_content = f.read()
        
        if 'critical' in logger_content and 'error' in logger_content:
            monitoring_components['has_advanced_logging'] = True
            features.append("Advanced Logging System")
            print("✅ Advanced logging system found")
        else:
            print("❌ Basic logging only")
    else:
        print("❌ No logging system found")
    
    # Check für Web Dashboard
    dashboard_paths = [
        'web/app.py',
        'dashboard/',
        'api/app.py'
    ]
    
    for path in dashboard_paths:
        if os.path.exists(path):
            monitoring_components['has_web_dashboard'] = True
            features.append(f"Web Dashboard: {path}")
            print(f"✅ Web dashboard found: {path}")
            break
    
    if not monitoring_components['has_web_dashboard']:
        print("❌ No web dashboard found")
    
    # Check für Health Monitoring
    health_files = [f for f in os.listdir('.') if 'health' in f.lower()]
    if health_files or os.path.exists('api/health.py'):
        monitoring_components['has_health_monitoring'] = True
        features.append(f"Health Monitoring: {health_files}")
        print(f"✅ Health monitoring found: {health_files}")
    else:
        print("❌ No health monitoring found")
    
    # Check für Performance Tracking
    performance_indicators = ['performance', 'metrics', 'analytics']
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules']]
        for file in files:
            if any(indicator in file.lower() for indicator in performance_indicators):
                monitoring_components['has_performance_tracking'] = True
                features.append(f"Performance tracking: {file}")
                print(f"✅ Performance tracking: {file}")
                break
        if monitoring_components['has_performance_tracking']:
            break
    
    if not monitoring_components['has_performance_tracking']:
        print("❌ No performance tracking found")
    
    # Check für Real-time Metrics
    realtime_files = [
        'dashboard/src/hooks/useWebSocket.js',
        'api/websocket/',
        'core/real_time_monitor.py'
    ]
    
    for file_path in realtime_files:
        if os.path.exists(file_path):
            monitoring_components['has_real_time_metrics'] = True
            features.append(f"Real-time metrics: {file_path}")
            print(f"✅ Real-time metrics: {file_path}")
            break
    
    if not monitoring_components['has_real_time_metrics']:
        print("❌ No real-time metrics found")
    
    return monitoring_components, features

def analyze_notification_channels():
    """Analysiere Benachrichtigungskanäle"""
    print("\n🔍 Analyzing Notification Channels...")
    print("=" * 50)
    
    notification_channels = {
        'telegram': False,
        'email': False,
        'slack': False,
        'webhook': False,
        'sms': False,
        'desktop': False
    }
    
    channel_files = []
    
    # Check utils/notifier files
    notifier_files = [
        'utils/notifier.py',
        'utils/notifier_clean.py', 
        'utils/notifier_final.py',
        'telegram_integration_ready.py',
        'setup_telegram.py'
    ]
    
    for file_path in notifier_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Analyzing {file_path}:")
            
            # Check für spezifische Kanäle
            if 'telegram' in content:
                notification_channels['telegram'] = True
                channel_files.append(f"Telegram: {file_path}")
                print("✅ Telegram integration")
            
            if 'email' in content or 'smtp' in content:
                notification_channels['email'] = True
                channel_files.append(f"Email: {file_path}")
                print("✅ Email integration")
            
            if 'slack' in content:
                notification_channels['slack'] = True
                channel_files.append(f"Slack: {file_path}")
                print("✅ Slack integration")
            
            if 'webhook' in content:
                notification_channels['webhook'] = True
                channel_files.append(f"Webhook: {file_path}")
                print("✅ Webhook integration")
    
    # Zusammenfassung
    active_channels = sum(notification_channels.values())
    print(f"\n📊 Active notification channels: {active_channels}/6")
    
    return notification_channels, channel_files, active_channels

def check_dashboard_capabilities():
    """Prüfe Dashboard-Funktionalitäten"""
    print("\n🔍 Checking Dashboard Capabilities...")
    print("=" * 50)
    
    dashboard_features = {
        'has_real_time_updates': False,
        'has_charts_visualization': False,
        'has_performance_metrics': False,
        'has_trade_history': False,
        'has_system_status': False,
        'has_strategy_controls': False,
        'has_risk_monitoring': False
    }
    
    # Check Dashboard-Verzeichnis
    dashboard_dir = 'dashboard/src/components'
    
    if os.path.exists(dashboard_dir):
        component_files = [f for f in os.listdir(dashboard_dir) if f.endswith('.js')]
        print(f"Found {len(component_files)} dashboard components")
        
        for component in component_files:
            component_path = os.path.join(dashboard_dir, component)
            with open(component_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Analyzing {component}:")
            
            # Check Features
            if 'websocket' in content or 'real' in content and 'time' in content:
                dashboard_features['has_real_time_updates'] = True
                print("✅ Real-time updates")
            
            if 'chart' in content or 'graph' in content:
                dashboard_features['has_charts_visualization'] = True
                print("✅ Charts/Visualization")
            
            if 'performance' in content or 'pnl' in content:
                dashboard_features['has_performance_metrics'] = True
                print("✅ Performance metrics")
            
            if 'trade' in content and 'history' in content:
                dashboard_features['has_trade_history'] = True
                print("✅ Trade history")
            
            if 'system' in content and ('health' in content or 'status' in content):
                dashboard_features['has_system_status'] = True
                print("✅ System status")
            
            if 'strategy' in content and ('control' in content or 'config' in content):
                dashboard_features['has_strategy_controls'] = True
                print("✅ Strategy controls")
            
            if 'risk' in content:
                dashboard_features['has_risk_monitoring'] = True
                print("✅ Risk monitoring")
    
    else:
        print("❌ No dashboard components found")
    
    return dashboard_features

def analyze_alert_rules():
    """Analysiere Alert-Regeln"""
    print("\n🔍 Analyzing Alert Rules...")
    print("=" * 50)
    
    alert_rules = {
        'drawdown_alerts': False,
        'strategy_error_alerts': False,
        'api_failure_alerts': False,
        'position_alerts': False,
        'liquidity_alerts': False,
        'performance_alerts': False
    }
    
    # Suche nach Alert-Regeln in relevanten Files
    alert_files = [
        'utils/notifier.py',
        'utils/notifier_clean.py',
        'utils/notifier_final.py',
        'core/safety_manager.py',
        'core/risk_manager.py'
    ]
    
    for file_path in alert_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            print(f"\n📄 Checking alert rules in {file_path}:")
            
            # Check spezifische Alert-Typen
            if 'drawdown' in content and ('alert' in content or 'notify' in content):
                alert_rules['drawdown_alerts'] = True
                print("✅ Drawdown alerts")
            
            if 'error' in content and ('strategy' in content or 'trading' in content):
                alert_rules['strategy_error_alerts'] = True
                print("✅ Strategy error alerts")
            
            if 'api' in content and ('fail' in content or 'disconnect' in content):
                alert_rules['api_failure_alerts'] = True
                print("✅ API failure alerts")
            
            if 'position' in content and ('large' in content or 'limit' in content):
                alert_rules['position_alerts'] = True
                print("✅ Position alerts")
            
            if 'liquidity' in content and ('low' in content or 'alert' in content):
                alert_rules['liquidity_alerts'] = True
                print("✅ Liquidity alerts")
            
            if 'performance' in content and ('poor' in content or 'anomaly' in content):
                alert_rules['performance_alerts'] = True
                print("✅ Performance alerts")
    
    configured_rules = sum(alert_rules.values())
    print(f"\n📊 Configured alert rules: {configured_rules}/6")
    
    return alert_rules, configured_rules

def check_log_analysis():
    """Prüfe Log-Analyse Capabilities"""
    print("\n🔍 Checking Log Analysis Capabilities...")
    print("=" * 50)
    
    log_analysis = {
        'has_structured_logging': False,
        'has_log_aggregation': False,
        'has_error_tracking': False,
        'has_performance_logs': False,
        'log_retention': 'unknown'
    }
    
    # Check logs directory
    if os.path.exists('logs'):
        log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
        print(f"Found {len(log_files)} log files: {log_files}")
        
        # Check log structure
        if log_files:
            log_path = os.path.join('logs', log_files[0])
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            if 'json' in log_content or '{' in log_content:
                log_analysis['has_structured_logging'] = True
                print("✅ Structured logging detected")
            
            if 'error' in log_content.lower():
                log_analysis['has_error_tracking'] = True
                print("✅ Error tracking in logs")
            
            if 'performance' in log_content.lower() or 'time' in log_content:
                log_analysis['has_performance_logs'] = True
                print("✅ Performance logging")
    
    else:
        print("❌ No logs directory found")
    
    # Check logger configuration
    if os.path.exists('utils/logger.py'):
        with open('utils/logger.py', 'r') as f:
            logger_content = f.read()
        
        if 'json' in logger_content.lower() or 'structured' in logger_content.lower():
            log_analysis['has_structured_logging'] = True
            print("✅ Structured logging configured")
        
        if 'aggregat' in logger_content.lower() or 'central' in logger_content.lower():
            log_analysis['has_log_aggregation'] = True
            print("✅ Log aggregation configured")
    
    return log_analysis

def generate_monitoring_score():
    """Generiere Monitoring & Alerting Score"""
    print("\n" + "="*60)
    print("📊 MONITORING & ALERTING AUDIT REPORT")
    print("="*60)
    
    # Führe alle Tests durch
    monitoring, monitoring_features = check_monitoring_system()
    notifications, channel_files, active_channels = analyze_notification_channels()
    dashboard = check_dashboard_capabilities()
    alert_rules, configured_rules = analyze_alert_rules()
    log_analysis = check_log_analysis()
    
    # Berechne Scores
    total_score = 0
    max_score = 100
    
    # Basic Monitoring (25 Punkte)
    monitoring_score = sum(monitoring.values()) * (25 / len(monitoring))
    total_score += monitoring_score
    
    # Notification Channels (20 Punkte)
    notification_score = (active_channels / 6) * 20
    total_score += notification_score
    
    # Dashboard Capabilities (25 Punkte)
    dashboard_score = sum(dashboard.values()) * (25 / len(dashboard))
    total_score += dashboard_score
    
    # Alert Rules (20 Punkte)
    alert_score = (configured_rules / 6) * 20
    total_score += alert_score
    
    # Log Analysis (10 Punkte)
    log_score = sum(1 for k, v in log_analysis.items() if isinstance(v, bool) and v) * (10 / 4)
    total_score += log_score
    
    print(f"\n📊 MONITORING & ALERTING SCORE: {total_score:.1f}/100")
    print(f"   Basic Monitoring: {monitoring_score:.1f}/25")
    print(f"   Notification Channels: {notification_score:.1f}/20")
    print(f"   Dashboard: {dashboard_score:.1f}/25")
    print(f"   Alert Rules: {alert_score:.1f}/20")
    print(f"   Log Analysis: {log_score:.1f}/10")
    
    # Detaillierte Bewertung
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    print(f"\nBasic Monitoring Components:")
    for key, value in monitoring.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nNotification Channels ({active_channels}/6):")
    for key, value in notifications.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.title()}")
    
    print(f"\nDashboard Capabilities:")
    for key, value in dashboard.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nAlert Rules ({configured_rules}/6):")
    for key, value in alert_rules.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}")
    
    print(f"\nLog Analysis:")
    for key, value in log_analysis.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}")
        else:
            print(f"   📊 {key.replace('_', ' ').title()}: {value}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if monitoring_score < 15:
        print("   🔧 Implement comprehensive monitoring system")
        print("   🔧 Add health check endpoints")
        print("   🔧 Set up performance tracking")
    
    if notification_score < 10:
        print("   🔧 Set up Telegram bot for alerts")
        print("   🔧 Configure email notifications")
        print("   🔧 Add webhook support for external integrations")
    
    if dashboard_score < 15:
        print("   🔧 Enhance dashboard with real-time updates")
        print("   🔧 Add comprehensive charts and visualizations")
        print("   🔧 Implement strategy control interface")
    
    if alert_score < 12:
        print("   🔧 Configure critical alert rules:")
        print("       - Drawdown > 10%")
        print("       - Strategy errors")
        print("       - API failures")
        print("       - Unusual position sizes")
    
    if log_score < 6:
        print("   🔧 Implement structured logging")
        print("   🔧 Set up log aggregation")
        print("   🔧 Add error tracking and alerting")
    
    # Priority TODOs
    priority_todos = []
    
    if not monitoring['has_alerting_system']:
        priority_todos.append("Create comprehensive alerting system")
    if active_channels < 2:
        priority_todos.append("Set up Telegram and email notifications")
    if not dashboard['has_real_time_updates']:
        priority_todos.append("Add real-time dashboard updates")
    if configured_rules < 3:
        priority_todos.append("Configure critical alert rules")
    if not log_analysis['has_structured_logging']:
        priority_todos.append("Implement structured logging")
    
    print(f"\n🎯 PRIORITY TODO LIST:")
    for i, todo in enumerate(priority_todos, 1):
        print(f"   {i}. {todo}")
    
    # Specific Implementation Guide
    if total_score < 70:
        print(f"\n📋 QUICK IMPLEMENTATION GUIDE:")
        print("   1. Setup Telegram Bot:")
        print("      - Create bot with @BotFather")
        print("      - Add token to environment")
        print("      - Test with simple_telegram_test.py")
        print("   2. Configure Alert Rules:")
        print("      - Drawdown alerts in safety_manager.py")
        print("      - API failure detection")
        print("   3. Enhance Dashboard:")
        print("      - Add WebSocket for real-time updates")
        print("      - Implement performance charts")
    
    return {
        'total_score': total_score,
        'monitoring_components': monitoring,
        'notification_channels': notifications,
        'dashboard_features': dashboard,
        'alert_rules': alert_rules,
        'log_analysis': log_analysis,
        'monitoring_features': monitoring_features,
        'channel_files': channel_files,
        'priority_todos': priority_todos
    }

if __name__ == "__main__":
    # Führe Monitoring & Alerting Audit durch
    report = generate_monitoring_score()
    
    # Speichere Report
    with open('monitoring_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: monitoring_audit.json")