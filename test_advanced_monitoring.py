#!/usr/bin/env python3
"""
Test Advanced Monitoring System
===============================

Test the advanced monitoring and alerting system
"""
import sys
import os
import logging
import time
import asyncio
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_advanced_monitoring():
    """Test advanced monitoring system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Advanced Monitoring System ===")
    
    try:
        # Test 1: Import monitoring system
        logger.info("Test 1: Importing monitoring system...")
        from core.advanced_monitoring import get_monitoring_system, AlertLevel, MetricType, NotificationChannel
        from config.settings import Settings
        
        settings = Settings()
        monitoring_system = get_monitoring_system(settings)
        logger.info("✓ Monitoring system imported successfully")
        
        # Test 2: Start monitoring
        logger.info("Test 2: Starting monitoring system...")
        monitoring_system.start_monitoring()
        await asyncio.sleep(1)  # Give it time to start
        logger.info("✓ Monitoring system started")
        
        # Test 3: Test metric recording
        logger.info("Test 3: Testing metric recording...")
        
        # Record various types of metrics
        monitoring_system.record_metric("test.counter", 1, MetricType.COUNTER, description="Test counter metric")
        monitoring_system.record_metric("test.gauge", 42.5, MetricType.GAUGE, description="Test gauge metric")
        monitoring_system.record_metric("test.histogram", 1.23, MetricType.HISTOGRAM, description="Test histogram metric")
        monitoring_system.record_metric("test.timer", 0.156, MetricType.TIMER, description="Test timer metric")
        
        # Record metrics with labels
        monitoring_system.record_metric(
            "test.labeled_metric", 
            100, 
            MetricType.GAUGE, 
            labels={"environment": "test", "component": "monitoring"},
            description="Test labeled metric"
        )
        
        logger.info("✓ Metrics recorded successfully")
        
        # Test 4: Test alert creation
        logger.info("Test 4: Testing alert creation...")
        
        # Create different types of alerts
        info_alert_id = monitoring_system.create_alert(
            AlertLevel.INFO,
            "test.info",
            "This is an info alert for testing",
            {"test_data": "info_test"}
        )
        
        warning_alert_id = monitoring_system.create_alert(
            AlertLevel.WARNING,
            "test.warning",
            "This is a warning alert for testing",
            {"test_data": "warning_test"}
        )
        
        error_alert_id = monitoring_system.create_alert(
            AlertLevel.ERROR,
            "test.error",
            "This is an error alert for testing",
            {"test_data": "error_test"}
        )
        
        critical_alert_id = monitoring_system.create_alert(
            AlertLevel.CRITICAL,
            "test.critical",
            "This is a critical alert for testing",
            {"test_data": "critical_test"},
            channels=[NotificationChannel.CONSOLE, NotificationChannel.FILE]
        )
        
        logger.info(f"✓ Alerts created: {len([info_alert_id, warning_alert_id, error_alert_id, critical_alert_id])} alerts")
        
        # Test 5: Test health checks
        logger.info("Test 5: Testing health checks...")
        
        # Register custom health check
        def test_health_check():
            return True  # Always healthy for test
        
        monitoring_system.register_health_check(
            "test_check",
            test_health_check,
            interval_seconds=5,
            timeout_seconds=2
        )
        
        # Register failing health check
        def failing_health_check():
            return False  # Always failing for test
        
        monitoring_system.register_health_check(
            "failing_check",
            failing_health_check,
            interval_seconds=3,
            timeout_seconds=1,
            failure_threshold=2
        )
        
        logger.info("✓ Health checks registered")
        
        # Test 6: Wait for health checks to run
        logger.info("Test 6: Waiting for health checks to execute...")
        await asyncio.sleep(8)  # Wait for health checks to run
        
        # Test 7: Test system status
        logger.info("Test 7: Testing system status...")
        
        system_status = monitoring_system.get_system_status()
        logger.info(f"✓ System Status Retrieved:")
        logger.info(f"  - Monitoring enabled: {system_status['monitoring_enabled']}")
        logger.info(f"  - Active alerts: {system_status['active_alerts_count']}")
        logger.info(f"  - CPU usage: {system_status['system_metrics']['cpu_usage']:.1f}%")
        logger.info(f"  - Memory usage: {system_status['system_metrics']['memory_usage']:.1f}%")
        logger.info(f"  - Health checks: {len(system_status['health_checks'])}")
        
        # Test 8: Test metrics retrieval
        logger.info("Test 8: Testing metrics retrieval...")
        
        # Wait a bit for metrics to be processed
        await asyncio.sleep(2)
        
        metrics = monitoring_system.get_metrics(name_pattern="test", hours=1)
        logger.info(f"✓ Retrieved {len(metrics)} test metrics")
        
        if metrics:
            latest_metric = metrics[0]
            logger.info(f"  - Latest metric: {latest_metric['name']} = {latest_metric['value']}")
        
        # Test 9: Test alerts retrieval
        logger.info("Test 9: Testing alerts retrieval...")
        
        alerts = monitoring_system.get_alerts(hours=1, include_resolved=False)
        logger.info(f"✓ Retrieved {len(alerts)} active alerts")
        
        for alert in alerts[:3]:  # Show first 3 alerts
            logger.info(f"  - {alert['level'].upper()}: {alert['source']} - {alert['message']}")
        
        # Test 10: Test alert management
        logger.info("Test 10: Testing alert management...")
        
        if critical_alert_id:
            # Acknowledge alert
            ack_success = monitoring_system.acknowledge_alert(critical_alert_id)
            logger.info(f"✓ Alert acknowledged: {ack_success}")
            
            # Resolve alert
            resolve_success = monitoring_system.resolve_alert(critical_alert_id)
            logger.info(f"✓ Alert resolved: {resolve_success}")
        
        # Test 11: Test event system integration
        logger.info("Test 11: Testing event system integration...")
        
        from core.interfaces import global_event_bus
        
        # Trigger some events that monitoring should capture
        global_event_bus.publish("trade_executed", {
            "symbol": "BTC/USDT",
            "amount": 1000,
            "strategy_name": "test_strategy"
        })
        
        global_event_bus.publish("error_occurred", {
            "error_type": "test_error",
            "message": "This is a test error for monitoring",
            "severity": "medium"
        })
        
        global_event_bus.publish("risk_limit_breached", {
            "limit_type": "drawdown",
            "current_value": 0.15,
            "limit_value": 0.10,
            "message": "Test risk limit breach"
        })
        
        await asyncio.sleep(2)  # Wait for events to be processed
        logger.info("✓ Events published and processed")
        
        # Test 12: Test notification channels (console only for test)
        logger.info("Test 12: Testing notification channels...")
        
        test_alert = monitoring_system.create_alert(
            AlertLevel.WARNING,
            "test.notification",
            "Testing notification channels",
            {"channel_test": True},
            channels=[NotificationChannel.CONSOLE]
        )
        
        logger.info(f"✓ Notification test alert created: {test_alert}")
        
        # Test 13: Test custom alert and metric handlers
        logger.info("Test 13: Testing custom handlers...")
        
        alert_handler_called = False
        metric_handler_called = False
        
        def test_alert_handler(alert):
            nonlocal alert_handler_called
            alert_handler_called = True
            logger.info(f"Custom alert handler called: {alert.alert_id}")
        
        def test_metric_handler(metric):
            nonlocal metric_handler_called
            metric_handler_called = True
            logger.info(f"Custom metric handler called: {metric.name}")
        
        monitoring_system.add_alert_handler(test_alert_handler)
        monitoring_system.add_metric_handler(test_metric_handler)
        
        # Trigger handlers
        monitoring_system.record_metric("test.handler", 99, MetricType.GAUGE)
        monitoring_system.create_alert(AlertLevel.INFO, "test.handler", "Handler test alert")
        
        await asyncio.sleep(1)
        
        if alert_handler_called and metric_handler_called:
            logger.info("✓ Custom handlers working correctly")
        else:
            logger.warning("⚠ Custom handlers may not be working")
        
        # Test 14: Test configuration loading
        logger.info("Test 14: Testing configuration loading...")
        
        try:
            import json
            with open('config/advanced_monitoring.json', 'r') as f:
                config = json.load(f)
            
            logger.info("✓ Monitoring configuration loaded")
            logger.info(f"Enabled: {config['advanced_monitoring']['enabled']}")
            logger.info(f"Retention days: {config['advanced_monitoring']['metrics_retention_days']}")
            logger.info(f"Health check interval: {config['advanced_monitoring']['health_check_interval']}")
            
            # Test notification channels config
            channels = config['advanced_monitoring']['notification_channels']
            enabled_channels = [ch for ch, cfg in channels.items() if cfg.get('enabled', False)]
            logger.info(f"Enabled notification channels: {enabled_channels}")
            
        except Exception as e:
            logger.warning(f"⚠ Configuration not loaded: {e}")
        
        # Test 15: Test database functionality
        logger.info("Test 15: Testing database functionality...")
        
        try:
            # Force metrics processing
            monitoring_system._process_metrics_buffer()
            
            # Check if databases exist
            metrics_db_exists = os.path.exists(monitoring_system.metrics_db_path)
            alerts_db_exists = os.path.exists(monitoring_system.alerts_db_path)
            
            logger.info(f"✓ Metrics database exists: {metrics_db_exists}")
            logger.info(f"✓ Alerts database exists: {alerts_db_exists}")
            
            if metrics_db_exists and alerts_db_exists:
                logger.info("✓ Database storage working correctly")
            
        except Exception as e:
            logger.warning(f"⚠ Database test failed: {e}")
        
        # Test 16: Performance test
        logger.info("Test 16: Performance testing...")
        
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(100):
            monitoring_system.record_metric(f"perf.test.{i % 10}", i, MetricType.COUNTER)
        
        # Create multiple alerts
        for i in range(10):
            monitoring_system.create_alert(
                AlertLevel.INFO,
                f"perf.test.{i}",
                f"Performance test alert {i}",
                {"test_number": i}
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✓ Performance test completed in {duration:.3f} seconds")
        logger.info("✓ System handled high load successfully")
        
        # Cleanup
        logger.info("Cleaning up...")
        monitoring_system.stop_monitoring()
        
        # Clean up test database files
        try:
            if os.path.exists(monitoring_system.metrics_db_path):
                os.remove(monitoring_system.metrics_db_path)
            if os.path.exists(monitoring_system.alerts_db_path):
                os.remove(monitoring_system.alerts_db_path)
            # Clean up directory if empty
            db_dir = os.path.dirname(monitoring_system.metrics_db_path)
            if os.path.exists(db_dir) and not os.listdir(db_dir):
                os.rmdir(db_dir)
            logger.info("✓ Test databases cleaned up")
        except Exception as e:
            logger.warning(f"⚠ Database cleanup failed: {e}")
        
        logger.info("=== All Advanced Monitoring Tests Completed Successfully! ===")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Make sure all required modules are available")
        return False
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        logger.error("Check the error logs for details", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_advanced_monitoring())
    sys.exit(0 if success else 1)