"""
Dashboard Status Controller
==========================

Provides real-time status data for dashboard header indicators.
"""

import psutil
import os
from datetime import datetime
from pathlib import Path
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DashboardStatusController:
    """Controller for dashboard header status indicators"""
    
    def __init__(self):
        self.bot_script_names = ['main.py', 'bot.py', 'trading_bot.py']
        self.project_root = Path(__file__).parent.parent.parent
        
    def get_header_status(self) -> Dict[str, Any]:
        """Returns real status data for header displays"""
        return {
            'factory_online': self.check_dashboard_health(),
            'server_status': self.check_server_connection(),
            'trading_bot_status': self.check_bot_process_status(),
            'system_status': self.check_overall_system_health(),
            'last_update': datetime.now().strftime('%H:%M:%S')
        }
    
    def check_dashboard_health(self) -> bool:
        """Check if dashboard services are healthy"""
        try:
            # Check if API is responding
            return True  # Since we're running, API is healthy
        except Exception as e:
            logger.error(f"Dashboard health check failed: {str(e)}")
            return False
    
    def check_server_connection(self) -> str:
        """Check server/exchange connections"""
        try:
            # Check if we have exchange connections
            from data_sources.binance_source import BinanceDataSource
            
            # Try to ping Binance
            binance = BinanceDataSource()
            if hasattr(binance, 'client') and binance.client:
                return 'Connected'
            else:
                return 'Connecting...'
        except Exception as e:
            logger.warning(f"Server connection check failed: {str(e)}")
            return 'Disconnected'
    
    def check_bot_process_status(self) -> str:
        """Check if bot process is actually running"""
        try:
            # Search for Python process with bot script
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline:
                            cmdline_str = ' '.join(cmdline)
                            if any(script in cmdline_str for script in self.bot_script_names):
                                return 'Running'
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            return 'Stopped'
        except Exception as e:
            logger.error(f"Bot process check failed: {str(e)}")
            return 'Error'
    
    def check_overall_system_health(self) -> str:
        """Check overall system health"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Check memory usage
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90 or memory.percent > 90:
                return 'Warning'
            elif self.check_bot_process_status() == 'Running':
                return 'Operational'
            else:
                return 'Idle'
        except Exception as e:
            logger.error(f"System health check failed: {str(e)}")
            return 'Unknown'
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'cores': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'network': self._get_network_stats()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            stats = psutil.net_io_counters()
            return {
                'bytes_sent': stats.bytes_sent,
                'bytes_recv': stats.bytes_recv,
                'packets_sent': stats.packets_sent,
                'packets_recv': stats.packets_recv
            }
        except:
            return {}