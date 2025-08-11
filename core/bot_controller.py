#!/usr/bin/env python3
"""
🤖 JanicsBotController - Complete Bot Management System
Vollständige Backend-Implementierung für Bot Process Management
"""

import os
import sys
import json
import time
import psutil
import asyncio
import logging
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import signal
from enum import Enum

logger = logging.getLogger(__name__)

class BotStatus(Enum):
    """Bot status states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class BotProcess:
    """Bot process information"""
    pid: int
    name: str
    cmdline: str
    status: BotStatus
    start_time: datetime
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'pid': self.pid,
            'name': self.name,
            'cmdline': self.cmdline,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_mb': self.memory_mb,
            'uptime_seconds': self.uptime_seconds
        }

@dataclass 
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    memory_available_gb: float
    process_count: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'memory_available_gb': self.memory_available_gb,
            'process_count': self.process_count,
            'timestamp': self.timestamp.isoformat()
        }

class JanicsBotController:
    """
    🚀 Janics Freedom Factory Bot Controller
    
    Complete bot process management system:
    - Start/Stop/Restart trading bots
    - Real-time process monitoring
    - System metrics collection
    - Bot health diagnostics
    - Process lifecycle management
    """
    
    def __init__(self, bot_script_paths: List[str] = None, working_directory: str = None):
        """
        Initialize the bot controller
        
        Args:
            bot_script_paths: List of possible bot script paths to search
            working_directory: Working directory for bot processes
        """
        self.working_directory = working_directory or os.getcwd()
        
        # Default bot script search paths
        self.bot_script_paths = bot_script_paths or [
            "main.py",
            "main_fixed.py", 
            "trading_bot.py",
            "core/trading_bot.py",
            "core/main.py",
            "bot.py",
            "start.py"
        ]
        
        # Bot process tracking
        self.tracked_processes: Dict[int, BotProcess] = {}
        self.last_scan_time = datetime.now()
        self.scan_interval = 5  # seconds
        
        # Bot identification patterns
        self.bot_patterns = [
            'main.py',
            'main_fixed.py',
            'trading_bot.py',
            'python main.py',
            'python3 main.py',
            'altcoin_trading_bot'
        ]
        
        # Exclude patterns (don't treat these as trading bots)
        self.exclude_patterns = [
            'run_intelligence_api.py',
            'intelligence_api',
            'dashboard',
            'api_server',
            'bot_controller.py'
        ]
        
        logger.info("🤖 JanicsBotController initialized")
        logger.info(f"📁 Working directory: {self.working_directory}")
        logger.info(f"🔍 Bot script paths: {self.bot_script_paths}")
    
    def scan_bot_processes(self) -> List[BotProcess]:
        """
        Scan system for trading bot processes
        
        Returns:
            List of detected bot processes
        """
        bot_processes = []
        current_time = datetime.now()
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
                try:
                    pinfo = proc.info
                    
                    if not pinfo['cmdline']:
                        continue
                    
                    cmdline = ' '.join(pinfo['cmdline'])
                    
                    # Skip if matches exclude patterns
                    if any(exclude in cmdline.lower() for exclude in self.exclude_patterns):
                        continue
                    
                    # Check if this looks like our trading bot
                    is_bot = any(pattern in cmdline.lower() or pattern in pinfo['name'].lower() 
                               for pattern in self.bot_patterns)
                    
                    if is_bot:
                        # Get additional process info
                        try:
                            process = psutil.Process(pinfo['pid'])
                            memory_info = process.memory_info()
                            
                            bot_process = BotProcess(
                                pid=pinfo['pid'],
                                name=pinfo['name'],
                                cmdline=cmdline,
                                status=BotStatus.RUNNING,
                                start_time=datetime.fromtimestamp(pinfo['create_time']),
                                cpu_percent=process.cpu_percent(),
                                memory_percent=process.memory_percent(),
                                memory_mb=memory_info.rss / 1024 / 1024,  # Convert to MB
                                uptime_seconds=int((current_time - datetime.fromtimestamp(pinfo['create_time'])).total_seconds())
                            )
                            
                            bot_processes.append(bot_process)
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process disappeared or access denied, skip
                            continue
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Update tracked processes
            self.tracked_processes = {proc.pid: proc for proc in bot_processes}
            self.last_scan_time = current_time
            
            logger.debug(f"🔍 Bot scan found {len(bot_processes)} processes")
            return bot_processes
            
        except Exception as e:
            logger.error(f"❌ Error scanning bot processes: {e}")
            return []
    
    def get_system_metrics(self) -> SystemMetrics:
        """
        Get current system performance metrics
        
        Returns:
            SystemMetrics object with current system stats
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / 1024 / 1024 / 1024
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
            except:
                disk_percent = 0.0
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                memory_available_gb=memory_available_gb,
                process_count=process_count,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, datetime.now())
    
    def find_bot_script(self, preferred_script: str = None) -> Optional[str]:
        """
        Find available bot script to execute
        
        Args:
            preferred_script: Preferred script name to look for first
            
        Returns:
            Path to bot script if found, None otherwise
        """
        search_paths = []
        
        # Add preferred script to front of search list
        if preferred_script:
            search_paths.append(preferred_script)
            search_paths.append(f"./{preferred_script}")
            search_paths.append(os.path.join(self.working_directory, preferred_script))
        
        # Add default search paths
        for script_path in self.bot_script_paths:
            search_paths.extend([
                script_path,
                f"./{script_path}",
                os.path.join(self.working_directory, script_path),
                os.path.join(self.working_directory, "core", script_path),
                os.path.join(self.working_directory, "src", script_path)
            ])
        
        # Search for existing file
        for path in search_paths:
            if os.path.isfile(path) and os.access(path, os.R_OK):
                logger.info(f"✅ Found bot script: {path}")
                return os.path.abspath(path)
        
        logger.warning(f"❌ No bot script found in search paths: {search_paths[:10]}...")
        return None
    
    def start_bot(self, script_name: str = "main.py", **kwargs) -> Dict[str, Any]:
        """
        Start a trading bot process
        
        Args:
            script_name: Name of bot script to start
            **kwargs: Additional process arguments
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            # Check if bot is already running
            current_processes = self.scan_bot_processes()
            if current_processes:
                return {
                    'success': False,
                    'message': f'Trading bot already running (PIDs: {[p.pid for p in current_processes]})',
                    'error_code': 'BOT_ALREADY_RUNNING',
                    'running_processes': [p.to_dict() for p in current_processes]
                }
            
            # Find bot script
            script_path = self.find_bot_script(script_name)
            if not script_path:
                # List available Python files for debugging
                try:
                    available_files = [f for f in os.listdir(self.working_directory) if f.endswith('.py')][:10]
                    return {
                        'success': False,
                        'message': f'Bot script "{script_name}" not found',
                        'error_code': 'SCRIPT_NOT_FOUND',
                        'available_files': available_files,
                        'search_directory': self.working_directory
                    }
                except:
                    return {
                        'success': False,
                        'message': f'Bot script "{script_name}" not found',
                        'error_code': 'SCRIPT_NOT_FOUND'
                    }
            
            # Prepare process environment
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{self.working_directory}:{env.get('PYTHONPATH', '')}"
            
            # Start bot process
            logger.info(f"🚀 Starting bot: {script_path}")
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=self.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process
                **kwargs
            )
            
            # Wait briefly to check startup
            time.sleep(3)
            
            if process.poll() is None:
                # Process is running
                logger.info(f"✅ Bot started successfully (PID: {process.pid})")
                
                # Create bot process tracking
                bot_process = BotProcess(
                    pid=process.pid,
                    name=os.path.basename(script_path),
                    cmdline=f"{sys.executable} {script_path}",
                    status=BotStatus.STARTING,
                    start_time=datetime.now()
                )
                self.tracked_processes[process.pid] = bot_process
                
                return {
                    'success': True,
                    'message': f'Trading bot started successfully',
                    'pid': process.pid,
                    'script_path': script_path,
                    'start_time': datetime.now().isoformat(),
                    'process_info': bot_process.to_dict()
                }
            else:
                # Process died immediately
                stdout, stderr = process.communicate()
                error_msg = stderr.decode('utf-8', errors='ignore')[:500]
                
                logger.error(f"❌ Bot failed to start: {error_msg}")
                
                return {
                    'success': False,
                    'message': f'Bot failed to start: {error_msg}',
                    'error_code': 'STARTUP_FAILED',
                    'exit_code': process.returncode,
                    'stderr': error_msg,
                    'script_path': script_path
                }
                
        except Exception as e:
            logger.error(f"❌ Exception starting bot: {e}")
            return {
                'success': False,
                'message': f'Failed to start bot: {str(e)}',
                'error_code': 'EXCEPTION',
                'exception_type': type(e).__name__
            }
    
    def stop_bot(self, pid: int = None, force: bool = False) -> Dict[str, Any]:
        """
        Stop trading bot process(es)
        
        Args:
            pid: Specific process ID to stop (if None, stops all bot processes)
            force: Whether to use force kill (SIGKILL vs SIGTERM)
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            # Get current bot processes
            current_processes = self.scan_bot_processes()
            
            if not current_processes:
                return {
                    'success': False,
                    'message': 'No trading bot processes found to stop',
                    'error_code': 'NO_PROCESSES_FOUND'
                }
            
            # Filter processes to stop
            processes_to_stop = current_processes
            if pid:
                processes_to_stop = [p for p in current_processes if p.pid == pid]
                if not processes_to_stop:
                    return {
                        'success': False,
                        'message': f'Process with PID {pid} not found or not a bot process',
                        'error_code': 'PROCESS_NOT_FOUND'
                    }
            
            stopped_processes = []
            failed_processes = []
            
            for bot_process in processes_to_stop:
                try:
                    process = psutil.Process(bot_process.pid)
                    process_name = process.name()
                    process_pid = process.pid
                    
                    logger.info(f"🛑 Stopping bot process: {process_name} (PID: {process_pid})")
                    
                    if force:
                        # Force kill
                        process.kill()
                        signal_used = "SIGKILL"
                    else:
                        # Graceful shutdown
                        process.terminate()
                        signal_used = "SIGTERM"
                    
                    # Wait for process to stop
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        if not force:
                            # Timeout on graceful shutdown, try force kill
                            logger.warning(f"⏰ Graceful shutdown timeout for PID {process_pid}, force killing")
                            process.kill()
                            process.wait(timeout=5)
                            signal_used = "SIGKILL (after timeout)"
                    
                    stopped_processes.append({
                        'pid': process_pid,
                        'name': process_name,
                        'signal_used': signal_used,
                        'stop_time': datetime.now().isoformat()
                    })
                    
                    # Remove from tracking
                    if process_pid in self.tracked_processes:
                        del self.tracked_processes[process_pid]
                    
                    logger.info(f"✅ Bot process stopped: {process_name} (PID: {process_pid})")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process already stopped or no access
                    stopped_processes.append({
                        'pid': bot_process.pid,
                        'name': bot_process.name,
                        'note': 'Process already stopped or access denied',
                        'stop_time': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    failed_processes.append({
                        'pid': bot_process.pid,
                        'name': bot_process.name,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    logger.error(f"❌ Failed to stop process {bot_process.pid}: {e}")
            
            # Prepare result
            if stopped_processes and not failed_processes:
                return {
                    'success': True,
                    'message': f'Successfully stopped {len(stopped_processes)} bot process(es)',
                    'stopped_processes': stopped_processes,
                    'stop_time': datetime.now().isoformat()
                }
            elif stopped_processes and failed_processes:
                return {
                    'success': True,  # Partial success
                    'message': f'Stopped {len(stopped_processes)} processes, failed to stop {len(failed_processes)}',
                    'stopped_processes': stopped_processes,
                    'failed_processes': failed_processes,
                    'stop_time': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to stop {len(failed_processes)} bot process(es)',
                    'error_code': 'STOP_FAILED',
                    'failed_processes': failed_processes
                }
                
        except Exception as e:
            logger.error(f"❌ Exception stopping bot: {e}")
            return {
                'success': False,
                'message': f'Failed to stop bot: {str(e)}',
                'error_code': 'EXCEPTION',
                'exception_type': type(e).__name__
            }
    
    def restart_bot(self, script_name: str = "main.py", force_stop: bool = False) -> Dict[str, Any]:
        """
        Restart trading bot (stop then start)
        
        Args:
            script_name: Bot script to start after stopping
            force_stop: Whether to force kill existing processes
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            logger.info("🔄 Restarting trading bot...")
            
            # Stop existing processes
            stop_result = self.stop_bot(force=force_stop)
            
            # Check if stop was successful (or no processes were running)
            if not stop_result['success'] and stop_result['error_code'] != 'NO_PROCESSES_FOUND':
                return {
                    'success': False,
                    'message': f'Failed to stop bot before restart: {stop_result["message"]}',
                    'error_code': 'RESTART_STOP_FAILED',
                    'stop_result': stop_result
                }
            
            # Wait a moment for cleanup
            time.sleep(2)
            
            # Start bot again
            start_result = self.start_bot(script_name)
            
            if start_result['success']:
                return {
                    'success': True,
                    'message': f'Trading bot restarted successfully (PID: {start_result["pid"]})',
                    'pid': start_result['pid'],
                    'restart_time': datetime.now().isoformat(),
                    'stop_result': stop_result,
                    'start_result': start_result
                }
            else:
                return {
                    'success': False,
                    'message': f'Bot stopped but failed to restart: {start_result["message"]}',
                    'error_code': 'RESTART_START_FAILED',
                    'stop_result': stop_result,
                    'start_result': start_result
                }
                
        except Exception as e:
            logger.error(f"❌ Exception restarting bot: {e}")
            return {
                'success': False,
                'message': f'Failed to restart bot: {str(e)}',
                'error_code': 'EXCEPTION',
                'exception_type': type(e).__name__
            }
    
    def get_bot_status(self) -> Dict[str, Any]:
        """
        Get comprehensive bot status information
        
        Returns:
            Complete status dictionary with processes, metrics, and health info
        """
        try:
            # Scan for current processes
            bot_processes = self.scan_bot_processes()
            system_metrics = self.get_system_metrics()
            
            # Calculate aggregated metrics
            total_cpu = sum(p.cpu_percent for p in bot_processes)
            total_memory_mb = sum(p.memory_mb for p in bot_processes)
            total_memory_percent = sum(p.memory_percent for p in bot_processes)
            
            # Determine overall status
            if not bot_processes:
                overall_status = BotStatus.STOPPED
            elif any(p.status == BotStatus.STARTING for p in bot_processes):
                overall_status = BotStatus.STARTING
            elif any(p.status == BotStatus.ERROR for p in bot_processes):
                overall_status = BotStatus.ERROR
            else:
                overall_status = BotStatus.RUNNING
            
            # Calculate uptime
            max_uptime = max([p.uptime_seconds for p in bot_processes]) if bot_processes else 0
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status.value,
                'process_count': len(bot_processes),
                'processes': [p.to_dict() for p in bot_processes],
                'aggregated_metrics': {
                    'total_cpu_percent': total_cpu,
                    'total_memory_mb': total_memory_mb,
                    'total_memory_percent': total_memory_percent,
                    'max_uptime_seconds': max_uptime,
                    'max_uptime_hours': max_uptime / 3600 if max_uptime > 0 else 0
                },
                'system_metrics': system_metrics.to_dict(),
                'last_scan_time': self.last_scan_time.isoformat(),
                'controller_info': {
                    'working_directory': self.working_directory,
                    'bot_script_paths': self.bot_script_paths,
                    'tracked_processes': len(self.tracked_processes)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting bot status: {e}")
            return {
                'success': False,
                'message': f'Failed to get bot status: {str(e)}',
                'error_code': 'STATUS_ERROR',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_bot_logs(self, pid: int = None, lines: int = 50) -> Dict[str, Any]:
        """
        Get recent bot logs (if available)
        
        Args:
            pid: Specific process PID to get logs for
            lines: Number of recent lines to return
            
        Returns:
            Dictionary with log information
        """
        try:
            # This is a placeholder implementation
            # In a real system, you'd read from log files or capture process output
            
            return {
                'success': True,
                'message': 'Log retrieval not fully implemented yet',
                'pid': pid,
                'lines_requested': lines,
                'logs': [
                    {'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': 'Bot process running normally'},
                    {'timestamp': (datetime.now() - timedelta(minutes=1)).isoformat(), 'level': 'DEBUG', 'message': 'Market data update received'},
                    {'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(), 'level': 'INFO', 'message': 'Strategy evaluation completed'}
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to get logs: {str(e)}',
                'error_code': 'LOG_ERROR'
            }
    
    def cleanup_orphaned_processes(self) -> Dict[str, Any]:
        """
        Cleanup any orphaned or zombie processes
        
        Returns:
            Result of cleanup operation
        """
        try:
            cleaned_processes = []
            
            for pid in list(self.tracked_processes.keys()):
                try:
                    if not psutil.pid_exists(pid):
                        # Process no longer exists, remove from tracking
                        process_info = self.tracked_processes.pop(pid)
                        cleaned_processes.append({
                            'pid': pid,
                            'name': process_info.name,
                            'action': 'removed_from_tracking'
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error checking PID {pid}: {e}")
            
            return {
                'success': True,
                'message': f'Cleanup completed, processed {len(cleaned_processes)} entries',
                'cleaned_processes': cleaned_processes
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Cleanup failed: {str(e)}',
                'error_code': 'CLEANUP_ERROR'
            }

# Global bot controller instance
bot_controller = JanicsBotController()

def get_bot_controller() -> JanicsBotController:
    """Get the global bot controller instance"""
    return bot_controller

def initialize_bot_controller(working_directory: str = None, bot_scripts: List[str] = None) -> JanicsBotController:
    """Initialize the global bot controller with specific configuration"""
    global bot_controller
    bot_controller = JanicsBotController(
        bot_script_paths=bot_scripts,
        working_directory=working_directory
    )
    return bot_controller