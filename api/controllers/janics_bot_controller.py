"""
Janics Bot Controller
====================

Controls bot lifecycle (Start/Stop/Restart) for the dashboard.
"""

import subprocess
import psutil
import os
import sys
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class JanicsBotController:
    """Controller for bot lifecycle management"""
    
    def __init__(self):
        self.bot_process: Optional[subprocess.Popen] = None
        self.project_root = Path(__file__).parent.parent.parent
        self.bot_script_path = self._find_bot_main_script()
        self.python_path = sys.executable
        self.bot_log_file = self.project_root / 'logs' / 'bot_process.log'
        self._ensure_log_directory()
        
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        log_dir = self.bot_log_file.parent
        log_dir.mkdir(exist_ok=True, parents=True)
        
    def _find_bot_main_script(self) -> Optional[Path]:
        """Find the main bot script automatically"""
        possible_files = ['main.py', 'bot.py', 'trading_bot.py', 'run.py']
        
        for file in possible_files:
            file_path = self.project_root / file
            if file_path.exists():
                logger.info(f"Found bot script: {file_path}")
                return file_path
                
        logger.error(f"No bot script found in {self.project_root}")
        return None
    
    def _find_running_bot_process(self) -> Optional[psutil.Process]:
        """Find currently running bot process"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline:
                            cmdline_str = ' '.join(cmdline)
                            if self.bot_script_path and str(self.bot_script_path) in cmdline_str:
                                return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error finding bot process: {str(e)}")
        return None
    
    def start_bot(self, mode: str = 'live', strategy: str = None, profile: str = None) -> Dict[str, Any]:
        """Start the trading bot with specified parameters"""
        try:
            # Check if bot is already running
            existing_process = self._find_running_bot_process()
            if existing_process:
                return {
                    'success': False,
                    'message': f'Bot is already running (PID: {existing_process.pid})',
                    'status': 'already_running',
                    'pid': existing_process.pid
                }
            
            if not self.bot_script_path:
                return {
                    'success': False,
                    'message': 'Bot script not found',
                    'status': 'error'
                }
            
            # Build command
            cmd = [self.python_path, str(self.bot_script_path)]
            
            # Add mode parameter
            cmd.extend(['--mode', mode])
            
            # Add optional parameters
            if strategy:
                cmd.extend(['--strategy', strategy])
            if profile:
                cmd.extend(['--profile', profile])
            
            # Open log file for bot output
            log_file = open(self.bot_log_file, 'a')
            log_file.write(f"\n\n{'='*50}\n")
            log_file.write(f"Starting bot at {datetime.now()}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"{'='*50}\n\n")
            log_file.flush()
            
            # Start the bot process
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(self.project_root),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            
            # Wait a bit to ensure process started
            time.sleep(3)
            
            # Check if process is still running
            if self.bot_process.poll() is None:
                logger.info(f"Bot started successfully with PID: {self.bot_process.pid}")
                return {
                    'success': True,
                    'message': 'Trading bot started successfully',
                    'pid': self.bot_process.pid,
                    'status': 'running',
                    'mode': mode,
                    'strategy': strategy,
                    'profile': profile
                }
            else:
                # Process died immediately
                return_code = self.bot_process.poll()
                log_file.close()
                
                # Read last lines of log for error info
                error_info = self._read_last_log_lines(20)
                
                return {
                    'success': False,
                    'message': f'Bot failed to start (exit code: {return_code})',
                    'status': 'failed',
                    'error_details': error_info
                }
                
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            return {
                'success': False,
                'message': f'Error starting bot: {str(e)}',
                'status': 'error'
            }
    
    def stop_bot(self) -> Dict[str, Any]:
        """Stop the trading bot gracefully"""
        try:
            # Find running bot process
            bot_process = self._find_running_bot_process()
            
            if not bot_process:
                return {
                    'success': True,
                    'message': 'No bot process found',
                    'status': 'already_stopped'
                }
            
            pid = bot_process.pid
            
            # Try graceful termination first
            logger.info(f"Attempting to stop bot process {pid} gracefully...")
            bot_process.terminate()
            
            # Wait for process to stop
            try:
                bot_process.wait(timeout=10)
                logger.info(f"Bot process {pid} stopped gracefully")
                return {
                    'success': True,
                    'message': 'Bot stopped successfully',
                    'status': 'stopped',
                    'pid': pid
                }
            except psutil.TimeoutExpired:
                # Force kill if graceful termination failed
                logger.warning(f"Bot process {pid} did not stop gracefully, forcing kill...")
                bot_process.kill()
                bot_process.wait()
                
                return {
                    'success': True,
                    'message': 'Bot force stopped',
                    'status': 'killed',
                    'pid': pid
                }
                
        except Exception as e:
            logger.error(f"Error stopping bot: {str(e)}")
            return {
                'success': False,
                'message': f'Error stopping bot: {str(e)}',
                'status': 'error'
            }
    
    def restart_bot(self, mode: str = 'live', strategy: str = None, profile: str = None) -> Dict[str, Any]:
        """Restart the trading bot"""
        try:
            # Stop the bot first
            stop_result = self.stop_bot()
            
            if not stop_result['success'] and stop_result['status'] != 'already_stopped':
                return {
                    'success': False,
                    'message': f"Failed to stop bot: {stop_result['message']}",
                    'status': 'stop_failed'
                }
            
            # Wait a moment before restarting
            time.sleep(2)
            
            # Start the bot again
            start_result = self.start_bot(mode=mode, strategy=strategy, profile=profile)
            
            if start_result['success']:
                return {
                    'success': True,
                    'message': 'Bot restarted successfully',
                    'status': 'restarted',
                    'pid': start_result['pid']
                }
            else:
                return start_result
                
        except Exception as e:
            logger.error(f"Error restarting bot: {str(e)}")
            return {
                'success': False,
                'message': f'Error restarting bot: {str(e)}',
                'status': 'error'
            }
    
    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status and details"""
        try:
            bot_process = self._find_running_bot_process()
            
            if not bot_process:
                return {
                    'running': False,
                    'status': 'stopped',
                    'details': {}
                }
            
            # Get process info
            with bot_process.oneshot():
                cpu_percent = bot_process.cpu_percent()
                memory_info = bot_process.memory_info()
                create_time = bot_process.create_time()
                
            uptime = time.time() - create_time
            
            return {
                'running': True,
                'status': 'running',
                'pid': bot_process.pid,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'uptime_seconds': int(uptime),
                    'uptime_formatted': self._format_uptime(uptime),
                    'started_at': datetime.fromtimestamp(create_time).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status: {str(e)}")
            return {
                'running': False,
                'status': 'error',
                'error': str(e)
            }
    
    def get_bot_logs(self, lines: int = 50) -> List[str]:
        """Get recent bot logs"""
        try:
            if not self.bot_log_file.exists():
                return []
            
            return self._read_last_log_lines(lines)
            
        except Exception as e:
            logger.error(f"Error reading bot logs: {str(e)}")
            return [f"Error reading logs: {str(e)}"]
    
    def _read_last_log_lines(self, lines: int) -> List[str]:
        """Read last N lines from log file"""
        try:
            with open(self.bot_log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except:
            return []
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)