"""
Bot Manager Service
==================

Manages trading bot lifecycle from API endpoints.
"""

import subprocess
import psutil
import json
import os
import signal
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from db.models import TradingDatabase
from utils.error_handler import secure_error_handler
from api.websocket.events import emit_bot_status_update, emit_bot_performance_update

logger = logging.getLogger(__name__)

class BotManager:
    """Manages trading bot lifecycle from API"""
    
    def __init__(self):
        self.bot_process = None
        self.bot_thread = None
        self.db = TradingDatabase()
        self.bot_status = {
            'is_running': False,
            'status': 'stopped',  # stopped, starting, running, error
            'pid': None,
            'strategy': None,
            'mode': 'paper',
            'symbol': None,
            'start_time': None,
            'config': {},
            'last_update': None,
            'last_error': None,
            'logs': [],
            'performance': {
                'total_pnl': 0,
                'daily_pnl': 0,
                'win_rate': 0,
                'total_trades': 0,
                'active_positions': 0
            }
        }
        
        # Perform startup cleanup and health check
        self._startup_health_check()
    
    def _startup_health_check(self):
        """Comprehensive startup health check and cleanup"""
        logger.info("Performing startup health check...")
        
        # 1. Clean up any stale PID files
        self._cleanup_pid_files()
        
        # 2. Kill any orphaned bot processes
        self._cleanup_zombie_processes()
        
        # 3. Reset status to clean state
        self._reset_status()
        
        # 4. Check for any actual running processes
        self._check_existing_process()
        
        logger.info("Startup health check completed")
    
    def _cleanup_pid_files(self):
        """Remove any stale PID files"""
        pid_files = ['bot.pid', 'temp_bot_config.json']
        for pid_file in pid_files:
            pid_path = Path.cwd() / pid_file
            if pid_path.exists():
                try:
                    pid_path.unlink()
                    logger.info(f"Removed stale file: {pid_file}")
                except Exception as e:
                    logger.warning(f"Could not remove {pid_file}: {e}")
    
    def _cleanup_zombie_processes(self):
        """Kill any orphaned bot processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('main.py' in str(cmd) for cmd in cmdline):
                        # Check if it's our trading bot
                        if any('trading' in str(cmd).lower() or 'bot' in str(cmd).lower() for cmd in cmdline):
                            logger.warning(f"Found orphaned bot process {proc.info['pid']}, terminating...")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.warning(f"Error during zombie process cleanup: {e}")
    
    def _reset_status(self):
        """Reset bot status to clean state"""
        self.bot_process = None
        self.bot_status = {
            'is_running': False,
            'pid': None,
            'strategy': None,
            'mode': 'paper',
            'symbol': None,
            'start_time': None,
            'config': {},
            'last_update': datetime.now(timezone.utc).isoformat(),
            'performance': {
                'total_pnl': 0,
                'daily_pnl': 0,
                'win_rate': 0,
                'total_trades': 0,
                'active_positions': 0
            }
        }
        logger.info("Bot status reset to clean state")
    
    def _check_existing_process(self):
        """Check if bot process is already running"""
        try:
            # Look for any running bot processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('main.py' in str(cmd) for cmd in cmdline):
                        # Verify it's actually our trading bot
                        if any('--daemon' in str(cmd) or 'api_mode' in str(cmd) for cmd in cmdline):
                            self.bot_process = proc
                            self.bot_status['is_running'] = True
                            self.bot_status['pid'] = proc.info['pid']
                            logger.info(f"Found existing bot process: PID {proc.info['pid']}")
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error checking existing process: {e}")
        
        return False
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if process with PID is running"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def is_actually_running(self) -> bool:
        """Verify bot is actually running (not just status flag)"""
        if not self.bot_status['is_running']:
            return False
        
        if self.bot_status['pid'] and not self._is_process_running(self.bot_status['pid']):
            # Process died, update status
            logger.warning("Bot process died, updating status")
            self._reset_status()
            return False
        
        return True
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force cleanup of any stale bot processes and reset status"""
        try:
            logger.info("Performing force cleanup...")
            
            # Kill any running bot process
            if self.bot_process:
                try:
                    self.bot_process.terminate()
                    self.bot_process.wait(timeout=5)
                except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                    try:
                        self.bot_process.kill()
                    except psutil.NoSuchProcess:
                        pass
            
            # Clean up zombie processes
            self._cleanup_zombie_processes()
            
            # Clean up files
            self._cleanup_pid_files()
            
            # Reset status
            self._reset_status()
            
            # Emit status update
            try:
                emit_bot_status_update({
                    'is_running': False,
                    'status': self.bot_status,
                    'event': 'force_cleanup',
                    'message': 'Bot force stopped and cleaned up'
                })
            except Exception as e:
                logger.warning(f"Failed to emit WebSocket event: {e}")
            
            logger.info("Force cleanup completed")
            return {
                'success': True,
                'message': 'Bot force stopped and cleaned up',
                'status': self.bot_status
            }
            
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
            return {
                'success': False,
                'message': f'Force cleanup failed: {str(e)}',
                'status': self.bot_status
            }
    
    def get_verified_status(self) -> Dict[str, Any]:
        """Get verified status with real-time process check"""
        # Update running status first
        self._update_running_status()
        
        return {
            'success': True,
            'status': self.bot_status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'verified': True
        }
    
    def start_bot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start bot with given configuration"""
        try:
            # Verify we're not actually running
            if self.is_actually_running():
                return {
                    'success': False,
                    'message': 'Bot is already running',
                    'status': self.bot_status,
                    'pid': self.bot_status['pid']
                }
            
            # Validate configuration
            validation_result = self._validate_config(config)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'message': f"Configuration validation failed: {validation_result['errors']}",
                    'status': self.bot_status
                }
            
            # Prepare bot command
            bot_config = {
                'mode': config.get('mode', 'paper'),
                'strategy': config.get('strategy', 'momentum'),
                'symbol': config.get('symbol', 'BTC/USDT'),
                'capital': config.get('capital', 10000),
                'risk_per_trade': config.get('risk_per_trade', 0.02),
                'strategy_params': config.get('strategy_params', {}),
                'api_mode': True,  # Flag to indicate started from API
                'log_level': 'INFO'
            }
            
            # Write config to temporary file
            config_file = Path.cwd() / 'temp_bot_config.json'
            with open(config_file, 'w') as f:
                json.dump(bot_config, f, indent=2)
            
            # Find the correct path to main.py
            project_root = Path(__file__).parent.parent.parent
            main_py_path = project_root / 'main.py'
            
            if not main_py_path.exists():
                raise FileNotFoundError(f"main.py not found at {main_py_path}")
            
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['BOT_MODE'] = config.get('mode', 'paper')
            env['PYTHONPATH'] = str(project_root)
            
            # Build command - use direct arguments instead of config file
            cmd = [
                sys.executable,
                str(main_py_path),
                '--mode', config.get('mode', 'paper'),
                '--strategy', config.get('strategy', 'momentum'),
                '--symbol', config.get('symbol', 'BTC/USDT'),
                '--config-json', json.dumps(config)
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Ensure logs directory exists
            logs_dir = project_root / 'logs'
            logs_dir.mkdir(exist_ok=True)
            
            # Start the process
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                env=env,
                cwd=str(project_root),
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Write PID file
            pid_file = project_root / 'bot.pid'
            with open(pid_file, 'w') as f:
                f.write(str(self.bot_process.pid))
            
            # Start monitoring thread
            self.bot_thread = threading.Thread(
                target=self._monitor_bot_output,
                daemon=True
            )
            self.bot_thread.start()
            
            # Wait to verify startup
            time.sleep(3)
            
            if self.bot_process.poll() is None:
                # Process is still running
                self.bot_status.update({
                    'is_running': True,
                    'status': 'running',
                    'pid': self.bot_process.pid,
                    'strategy': config.get('strategy'),
                    'mode': config.get('mode'),
                    'symbol': config.get('symbol'),
                    'start_time': datetime.now(timezone.utc).isoformat(),
                    'config': config,
                    'last_update': datetime.now(timezone.utc).isoformat()
                })
                
                # Log bot start event
                self.db.log_system_event(
                    'INFO', 
                    f"Trading bot started: {bot_config['strategy']} on {bot_config['symbol']} ({bot_config['mode']} mode)",
                    'bot_manager',
                    config=bot_config
                )
                
                logger.info(f"Bot started successfully: PID {self.bot_process.pid}")
                
                # Emit WebSocket event
                try:
                    emit_bot_status_update({
                        'is_running': True,
                        'status': self.bot_status,
                        'event': 'bot_started',
                        'message': 'Bot started successfully'
                    })
                except Exception as e:
                    logger.warning(f"Failed to emit WebSocket event: {e}")
                
                return {
                    'success': True,
                    'message': 'Bot started successfully',
                    'status': self.bot_status
                }
            else:
                # Process failed to start
                stdout, stderr = self.bot_process.communicate()
                error_msg = stderr.decode() if stderr else "Unknown error"
                
                logger.error(f"Bot failed to start: {error_msg}")
                
                return {
                    'success': False,
                    'message': f'Bot failed to start: {error_msg}',
                    'status': self.bot_status
                }
                
        except Exception as e:
            error_response = secure_error_handler.handle_critical_error(
                error=e,
                context={
                    "operation": "start_bot",
                    "config": config
                }
            )
            logger.error(f"Error starting bot - ID: {error_response.error_id}")
            
            return {
                'success': False,
                'message': f'Failed to start bot: {str(e)}',
                'status': self.bot_status
            }
    
    def stop_bot(self) -> Dict[str, Any]:
        """Gracefully stop the bot"""
        try:
            if not self.bot_status['is_running']:
                return {
                    'success': False,
                    'message': 'Bot is not running',
                    'status': self.bot_status
                }
            
            if self.bot_process and self.bot_process.poll() is None:
                # Send SIGTERM for graceful shutdown
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.bot_process.pid), signal.SIGTERM)
                else:
                    self.bot_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.bot_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Bot didn't stop gracefully, force killing...")
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.bot_process.pid), signal.SIGKILL)
                    else:
                        self.bot_process.kill()
                    self.bot_process.wait()
            
            # Clean up
            self._cleanup_bot_status()
            
            # Log bot stop event
            self.db.log_system_event(
                'INFO', 
                "Trading bot stopped via API",
                'bot_manager'
            )
            
            logger.info("Bot stopped successfully")
            
            # Emit WebSocket event
            try:
                emit_bot_status_update({
                    'is_running': False,
                    'status': self.bot_status,
                    'event': 'bot_stopped',
                    'message': 'Bot stopped successfully'
                })
            except Exception as e:
                logger.warning(f"Failed to emit WebSocket event: {e}")
            
            return {
                'success': True,
                'message': 'Bot stopped successfully',
                'status': self.bot_status
            }
            
        except Exception as e:
            error_response = secure_error_handler.handle_critical_error(
                error=e,
                context={"operation": "stop_bot"}
            )
            logger.error(f"Error stopping bot - ID: {error_response.error_id}")
            
            # Force cleanup on error
            self._cleanup_bot_status()
            
            return {
                'success': False,
                'message': f'Error stopping bot: {str(e)}',
                'status': self.bot_status
            }
    
    def restart_bot(self, new_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Restart with optional new config"""
        try:
            # Stop current bot
            stop_result = self.stop_bot()
            if not stop_result['success'] and self.bot_status['is_running']:
                return stop_result
            
            # Wait a moment
            time.sleep(1)
            
            # Start with new or existing config
            config = new_config if new_config else self.bot_status['config']
            return self.start_bot(config)
            
        except Exception as e:
            logger.error(f"Error restarting bot: {e}")
            return {
                'success': False,
                'message': f'Error restarting bot: {str(e)}',
                'status': self.bot_status
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            # Update running status
            self._update_running_status()
            
            # Get performance data from database
            if self.bot_status['is_running']:
                self._update_performance_metrics()
            
            self.bot_status['last_update'] = datetime.now(timezone.utc).isoformat()
            
            # Emit periodic status update via WebSocket
            try:
                emit_bot_status_update({
                    'is_running': self.bot_status['is_running'],
                    'status': self.bot_status,
                    'event': 'status_update'
                })
                
                if self.bot_status['is_running']:
                    emit_bot_performance_update({
                        'performance': self.bot_status['performance']
                    })
            except Exception as e:
                logger.warning(f"Failed to emit WebSocket status update: {e}")
            
            return {
                'success': True,
                'status': self.bot_status
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'success': False,
                'message': f'Error getting status: {str(e)}',
                'status': self.bot_status
            }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status including trades and positions"""
        try:
            basic_status = self.get_status()
            
            if not basic_status['success']:
                return basic_status
            
            # Get recent trades
            recent_trades = self.db.get_trades(limit=10)
            
            # Get performance metrics
            performance_metrics = self.db.get_performance_metrics(days=7)
            
            # Get system logs
            recent_logs = self.db.get_system_logs(limit=20)
            
            detailed_status = basic_status['status'].copy()
            detailed_status.update({
                'recent_trades': recent_trades,
                'performance_history': performance_metrics,
                'recent_logs': recent_logs
            })
            
            return {
                'success': True,
                'status': detailed_status
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed status: {e}")
            return {
                'success': False,
                'message': f'Error getting detailed status: {str(e)}',
                'status': self.bot_status
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bot configuration"""
        errors = []
        
        # Required fields
        required_fields = ['mode', 'strategy', 'symbol']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Mode validation
        if config.get('mode') not in ['paper', 'live']:
            errors.append("Mode must be 'paper' or 'live'")
        
        # Strategy validation
        valid_strategies = ['momentum', 'mean_reversion', 'grid_trading', 'arbitrage', 'defi_yield', 'liquidation', 'ml_strategy']
        if config.get('strategy') not in valid_strategies:
            errors.append(f"Strategy must be one of: {', '.join(valid_strategies)}")
        
        # Symbol validation
        symbol = config.get('symbol', '')
        if not symbol or '/' not in symbol:
            errors.append("Symbol must be in format 'BASE/QUOTE' (e.g., 'BTC/USDT')")
        
        # Capital validation
        capital = config.get('capital', 0)
        if capital < 100 or capital > 1000000:
            errors.append("Capital must be between 100 and 1,000,000")
        
        # Risk validation
        risk = config.get('risk_per_trade', 0)
        if risk < 0.001 or risk > 0.1:
            errors.append("Risk per trade must be between 0.1% and 10%")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _update_running_status(self):
        """Check if bot process is still running"""
        if self.bot_status['is_running'] and self.bot_status['pid']:
            if not psutil.pid_exists(self.bot_status['pid']):
                logger.warning("Bot process no longer exists, updating status")
                self._cleanup_bot_status()
            else:
                try:
                    process = psutil.Process(self.bot_status['pid'])
                    if process.status() == psutil.STATUS_ZOMBIE:
                        logger.warning("Bot process is zombie, cleaning up")
                        self._cleanup_bot_status()
                except psutil.NoSuchProcess:
                    logger.warning("Bot process not found, cleaning up")
                    self._cleanup_bot_status()
    
    def _update_performance_metrics(self):
        """Update performance metrics from database"""
        try:
            # Get recent trades for performance calculation
            recent_trades = self.db.get_trades(limit=100)
            
            if recent_trades:
                total_trades = len(recent_trades)
                winning_trades = len([t for t in recent_trades if t.get('side') == 'sell' and t.get('total_value', 0) > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Calculate P&L (simplified)
                total_pnl = sum([
                    t.get('total_value', 0) - t.get('fees', 0) 
                    for t in recent_trades if t.get('side') == 'sell'
                ]) - sum([
                    t.get('total_value', 0) + t.get('fees', 0) 
                    for t in recent_trades if t.get('side') == 'buy'
                ])
                
                # Update status
                self.bot_status['performance'].update({
                    'total_pnl': round(total_pnl, 2),
                    'win_rate': round(win_rate, 1),
                    'total_trades': total_trades,
                    'active_positions': 0  # Would need position manager integration
                })
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _monitor_bot_output(self):
        """Monitor bot output in background"""
        try:
            while self.bot_process and self.bot_process.poll() is None:
                line = self.bot_process.stdout.readline()
                if line:
                    logger.info(f"Bot output: {line.strip()}")
                    self.bot_status['logs'].append({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'message': line.strip()
                    })
                    # Keep only last 100 logs
                    if len(self.bot_status['logs']) > 100:
                        self.bot_status['logs'] = self.bot_status['logs'][-100:]
        except Exception as e:
            logger.error(f"Error monitoring bot output: {e}")
        finally:
            # Bot stopped
            self.bot_status['is_running'] = False
            self.bot_status['status'] = 'stopped'
            if self.bot_process:
                self.bot_status['last_error'] = f"Bot process exited with code {self.bot_process.returncode}"
    
    def _cleanup_bot_status(self):
        """Clean up bot status when process stops"""
        self.bot_process = None
        self.bot_status.update({
            'is_running': False,
            'status': 'stopped',
            'pid': None,
            'last_update': datetime.now(timezone.utc).isoformat()
        })
        
        # Clean up PID file
        pid_file = Path.cwd() / 'bot.pid'
        if pid_file.exists():
            try:
                pid_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove PID file: {e}")
        
        # Clean up temp config file
        config_file = Path.cwd() / 'temp_bot_config.json'
        if config_file.exists():
            try:
                config_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove temp config file: {e}")

# Global bot manager instance
bot_manager = BotManager()