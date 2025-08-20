"""
Windows Service Setup für Trading Bot
====================================

Erstellt einen Windows Service für 24/7 Bot-Betrieb.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Windows Service Dependencies (installiere mit: pip install pywin32)
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    WINDOWS_SERVICE_AVAILABLE = True
except ImportError:
    WINDOWS_SERVICE_AVAILABLE = False
    print("Windows Service dependencies not available.")
    print("Install with: pip install pywin32")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TradingBotWindowsService(win32serviceutil.ServiceFramework):
    """Windows Service für Trading Bot"""
    
    _svc_name_ = "JanicsTradingBot"
    _svc_display_name_ = "Janics Trading Bot Service"
    _svc_description_ = "Automated cryptocurrency trading bot with mobile dashboard"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True
        
        # Configure logging for service
        log_path = Path(__file__).parent.parent / 'logs' / 'service.log'
        log_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def SvcStop(self):
        """Service Stop Handler"""
        self.logger.info("Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.running = False
        win32event.SetEvent(self.hWaitStop)
    
    def SvcDoRun(self):
        """Service Main Loop"""
        try:
            self.logger.info("Trading Bot Service starting...")
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            
            # Start the dashboard server
            self.start_dashboard_server()
            
        except Exception as e:
            self.logger.error(f"Service error: {str(e)}")
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_ERROR_TYPE,
                servicemanager.PYS_SERVICE_STOPPED,
                (self._svc_name_, str(e))
            )
    
    def start_dashboard_server(self):
        """Startet den Dashboard Server"""
        try:
            from server.dashboard_api import app
            
            # Run Flask app in service context
            app.run(
                host='0.0.0.0',
                port=5000,
                debug=False,
                threaded=True,
                use_reloader=False
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {str(e)}")
            raise


def install_service():
    """Installiert den Windows Service"""
    if not WINDOWS_SERVICE_AVAILABLE:
        print("ERROR: pywin32 not installed. Run: pip install pywin32")
        return False
    
    try:
        # Service installieren
        win32serviceutil.InstallService(
            TradingBotWindowsService._svc_reg_class_,
            TradingBotWindowsService._svc_name_,
            TradingBotWindowsService._svc_display_name_,
            description=TradingBotWindowsService._svc_description_
        )
        print(f"✅ Service '{TradingBotWindowsService._svc_display_name_}' installed successfully!")
        print("Use 'python server/windows_service.py start' to start the service")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install service: {str(e)}")
        return False

def remove_service():
    """Entfernt den Windows Service"""
    if not WINDOWS_SERVICE_AVAILABLE:
        print("ERROR: pywin32 not installed")
        return False
    
    try:
        win32serviceutil.RemoveService(TradingBotWindowsService._svc_name_)
        print(f"✅ Service '{TradingBotWindowsService._svc_display_name_}' removed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to remove service: {str(e)}")
        return False

def main():
    """Main entry point für Service-Management"""
    if len(sys.argv) == 1:
        # Keine Argumente - zeige Hilfe
        print("Trading Bot Windows Service Management")
        print("=====================================")
        print("Usage:")
        print("  python server/windows_service.py install    - Install service")
        print("  python server/windows_service.py remove     - Remove service")
        print("  python server/windows_service.py start      - Start service")
        print("  python server/windows_service.py stop       - Stop service")
        print("  python server/windows_service.py restart    - Restart service")
        print("")
        print("After installation, the service will auto-start on Windows boot.")
        print("Dashboard will be available at: http://localhost:5000")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'install':
        install_service()
    elif command == 'remove':
        remove_service()
    elif command in ['start', 'stop', 'restart']:
        if WINDOWS_SERVICE_AVAILABLE:
            # Use win32serviceutil command line handler
            win32serviceutil.HandleCommandLine(TradingBotWindowsService)
        else:
            print("ERROR: pywin32 not installed")
    else:
        print(f"Unknown command: {command}")
        print("Use 'python server/windows_service.py' for help")

if __name__ == '__main__':
    if WINDOWS_SERVICE_AVAILABLE:
        if len(sys.argv) == 1:
            main()
        else:
            # Let win32serviceutil handle service commands
            try:
                win32serviceutil.HandleCommandLine(TradingBotWindowsService)
            except:
                main()
    else:
        main()