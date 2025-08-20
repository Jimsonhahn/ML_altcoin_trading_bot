"""
Start Server Script - Startet Bot API und Dashboard
==================================================

Dieses Script startet die Dashboard API für Windows Server.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the dashboard API server"""
    try:
        logger.info("Starting Trading Bot Dashboard Server...")
        
        # Import and run the Flask app
        from server.dashboard_api import app
        
        # Get host and port from environment or use defaults
        host = os.getenv('DASHBOARD_HOST', '0.0.0.0')
        port = int(os.getenv('DASHBOARD_PORT', 5000))
        
        logger.info(f"Dashboard will be available at:")
        logger.info(f"- Local: http://localhost:{port}")
        logger.info(f"- Network: http://YOUR_SERVER_IP:{port}")
        logger.info(f"- Mobile Dashboard: http://YOUR_SERVER_IP:{port}/server/mobile_dashboard.html")
        
        # Serve the mobile dashboard
        @app.route('/')
        def serve_dashboard():
            """Serve the mobile dashboard"""
            dashboard_path = Path(__file__).parent / 'mobile_dashboard.html'
            with open(dashboard_path, 'r') as f:
                return f.read()
        
        # Start the server
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()