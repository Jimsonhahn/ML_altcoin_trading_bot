#!/usr/bin/env python3
"""
🚀 Bulletproof Simple Trading Bot
All-in-One Lösung für Mac Development + Windows Server Deployment

Features:
- SQLite Database für einfache Datenspeicherung
- Built-in Web Dashboard (Flask)
- REST API für Smartphone Zugriff
- Automatische Datensynchronisation
- 24/7 Server Betrieb ready
- Keine komplexen Abhängigkeiten
"""

import os
import json
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import time

# Web Framework - Flask für Einfachheit
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS

# Trading APIs
import ccxt
import requests

# Konfiguration
CONFIG = {
    'DATABASE_PATH': 'trading_data.db',
    'LOG_PATH': 'trading_bot.log',
    'DATA_SYNC_PATH': 'sync_data/',
    'WEB_PORT': 5000,
    'API_PORT': 8001,
    'REFRESH_INTERVAL': 30,
    'DEBUG': True,
    'AUTO_START': True
}

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['LOG_PATH']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTradingDatabase:
    """Einfache SQLite Database für Trading Daten"""
    
    def __init__(self, db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Erstelle Database Tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trading Insights Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT,
                title TEXT,
                description TEXT,
                confidence REAL,
                strategy TEXT,
                data TEXT
            )
        ''')
        
        # Performance Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy TEXT,
                profit_loss REAL,
                win_rate REAL,
                total_trades INTEGER,
                data TEXT
            )
        ''')
        
        # Market Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                price REAL,
                volume REAL,
                change_24h REAL,
                data TEXT
            )
        ''')
        
        # System Health Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                status TEXT,
                data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    
    def add_insight(self, insight_type: str, title: str, description: str, 
                   confidence: float, strategy: str = None, data: dict = None):
        """Füge Trading Insight hinzu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO insights (type, title, description, confidence, strategy, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (insight_type, title, description, confidence, strategy, 
              json.dumps(data) if data else None))
        
        conn.commit()
        conn.close()
    
    def get_latest_insights(self, limit: int = 10) -> List[Dict]:
        """Hole neueste Insights"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM insights 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        insights = []
        for row in rows:
            insights.append({
                'id': row[0],
                'timestamp': row[1],
                'type': row[2],
                'title': row[3],
                'description': row[4],
                'confidence': row[5],
                'strategy': row[6],
                'data': json.loads(row[7]) if row[7] else {}
            })
        
        return insights
    
    def add_performance_data(self, strategy: str, profit_loss: float, 
                           win_rate: float, total_trades: int, data: dict = None):
        """Füge Performance Daten hinzu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance (strategy, profit_loss, win_rate, total_trades, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (strategy, profit_loss, win_rate, total_trades,
              json.dumps(data) if data else None))
        
        conn.commit()
        conn.close()
    
    def get_performance_data(self, days: int = 7) -> List[Dict]:
        """Hole Performance Daten"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM performance 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (since_date.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        performance = []
        for row in rows:
            performance.append({
                'id': row[0],
                'timestamp': row[1],
                'strategy': row[2],
                'profit_loss': row[3],
                'win_rate': row[4],
                'total_trades': row[5],
                'data': json.loads(row[6]) if row[6] else {}
            })
        
        return performance

class SimpleMarketMonitor:
    """Einfacher Market Data Monitor"""
    
    def __init__(self, database: SimpleTradingDatabase):
        self.db = database
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        self.running = False
    
    def start_monitoring(self):
        """Starte Market Monitoring"""
        self.running = True
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        logger.info("✅ Market monitoring started")
    
    def _monitor_loop(self):
        """Market Monitoring Loop"""
        try:
            exchange = ccxt.binance()
            
            while self.running:
                try:
                    for symbol in self.symbols:
                        ticker = exchange.fetch_ticker(symbol)
                        
                        # Speichere Market Data
                        conn = sqlite3.connect(self.db.db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute('''
                            INSERT INTO market_data (symbol, price, volume, change_24h, data)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (symbol, ticker['last'], ticker['baseVolume'], 
                              ticker['percentage'], json.dumps(ticker)))
                        
                        conn.commit()
                        conn.close()
                        
                        # Generiere einfache Insights
                        if abs(ticker['percentage']) > 5:
                            self.db.add_insight(
                                'price_alert',
                                f'{symbol} Preisbewegung',
                                f'{symbol} hat sich um {ticker["percentage"]:.2f}% in 24h bewegt',
                                0.8,
                                'momentum_strategy',
                                {'symbol': symbol, 'change': ticker['percentage']}
                            )
                    
                    time.sleep(CONFIG['REFRESH_INTERVAL'])
                    
                except Exception as e:
                    logger.error(f"Market monitoring error: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            logger.error(f"Market monitor failed: {e}")

class SimpleWebDashboard:
    """Einfaches Web Dashboard mit Flask"""
    
    def __init__(self, database: SimpleTradingDatabase):
        self.db = database
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask Routes"""
        
        @self.app.route('/')
        def dashboard():
            """Hauptseite Dashboard"""
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/health')
        def health():
            """Health Check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'database': os.path.exists(self.db.db_path),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/insights')
        def get_insights():
            """Get Latest Insights"""
            limit = request.args.get('limit', 10, type=int)
            insights = self.db.get_latest_insights(limit)
            return jsonify(insights)
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get Performance Data"""
            days = request.args.get('days', 7, type=int)
            performance = self.db.get_performance_data(days)
            return jsonify(performance)
        
        @self.app.route('/api/market-data')
        def get_market_data():
            """Get Market Data"""
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, price, change_24h, timestamp
                FROM market_data 
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            market_data = []
            for row in rows:
                market_data.append({
                    'symbol': row[0],
                    'price': row[1],
                    'change_24h': row[2],
                    'timestamp': row[3]
                })
            
            return jsonify(market_data)
        
        @self.app.route('/api/sync-data')
        def sync_data():
            """Synchronisiere Daten für Mac/Claude Code"""
            try:
                # Erstelle Sync Directory
                sync_dir = Path(CONFIG['DATA_SYNC_PATH'])
                sync_dir.mkdir(exist_ok=True)
                
                # Exportiere alle Daten
                insights = self.db.get_latest_insights(100)
                performance = self.db.get_performance_data(30)
                
                # Schreibe JSON Files
                with open(sync_dir / 'insights.json', 'w') as f:
                    json.dump(insights, f, indent=2, default=str)
                
                with open(sync_dir / 'performance.json', 'w') as f:
                    json.dump(performance, f, indent=2, default=str)
                
                # System Status
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'total_insights': len(insights),
                    'total_performance_records': len(performance),
                    'database_size': os.path.getsize(self.db.db_path),
                    'uptime': 'running'
                }
                
                with open(sync_dir / 'status.json', 'w') as f:
                    json.dump(status, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': 'Data synchronized successfully',
                    'files_created': ['insights.json', 'performance.json', 'status.json'],
                    'sync_path': str(sync_dir)
                })
                
            except Exception as e:
                logger.error(f"Data sync error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def get_dashboard_html(self):
        """HTML Template für Dashboard"""
        return '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Simple Trading Bot Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow-md p-6 mb-4; }
        .metric { @apply text-center p-4 bg-gray-50 rounded; }
        .insight-item { @apply border-l-4 border-blue-500 pl-4 mb-3; }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="card">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">🚀 Simple Trading Bot</h1>
            <p class="text-gray-600">Bulletproof Simple & Efficient | 
                <span id="status" class="text-green-600">● Online</span> | 
                <span id="lastUpdate">Aktualisiert: --</span>
            </p>
        </div>

        <!-- Metrics Dashboard -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="metric">
                <div class="text-2xl font-bold text-blue-600" id="totalInsights">--</div>
                <div class="text-sm text-gray-600">Insights Today</div>
            </div>
            <div class="metric">
                <div class="text-2xl font-bold text-green-600" id="totalProfit">--</div>
                <div class="text-sm text-gray-600">Total Profit</div>
            </div>
            <div class="metric">
                <div class="text-2xl font-bold text-purple-600" id="winRate">--</div>
                <div class="text-sm text-gray-600">Win Rate</div>
            </div>
            <div class="metric">
                <div class="text-2xl font-bold text-yellow-600" id="activeTrades">--</div>
                <div class="text-sm text-gray-600">Active Trades</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Latest Insights -->
            <div class="card">
                <h2 class="text-xl font-bold mb-4">🧠 Latest Insights</h2>
                <div id="insightsList">
                    <div class="text-center text-gray-500">Loading insights...</div>
                </div>
            </div>

            <!-- Market Data -->
            <div class="card">
                <h2 class="text-xl font-bold mb-4">📈 Market Data</h2>
                <div id="marketData">
                    <div class="text-center text-gray-500">Loading market data...</div>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="card lg:col-span-2">
                <h2 class="text-xl font-bold mb-4">📊 Performance Evolution</h2>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="card">
            <h2 class="text-xl font-bold mb-4">🎛️ Control Panel</h2>
            <div class="flex flex-wrap gap-2">
                <button onclick="refreshData()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                    🔄 Refresh Data
                </button>
                <button onclick="syncData()" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                    📤 Sync to Mac
                </button>
                <button onclick="downloadData()" class="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
                    💾 Download Data
                </button>
                <button onclick="showSystemInfo()" class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600">
                    ℹ️ System Info
                </button>
            </div>
        </div>
    </div>

    <script>
        let performanceChart;

        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);

        // Load data on page load
        document.addEventListener('DOMContentLoaded', refreshData);

        async function refreshData() {
            try {
                // Update status
                document.getElementById('lastUpdate').textContent = 
                    'Aktualisiert: ' + new Date().toLocaleTimeString();

                // Load insights
                const insightsResponse = await fetch('/api/insights');
                const insights = await insightsResponse.json();
                displayInsights(insights);

                // Load market data
                const marketResponse = await fetch('/api/market-data');
                const marketData = await marketResponse.json();
                displayMarketData(marketData);

                // Load performance data
                const performanceResponse = await fetch('/api/performance');
                const performance = await performanceResponse.json();
                updatePerformanceChart(performance);

                // Update metrics
                updateMetrics(insights, performance);

            } catch (error) {
                console.error('Error refreshing data:', error);
                document.getElementById('status').innerHTML = '● <span class="text-red-600">Error</span>';
            }
        }

        function displayInsights(insights) {
            const container = document.getElementById('insightsList');
            
            if (insights.length === 0) {
                container.innerHTML = '<div class="text-gray-500">No insights available</div>';
                return;
            }

            container.innerHTML = insights.slice(0, 5).map(insight => `
                <div class="insight-item">
                    <div class="font-semibold text-gray-800">${insight.title}</div>
                    <div class="text-sm text-gray-600">${insight.description}</div>
                    <div class="text-xs text-gray-400 mt-1">
                        ${new Date(insight.timestamp).toLocaleString()} | 
                        Confidence: ${(insight.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `).join('');
        }

        function displayMarketData(marketData) {
            const container = document.getElementById('marketData');
            
            // Group by symbol and get latest
            const latest = {};
            marketData.forEach(item => {
                if (!latest[item.symbol] || new Date(item.timestamp) > new Date(latest[item.symbol].timestamp)) {
                    latest[item.symbol] = item;
                }
            });

            const symbols = Object.values(latest);
            
            if (symbols.length === 0) {
                container.innerHTML = '<div class="text-gray-500">No market data available</div>';
                return;
            }

            container.innerHTML = symbols.map(item => `
                <div class="flex justify-between items-center py-2 border-b">
                    <span class="font-semibold">${item.symbol}</span>
                    <div class="text-right">
                        <div class="font-bold">$${parseFloat(item.price).toFixed(2)}</div>
                        <div class="text-sm ${item.change_24h >= 0 ? 'text-green-600' : 'text-red-600'}">
                            ${item.change_24h >= 0 ? '+' : ''}${parseFloat(item.change_24h).toFixed(2)}%
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updatePerformanceChart(performance) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }

            const labels = performance.map(p => new Date(p.timestamp).toLocaleDateString());
            const profits = performance.map(p => p.profit_loss);

            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Profit/Loss',
                        data: profits,
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function updateMetrics(insights, performance) {
            // Count insights from today
            const today = new Date().toDateString();
            const todayInsights = insights.filter(i => 
                new Date(i.timestamp).toDateString() === today
            ).length;

            document.getElementById('totalInsights').textContent = todayInsights;

            if (performance.length > 0) {
                const totalProfit = performance.reduce((sum, p) => sum + p.profit_loss, 0);
                const avgWinRate = performance.reduce((sum, p) => sum + (p.win_rate || 0), 0) / performance.length;
                
                document.getElementById('totalProfit').textContent = 
                    (totalProfit >= 0 ? '+' : '') + totalProfit.toFixed(2) + '%';
                document.getElementById('winRate').textContent = 
                    avgWinRate.toFixed(1) + '%';
            }

            document.getElementById('activeTrades').textContent = '3'; // Mock data
        }

        async function syncData() {
            try {
                const response = await fetch('/api/sync-data');
                const result = await response.json();
                
                if (result.success) {
                    alert('✅ Daten erfolgreich synchronisiert!\\nPfad: ' + result.sync_path);
                } else {
                    alert('❌ Synchronisation fehlgeschlagen: ' + result.error);
                }
            } catch (error) {
                alert('❌ Sync Error: ' + error.message);
            }
        }

        function downloadData() {
            window.open('/api/sync-data', '_blank');
            alert('📥 Download gestartet. Daten werden synchronisiert.');
        }

        function showSystemInfo() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    alert(`🖥️ System Info:\\n` +
                          `Status: ${data.status}\\n` +
                          `Version: ${data.version}\\n` +
                          `Database: ${data.database ? 'Connected' : 'Disconnected'}\\n` +
                          `Time: ${new Date(data.timestamp).toLocaleString()}`);
                });
        }
    </script>
</body>
</html>
        '''

class SimpleTradingBot:
    """Hauptklasse für Simple Trading Bot"""
    
    def __init__(self):
        self.database = SimpleTradingDatabase(CONFIG['DATABASE_PATH'])
        self.market_monitor = SimpleMarketMonitor(self.database)
        self.web_dashboard = SimpleWebDashboard(self.database)
        self.running = False
        
        logger.info("🚀 Simple Trading Bot initialized")
    
    def start(self):
        """Starte alle Services"""
        try:
            logger.info("🚀 Starting Simple Trading Bot...")
            
            # Erstelle Sync Directory
            Path(CONFIG['DATA_SYNC_PATH']).mkdir(exist_ok=True)
            
            # Starte Market Monitor
            self.market_monitor.start_monitoring()
            
            # Füge Demo Daten hinzu
            self._add_demo_data()
            
            # Starte Web Dashboard
            logger.info(f"🌐 Starting web dashboard on port {CONFIG['WEB_PORT']}")
            self.running = True
            
            self.web_dashboard.app.run(
                host='0.0.0.0',
                port=CONFIG['WEB_PORT'],
                debug=CONFIG['DEBUG'],
                threaded=True
            )
            
        except KeyboardInterrupt:
            logger.info("👋 Shutting down gracefully...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            raise
    
    def stop(self):
        """Stoppe alle Services"""
        self.running = False
        self.market_monitor.running = False
        logger.info("✅ Simple Trading Bot stopped")
    
    def _add_demo_data(self):
        """Füge Demo Daten für Testing hinzu"""
        # Demo Insights
        demo_insights = [
            {
                'type': 'pattern_detection',
                'title': 'BTC Bullish Pattern Detected',
                'description': 'Strong support level at $65,000 with increasing volume',
                'confidence': 0.85,
                'strategy': 'momentum_strategy'
            },
            {
                'type': 'risk_alert',
                'title': 'High Volatility Warning',
                'description': 'Market volatility increased by 15% in last 4 hours',
                'confidence': 0.92,
                'strategy': 'defensive_strategy'
            },
            {
                'type': 'opportunity',
                'title': 'Arbitrage Opportunity',
                'description': 'ETH price difference of 0.3% between exchanges',
                'confidence': 0.78,
                'strategy': 'arbitrage_strategy'
            }
        ]
        
        for insight in demo_insights:
            self.database.add_insight(**insight)
        
        # Demo Performance Data
        demo_performance = [
            {'strategy': 'momentum_strategy', 'profit_loss': 2.5, 'win_rate': 68.5, 'total_trades': 15},
            {'strategy': 'defensive_strategy', 'profit_loss': 1.2, 'win_rate': 72.0, 'total_trades': 8},
            {'strategy': 'arbitrage_strategy', 'profit_loss': 0.8, 'win_rate': 85.0, 'total_trades': 23}
        ]
        
        for perf in demo_performance:
            self.database.add_performance_data(**perf)
        
        logger.info("✅ Demo data added to database")

def main():
    """Main Entry Point"""
    print("=" * 60)
    print("🚀 BULLETPROOF SIMPLE TRADING BOT")
    print("=" * 60)
    print("✅ All-in-One Solution für Mac + Windows Server")
    print("✅ SQLite Database - Keine komplexen Dependencies")
    print("✅ Built-in Web Dashboard - Flask powered")
    print("✅ REST API für Smartphone Access") 
    print("✅ Automatische Datensynchronisation")
    print("✅ 24/7 Server Ready")
    print("=" * 60)
    print(f"📍 Dashboard: http://localhost:{CONFIG['WEB_PORT']}")
    print(f"📍 API Health: http://localhost:{CONFIG['WEB_PORT']}/api/health")
    print(f"📍 Data Sync: http://localhost:{CONFIG['WEB_PORT']}/api/sync-data")
    print("=" * 60)
    
    try:
        bot = SimpleTradingBot()
        bot.start()
    except Exception as e:
        logger.error(f"❌ Bot startup failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check if port 5000 is available")
        print("2. Install required packages: pip install flask flask-cors ccxt requests")
        print("3. Run with: python simple_bulletproof_trading_bot.py")

if __name__ == "__main__":
    main()