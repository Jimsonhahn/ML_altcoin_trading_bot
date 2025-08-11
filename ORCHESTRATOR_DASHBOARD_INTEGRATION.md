# Orchestrator Dashboard Integration Guide
=========================================

## 🚀 Quick Start

### Start Everything with One Command:
```bash
# Paper Trading Mode (default)
./start_with_dashboard.sh

# Live Trading Mode
./start_with_dashboard.sh live

# Hybrid Mode
./start_with_dashboard.sh hybrid
```

This starts:
1. API Server (Port 5000)
2. Orchestrator Background Worker
3. Dashboard (Port 3000) - if available

## 📊 Dashboard Features

### 1. **Orchestrator Overview** 
Real-time display of:
- Current trading mode (Paper/Live/Hybrid)
- Total discovered strategies
- Active strategies and allocations
- Market regime detection
- Portfolio performance

### 2. **Portfolio Management**
- Total portfolio value
- Cash balance
- Open positions
- P&L tracking (absolute and percentage)
- Win rate and Sharpe ratio
- Maximum drawdown

### 3. **Strategy DNA Profiles**
For each discovered strategy:
- Risk level (Conservative/Moderate/Aggressive/Extreme)
- Timeframe (Scalping/Intraday/Swing/Position)
- Signal sources
- Expected win rate
- Cooperation score
- Conflict detection

### 4. **Health Monitoring**
- Real-time health scores for each strategy
- Performance degradation alerts
- Emergency stop notifications
- Risk warnings

### 5. **A/B Testing**
- Active test tracking
- Performance comparisons
- Statistical significance
- Test recommendations (Adopt/Reject/Extend)

## 🔌 API Endpoints

### Orchestrator Status
```bash
GET /api/v1/orchestrator/status
Authorization: Bearer <token>
```

### Discovered Strategies
```bash
GET /api/v1/orchestrator/strategies
Authorization: Bearer <token>
```

### Portfolio Details
```bash
GET /api/v1/orchestrator/portfolio
Authorization: Bearer <token>
```

### Market Analysis
```bash
GET /api/v1/orchestrator/market-analysis
Authorization: Bearer <token>
```

### Strategy Allocation
```bash
GET /api/v1/orchestrator/strategy-allocation
Authorization: Bearer <token>
```

### Health Metrics
```bash
GET /api/v1/orchestrator/health-metrics/<strategy_name>
Authorization: Bearer <token>
```

### A/B Tests
```bash
GET /api/v1/orchestrator/ab-tests
Authorization: Bearer <token>
```

### Performance History
```bash
GET /api/v1/orchestrator/performance-history?hours=24
Authorization: Bearer <token>
```

### Switch Trading Mode
```bash
POST /api/v1/orchestrator/switch-mode
Authorization: Bearer <token>
Content-Type: application/json

{
    "mode": "live",
    "transfer_positions": true
}
```

## 🔄 WebSocket Events

### Subscribe to Updates
```javascript
// Connect to WebSocket
const socket = io('http://localhost:5000/ws');

// Subscribe to orchestrator updates
socket.emit('subscribe_orchestrator');

// Listen for updates
socket.on('orchestrator_update', (data) => {
    console.log('Orchestrator update:', data);
});

socket.on('portfolio_update', (data) => {
    console.log('Portfolio update:', data);
});

socket.on('health_alert', (data) => {
    console.log('Health alert:', data);
});
```

### WebSocket Events Available:
- `orchestrator_update` - Strategy allocations and market analysis
- `portfolio_update` - Portfolio value and position changes
- `health_alert` - Strategy health warnings and emergency stops
- `orchestrator_status_update` - General status updates

## 📈 Dashboard Components

### 1. **OrchestratorOverview Component**
```jsx
// Shows main orchestrator status
<OrchestratorOverview 
    mode={tradingMode}
    discoveredStrategies={strategies}
    marketRegime={regime}
    portfolioValue={value}
/>
```

### 2. **PortfolioTracker Component**
```jsx
// Real-time portfolio tracking
<PortfolioTracker
    totalValue={portfolioValue}
    pnl={totalPnl}
    positions={openPositions}
    winRate={winRate}
/>
```

### 3. **StrategyDNA Component**
```jsx
// Strategy DNA visualization
<StrategyDNA
    strategies={discoveredStrategies}
    allocations={strategyAllocations}
/>
```

### 4. **HealthMonitor Component**
```jsx
// Health monitoring dashboard
<HealthMonitor
    strategies={strategies}
    alerts={healthAlerts}
    emergencyStops={stops}
/>
```

### 5. **ABTestDashboard Component**
```jsx
// A/B testing results
<ABTestDashboard
    activeTests={activeTests}
    completedTests={completedTests}
/>
```

## 🛠️ Configuration

### Environment Variables
```bash
# API Configuration
FLASK_PORT=5000
ORCHESTRATOR_MODE=paper
ORCHESTRATOR_CAPITAL=10000

# Dashboard Configuration  
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000
```

### Configuration File (orchestrator_config.json)
```json
{
  "trading_mode": {
    "default": "paper"
  },
  "portfolio_management": {
    "initial_capital": 10000.0,
    "max_positions": 20
  }
}
```

## 🔍 Monitoring

### Check Service Status
```bash
# API Health
curl http://localhost:5000/health

# Orchestrator Status (requires auth)
curl -H "Authorization: Bearer <token>" \
     http://localhost:5000/api/v1/orchestrator/status
```

### View Logs
```bash
# API Logs
tail -f logs/api.log

# Orchestrator Logs
tail -f logs/orchestrator.log
```

## 🚨 Troubleshooting

### Dashboard Not Updating
1. Check WebSocket connection in browser console
2. Verify API is running: `http://localhost:5000/health`
3. Check orchestrator worker is running

### Strategies Not Discovered
1. Ensure strategies are in the `strategies/` directory
2. Check for Python syntax errors in strategy files
3. Verify orchestrator has proper permissions

### Portfolio Not Tracking
1. Check trading mode configuration
2. Verify initial capital is set
3. Review position limits in config

## 📱 Mobile Dashboard Support

The dashboard is responsive and works on mobile devices:
- Real-time updates via WebSocket
- Touch-friendly interface
- Optimized data display for small screens

## 🔐 Security

- All API endpoints require JWT authentication
- WebSocket connections are authenticated
- Sensitive data is encrypted
- Rate limiting on API endpoints

## 🎯 Next Steps

1. Customize dashboard layout for your needs
2. Add custom indicators and charts
3. Integrate with more exchanges
4. Set up alerts and notifications
5. Configure automated reporting

## 💡 Tips

- Start in Paper mode to test strategies
- Use Hybrid mode to gradually move to live trading
- Monitor health scores regularly
- Review A/B test results before adopting changes
- Keep an eye on the emergency stop indicators