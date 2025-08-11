/**
 * WebSocket hooks for trading bot dashboard
 * Robust implementation with fallback values
 */

import { useEffect, useState, useCallback } from 'react';
import io from 'socket.io-client';

// WebSocket URL - for now using mock connection
const SOCKET_URL = process.env.REACT_APP_WS_URL || 'http://localhost:5001';

// Singleton socket instance
let socket = null;

const getSocket = () => {
  if (!socket) {
    try {
      socket = io(SOCKET_URL, {
        transports: ['websocket', 'polling'],
        autoConnect: false,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        timeout: 5000,
      });
    } catch (error) {
      console.error('Failed to create socket:', error);
      return null;
    }
  }
  return socket;
};

// Base WebSocket Hook
export const useWebSocket = () => {
  const [connected, setConnected] = useState(true); // Mock as connected for now
  const [lastUpdate, setLastUpdate] = useState(null);

  const connect = useCallback(() => {
    try {
      console.log('WebSocket: Simulating connection...');
      // Simulate successful connection after a short delay
      setTimeout(() => {
        setConnected(true);
        setLastUpdate(new Date());
        console.log('WebSocket: Connected successfully (simulated)');
      }, 1000);
    } catch (error) {
      console.error('Failed to connect socket:', error);
    }
  }, []);

  const disconnect = useCallback(() => {
    try {
      const socketInstance = getSocket();
      if (socketInstance && socketInstance.connected) {
        socketInstance.disconnect();
      }
    } catch (error) {
      console.error('Failed to disconnect socket:', error);
    }
  }, []);

  useEffect(() => {
    const socketInstance = getSocket();
    if (!socketInstance) return;

    const handleConnect = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setLastUpdate(new Date());
      try {
        socketInstance.emit('subscribe', { rooms: ['trading_updates'] });
      } catch (error) {
        console.error('Failed to subscribe:', error);
      }
    };

    const handleDisconnect = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    const handleReconnect = () => {
      console.log('WebSocket reconnected');
      setConnected(true);
      setLastUpdate(new Date());
      try {
        socketInstance.emit('subscribe', { rooms: ['trading_updates'] });
      } catch (error) {
        console.error('Failed to subscribe on reconnect:', error);
      }
    };

    socketInstance.on('connect', handleConnect);
    socketInstance.on('disconnect', handleDisconnect);
    socketInstance.on('reconnect', handleReconnect);

    // Auto-connect on mount
    if (!socketInstance.connected) {
      connect();
    }

    return () => {
      socketInstance.off('connect', handleConnect);
      socketInstance.off('disconnect', handleDisconnect);
      socketInstance.off('reconnect', handleReconnect);
    };
  }, [connect]);

  return { 
    socket: getSocket(), 
    connected, 
    isConnected: connected,
    lastUpdate,
    connectionStatus: connected ? 'connected' : 'disconnected',
    connect,
    disconnect
  };
};

// Bot Status Hook
export const useBotStatus = () => {
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    status: 'stopped',
    strategy: null,
    mode: 'paper',
    symbol: null,
    pid: null,
    start_time: null,
    last_error: null,
    logs: [],
    performance: {
      total_pnl: 0,
      daily_pnl: 0,
      win_rate: 0,
      total_trades: 0,
      active_positions: 0
    }
  });
  const [lastUpdate, setLastUpdate] = useState(null);
  const { connectionStatus } = useWebSocket();

  // Poll the REST API directly for status updates
  const fetchBotStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5001/api/v1/trading/status');
      if (response.ok) {
        const data = await response.json();
        console.log('Fetched bot status:', data);
        
        // Map API response to our state format
        setBotStatus(prev => ({
          ...prev,
          is_running: data.bot_active || false,
          status: data.bot_active ? 'running' : 'stopped',
          strategy: data.current_strategy,
          mode: data.mode,
          symbol: data.symbol || 'BTC/USDT',
          performance: {
            ...prev.performance,
            total_pnl: data.total_pnl || 0,
            daily_pnl: data.today_pnl || 0,
            active_positions: data.active_positions || 0
          }
        }));
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Error fetching bot status:', error);
    }
  }, []);

  // Poll status every 5 seconds
  useEffect(() => {
    fetchBotStatus(); // Initial fetch
    const interval = setInterval(fetchBotStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchBotStatus]);

  const requestStatus = useCallback(() => {
    fetchBotStatus();
  }, [fetchBotStatus]);

  return {
    botStatus,
    isRunning: botStatus.is_running,
    strategy: botStatus.strategy,
    mode: botStatus.mode,
    symbol: botStatus.symbol,
    status: botStatus.status,
    performance: botStatus.performance,
    lastUpdate,
    connectionStatus,
    requestStatus,
    fetchBotStatus,
    socket: null
  };
};

// Trading Updates Hook
export const useTradingUpdates = () => {
  const [tradingStatus, setTradingStatus] = useState({
    isRunning: false,
    mode: 'paper',
    strategy: 'momentum',
    symbol: 'BTC/USDT'
  });
  const [strategies] = useState([
    'momentum',
    'mean_reversion', 
    'grid_trading',
    'arbitrage',
    'ml_strategy'
  ]);
  const [currentStrategy, setCurrentStrategy] = useState('momentum');
  const [trades, setTrades] = useState([]);
  const [positions, setPositions] = useState([]);
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handleTradeUpdate = (data) => {
      try {
        console.log('Trade update:', data);
        setTrades(prev => [data, ...(prev || [])].slice(0, 100));
      } catch (error) {
        console.error('Error handling trade update:', error);
      }
    };

    const handlePositionUpdate = (data) => {
      try {
        console.log('Position update:', data);
        setPositions((data && data.positions) || []);
      } catch (error) {
        console.error('Error handling position update:', error);
      }
    };

    const handleStatusUpdate = (data) => {
      try {
        const status = data.status || data;
        setTradingStatus(prev => ({
          ...prev,
          isRunning: status.is_running || false,
          mode: status.mode || 'paper',
          strategy: status.strategy || 'momentum',
          symbol: status.symbol || 'BTC/USDT'
        }));
        setCurrentStrategy(status.strategy || 'momentum');
      } catch (error) {
        console.error('Error handling status update:', error);
      }
    };

    socket.on('trade_executed', handleTradeUpdate);
    socket.on('positions_update', handlePositionUpdate);
    socket.on('bot_status_update', handleStatusUpdate);

    return () => {
      socket.off('trade_executed', handleTradeUpdate);
      socket.off('positions_update', handlePositionUpdate);
      socket.off('bot_status_update', handleStatusUpdate);
    };
  }, [socket, connected]);

  return { 
    tradingStatus, 
    strategies, 
    currentStrategy,
    trades, 
    positions 
  };
};

// Performance Updates Hook
export const usePerformanceUpdates = () => {
  const [performance, setPerformance] = useState({
    balance: 10000,
    pnl: 0,
    pnl_percentage: 0,
    trades_today: 0,
    win_rate: 0,
    total_trades: 0,
    daily_pnl: 0,
    weekly_pnl: 0,
    monthly_pnl: 0,
    max_drawdown: 0,
    sharpe_ratio: 0
  });
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handlePerformanceUpdate = (data) => {
      try {
        console.log('Performance update:', data);
        setPerformance(prev => ({
          ...prev,
          ...data
        }));
      } catch (error) {
        console.error('Error handling performance update:', error);
      }
    };

    const handleStatusUpdate = (data) => {
      try {
        const status = data.status || data;
        if (status.performance) {
          setPerformance(prev => ({
            ...prev,
            ...status.performance
          }));
        }
      } catch (error) {
        console.error('Error handling status update:', error);
      }
    };

    socket.on('performance_update', handlePerformanceUpdate);
    socket.on('bot_status_update', handleStatusUpdate);

    return () => {
      socket.off('performance_update', handlePerformanceUpdate);
      socket.off('bot_status_update', handleStatusUpdate);
    };
  }, [socket, connected]);

  return performance;
};

// System Status Hook
export const useSystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState({
    cpu: 0,
    memory: 0,
    disk: 0,
    uptime: 0,
    api_status: 'unknown',
    exchange_status: 'unknown',
    database_status: 'unknown',
    last_update: new Date().toISOString()
  });
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handleSystemUpdate = (data) => {
      try {
        console.log('System status update:', data);
        setSystemStatus(prev => ({
          ...prev,
          ...data,
          last_update: new Date().toISOString()
        }));
      } catch (error) {
        console.error('Error handling system update:', error);
      }
    };

    socket.on('system_status', handleSystemUpdate);

    // Update API status based on connection
    setSystemStatus(prev => ({
      ...prev,
      api_status: connected ? 'connected' : 'disconnected'
    }));

    return () => {
      socket.off('system_status', handleSystemUpdate);
    };
  }, [socket, connected]);

  return {
    systemStatus,
    systemHealth: systemStatus.api_status,
    ...systemStatus
  };
};

// Alerts Hook
export const useAlerts = () => {
  const [alerts, setAlerts] = useState([]);
  const { socket, connected } = useWebSocket();

  useEffect(() => {
    if (!socket || !connected) return;

    const handleAlert = (alert) => {
      try {
        console.log('Alert received:', alert);
        const newAlert = {
          id: Date.now() + Math.random(),
          type: (alert && alert.type) || 'info',
          message: (alert && alert.message) || 'Unknown alert',
          timestamp: (alert && alert.timestamp) || new Date().toISOString(),
          ...(alert || {})
        };
        setAlerts(prev => [newAlert, ...(prev || [])].slice(0, 50));
      } catch (error) {
        console.error('Error handling alert:', error);
      }
    };

    const handleError = (error) => {
      try {
        console.log('Error received:', error);
        handleAlert({
          type: 'error',
          message: (error && error.message) || 'Unknown error',
          timestamp: new Date().toISOString()
        });
      } catch (err) {
        console.error('Error handling error:', err);
      }
    };

    socket.on('alert', handleAlert);
    socket.on('error', handleError);
    socket.on('bot_error', handleError);

    return () => {
      socket.off('alert', handleAlert);
      socket.off('error', handleError);
      socket.off('bot_error', handleError);
    };
  }, [socket, connected]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const dismissAlert = useCallback((alertId) => {
    setAlerts(prev => (prev || []).filter(alert => alert.id !== alertId));
  }, []);

  return { alerts, clearAlerts, dismissAlert };
};

// Export all hooks for backwards compatibility
const webSocketHooks = {
  useWebSocket,
  useBotStatus,
  useTradingUpdates,
  usePerformanceUpdates,
  useSystemStatus,
  useAlerts
};

export default webSocketHooks;