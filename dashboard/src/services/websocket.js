/**
 * WebSocket Service for Real-time Updates
 * Handles WebSocket connection and event management
 */

import { io } from 'socket.io-client';
import apiService from './api';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = new Map();
    this.isConnected = false;
    this.subscriptions = new Set();
  }

  connect() {
    const token = apiService.getToken();
    
    if (!token) {
      console.error('No auth token available for WebSocket connection');
      return;
    }

    if (this.socket) {
      this.disconnect();
    }

    const socketUrl = process.env.REACT_APP_WS_URL || 'http://localhost:5000';
    
    this.socket = io(socketUrl, {
      auth: {
        token: token
      },
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
    });

    this.setupEventHandlers();
  }

  setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
      
      // Re-subscribe to channels after reconnection
      if (this.subscriptions.size > 0) {
        this.subscribe([...this.subscriptions]);
      }
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.emit('connection_status', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        this.emit('connection_error', { error: 'Max reconnection attempts reached' });
      }
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', { error });
    });

    // Trading events
    this.socket.on('trade_update', (data) => {
      this.emit('trade_update', data);
    });

    this.socket.on('order_update', (data) => {
      this.emit('order_update', data);
    });

    this.socket.on('position_update', (data) => {
      this.emit('position_update', data);
    });

    this.socket.on('strategy_update', (data) => {
      this.emit('strategy_update', data);
    });

    // Market data events
    this.socket.on('market_data', (data) => {
      this.emit('market_data', data);
    });

    // Performance events
    this.socket.on('performance_update', (data) => {
      this.emit('performance_update', data);
    });

    // Alert events
    this.socket.on('alert', (data) => {
      this.emit('alert', data);
    });

    // System events
    this.socket.on('system_status', (data) => {
      this.emit('system_status', data);
    });

    this.socket.on('error_event', (data) => {
      this.emit('error_event', data);
    });

    this.socket.on('connection_stats', (data) => {
      this.emit('connection_stats', data);
    });

    // Subscription confirmations
    this.socket.on('subscription_confirmed', (data) => {
      console.log('Subscription confirmed:', data);
      data.channels.forEach(channel => this.subscriptions.add(channel));
    });

    this.socket.on('unsubscription_confirmed', (data) => {
      console.log('Unsubscription confirmed:', data);
      data.channels.forEach(channel => this.subscriptions.delete(channel));
    });

    // Status updates
    this.socket.on('status_update', (data) => {
      this.emit('status_update', data);
    });

    // Ping/Pong for connection health
    this.socket.on('pong', (data) => {
      this.emit('pong', data);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.subscriptions.clear();
    }
  }

  subscribe(channels) {
    if (!this.socket || !this.isConnected) {
      console.warn('WebSocket not connected, cannot subscribe');
      return;
    }

    if (typeof channels === 'string') {
      channels = [channels];
    }

    this.socket.emit('subscribe', { channels });
  }

  unsubscribe(channels) {
    if (!this.socket || !this.isConnected) {
      console.warn('WebSocket not connected, cannot unsubscribe');
      return;
    }

    if (typeof channels === 'string') {
      channels = [channels];
    }

    this.socket.emit('unsubscribe', { channels });
  }

  getStatus() {
    if (!this.socket || !this.isConnected) {
      console.warn('WebSocket not connected, cannot get status');
      return;
    }

    this.socket.emit('get_status');
  }

  ping() {
    if (!this.socket || !this.isConnected) {
      console.warn('WebSocket not connected, cannot ping');
      return;
    }

    this.socket.emit('ping');
  }

  // Event listener management
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
      if (this.listeners.get(event).size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  // Utility methods
  isWebSocketConnected() {
    return this.isConnected;
  }

  getSubscriptions() {
    return [...this.subscriptions];
  }

  reconnect() {
    this.disconnect();
    setTimeout(() => this.connect(), 1000);
  }

  // Predefined channel subscriptions
  subscribeToTradingUpdates() {
    this.subscribe(['trading_updates']);
  }

  subscribeToMarketData() {
    this.subscribe(['market_data']);
  }

  subscribeToAlerts() {
    this.subscribe(['alerts']);
  }

  subscribeToPerformance() {
    this.subscribe(['performance']);
  }

  subscribeToAll() {
    this.subscribe(['trading_updates', 'market_data', 'alerts', 'performance']);
  }

  // Connection health monitoring
  startHealthCheck() {
    this.healthCheckInterval = setInterval(() => {
      if (this.isConnected) {
        this.ping();
      }
    }, 30000); // Ping every 30 seconds
  }

  stopHealthCheck() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }
}

export default new WebSocketService();