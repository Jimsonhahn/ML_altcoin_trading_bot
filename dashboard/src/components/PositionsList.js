/**
 * PositionsList Component
 * Displays trading positions with real-time updates
 */

import React, { useState, useEffect } from 'react';
import { useTradingUpdates } from '../hooks/useWebSocket';
import robustAPI from '../services/robustApi';
import PositionModal from './PositionModal';

const PositionsList = ({ className = '' }) => {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPrices, setCurrentPrices] = useState({});
  const [selectedPosition, setSelectedPosition] = useState(null);
  const [modalAction, setModalAction] = useState(null);
  const { positions: wsPositions } = useTradingUpdates();

  useEffect(() => {
    loadPositions();
  }, []);

  useEffect(() => {
    // Update positions from WebSocket
    if (wsPositions && wsPositions.length > 0) {
      setPositions(wsPositions);
    }
  }, [wsPositions]);

  const loadPositions = async () => {
    try {
      setLoading(true);
      console.log('🔄 Loading positions...');
      
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/positions');
      const positionsArray = response.positions || [];
      
      console.log('📈 Positions loaded:', positionsArray);
      
      setPositions(positionsArray);
      setError(null);
      
      if (robustAPI.isInFallbackMode()) {
        setError('Using cached positions - API temporarily unavailable');
      }
    } catch (err) {
      setError(`Failed to load positions: ${err.message}`);
      console.error('❌ Error loading positions:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount, currency = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  const formatPercentage = (value) => {
    const percentage = (value * 100).toFixed(2);
    return `${percentage}%`;
  };

  const getPnLColor = (pnl) => {
    if (pnl > 0) return 'text-success-600 dark:text-success-400';
    if (pnl < 0) return 'text-danger-600 dark:text-danger-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const getPositionSizeColor = (amount) => {
    if (amount > 0) return 'text-success-600 dark:text-success-400';
    if (amount < 0) return 'text-danger-600 dark:text-danger-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const handleAddPosition = (position) => {
    setSelectedPosition(position);
    setModalAction('add');
  };

  const handleReducePosition = (position) => {
    setSelectedPosition(position);
    setModalAction('reduce');
  };

  const handleClosePosition = (position) => {
    setSelectedPosition(position);
    setModalAction('close');
  };

  const handlePositionAction = async (action, amount) => {
    if (!selectedPosition) return;

    try {
      switch (action) {
        case 'add':
          await apiService.addToPosition(selectedPosition.symbol, amount);
          break;
        case 'reduce':
          await apiService.reducePosition(selectedPosition.symbol, amount);
          break;
        case 'close':
          await apiService.closePosition(selectedPosition.symbol);
          break;
        default:
          break;
      }
      
      // Refresh positions after action
      await loadPositions();
      
      // Close modal
      setSelectedPosition(null);
      setModalAction(null);
    } catch (err) {
      console.error('Error performing position action:', err);
      setError(err.message || 'Failed to perform position action');
    }
  };

  const closeModal = () => {
    setSelectedPosition(null);
    setModalAction(null);
  };

  if (loading) {
    return (
      <div className={`card ${className}`}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Open Positions
          </h2>
          <div className="loading-spinner"></div>
        </div>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="flex items-center justify-between p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full"></div>
                  <div className="space-y-2">
                    <div className="w-20 h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
                    <div className="w-16 h-3 bg-gray-300 dark:bg-gray-600 rounded"></div>
                  </div>
                </div>
                <div className="w-24 h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`card ${className}`}>
        <div className="text-center py-8">
          <div className="text-danger-600 dark:text-danger-400 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={loadPositions}
            className="btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Open Positions
        </h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {(positions || []).length} position{(positions || []).length !== 1 ? 's' : ''}
          </span>
          <button
            onClick={loadPositions}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {(positions || []).length === 0 ? (
        <div className="text-center py-8">
          <div className="text-gray-400 dark:text-gray-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400">No open positions</p>
        </div>
      ) : (
        <div className="space-y-4">
          {(positions || []).map((position, index) => (
            <div
              key={position.symbol || index}
              className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center">
                    <span className="text-primary-600 dark:text-primary-400 font-medium text-sm">
                      {position.symbol?.split('/')[0] || 'N/A'}
                    </span>
                  </div>
                </div>
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {position.symbol || 'Unknown'}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Entry: {formatCurrency(position.entry_price)}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-6">
                <div className="text-right">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Size</p>
                  <p className={`font-medium ${getPositionSizeColor(position.amount)}`}>
                    {position.amount?.toFixed(6) || '0.000000'}
                  </p>
                </div>

                <div className="text-right">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Current</p>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {formatCurrency(position.current_price)}
                  </p>
                </div>

                <div className="text-right">
                  <p className="text-sm text-gray-500 dark:text-gray-400">P&L</p>
                  <p className={`font-medium ${getPnLColor(position.unrealized_pnl)}`}>
                    {formatCurrency(position.unrealized_pnl || 0)}
                  </p>
                </div>

                <div className="text-right">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Change</p>
                  <p className={`font-medium ${getPnLColor(position.unrealized_pnl)}`}>
                    {position.entry_price && position.current_price
                      ? formatPercentage((position.current_price - position.entry_price) / position.entry_price)
                      : '0.00%'
                    }
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  {/* Add to Position */}
                  <button
                    onClick={() => handleAddPosition(position)}
                    className="p-2 text-green-500 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/20 transition-colors"
                    title="Add to position"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                  </button>
                  
                  {/* Reduce Position */}
                  <button
                    onClick={() => handleReducePosition(position)}
                    className="p-2 text-orange-500 hover:text-orange-700 dark:text-orange-400 dark:hover:text-orange-300 rounded-lg hover:bg-orange-50 dark:hover:bg-orange-900/20 transition-colors"
                    title="Reduce position"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 12H6" />
                    </svg>
                  </button>
                  
                  {/* Close Position */}
                  <button
                    onClick={() => handleClosePosition(position)}
                    className="p-2 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                    title="Close position"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {(positions || []).length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">
              Total Unrealized P&L:
            </span>
            <span className={`font-medium ${getPnLColor(
              positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0)
            )}`}>
              {formatCurrency(
                positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0)
              )}
            </span>
          </div>
        </div>
      )}
      
      {/* Position Modal */}
      {selectedPosition && modalAction && (
        <PositionModal
          position={selectedPosition}
          action={modalAction}
          onConfirm={handlePositionAction}
          onClose={closeModal}
        />
      )}
    </div>
  );
};

export default PositionsList;