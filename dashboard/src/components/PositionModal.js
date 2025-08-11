import React, { useState } from 'react';
import { X, AlertTriangle } from 'lucide-react';

const PositionModal = ({ position, action, onConfirm, onClose }) => {
  const [amount, setAmount] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const getActionTitle = () => {
    switch (action) {
      case 'add':
        return 'Position erweitern';
      case 'reduce':
        return 'Position reduzieren';
      case 'close':
        return 'Position schließen';
      default:
        return 'Position verwalten';
    }
  };

  const getActionColor = () => {
    switch (action) {
      case 'add':
        return 'bg-green-600 hover:bg-green-700';
      case 'reduce':
        return 'bg-orange-600 hover:bg-orange-700';
      case 'close':
        return 'bg-red-600 hover:bg-red-700';
      default:
        return 'bg-gray-600 hover:bg-gray-700';
    }
  };

  const getActionButtonText = () => {
    switch (action) {
      case 'add':
        return 'Erweitern';
      case 'reduce':
        return 'Reduzieren';
      case 'close':
        return 'Schließen';
      default:
        return 'Bestätigen';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (action !== 'close' && (!amount || isNaN(amount) || parseFloat(amount) <= 0)) {
      setError('Bitte geben Sie einen gültigen Betrag ein');
      return;
    }

    if (action === 'reduce' && parseFloat(amount) > Math.abs(position.amount)) {
      setError('Betrag kann nicht größer als die aktuelle Position sein');
      return;
    }

    setIsLoading(true);

    try {
      await onConfirm(action, action === 'close' ? null : parseFloat(amount));
    } catch (err) {
      setError(err.message || 'Fehler beim Ausführen der Aktion');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount, currency = 'USD') => {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            {getActionTitle()}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Position Info */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Symbol:
              </span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {position.symbol}
              </span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Aktuelle Größe:
              </span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {position.amount?.toFixed(6)} {position.symbol?.split('/')[0]}
              </span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Aktueller Preis:
              </span>
              <span className="text-sm font-semibold text-gray-900 dark:text-white">
                {formatCurrency(position.current_price)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Unrealized P&L:
              </span>
              <span className={`text-sm font-semibold ${
                position.unrealized_pnl >= 0 
                  ? 'text-green-600 dark:text-green-400' 
                  : 'text-red-600 dark:text-red-400'
              }`}>
                {formatCurrency(position.unrealized_pnl || 0)}
              </span>
            </div>
          </div>

          {/* Amount Input for Add/Reduce */}
          {action !== 'close' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {action === 'add' ? 'Hinzufügen' : 'Reduzieren'} ({position.symbol?.split('/')[0]})
              </label>
              <input
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                step="0.000001"
                min="0"
                max={action === 'reduce' ? Math.abs(position.amount) : undefined}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="0.000000"
                required
              />
              {action === 'reduce' && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Maximal: {Math.abs(position.amount)?.toFixed(6)} {position.symbol?.split('/')[0]}
                </p>
              )}
            </div>
          )}

          {/* Warning for Close */}
          {action === 'close' && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
                <span className="text-sm text-red-600 dark:text-red-400">
                  Diese Aktion schließt die gesamte Position permanent.
                </span>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
                <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
              </div>
            </div>
          )}

          {/* Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 font-medium transition-colors disabled:opacity-50"
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={isLoading || (action !== 'close' && !amount)}
              className={`flex-1 px-4 py-2 ${getActionColor()} text-white rounded-md font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isLoading ? 'Wird ausgeführt...' : getActionButtonText()}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PositionModal;