import React, { useState, useEffect } from 'react';
import { Play, Square, Pause, AlertTriangle, Settings, TrendingUp, TrendingDown } from 'lucide-react';
import { useTradingUpdates } from '../hooks/useTradingUpdates';
import apiService from '../services/api';
import { useTheme } from '../hooks/useTheme';

const TradingControls = ({ className = '' }) => {
  const { tradingStatus, strategies, currentStrategy } = useTradingUpdates();
  const { isDark } = useTheme();
  const [isLoading, setIsLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(currentStrategy || '');
  const [selectedMode, setSelectedMode] = useState('paper');
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setSelectedStrategy(currentStrategy || '');
  }, [currentStrategy]);

  const handleStart = async () => {
    if (!selectedStrategy) {
      setError('Bitte wählen Sie eine Strategie aus');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      await apiService.startTrading(selectedMode, selectedStrategy);
    } catch (err) {
      setError(err.message || 'Fehler beim Starten des Trading Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    setError('');

    try {
      await apiService.stopTrading();
    } catch (err) {
      setError(err.message || 'Fehler beim Stoppen des Trading Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePause = async () => {
    setIsLoading(true);
    setError('');

    try {
      await apiService.pauseTrading();
    } catch (err) {
      setError(err.message || 'Fehler beim Pausieren des Trading Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const isRunning = tradingStatus === 'running';
  const isPaused = tradingStatus === 'paused';
  const isStopped = tradingStatus === 'stopped';

  const getStatusColor = () => {
    switch (tradingStatus) {
      case 'running':
        return 'text-green-600 dark:text-green-400';
      case 'paused':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'stopped':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-500 dark:text-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (tradingStatus) {
      case 'running':
        return <TrendingUp className="w-4 h-4" />;
      case 'paused':
        return <Pause className="w-4 h-4" />;
      case 'stopped':
        return <Square className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Trading Controls
        </h2>
        <div className={`flex items-center gap-2 ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="text-sm font-medium capitalize">
            {tradingStatus || 'Unbekannt'}
          </span>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
            <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {/* Strategy Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Strategie
          </label>
          <select
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            disabled={isRunning || isPaused || isLoading}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
          >
            <option value="">Strategie wählen...</option>
            {strategies.map((strategy) => (
              <option key={strategy.name} value={strategy.name}>
                {strategy.display_name || strategy.name}
              </option>
            ))}
          </select>
        </div>

        {/* Mode Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Modus
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedMode('paper')}
              disabled={isRunning || isPaused || isLoading}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                selectedMode === 'paper'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              } disabled:opacity-50`}
            >
              Paper Trading
            </button>
            <button
              onClick={() => setSelectedMode('live')}
              disabled={isRunning || isPaused || isLoading}
              className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                selectedMode === 'live'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              } disabled:opacity-50`}
            >
              Live Trading
            </button>
          </div>
          {selectedMode === 'live' && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              ⚠️ Live Trading verwendet echtes Geld!
            </p>
          )}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-2">
          {isStopped && (
            <button
              onClick={handleStart}
              disabled={isLoading || !selectedStrategy}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
            >
              <Play className="w-4 h-4" />
              {isLoading ? 'Startet...' : 'Starten'}
            </button>
          )}

          {isRunning && (
            <>
              <button
                onClick={handlePause}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                <Pause className="w-4 h-4" />
                {isLoading ? 'Pausiert...' : 'Pausieren'}
              </button>
              <button
                onClick={handleStop}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                <Square className="w-4 h-4" />
                {isLoading ? 'Stoppt...' : 'Stoppen'}
              </button>
            </>
          )}

          {isPaused && (
            <>
              <button
                onClick={handleStart}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                <Play className="w-4 h-4" />
                {isLoading ? 'Startet...' : 'Fortsetzen'}
              </button>
              <button
                onClick={handleStop}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                <Square className="w-4 h-4" />
                {isLoading ? 'Stoppt...' : 'Stoppen'}
              </button>
            </>
          )}
        </div>

        {/* Settings Button */}
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-md font-medium transition-colors"
        >
          <Settings className="w-4 h-4" />
          Erweiterte Einstellungen
        </button>

        {/* Strategy Info */}
        {selectedStrategy && (
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-md">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
              Strategie-Info
            </h3>
            {strategies.find(s => s.name === selectedStrategy) && (
              <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <p>
                  <span className="font-medium">Typ:</span>{' '}
                  {strategies.find(s => s.name === selectedStrategy)?.type || 'Unbekannt'}
                </p>
                <p>
                  <span className="font-medium">Beschreibung:</span>{' '}
                  {strategies.find(s => s.name === selectedStrategy)?.description || 'Keine Beschreibung verfügbar'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingControls;