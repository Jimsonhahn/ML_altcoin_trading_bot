import React, { useState, useEffect } from 'react';
import { Settings, Play, Square, AlertTriangle, Info, Save, RotateCcw, TrendingUp, CheckCircle, XCircle, Zap } from 'lucide-react';
import robustAPI from '../services/robustApi';
import { useTheme } from '../hooks/useTheme';
import { useBotStatus } from '../hooks/useWebSocket';
import { validateTradingConfig, validateField, formatValidationMessage, getValidationSeverity } from '../utils/validation';

const StrategyConfiguration = ({ className = '' }) => {
  const { isDark } = useTheme();
  const { botStatus, isRunning, strategy: currentStrategy, mode: currentMode, symbol: currentSymbol, performance, lastUpdate, connectionStatus, fetchBotStatus } = useBotStatus();
  const [selectedStrategy, setSelectedStrategy] = useState('super_lazy_billionaire');
  const [selectedMode, setSelectedMode] = useState('paper');
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [capital, setCapital] = useState(10000);
  const [riskPerTrade, setRiskPerTrade] = useState(0.02);
  const [strategyParams, setStrategyParams] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [validationErrors, setValidationErrors] = useState([]);
  const [validationWarnings, setValidationWarnings] = useState([]);
  const [fieldValidation, setFieldValidation] = useState({});

  // Strategy definitions with parameters
  const strategies = {
    super_lazy_billionaire: {
      name: '🚀 Super Lazy Billionaire Strategy',
      description: 'AI-Driven Master Trading Strategy mit Kelly Criterion, ML-Enhanced Entry/Exit und dynamischer Allokation',
      params: {
        capital_allocation: { label: 'Kapital Allokation', type: 'number', default: 0.95, min: 0.1, max: 1.0, step: 0.05 },
        kelly_factor: { label: 'Kelly Faktor', type: 'number', default: 0.25, min: 0.1, max: 0.5, step: 0.05 },
        ml_confidence_threshold: { label: 'ML Konfidenz Schwelle', type: 'number', default: 0.7, min: 0.5, max: 0.95, step: 0.05 },
        risk_per_trade: { label: 'Risiko pro Trade', type: 'number', default: 0.02, min: 0.005, max: 0.05, step: 0.005 },
        max_positions: { label: 'Max. Positionen', type: 'number', default: 5, min: 1, max: 10 },
        rebalance_threshold: { label: 'Rebalancing Schwelle', type: 'number', default: 0.1, min: 0.05, max: 0.3, step: 0.05 }
      }
    },
    momentum: {
      name: 'Momentum Trading',
      description: 'Folgt starken Kursbewegungen und Trends',
      params: {
        rsi_period: { label: 'RSI Periode', type: 'number', default: 14, min: 5, max: 30 },
        rsi_overbought: { label: 'RSI Überkauft', type: 'number', default: 70, min: 60, max: 90 },
        rsi_oversold: { label: 'RSI Überverkauft', type: 'number', default: 30, min: 10, max: 40 },
        sma_short: { label: 'Kurzer SMA', type: 'number', default: 5, min: 3, max: 15 },
        sma_long: { label: 'Langer SMA', type: 'number', default: 20, min: 10, max: 50 },
        volume_spike_threshold: { label: 'Volumen Schwelle', type: 'number', default: 1.5, min: 1.0, max: 3.0, step: 0.1 }
      }
    },
    mean_reversion: {
      name: 'Mean Reversion',
      description: 'Kauft bei Überverkauft, verkauft bei Überkauft',
      params: {
        bollinger_period: { label: 'Bollinger Periode', type: 'number', default: 20, min: 10, max: 50 },
        bollinger_std: { label: 'Bollinger Std Dev', type: 'number', default: 2.0, min: 1.0, max: 3.0, step: 0.1 },
        rsi_period: { label: 'RSI Periode', type: 'number', default: 14, min: 5, max: 30 },
        use_rsi_filter: { label: 'RSI Filter aktivieren', type: 'boolean', default: true }
      }
    },
    grid_trading: {
      name: 'Grid Trading',
      description: 'Platziert Kauf-/Verkaufsorders in einem Raster',
      params: {
        num_grids: { label: 'Anzahl Grids', type: 'number', default: 10, min: 5, max: 20 },
        price_range_multiplier: { label: 'Preis-Range Multiplikator', type: 'number', default: 0.05, min: 0.01, max: 0.1, step: 0.01 },
        grid_size_percent: { label: 'Grid Größe (%)', type: 'number', default: 0.01, min: 0.005, max: 0.05, step: 0.005 }
      }
    },
    arbitrage: {
      name: 'Arbitrage',
      description: 'Nutzt Preisunterschiede zwischen Börsen',
      params: {
        min_profit_threshold: { label: 'Min. Gewinn Schwelle (%)', type: 'number', default: 0.5, min: 0.1, max: 2.0, step: 0.1 },
        max_slippage: { label: 'Max. Slippage (%)', type: 'number', default: 0.2, min: 0.05, max: 1.0, step: 0.05 }
      }
    },
    ml_strategy: {
      name: 'ML Strategy',
      description: 'Verwendet Machine Learning für Vorhersagen',
      params: {
        model_confidence_threshold: { label: 'Model Confidence', type: 'number', default: 0.7, min: 0.5, max: 0.9, step: 0.05 },
        lookback_window: { label: 'Lookback Fenster', type: 'number', default: 20, min: 10, max: 50 },
        feature_importance_min: { label: 'Min. Feature Importance', type: 'number', default: 0.1, min: 0.01, max: 0.5, step: 0.01 }
      }
    }
  };

  const symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT'];

  useEffect(() => {
    // Initialize strategy params when strategy changes
    const strategy = strategies[selectedStrategy];
    if (strategy) {
      const defaultParams = {};
      Object.entries(strategy.params).forEach(([key, config]) => {
        defaultParams[key] = config.default;
      });
      setStrategyParams(defaultParams);
    }
  }, [selectedStrategy]);

  // Initial status fetch with error recovery
  useEffect(() => {
    const fetchInitialStatus = async () => {
      try {
        const response = await robustAPI.fetchWithRetry('/api/v1/trading/status');
        console.log('Initial bot status:', response);
        
        // If we get a status but there's an error, try force cleanup
        if (!response.success || (response.status && response.status.is_running && !response.verified)) {
          console.warn('Status may be inconsistent, checking...');
          setError('Status wird überprüft...');
          
          // Wait a moment then check again
          setTimeout(async () => {
            try {
              const recheck = await robustAPI.fetchWithRetry('/api/v1/trading/status');
              if (!recheck.success || !recheck.verified) {
                setError('Status inkonsistent. Verwenden Sie "Force Cleanup" falls nötig.');
              } else {
                setError('');
              }
            } catch (err) {
              setError('Kann Bot-Status nicht abrufen');
            }
          }, 2000);
        }
      } catch (err) {
        console.error('Error fetching initial bot status:', err);
        setError('Kann Bot-Status nicht abrufen');
      }
    };
    fetchInitialStatus();
  }, []);

  const handleParamChange = (paramKey, value) => {
    setStrategyParams(prev => ({
      ...prev,
      [paramKey]: value
    }));
    
    // Real-time validation for strategy parameters
    validateConfiguration();
  };

  const validateConfiguration = () => {
    const config = {
      mode: selectedMode,
      strategy: selectedStrategy,
      symbol: selectedSymbol,
      capital: capital,
      risk_per_trade: riskPerTrade,
      strategy_params: strategyParams
    };

    const validationResult = validateTradingConfig(config);
    setValidationErrors(validationResult.errors);
    setValidationWarnings(validationResult.warnings);

    // Update field-specific validation
    const newFieldValidation = {};
    validationResult.errors.forEach(error => {
      newFieldValidation[error.field] = {
        type: 'error',
        message: formatValidationMessage(error)
      };
    });
    validationResult.warnings.forEach(warning => {
      if (!newFieldValidation[warning.field]) {
        newFieldValidation[warning.field] = {
          type: 'warning',
          message: warning.message
        };
      }
    });
    setFieldValidation(newFieldValidation);

    return validationResult.isValid;
  };

  const handleFieldChange = (fieldName, value) => {
    // Update state
    switch (fieldName) {
      case 'mode':
        setSelectedMode(value);
        break;
      case 'strategy':
        setSelectedStrategy(value);
        break;
      case 'symbol':
        setSelectedSymbol(value);
        break;
      case 'capital':
        setCapital(value);
        break;
      case 'risk_per_trade':
        setRiskPerTrade(value);
        break;
    }

    // Validate field
    setTimeout(() => validateConfiguration(), 100);
  };

  // Validate on component mount and when dependencies change
  useEffect(() => {
    validateConfiguration();
  }, [selectedMode, selectedStrategy, selectedSymbol, capital, riskPerTrade, strategyParams]);

  const handleStartBot = async () => {
    setIsLoading(true);
    setError('');
    setSuccess('');

    // Validate configuration before starting
    if (!validateConfiguration()) {
      setError('Konfiguration ist ungültig. Bitte korrigieren Sie die Fehler bevor Sie fortfahren.');
      setIsLoading(false);
      return;
    }

    try {
      const config = {
        mode: selectedMode,
        strategy: selectedStrategy,
        symbol: selectedSymbol,
        capital: capital,
        risk_per_trade: riskPerTrade,
        strategy_params: strategyParams
      };

      const response = await robustAPI.fetchWithRetry('/api/v1/trading/start', {
        method: 'POST',
        body: config
      });
      const result = response.data || response;
      
      if (result.status === 'started' || result.success) {
        setSuccess('Trading bot started successfully');
        // Clear validation errors on successful start
        setValidationErrors([]);
        setValidationWarnings([]);
        
        // Refresh bot status after successful start
        setTimeout(async () => {
          try {
            await checkBotStatus();
          } catch (error) {
            console.error('Error refreshing bot status:', error);
          }
        }, 1000);
      } else {
        setError(result.message || 'Fehler beim Starten des Bots');
      }
    } catch (err) {
      setError(err.message || 'Fehler beim Starten des Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopBot = async () => {
    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/stop', {
        method: 'POST'
      });
      const result = response.data || response;
      
      if (result.status === 'stopped' || result.success) {
        setSuccess('Trading bot stopped successfully');
        
        // Refresh bot status after successful stop
        setTimeout(async () => {
          try {
            await checkBotStatus();
          } catch (error) {
            console.error('Error refreshing bot status:', error);
          }
        }, 1000);
      } else {
        setError(result.message || 'Fehler beim Stoppen des Bots');
      }
    } catch (err) {
      setError(err.message || 'Fehler beim Stoppen des Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestartBot = async () => {
    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const config = {
        mode: selectedMode,
        strategy: selectedStrategy,
        symbol: selectedSymbol,
        capital: capital,
        risk_per_trade: riskPerTrade,
        strategy_params: strategyParams
      };

      const result = await robustAPI.fetchWithRetry('/api/v1/trading/restart', {
        method: 'POST',
        body: config
      });
      if (result.success) {
        setSuccess('Bot erfolgreich neu gestartet!');
      } else {
        setError(result.message || 'Fehler beim Neustarten des Bots');
      }
    } catch (err) {
      setError(err.message || 'Fehler beim Neustarten des Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const checkBotStatus = async () => {
    try {
      if (fetchBotStatus) {
        await fetchBotStatus();
        console.log('Refreshed bot status');
      }
    } catch (error) {
      console.error('Error checking bot status:', error);
    }
  };

  const handleForceStop = async () => {
    if (!window.confirm('Bot wirklich force stoppen? Dies beendet alle laufenden Prozesse.')) {
      return;
    }

    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await robustAPI.fetchWithRetry('/api/v1/trading/force-stop', {
        method: 'POST'
      });
      const result = response.data || response;
      
      if (result.status === 'force_stopped' || result.success) {
        setSuccess('Trading bot force stopped and cleaned up successfully');
        // Clear any validation errors
        setValidationErrors([]);
        setValidationWarnings([]);
        
        // Refresh bot status after force stop
        setTimeout(async () => {
          try {
            await checkBotStatus();
          } catch (error) {
            console.error('Error refreshing bot status:', error);
          }
        }, 1000);
      } else {
        setError(result.message || 'Fehler beim Force-Stop des Bots');
      }
    } catch (err) {
      setError(err.message || 'Fehler beim Force-Stop des Bots');
    } finally {
      setIsLoading(false);
    }
  };

  const renderParameterInput = (paramKey, paramConfig) => {
    const value = strategyParams[paramKey];

    if (paramConfig.type === 'boolean') {
      return (
        <div key={paramKey} className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {paramConfig.label}
          </label>
          <button
            onClick={() => handleParamChange(paramKey, !value)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              value ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                value ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      );
    }

    return (
      <div key={paramKey}>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          {paramConfig.label}
        </label>
        <input
          type="number"
          value={value || ''}
          onChange={(e) => handleParamChange(paramKey, parseFloat(e.target.value) || 0)}
          min={paramConfig.min}
          max={paramConfig.max}
          step={paramConfig.step || 1}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>
    );
  };

  // Real-time connection status indicator
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-green-600 dark:text-green-400';
      case 'connecting':
        return 'text-yellow-600 dark:text-yellow-400';
      default:
        return 'text-red-600 dark:text-red-400';
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Strategie Konfiguration
        </h2>
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 ${isRunning ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
            <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm font-medium">
              {isRunning ? 'Läuft' : 'Gestoppt'}
            </span>
          </div>
          <div className={`flex items-center gap-1 ${getConnectionStatusColor()}`}>
            <div className={`w-1.5 h-1.5 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs">
              {connectionStatus === 'connected' ? 'Live' : 'Offline'}
            </span>
          </div>
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

      {success && (
        <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
            <span className="text-sm text-green-600 dark:text-green-400">{success}</span>
          </div>
        </div>
      )}

      {/* Current Status */}
      {isRunning && (
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span className="text-sm font-medium text-blue-900 dark:text-blue-100">Aktueller Status</span>
          </div>
          <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
            <p><span className="font-medium">Strategie:</span> {strategies[currentStrategy]?.name || currentStrategy}</p>
            <p><span className="font-medium">Modus:</span> {currentMode === 'paper' ? 'Paper Trading' : 'Live Trading'}</p>
            <p><span className="font-medium">Symbol:</span> {currentSymbol}</p>
            {lastUpdate && (
              <p><span className="font-medium">Letztes Update:</span> {lastUpdate.toLocaleTimeString()}</p>
            )}
          </div>
        </div>
      )}

      {/* Validation Summary */}
      {((validationErrors || []).length > 0 || (validationWarnings || []).length > 0) && (
        <div className="mb-4 space-y-3">
          {(validationErrors || []).length > 0 && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
                <h3 className="text-sm font-medium text-red-900 dark:text-red-100">
                  Konfigurationsfehler ({(validationErrors || []).length})
                </h3>
              </div>
              <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                {(validationErrors || []).slice(0, 5).map((error, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="w-1 h-1 bg-red-500 rounded-full mt-2 flex-shrink-0" />
                    <span>{formatValidationMessage(error)}</span>
                  </li>
                ))}
                {(validationErrors || []).length > 5 && (
                  <li className="text-red-600 dark:text-red-400 font-medium">
                    ... und {(validationErrors || []).length - 5} weitere Fehler
                  </li>
                )}
              </ul>
            </div>
          )}

          {(validationWarnings || []).length > 0 && (
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
                <h3 className="text-sm font-medium text-yellow-900 dark:text-yellow-100">
                  Warnungen ({(validationWarnings || []).length})
                </h3>
              </div>
              <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
                {(validationWarnings || []).slice(0, 3).map((warning, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="w-1 h-1 bg-yellow-500 rounded-full mt-2 flex-shrink-0" />
                    <span>{warning.message}</span>
                  </li>
                ))}
                {(validationWarnings || []).length > 3 && (
                  <li className="text-yellow-600 dark:text-yellow-400 font-medium">
                    ... und {(validationWarnings || []).length - 3} weitere Warnungen
                  </li>
                )}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="space-y-6">
        {/* Basic Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Strategie
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              disabled={isRunning || isLoading}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              {Object.entries(strategies).map(([key, strategy]) => (
                <option key={key} value={key}>
                  {strategy.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Symbol
            </label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              disabled={isRunning || isLoading}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              {symbols.map((symbol) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Kapital (USDT)
            </label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(parseFloat(e.target.value) || 0)}
              disabled={isRunning || isLoading}
              min="100"
              max="1000000"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Risiko pro Trade (%)
            </label>
            <input
              type="number"
              value={riskPerTrade * 100}
              onChange={(e) => setRiskPerTrade((parseFloat(e.target.value) || 0) / 100)}
              disabled={isRunning || isLoading}
              min="0.1"
              max="10"
              step="0.1"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
        </div>

        {/* Mode Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Trading Modus
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedMode('paper')}
              disabled={isRunning || isLoading}
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
              disabled={isRunning || isLoading}
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

        {/* Strategy Parameters */}
        <div>
          <h3 className="text-md font-medium text-gray-900 dark:text-white mb-4">
            {strategies[selectedStrategy]?.name} Parameter
          </h3>
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-md">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {strategies[selectedStrategy]?.description}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(strategies[selectedStrategy]?.params || {}).map(([key, config]) =>
                renderParameterInput(key, config)
              )}
            </div>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="space-y-3">
          <div className="flex gap-2">
            {!isRunning ? (
              <button
                onClick={handleStartBot}
                disabled={isLoading || (validationErrors || []).length > 0}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
              >
                <Play className="w-4 h-4" />
                {isLoading ? 'Startet...' : 'Bot Starten'}
              </button>
            ) : (
              <>
                <button
                  onClick={handleRestartBot}
                  disabled={isLoading}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                  {isLoading ? 'Startet neu...' : 'Neu starten'}
                </button>
                <button
                  onClick={handleStopBot}
                  disabled={isLoading}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors"
                >
                  <Square className="w-4 h-4" />
                  {isLoading ? 'Stoppt...' : 'Bot Stoppen'}
                </button>
              </>
            )}
          </div>

          {/* Force Stop Button - always available for emergencies */}
          <div className="flex justify-center">
            <button
              onClick={handleForceStop}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors text-sm"
            >
              <Zap className="w-4 h-4" />
              {isLoading ? 'Force Stoppe...' : 'Force Cleanup'}
            </button>
          </div>
          
          {(validationErrors || []).length > 0 && (
            <p className="text-xs text-red-600 dark:text-red-400 text-center">
              Start deaktiviert: Konfigurationsfehler beheben
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyConfiguration;