import React, { useState, useEffect } from 'react';
import { Settings, AlertTriangle, Shield, TrendingUp, TrendingDown, DollarSign, Percent } from 'lucide-react';
import apiService from '../services/api';
import { useTheme } from '../hooks/useTheme';

const RiskParameters = ({ className = '' }) => {
  const { isDark } = useTheme();
  const [riskSettings, setRiskSettings] = useState({
    max_position_size: 0.1,
    max_daily_loss: 0.05,
    max_drawdown: 0.15,
    stop_loss_percentage: 0.02,
    take_profit_percentage: 0.04,
    max_open_positions: 5,
    leverage_limit: 1.0,
    min_profit_target: 0.01,
    risk_per_trade: 0.02,
    correlation_limit: 0.7
  });
  
  const [originalSettings, setOriginalSettings] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    loadRiskSettings();
  }, []);

  useEffect(() => {
    const changed = JSON.stringify(riskSettings) !== JSON.stringify(originalSettings);
    setHasChanges(changed);
  }, [riskSettings, originalSettings]);

  const loadRiskSettings = async () => {
    setIsLoading(true);
    setError('');

    try {
      const response = await apiService.getRiskSettings();
      setRiskSettings(response.settings || riskSettings);
      setOriginalSettings(response.settings || riskSettings);
    } catch (err) {
      setError(err.message || 'Fehler beim Laden der Risk-Parameter');
    } finally {
      setIsLoading(false);
    }
  };

  const saveRiskSettings = async () => {
    setIsSaving(true);
    setError('');
    setSuccess('');

    try {
      await apiService.updateRiskSettings(riskSettings);
      setOriginalSettings({ ...riskSettings });
      setSuccess('Risk-Parameter erfolgreich gespeichert');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message || 'Fehler beim Speichern der Risk-Parameter');
    } finally {
      setIsSaving(false);
    }
  };

  const resetToDefaults = () => {
    const defaultSettings = {
      max_position_size: 0.1,
      max_daily_loss: 0.05,
      max_drawdown: 0.15,
      stop_loss_percentage: 0.02,
      take_profit_percentage: 0.04,
      max_open_positions: 5,
      leverage_limit: 1.0,
      min_profit_target: 0.01,
      risk_per_trade: 0.02,
      correlation_limit: 0.7
    };
    setRiskSettings(defaultSettings);
  };

  const resetToOriginal = () => {
    setRiskSettings({ ...originalSettings });
  };

  const handleInputChange = (key, value) => {
    setRiskSettings(prev => ({
      ...prev,
      [key]: parseFloat(value) || 0
    }));
  };

  const formatPercentage = (value) => {
    return (value * 100).toFixed(1);
  };

  const getRiskLevel = (key, value) => {
    const riskThresholds = {
      max_position_size: { low: 0.05, high: 0.2 },
      max_daily_loss: { low: 0.02, high: 0.1 },
      max_drawdown: { low: 0.1, high: 0.25 },
      stop_loss_percentage: { low: 0.01, high: 0.05 },
      risk_per_trade: { low: 0.01, high: 0.05 },
      leverage_limit: { low: 1.0, high: 3.0 }
    };

    const threshold = riskThresholds[key];
    if (!threshold) return 'medium';

    if (value <= threshold.low) return 'low';
    if (value >= threshold.high) return 'high';
    return 'medium';
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'low':
        return 'text-green-600 dark:text-green-400';
      case 'high':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-yellow-600 dark:text-yellow-400';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'low':
        return <Shield className="w-4 h-4" />;
      case 'high':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <TrendingUp className="w-4 h-4" />;
    }
  };

  const riskParameterGroups = [
    {
      title: 'Position Management',
      icon: <DollarSign className="w-5 h-5" />,
      parameters: [
        {
          key: 'max_position_size',
          label: 'Max Position Size',
          description: 'Maximale Positionsgröße als Anteil des Portfolios',
          unit: '%',
          min: 0.01,
          max: 1.0,
          step: 0.01
        },
        {
          key: 'max_open_positions',
          label: 'Max Open Positions',
          description: 'Maximale Anzahl gleichzeitig offener Positionen',
          unit: '',
          min: 1,
          max: 20,
          step: 1,
          isInteger: true
        },
        {
          key: 'leverage_limit',
          label: 'Leverage Limit',
          description: 'Maximaler Hebel für Positionen',
          unit: 'x',
          min: 1.0,
          max: 10.0,
          step: 0.1
        }
      ]
    },
    {
      title: 'Loss Management',
      icon: <TrendingDown className="w-5 h-5" />,
      parameters: [
        {
          key: 'max_daily_loss',
          label: 'Max Daily Loss',
          description: 'Maximaler täglicher Verlust als Anteil des Portfolios',
          unit: '%',
          min: 0.01,
          max: 0.5,
          step: 0.01
        },
        {
          key: 'max_drawdown',
          label: 'Max Drawdown',
          description: 'Maximaler Drawdown bevor Trading gestoppt wird',
          unit: '%',
          min: 0.05,
          max: 0.5,
          step: 0.01
        },
        {
          key: 'stop_loss_percentage',
          label: 'Stop Loss',
          description: 'Standard Stop-Loss Prozentsatz',
          unit: '%',
          min: 0.005,
          max: 0.1,
          step: 0.005
        }
      ]
    },
    {
      title: 'Profit Management',
      icon: <TrendingUp className="w-5 h-5" />,
      parameters: [
        {
          key: 'take_profit_percentage',
          label: 'Take Profit',
          description: 'Standard Take-Profit Prozentsatz',
          unit: '%',
          min: 0.01,
          max: 0.2,
          step: 0.01
        },
        {
          key: 'min_profit_target',
          label: 'Min Profit Target',
          description: 'Minimales Profit-Ziel für Trade-Signale',
          unit: '%',
          min: 0.005,
          max: 0.1,
          step: 0.005
        },
        {
          key: 'risk_per_trade',
          label: 'Risk per Trade',
          description: 'Risiko pro Trade als Anteil des Portfolios',
          unit: '%',
          min: 0.005,
          max: 0.1,
          step: 0.005
        }
      ]
    },
    {
      title: 'Advanced Settings',
      icon: <Settings className="w-5 h-5" />,
      parameters: [
        {
          key: 'correlation_limit',
          label: 'Correlation Limit',
          description: 'Maximale Korrelation zwischen Positionen',
          unit: '',
          min: 0.1,
          max: 1.0,
          step: 0.1
        }
      ]
    }
  ];

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Risk Parameter
        </h2>
        <div className="flex items-center gap-2">
          {hasChanges && (
            <span className="text-sm text-orange-600 dark:text-orange-400">
              Nicht gespeicherte Änderungen
            </span>
          )}
          <Settings className="w-5 h-5 text-gray-500 dark:text-gray-400" />
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
            <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
            <span className="text-sm text-green-600 dark:text-green-400">{success}</span>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {riskParameterGroups.map((group, groupIndex) => (
          <div key={groupIndex} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-4">
              <div className="text-blue-600 dark:text-blue-400">
                {group.icon}
              </div>
              <h3 className="text-base font-medium text-gray-900 dark:text-white">
                {group.title}
              </h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {group.parameters.map((param) => {
                const riskLevel = getRiskLevel(param.key, riskSettings[param.key]);
                const value = riskSettings[param.key];
                
                return (
                  <div key={param.key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {param.label}
                      </label>
                      <div className={`flex items-center gap-1 text-xs ${getRiskColor(riskLevel)}`}>
                        {getRiskIcon(riskLevel)}
                        <span>{riskLevel.toUpperCase()}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        value={value}
                        onChange={(e) => handleInputChange(param.key, e.target.value)}
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                      />
                      <span className="text-sm text-gray-500 dark:text-gray-400 min-w-[30px]">
                        {param.unit === '%' ? `${formatPercentage(value)}%` : `${value}${param.unit}`}
                      </span>
                    </div>
                    
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {param.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={resetToDefaults}
          disabled={isSaving}
          className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors disabled:opacity-50"
        >
          Defaults
        </button>
        
        <button
          onClick={resetToOriginal}
          disabled={isSaving || !hasChanges}
          className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors disabled:opacity-50"
        >
          Zurücksetzen
        </button>
        
        <button
          onClick={saveRiskSettings}
          disabled={isSaving || !hasChanges}
          className="flex-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 rounded-md transition-colors disabled:opacity-50"
        >
          {isSaving ? 'Speichert...' : 'Speichern'}
        </button>
      </div>
    </div>
  );
};

export default RiskParameters;