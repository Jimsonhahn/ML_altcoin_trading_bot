import React, { useState } from 'react';
import { BookOpen, Star, Download, Upload, Trash2, Edit, Plus } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

const StrategyPresets = ({ className = '', onLoadPreset }) => {
  const { isDark } = useTheme();
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');

  // Predefined strategy presets
  const presets = [
    {
      id: 'conservative',
      name: 'Conservative Trading',
      description: 'Niedrig-Risiko Momentum-Strategie für stabile Gewinne',
      strategy: 'momentum',
      favorite: true,
      config: {
        mode: 'paper',
        symbol: 'BTC/USDT',
        capital: 10000,
        risk_per_trade: 0.01,
        strategy_params: {
          rsi_period: 21,
          rsi_overbought: 75,
          rsi_oversold: 25,
          sma_short: 8,
          sma_long: 21,
          volume_spike_threshold: 1.2
        }
      },
      performance: {
        backtest_return: '+8.5%',
        win_rate: '72%',
        max_drawdown: '3.2%'
      }
    },
    {
      id: 'aggressive',
      name: 'Aggressive Growth',
      description: 'Hoch-Risiko Strategie für maximale Gewinne',
      strategy: 'momentum',
      favorite: false,
      config: {
        mode: 'paper',
        symbol: 'ETH/USDT',
        capital: 10000,
        risk_per_trade: 0.05,
        strategy_params: {
          rsi_period: 14,
          rsi_overbought: 65,
          rsi_oversold: 35,
          sma_short: 3,
          sma_long: 12,
          volume_spike_threshold: 2.0
        }
      },
      performance: {
        backtest_return: '+24.2%',
        win_rate: '58%',
        max_drawdown: '12.8%'
      }
    },
    {
      id: 'scalping',
      name: 'Scalping Master',
      description: 'Kurzfristige Trades für schnelle Gewinne',
      strategy: 'mean_reversion',
      favorite: true,
      config: {
        mode: 'paper',
        symbol: 'ADA/USDT',
        capital: 5000,
        risk_per_trade: 0.02,
        strategy_params: {
          bollinger_period: 15,
          bollinger_std: 1.8,
          rsi_period: 10,
          use_rsi_filter: true
        }
      },
      performance: {
        backtest_return: '+15.7%',
        win_rate: '67%',
        max_drawdown: '5.1%'
      }
    },
    {
      id: 'grid_stable',
      name: 'Stable Grid',
      description: 'Grid Trading für stabile, seitwärts bewegende Märkte',
      strategy: 'grid_trading',
      favorite: false,
      config: {
        mode: 'paper',
        symbol: 'BNB/USDT',
        capital: 15000,
        risk_per_trade: 0.015,
        strategy_params: {
          num_grids: 15,
          price_range_multiplier: 0.03,
          grid_size_percent: 0.008
        }
      },
      performance: {
        backtest_return: '+11.3%',
        win_rate: '78%',
        max_drawdown: '4.5%'
      }
    },
    {
      id: 'ml_balanced',
      name: 'AI Balanced',
      description: 'Machine Learning Strategie mit ausgewogenem Risiko',
      strategy: 'ml_strategy',
      favorite: true,
      config: {
        mode: 'paper',
        symbol: 'SOL/USDT',
        capital: 12000,
        risk_per_trade: 0.025,
        strategy_params: {
          model_confidence_threshold: 0.75,
          lookback_window: 25,
          feature_importance_min: 0.15
        }
      },
      performance: {
        backtest_return: '+19.4%',
        win_rate: '71%',
        max_drawdown: '7.2%'
      }
    }
  ];

  const handleLoadPreset = (preset) => {
    if (onLoadPreset) {
      onLoadPreset(preset.config);
    }
    setSelectedPreset(preset.id);
  };

  const toggleFavorite = (presetId) => {
    // In a real app, this would update the backend
    console.log('Toggle favorite for preset:', presetId);
  };

  const deletePreset = (presetId) => {
    // In a real app, this would delete from backend
    console.log('Delete preset:', presetId);
  };

  const handleCreatePreset = () => {
    if (newPresetName.trim()) {
      // In a real app, this would save to backend
      console.log('Create preset:', { name: newPresetName, description: newPresetDescription });
      setShowCreateModal(false);
      setNewPresetName('');
      setNewPresetDescription('');
    }
  };

  const handleExportPresets = () => {
    const dataStr = JSON.stringify(presets, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'strategy_presets.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleImportPresets = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedPresets = JSON.parse(e.target.result);
          console.log('Imported presets:', importedPresets);
          // In a real app, validate and merge with existing presets
        } catch (error) {
          console.error('Error importing presets:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5" />
          Strategie Presets
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
          >
            <Plus className="w-4 h-4" />
            Neu
          </button>
          <div className="flex items-center gap-1">
            <button
              onClick={handleExportPresets}
              className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Presets exportieren"
            >
              <Download className="w-4 h-4" />
            </button>
            <label className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors cursor-pointer" title="Presets importieren">
              <Upload className="w-4 h-4" />
              <input
                type="file"
                accept=".json"
                onChange={handleImportPresets}
                className="hidden"
              />
            </label>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {presets.map((preset) => (
          <div
            key={preset.id}
            className={`border rounded-lg p-4 transition-all duration-200 cursor-pointer ${
              selectedPreset === preset.id
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
            }`}
            onClick={() => handleLoadPreset(preset)}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <h3 className="font-medium text-gray-900 dark:text-white">
                  {preset.name}
                </h3>
                {preset.favorite && (
                  <Star className="w-4 h-4 text-yellow-500 fill-current" />
                )}
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleFavorite(preset.id);
                  }}
                  className="p-1 text-gray-400 hover:text-yellow-500 transition-colors"
                >
                  <Star className={`w-4 h-4 ${preset.favorite ? 'fill-current text-yellow-500' : ''}`} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deletePreset(preset.id);
                  }}
                  className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              {preset.description}
            </p>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500 dark:text-gray-400">Strategie:</span>
                <span className="font-medium text-gray-900 dark:text-white capitalize">
                  {preset.strategy.replace('_', ' ')}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500 dark:text-gray-400">Symbol:</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {preset.config.symbol}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500 dark:text-gray-400">Risiko:</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {(preset.config.risk_per_trade * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {preset.performance && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center">
                    <div className="text-gray-500 dark:text-gray-400">Return</div>
                    <div className={`font-medium ${
                      preset.performance.backtest_return.startsWith('+') 
                        ? 'text-green-600 dark:text-green-400' 
                        : 'text-red-600 dark:text-red-400'
                    }`}>
                      {preset.performance.backtest_return}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-gray-500 dark:text-gray-400">Win Rate</div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {preset.performance.win_rate}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-gray-500 dark:text-gray-400">Max DD</div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {preset.performance.max_drawdown}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Create Preset Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Neues Preset erstellen
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Preset Name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Beschreibung
                </label>
                <textarea
                  value={newPresetDescription}
                  onChange={(e) => setNewPresetDescription(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="3"
                  placeholder="Beschreibung..."
                />
              </div>
            </div>
            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                Abbrechen
              </button>
              <button
                onClick={handleCreatePreset}
                disabled={!newPresetName.trim()}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-md transition-colors"
              >
                Erstellen
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyPresets;