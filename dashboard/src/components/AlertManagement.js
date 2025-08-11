import React, { useState, useEffect } from 'react';
import { Bell, BellOff, AlertTriangle, CheckCircle, XCircle, Info, Settings, Plus, Trash2, Edit3, Volume2, VolumeX } from 'lucide-react';
import apiService from '../services/api';
import { useTheme } from '../hooks/useTheme';

const AlertManagement = ({ className = '' }) => {
  const { isDark } = useTheme();
  const [alerts, setAlerts] = useState([]);
  const [alertSettings, setAlertSettings] = useState({
    enabled: true,
    email_notifications: true,
    push_notifications: true,
    telegram_notifications: false,
    sound_enabled: true,
    min_priority: 'medium'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCreateAlert, setShowCreateAlert] = useState(false);
  const [editingAlert, setEditingAlert] = useState(null);
  const [newAlert, setNewAlert] = useState({
    name: '',
    type: 'price',
    condition: 'above',
    value: '',
    symbol: '',
    priority: 'medium',
    enabled: true
  });

  // Filter states
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');

  useEffect(() => {
    loadAlerts();
    loadAlertSettings();
  }, []);

  const loadAlerts = async () => {
    try {
      const response = await apiService.getAlerts();
      setAlerts(response.alerts || []);
    } catch (err) {
      setError(err.message || 'Fehler beim Laden der Alerts');
    }
  };

  const loadAlertSettings = async () => {
    try {
      const response = await apiService.getAlertSettings();
      setAlertSettings(response.settings || alertSettings);
    } catch (err) {
      console.error('Error loading alert settings:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const saveAlertSettings = async () => {
    try {
      await apiService.updateAlertSettings(alertSettings);
    } catch (err) {
      setError(err.message || 'Fehler beim Speichern der Einstellungen');
    }
  };

  const createAlert = async () => {
    try {
      await apiService.createAlert(newAlert);
      setShowCreateAlert(false);
      setNewAlert({
        name: '',
        type: 'price',
        condition: 'above',
        value: '',
        symbol: '',
        priority: 'medium',
        enabled: true
      });
      await loadAlerts();
    } catch (err) {
      setError(err.message || 'Fehler beim Erstellen des Alerts');
    }
  };

  const updateAlert = async (alertId, updates) => {
    try {
      await apiService.updateAlert(alertId, updates);
      await loadAlerts();
    } catch (err) {
      setError(err.message || 'Fehler beim Aktualisieren des Alerts');
    }
  };

  const deleteAlert = async (alertId) => {
    if (window.confirm('Möchten Sie diesen Alert wirklich löschen?')) {
      try {
        await apiService.deleteAlert(alertId);
        await loadAlerts();
      } catch (err) {
        setError(err.message || 'Fehler beim Löschen des Alerts');
      }
    }
  };

  const acknowledgeAlert = async (alertId) => {
    try {
      await apiService.acknowledgeAlert(alertId);
      await loadAlerts();
    } catch (err) {
      setError(err.message || 'Fehler beim Bestätigen des Alerts');
    }
  };

  const dismissAlert = async (alertId) => {
    try {
      await apiService.dismissAlert(alertId);
      await loadAlerts();
    } catch (err) {
      setError(err.message || 'Fehler beim Verwerfen des Alerts');
    }
  };

  const testAlert = async (alertId) => {
    try {
      await apiService.testAlert(alertId);
    } catch (err) {
      setError(err.message || 'Fehler beim Testen des Alerts');
    }
  };

  const getAlertIcon = (type) => {
    switch (type) {
      case 'price':
        return <AlertTriangle className="w-4 h-4" />;
      case 'pnl':
        return <Info className="w-4 h-4" />;
      case 'position':
        return <Bell className="w-4 h-4" />;
      case 'system':
        return <Settings className="w-4 h-4" />;
      default:
        return <Bell className="w-4 h-4" />;
    }
  };

  const getAlertColor = (priority, status) => {
    if (status === 'acknowledged') return 'text-green-600 dark:text-green-400';
    if (status === 'dismissed') return 'text-gray-600 dark:text-gray-400';
    
    switch (priority) {
      case 'high':
        return 'text-red-600 dark:text-red-400';
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'low':
        return 'text-blue-600 dark:text-blue-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getAlertBgColor = (priority, status) => {
    if (status === 'acknowledged') return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
    if (status === 'dismissed') return 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    
    switch (priority) {
      case 'high':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'medium':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'low':
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
      default:
        return 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  const formatDateTime = (timestamp) => {
    return new Date(timestamp).toLocaleString('de-DE', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filterType !== 'all' && alert.type !== filterType) return false;
    if (filterStatus !== 'all' && alert.status !== filterStatus) return false;
    if (filterPriority !== 'all' && alert.priority !== filterPriority) return false;
    return true;
  });

  const alertTypes = [
    { value: 'price', label: 'Preis Alert' },
    { value: 'pnl', label: 'P&L Alert' },
    { value: 'position', label: 'Position Alert' },
    { value: 'system', label: 'System Alert' }
  ];

  const conditionOptions = {
    price: [
      { value: 'above', label: 'Über' },
      { value: 'below', label: 'Unter' },
      { value: 'change', label: 'Änderung um' }
    ],
    pnl: [
      { value: 'profit', label: 'Gewinn über' },
      { value: 'loss', label: 'Verlust über' }
    ],
    position: [
      { value: 'opened', label: 'Position eröffnet' },
      { value: 'closed', label: 'Position geschlossen' },
      { value: 'size', label: 'Positionsgröße über' }
    ],
    system: [
      { value: 'error', label: 'Fehler aufgetreten' },
      { value: 'connected', label: 'Verbindung wiederhergestellt' },
      { value: 'disconnected', label: 'Verbindung unterbrochen' }
    ]
  };

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
          Alert Management
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAlertSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
            className={`p-2 rounded-lg transition-colors ${
              alertSettings.enabled 
                ? 'text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20' 
                : 'text-gray-400 dark:text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-900/20'
            }`}
            title={alertSettings.enabled ? 'Alerts deaktivieren' : 'Alerts aktivieren'}
          >
            {alertSettings.enabled ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setShowCreateAlert(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Neuer Alert
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
        </div>
      )}

      {/* Alert Settings */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <h3 className="text-base font-medium text-gray-900 dark:text-white mb-3">
          Benachrichtigungseinstellungen
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={alertSettings.email_notifications}
              onChange={(e) => setAlertSettings(prev => ({ ...prev, email_notifications: e.target.checked }))}
              className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Email-Benachrichtigungen</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={alertSettings.push_notifications}
              onChange={(e) => setAlertSettings(prev => ({ ...prev, push_notifications: e.target.checked }))}
              className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Push-Benachrichtigungen</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={alertSettings.telegram_notifications}
              onChange={(e) => setAlertSettings(prev => ({ ...prev, telegram_notifications: e.target.checked }))}
              className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">Telegram-Benachrichtigungen</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={alertSettings.sound_enabled}
              onChange={(e) => setAlertSettings(prev => ({ ...prev, sound_enabled: e.target.checked }))}
              className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300 flex items-center gap-1">
              {alertSettings.sound_enabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              Sound aktiviert
            </span>
          </label>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Minimale Priorität
            </label>
            <select
              value={alertSettings.min_priority}
              onChange={(e) => setAlertSettings(prev => ({ ...prev, min_priority: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="low">Niedrig</option>
              <option value="medium">Mittel</option>
              <option value="high">Hoch</option>
            </select>
          </div>
        </div>
        <div className="mt-4">
          <button
            onClick={saveAlertSettings}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm"
          >
            Einstellungen speichern
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6 flex flex-wrap gap-4">
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        >
          <option value="all">Alle Typen</option>
          {alertTypes.map(type => (
            <option key={type.value} value={type.value}>{type.label}</option>
          ))}
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        >
          <option value="all">Alle Status</option>
          <option value="active">Aktiv</option>
          <option value="triggered">Ausgelöst</option>
          <option value="acknowledged">Bestätigt</option>
          <option value="dismissed">Verworfen</option>
        </select>
        <select
          value={filterPriority}
          onChange={(e) => setFilterPriority(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
        >
          <option value="all">Alle Prioritäten</option>
          <option value="high">Hoch</option>
          <option value="medium">Mittel</option>
          <option value="low">Niedrig</option>
        </select>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {filteredAlerts.map((alert) => (
          <div
            key={alert.id}
            className={`p-4 rounded-lg border ${getAlertBgColor(alert.priority, alert.status)}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3">
                <div className={`mt-1 ${getAlertColor(alert.priority, alert.status)}`}>
                  {getAlertIcon(alert.type)}
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="font-medium text-gray-900 dark:text-white">
                      {alert.name}
                    </h3>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.priority === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400' :
                      alert.priority === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
                      'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                    }`}>
                      {alert.priority.toUpperCase()}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.status === 'active' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
                      alert.status === 'triggered' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400' :
                      alert.status === 'acknowledged' ? 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400' :
                      'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
                    }`}>
                      {alert.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {alert.description}
                  </p>
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                    <span>Typ: {alertTypes.find(t => t.value === alert.type)?.label}</span>
                    <span>Symbol: {alert.symbol}</span>
                    <span>Erstellt: {formatDateTime(alert.created_at)}</span>
                    {alert.triggered_at && (
                      <span>Ausgelöst: {formatDateTime(alert.triggered_at)}</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {alert.status === 'triggered' && (
                  <>
                    <button
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="p-2 text-green-600 hover:text-green-800 dark:text-green-400 dark:hover:text-green-300 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/20 transition-colors"
                      title="Bestätigen"
                    >
                      <CheckCircle className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => dismissAlert(alert.id)}
                      className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-900/20 transition-colors"
                      title="Verwerfen"
                    >
                      <XCircle className="w-4 h-4" />
                    </button>
                  </>
                )}
                <button
                  onClick={() => testAlert(alert.id)}
                  className="p-2 text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                  title="Testen"
                >
                  <Bell className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setEditingAlert(alert)}
                  className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-900/20 transition-colors"
                  title="Bearbeiten"
                >
                  <Edit3 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => deleteAlert(alert.id)}
                  className="p-2 text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                  title="Löschen"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredAlerts.length === 0 && (
        <div className="text-center py-8">
          <Bell className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">
            Keine Alerts gefunden
          </p>
        </div>
      )}

      {/* Create Alert Modal */}
      {showCreateAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Neuer Alert
              </h3>
              <button
                onClick={() => setShowCreateAlert(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  value={newAlert.name}
                  onChange={(e) => setNewAlert(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Alert Name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Typ
                </label>
                <select
                  value={newAlert.type}
                  onChange={(e) => setNewAlert(prev => ({ ...prev, type: e.target.value, condition: conditionOptions[e.target.value][0].value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {alertTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Bedingung
                </label>
                <select
                  value={newAlert.condition}
                  onChange={(e) => setNewAlert(prev => ({ ...prev, condition: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {conditionOptions[newAlert.type]?.map(option => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </div>
              {newAlert.type === 'price' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Symbol
                  </label>
                  <input
                    type="text"
                    value={newAlert.symbol}
                    onChange={(e) => setNewAlert(prev => ({ ...prev, symbol: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="BTC/USDT"
                  />
                </div>
              )}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Wert
                </label>
                <input
                  type="number"
                  value={newAlert.value}
                  onChange={(e) => setNewAlert(prev => ({ ...prev, value: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="0.00"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Priorität
                </label>
                <select
                  value={newAlert.priority}
                  onChange={(e) => setNewAlert(prev => ({ ...prev, priority: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="low">Niedrig</option>
                  <option value="medium">Mittel</option>
                  <option value="high">Hoch</option>
                </select>
              </div>
              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setShowCreateAlert(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 font-medium transition-colors"
                >
                  Abbrechen
                </button>
                <button
                  onClick={createAlert}
                  disabled={!newAlert.name || !newAlert.value}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-md font-medium transition-colors disabled:opacity-50"
                >
                  Erstellen
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertManagement;