/**
 * Main Dashboard Component
 * Combines all dashboard components with responsive layout
 */

import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { useTheme } from '../hooks/useTheme';
import PositionsList from './PositionsList';
import PnLChart from './PnLChart';
import StrategyConfiguration from './StrategyConfiguration';
import SystemHealth from './SystemHealth';
import ThemeToggle from './ThemeToggle';
import ConnectionStatus from './ConnectionStatus';
import DashboardMetrics from './DashboardMetrics';
import SuperLazyBillionairePanel from './SuperLazyBillionairePanel';
import DebugPanel from './DebugPanel';
import OrchestratorDashboard from './OrchestratorDashboard';
import IntelligenceDashboard from './IntelligenceDashboard';
import { MainContentRow, BottomRow } from './layout/DashboardGrid';
import SidebarLogo from './ui/SidebarLogo';
import robustAPI from '../services/robustApi';

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const { isConnected, connect, disconnect } = useWebSocket();

  useEffect(() => {
    loadUserProfile();
    
    // Connect to WebSocket
    if (connect) {
      connect();
    }

    return () => {
      if (disconnect) {
        disconnect();
      }
    };
  }, [connect, disconnect]);

  const loadUserProfile = async () => {
    try {
      console.log('🔄 Loading user profile...');
      
      // Mock user data for demo purposes
      setUser({
        username: 'admin',
        roles: ['admin', 'trader'],
        authenticated: true
      });

      console.log('✅ User profile loaded');
    } catch (err) {
      console.error('❌ Failed to load user profile:', err);
      setError('Failed to load user profile');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      console.log('🔄 Logging out...');
      setUser(null);
      // For demo purposes, just clear the user state
      console.log('✅ Logged out successfully');
    } catch (err) {
      console.error('❌ Logout error:', err);
    }
  };

  // Removed unused handleStrategyChange function

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner w-8 h-8 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-danger-600 dark:text-danger-400 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-900">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className="absolute inset-0 bg-gray-600 opacity-75"></div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-surface-800 transform ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 shadow-xl flex flex-col h-full`}>
        {/* Professional Logo/Branding Area */}
        <SidebarLogo 
          isConnected={isConnected} 
          onClose={() => setSidebarOpen(false)}
          variant="professional"
        />

        {/* Navigation - Scrollable Middle Section */}
        <nav className="flex-1 overflow-y-auto px-4 py-6">
          <div className="space-y-2">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`${activeTab === 'dashboard' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-primary-500 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
              </svg>
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('orchestrator')}
              className={`${activeTab === 'orchestrator' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              Orchestrator
            </button>
            <button
              onClick={() => setActiveTab('trading')}
              className={`${activeTab === 'trading' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Trading
            </button>
            <button
              onClick={() => setActiveTab('strategies')}
              className={`${activeTab === 'strategies' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Strategies
            </button>
            <button
              onClick={() => setActiveTab('intelligence')}
              className={`${activeTab === 'intelligence' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Intelligence
            </button>
            <button
              onClick={() => setActiveTab('monitoring')}
              className={`${activeTab === 'monitoring' 
                ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' 
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white'
              } group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left`}
            >
              <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Monitoring
            </button>
          </div>
        </nav>

        {/* Divider */}
        <div className="mx-4 border-t border-gray-200 dark:border-surface-700"></div>

        {/* User Section - Fixed Bottom */}
        <div className="p-4 space-y-2">
          {/* User Info */}
          <div className="flex items-center px-3 py-2.5 rounded-lg hover:bg-gray-50 dark:hover:bg-surface-700 transition-all duration-200">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user?.username?.charAt(0).toUpperCase() || 'A'}
                </span>
              </div>
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {user?.username || 'admin'}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {user?.roles?.join(', ') || 'Administrator'}
              </p>
            </div>
            <button
              onClick={handleLogout}
              className="flex-shrink-0 p-1.5 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-surface-600 transition-all duration-200"
              title="Logout"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
          
          {/* Settings Link */}
          <button
            className="text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-surface-700 hover:text-gray-900 dark:hover:text-white group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 w-full text-left"
          >
            <svg className="text-gray-400 mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Settings
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top navigation */}
        <div className="sticky top-0 z-10 card-glass shadow-lg border-b border-gray-200 dark:border-surface-700">
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-surface-700 transition-colors duration-200"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                </button>
                <h1 className="ml-4 lg:ml-0 text-xl font-bold text-gradient">
                  Trading Dashboard
                </h1>
              </div>
              
              <div className="flex items-center space-x-4">
                <ConnectionStatus isConnected={isConnected} />
                <button
                  onClick={() => setShowDebugPanel(true)}
                  className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  title="Open Debug Panel"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                  </svg>
                </button>
                <ThemeToggle />
              </div>
            </div>
          </div>
        </div>

        {/* Dashboard content */}
        <main className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-7xl mx-auto space-y-8">
            {activeTab === 'dashboard' && (
              <>
                {/* Metrics Row */}
                <DashboardMetrics />

                {/* SuperLazyBillionaire Strategy Panel */}
                <SuperLazyBillionairePanel />

                {/* Main Content Row */}
                <MainContentRow>
                  {/* Strategy Configuration */}
                  <div className="lg:col-span-6">
                    <StrategyConfiguration />
                  </div>

                  {/* P&L Chart */}
                  <div className="lg:col-span-6">
                    <PnLChart />
                  </div>
                </MainContentRow>

                {/* Bottom Row */}
                <BottomRow>
                  {/* System Health */}
                  <div className="xl:col-span-4">
                    <SystemHealth />
                  </div>

                  {/* Positions List */}
                  <div className="xl:col-span-8">
                    <PositionsList />
                  </div>
                </BottomRow>
              </>
            )}

            {activeTab === 'orchestrator' && (
              <OrchestratorDashboard />
            )}

            {activeTab === 'intelligence' && (
              <IntelligenceDashboard />
            )}

            {activeTab === 'trading' && (
              <div className="text-center py-12">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">Trading Panel</h2>
                <p className="text-gray-600 dark:text-gray-400 mt-2">Trading features coming soon...</p>
              </div>
            )}

            {activeTab === 'strategies' && (
              <div className="text-center py-12">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">Strategy Management</h2>
                <p className="text-gray-600 dark:text-gray-400 mt-2">Strategy management features coming soon...</p>
              </div>
            )}

            {activeTab === 'monitoring' && (
              <div className="text-center py-12">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">Advanced Monitoring</h2>
                <p className="text-gray-600 dark:text-gray-400 mt-2">Advanced monitoring features coming soon...</p>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Debug Panel Modal */}
      <DebugPanel 
        isVisible={showDebugPanel} 
        onClose={() => setShowDebugPanel(false)} 
      />
    </div>
  );
};

export default Dashboard;