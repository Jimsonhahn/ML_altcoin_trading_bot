/**
 * SidebarLogo Component
 * Professional logo/branding area for sidebar
 */

import React from 'react';

const SidebarLogo = ({ 
  isConnected = true, 
  collapsed = false, 
  onClose = null,
  variant = 'professional' // 'professional', 'minimal', 'animated'
}) => {
  
  const renderProfessionalLogo = () => (
    <div className="p-6 border-b border-surface-700 bg-gradient-to-r from-surface-800 to-surface-900 logo-area">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {/* Animated Logo with Status Indicator */}
          <div className="relative flex-shrink-0 group">
            <div className="logo-glow"></div>
            <div className="w-12 h-12 bg-gradient-to-br from-primary-500 via-primary-600 to-blue-600 
                            rounded-xl flex items-center justify-center shadow-lg shadow-primary-500/25
                            logo-icon relative z-10 logo-pulse">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} 
                      d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            {/* Enhanced Status Indicator with Pulse Animation */}
            <div className={`absolute -bottom-1 -right-1 w-3.5 h-3.5 rounded-full border-2 border-surface-800 ${
              isConnected 
                ? 'bg-success-500 status-pulse' 
                : 'bg-danger-500 animate-pulse'
            }`}>
              {/* Inner glow effect for connected status */}
              {isConnected && (
                <div className="absolute inset-0 bg-success-400 rounded-full animate-ping opacity-75"></div>
              )}
            </div>
          </div>
          
          {/* Brand Text with Enhanced Typography */}
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <h1 className="text-lg font-bold text-white tracking-tight hover:text-primary-100 transition-colors duration-300">
                AlgoTrader Pro
              </h1>
              <div className="flex items-center space-x-2 text-xs">
                <span className={`font-medium flex items-center space-x-1 ${
                  isConnected ? 'text-success-400' : 'text-danger-400'
                }`}>
                  <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                    isConnected ? 'bg-success-400 animate-pulse' : 'bg-danger-400'
                  }`}></span>
                  <span>{isConnected ? 'Live Trading' : 'Disconnected'}</span>
                </span>
                <span className="text-surface-500">•</span>
                <span className="text-surface-400 hover:text-surface-300 transition-colors duration-200">v2.0.1</span>
              </div>
            </div>
          )}
        </div>
        
        {/* Mobile Close Button */}
        {onClose && (
          <button
            onClick={onClose}
            className="lg:hidden p-2 rounded-lg text-surface-400 hover:text-surface-300 
                       hover:bg-surface-700 transition-all duration-200 hover:scale-105"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );

  const renderMinimalLogo = () => (
    <div className="h-16 border-b border-surface-700 flex items-center px-6 bg-surface-800 hover:bg-surface-700 transition-colors duration-300">
      <div className="flex items-center space-x-3">
        <div className="relative">
          <div className="text-2xl font-black bg-gradient-to-r from-primary-400 to-blue-400 
                          bg-clip-text text-transparent hover:from-primary-300 hover:to-blue-300 
                          transition-all duration-300 transform hover:scale-105">
            AT
          </div>
          {/* Minimal status indicator */}
          <div className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${
            isConnected ? 'bg-success-500' : 'bg-danger-500'
          }`}></div>
        </div>
        {!collapsed && (
          <>
            <div className="w-px h-8 bg-surface-700"></div>
            <div>
              <div className="text-sm font-semibold text-white hover:text-primary-100 transition-colors duration-200">
                AlgoTrader
              </div>
              <div className="text-xs text-surface-500 hover:text-surface-400 transition-colors duration-200">
                Professional
              </div>
            </div>
          </>
        )}
      </div>
      
      {onClose && (
        <button
          onClick={onClose}
          className="lg:hidden ml-auto p-2 rounded-lg text-surface-400 hover:text-surface-300 
                     hover:bg-surface-700 transition-all duration-200 hover:scale-105"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );

  const renderAnimatedLogo = () => (
    <div className="p-6 border-b border-surface-700 bg-surface-800">
      <div className="relative">
        {/* Enhanced Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-r from-primary-600/20 to-blue-600/20 
                        blur-xl animate-pulse opacity-50"></div>
        <div className="absolute inset-0 bg-gradient-to-l from-transparent via-white/5 to-transparent 
                        animate-shimmer opacity-30"></div>
        
        {/* Content */}
        <div className="relative flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="w-11 h-11 bg-surface-700 rounded-xl flex items-center justify-center 
                              ring-2 ring-primary-500/50 transform transition-all duration-300 hover:scale-105 
                              hover:ring-primary-400/70 hover:shadow-lg hover:shadow-primary-500/25">
                <span className="text-xl font-bold bg-gradient-to-br from-primary-400 to-blue-400 
                                bg-clip-text text-transparent animate-pulse">
                  TB
                </span>
              </div>
              {/* Enhanced status indicator */}
              <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border border-surface-800 ${
                isConnected 
                  ? 'bg-success-500 animate-pulse shadow-lg shadow-success-500/50' 
                  : 'bg-danger-500 animate-bounce'
              }`}></div>
            </div>
            {!collapsed && (
              <div>
                <h1 className="text-base font-semibold text-white hover:text-primary-100 transition-colors duration-300">
                  Trading Bot
                </h1>
                <div className="flex items-center space-x-2 text-xs">
                  <span className={`flex items-center space-x-1 ${
                    isConnected ? 'text-success-400' : 'text-danger-400'
                  }`}>
                    <span className={`inline-block w-1 h-1 rounded-full ${
                      isConnected ? 'bg-success-400 animate-ping' : 'bg-danger-400'
                    }`}></span>
                    <span>{isConnected ? 'Live' : 'Offline'}</span>
                  </span>
                  <span className="text-surface-500">•</span>
                  <span className="text-surface-400 hover:text-surface-300 transition-colors duration-200">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            )}
          </div>
          
          {onClose && (
            <button
              onClick={onClose}
              className="lg:hidden p-2 rounded-lg text-surface-400 hover:text-surface-300 
                         hover:bg-surface-700 transition-all duration-200 hover:scale-105 
                         hover:rotate-90"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );

  const renderCollapsedLogo = () => (
    <div className="w-16 flex justify-center py-6 border-b border-surface-700 bg-surface-800">
      <div className="relative">
        <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-blue-600 
                        rounded-xl flex items-center justify-center shadow-lg hover:scale-105 
                        transition-all duration-300 hover:shadow-xl hover:shadow-primary-500/25 
                        hover:rotate-2">
          <span className="text-lg font-bold text-white">A</span>
        </div>
        {/* Collapsed status indicator */}
        <div className={`absolute -bottom-1 -right-1 w-2.5 h-2.5 rounded-full border border-surface-800 ${
          isConnected 
            ? 'bg-success-500 animate-pulse' 
            : 'bg-danger-500'
        }`}></div>
      </div>
    </div>
  );

  if (collapsed) {
    return renderCollapsedLogo();
  }

  switch (variant) {
    case 'minimal':
      return renderMinimalLogo();
    case 'animated':
      return renderAnimatedLogo();
    default:
      return renderProfessionalLogo();
  }
};

export default SidebarLogo;