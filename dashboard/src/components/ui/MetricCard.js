/**
 * MetricCard Component
 * Modern metric display card with animations and gradients
 */

import React from 'react';

const MetricCard = ({ 
  title, 
  value, 
  change, 
  changeType = 'neutral', 
  icon, 
  loading = false, 
  className = '',
  trend = null,
  subtitle = null
}) => {
  const getChangeColor = (type) => {
    switch (type) {
      case 'positive':
        return 'text-success-500';
      case 'negative':
        return 'text-danger-500';
      case 'warning':
        return 'text-warning-500';
      default:
        return 'text-gray-500 dark:text-gray-400';
    }
  };

  const getTrendIcon = (type) => {
    switch (type) {
      case 'up':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11l5-5m0 0l5 5m-5-5v12" />
          </svg>
        );
      case 'down':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 13l-5 5m0 0l-5-5m5 5V6" />
          </svg>
        );
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className={`metric-card ${className}`}>
        <div className="animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-3"></div>
              <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/4"></div>
            </div>
            <div className="w-12 h-12 bg-gray-300 dark:bg-gray-600 rounded-full"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`metric-card metric-card-hover group ${className}`}>
      {/* Background decoration */}
      <div className="absolute -right-8 -bottom-8 w-24 h-24 rounded-full bg-gradient-to-br from-primary-500/10 to-primary-600/10 group-hover:from-primary-500/20 group-hover:to-primary-600/20 transition-all duration-300"></div>
      
      {/* Content */}
      <div className="relative z-10 flex items-center justify-between">
        <div className="flex-1">
          {/* Title */}
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
            {title}
          </p>
          
          {/* Value */}
          <p className="text-2xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors duration-300">
            {value}
          </p>
          
          {/* Change indicator */}
          {change && (
            <div className={`flex items-center space-x-1 text-sm ${getChangeColor(changeType)}`}>
              {getTrendIcon(trend)}
              <span className="font-medium">{change}</span>
            </div>
          )}
          
          {/* Subtitle */}
          {subtitle && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {subtitle}
            </p>
          )}
        </div>
        
        {/* Icon */}
        {icon && (
          <div className="flex-shrink-0 p-3 rounded-full bg-gradient-to-br from-primary-500/10 to-primary-600/10 group-hover:from-primary-500/20 group-hover:to-primary-600/20 transition-all duration-300">
            <div className="w-6 h-6 text-primary-600 dark:text-primary-400 group-hover:scale-110 transition-transform duration-300">
              {icon}
            </div>
          </div>
        )}
      </div>
      
      {/* Shimmer effect on hover */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 animate-shimmer"></div>
      </div>
    </div>
  );
};

export default MetricCard;