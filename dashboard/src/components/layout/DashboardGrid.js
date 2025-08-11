/**
 * DashboardGrid Component
 * Modern responsive grid layout for dashboard components
 */

import React from 'react';

const DashboardGrid = ({ children, className = '' }) => {
  return (
    <div className={`dashboard-grid ${className}`}>
      {children}
    </div>
  );
};

// Grid item wrapper components
const GridItem = ({ children, size = 'medium', className = '' }) => {
  const sizeClasses = {
    small: 'grid-item-small',
    medium: 'grid-item-medium', 
    large: 'grid-item-large',
    full: 'grid-item-full'
  };
  
  return (
    <div className={`${sizeClasses[size]} ${className}`}>
      {children}
    </div>
  );
};

// Specific grid components
const MetricsRow = ({ children, className = '' }) => (
  <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 gap-6 mb-8 ${className}`}>
    {children}
  </div>
);

const MainContentRow = ({ children, className = '' }) => (
  <div className={`grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8 ${className}`}>
    {children}
  </div>
);

const BottomRow = ({ children, className = '' }) => (
  <div className={`grid grid-cols-1 xl:grid-cols-12 gap-6 ${className}`}>
    {children}
  </div>
);

export default DashboardGrid;
export { GridItem, MetricsRow, MainContentRow, BottomRow };