/**
 * Main App Component
 * Root component with routing and theme provider
 */

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import { useTheme } from './hooks/useTheme';
import apiService from './services/api';

function App() {
  const { theme } = useTheme();
  const [isAuthenticated, setIsAuthenticated] = useState(apiService.isAuthenticated());

  // Listen for authentication changes
  useEffect(() => {
    const checkAuth = () => {
      setIsAuthenticated(apiService.isAuthenticated());
    };

    // Check authentication on mount and when localStorage changes
    checkAuth();
    
    // Listen for localStorage changes (token updates)
    window.addEventListener('storage', checkAuth);
    
    // Custom event for token changes within the same tab
    window.addEventListener('authChange', checkAuth);

    return () => {
      window.removeEventListener('storage', checkAuth);
      window.removeEventListener('authChange', checkAuth);
    };
  }, []);

  return (
    <div className={`App ${theme}`}>
      <Router>
        <Routes>
          <Route 
            path="/login" 
            element={!isAuthenticated ? <Login /> : <Navigate to="/dashboard" />} 
          />
          <Route 
            path="/dashboard" 
            element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/" 
            element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} 
          />
        </Routes>
      </Router>
    </div>
  );
}

export default App;