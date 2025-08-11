/**
 * Comprehensive validation utilities for the trading bot dashboard
 */

// Trading configuration validation schemas
export const VALIDATION_RULES = {
  // Trading modes
  modes: ['paper', 'live'],
  
  // Supported strategies
  strategies: ['super_lazy_billionaire', 'momentum', 'mean_reversion', 'grid_trading', 'arbitrage', 'defi_yield', 'liquidation', 'ml_strategy'],
  
  // Supported symbols
  symbols: ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'AVAX/USDT', 'LINK/USDT'],
  
  // Capital limits
  capital: {
    min: 100,
    max: 1000000
  },
  
  // Risk limits
  risk_per_trade: {
    min: 0.001, // 0.1%
    max: 0.1    // 10%
  },
  
  // Strategy-specific parameter limits
  strategy_params: {
    super_lazy_billionaire: {
      capital_allocation: { min: 0.1, max: 1.0, type: 'number' },
      kelly_factor: { min: 0.1, max: 0.5, type: 'number' },
      ml_confidence_threshold: { min: 0.5, max: 0.95, type: 'number' },
      risk_per_trade: { min: 0.005, max: 0.05, type: 'number' },
      max_positions: { min: 1, max: 10, type: 'integer' },
      rebalance_threshold: { min: 0.05, max: 0.3, type: 'number' }
    },
    momentum: {
      rsi_period: { min: 5, max: 30, type: 'integer' },
      rsi_overbought: { min: 60, max: 90, type: 'number' },
      rsi_oversold: { min: 10, max: 40, type: 'number' },
      sma_short: { min: 3, max: 15, type: 'integer' },
      sma_long: { min: 10, max: 50, type: 'integer' },
      volume_spike_threshold: { min: 1.0, max: 3.0, type: 'number' }
    },
    mean_reversion: {
      bollinger_period: { min: 10, max: 50, type: 'integer' },
      bollinger_std: { min: 1.0, max: 3.0, type: 'number' },
      rsi_period: { min: 5, max: 30, type: 'integer' },
      use_rsi_filter: { type: 'boolean' }
    },
    grid_trading: {
      num_grids: { min: 5, max: 20, type: 'integer' },
      price_range_multiplier: { min: 0.01, max: 0.1, type: 'number' },
      grid_size_percent: { min: 0.005, max: 0.05, type: 'number' }
    },
    arbitrage: {
      min_profit_threshold: { min: 0.1, max: 2.0, type: 'number' },
      max_slippage: { min: 0.05, max: 1.0, type: 'number' }
    },
    ml_strategy: {
      model_confidence_threshold: { min: 0.5, max: 0.9, type: 'number' },
      lookback_window: { min: 10, max: 50, type: 'integer' },
      feature_importance_min: { min: 0.01, max: 0.5, type: 'number' }
    }
  }
};

// Error types
export const VALIDATION_ERRORS = {
  REQUIRED_FIELD: 'REQUIRED_FIELD',
  INVALID_TYPE: 'INVALID_TYPE',
  OUT_OF_RANGE: 'OUT_OF_RANGE',
  INVALID_VALUE: 'INVALID_VALUE',
  DEPENDENCY_ERROR: 'DEPENDENCY_ERROR',
  FORMAT_ERROR: 'FORMAT_ERROR'
};

// Validation result class
export class ValidationResult {
  constructor() {
    this.isValid = true;
    this.errors = [];
    this.warnings = [];
  }

  addError(field, type, message, value = null) {
    this.isValid = false;
    this.errors.push({
      field,
      type,
      message,
      value,
      timestamp: new Date().toISOString()
    });
  }

  addWarning(field, message, value = null) {
    this.warnings.push({
      field,
      message,
      value,
      timestamp: new Date().toISOString()
    });
  }

  getFieldErrors(field) {
    return this.errors.filter(error => error.field === field);
  }

  hasFieldErrors(field) {
    return this.getFieldErrors(field).length > 0;
  }

  getErrorSummary() {
    return {
      total: this.errors.length,
      byType: this.errors.reduce((acc, error) => {
        acc[error.type] = (acc[error.type] || 0) + 1;
        return acc;
      }, {}),
      byField: this.errors.reduce((acc, error) => {
        acc[error.field] = (acc[error.field] || 0) + 1;
        return acc;
      }, {})
    };
  }
}

// Core validation functions
export const validateRequired = (value, fieldName) => {
  if (value === null || value === undefined || value === '') {
    return { valid: false, message: `${fieldName} ist erforderlich` };
  }
  return { valid: true };
};

export const validateType = (value, expectedType, fieldName) => {
  const actualType = typeof value;
  
  if (expectedType === 'integer') {
    if (!Number.isInteger(value)) {
      return { valid: false, message: `${fieldName} muss eine ganze Zahl sein` };
    }
  } else if (expectedType === 'number') {
    if (typeof value !== 'number' || isNaN(value)) {
      return { valid: false, message: `${fieldName} muss eine Zahl sein` };
    }
  } else if (actualType !== expectedType) {
    return { valid: false, message: `${fieldName} muss vom Typ ${expectedType} sein` };
  }
  
  return { valid: true };
};

export const validateRange = (value, min, max, fieldName) => {
  if (typeof value !== 'number') {
    return { valid: false, message: `${fieldName} muss eine Zahl sein` };
  }
  
  if (value < min || value > max) {
    return { valid: false, message: `${fieldName} muss zwischen ${min} und ${max} liegen` };
  }
  
  return { valid: true };
};

export const validateEnum = (value, allowedValues, fieldName) => {
  if (!allowedValues.includes(value)) {
    return { valid: false, message: `${fieldName} muss einer der folgenden Werte sein: ${allowedValues.join(', ')}` };
  }
  return { valid: true };
};

export const validateSymbolFormat = (symbol) => {
  const symbolPattern = /^[A-Z]{2,10}\/[A-Z]{2,10}$/;
  if (!symbolPattern.test(symbol)) {
    return { valid: false, message: 'Symbol muss im Format BASE/QUOTE sein (z.B. BTC/USDT)' };
  }
  return { valid: true };
};

// Strategy-specific validations
export const validateStrategyDependencies = (strategy, params) => {
  const result = new ValidationResult();
  
  switch (strategy) {
    case 'super_lazy_billionaire':
      // Kelly factor should be conservative
      if (params.kelly_factor && params.kelly_factor > 0.3) {
        result.addWarning('kelly_factor', 
          'Hoher Kelly-Faktor kann zu aggressiven Positionsgrößen führen');
      }
      
      // ML confidence threshold validation
      if (params.ml_confidence_threshold && params.ml_confidence_threshold < 0.6) {
        result.addWarning('ml_confidence_threshold',
          'Niedrige ML-Konfidenz kann zu schlechten Signalen führen');
      }
      
      // Capital allocation should be reasonable
      if (params.capital_allocation && params.capital_allocation > 0.9) {
        result.addWarning('capital_allocation',
          'Hohe Kapitalallokation lässt wenig Reserve für Opportunitäten');
      }
      break;
      
    case 'momentum':
      // SMA dependency: short period must be less than long period
      if (params.sma_short && params.sma_long && params.sma_short >= params.sma_long) {
        result.addError('sma_short', VALIDATION_ERRORS.DEPENDENCY_ERROR, 
          'Kurzer SMA Zeitraum muss kleiner als langer SMA Zeitraum sein');
      }
      
      // RSI overbought must be greater than oversold
      if (params.rsi_overbought && params.rsi_oversold && params.rsi_overbought <= params.rsi_oversold) {
        result.addError('rsi_overbought', VALIDATION_ERRORS.DEPENDENCY_ERROR,
          'RSI Überkauft-Level muss größer als Überverkauft-Level sein');
      }
      break;
      
    case 'grid_trading':
      // Grid size validation
      if (params.grid_size_percent && params.price_range_multiplier) {
        const maxGridSize = params.price_range_multiplier / 2;
        if (params.grid_size_percent > maxGridSize) {
          result.addWarning('grid_size_percent',
            `Grid-Größe ist möglicherweise zu groß für den Preis-Range. Empfohlen: max. ${maxGridSize.toFixed(3)}`);
        }
      }
      break;
      
    case 'ml_strategy':
      // Model confidence should be reasonable
      if (params.model_confidence_threshold && params.model_confidence_threshold > 0.85) {
        result.addWarning('model_confidence_threshold',
          'Sehr hohe Confidence-Schwelle kann zu wenigen Trades führen');
      }
      break;
  }
  
  return result;
};

// Main validation function for trading configuration
export const validateTradingConfig = (config) => {
  const result = new ValidationResult();
  
  // Validate required fields
  const requiredFields = ['mode', 'strategy', 'symbol', 'capital', 'risk_per_trade'];
  for (const field of requiredFields) {
    const validation = validateRequired(config[field], field);
    if (!validation.valid) {
      result.addError(field, VALIDATION_ERRORS.REQUIRED_FIELD, validation.message, config[field]);
    }
  }
  
  // Validate mode
  if (config.mode) {
    const validation = validateEnum(config.mode, VALIDATION_RULES.modes, 'Modus');
    if (!validation.valid) {
      result.addError('mode', VALIDATION_ERRORS.INVALID_VALUE, validation.message, config.mode);
    }
  }
  
  // Validate strategy
  if (config.strategy) {
    const validation = validateEnum(config.strategy, VALIDATION_RULES.strategies, 'Strategie');
    if (!validation.valid) {
      result.addError('strategy', VALIDATION_ERRORS.INVALID_VALUE, validation.message, config.strategy);
    }
  }
  
  // Validate symbol
  if (config.symbol) {
    const formatValidation = validateSymbolFormat(config.symbol);
    if (!formatValidation.valid) {
      result.addError('symbol', VALIDATION_ERRORS.FORMAT_ERROR, formatValidation.message, config.symbol);
    } else {
      const enumValidation = validateEnum(config.symbol, VALIDATION_RULES.symbols, 'Symbol');
      if (!enumValidation.valid) {
        result.addWarning('symbol', `Symbol wird möglicherweise nicht unterstützt: ${config.symbol}`);
      }
    }
  }
  
  // Validate capital
  if (config.capital !== undefined) {
    const typeValidation = validateType(config.capital, 'number', 'Kapital');
    if (!typeValidation.valid) {
      result.addError('capital', VALIDATION_ERRORS.INVALID_TYPE, typeValidation.message, config.capital);
    } else {
      const rangeValidation = validateRange(config.capital, VALIDATION_RULES.capital.min, VALIDATION_RULES.capital.max, 'Kapital');
      if (!rangeValidation.valid) {
        result.addError('capital', VALIDATION_ERRORS.OUT_OF_RANGE, rangeValidation.message, config.capital);
      }
    }
  }
  
  // Validate risk per trade
  if (config.risk_per_trade !== undefined) {
    const typeValidation = validateType(config.risk_per_trade, 'number', 'Risiko pro Trade');
    if (!typeValidation.valid) {
      result.addError('risk_per_trade', VALIDATION_ERRORS.INVALID_TYPE, typeValidation.message, config.risk_per_trade);
    } else {
      const rangeValidation = validateRange(config.risk_per_trade, VALIDATION_RULES.risk_per_trade.min, VALIDATION_RULES.risk_per_trade.max, 'Risiko pro Trade');
      if (!rangeValidation.valid) {
        result.addError('risk_per_trade', VALIDATION_ERRORS.OUT_OF_RANGE, rangeValidation.message, config.risk_per_trade);
      }
      
      // Warning for high risk
      if (config.risk_per_trade > 0.05) {
        result.addWarning('risk_per_trade', 'Hoher Risikofaktor kann zu erheblichen Verlusten führen');
      }
    }
  }
  
  // Validate strategy parameters
  if (config.strategy && config.strategy_params && VALIDATION_RULES.strategy_params[config.strategy]) {
    const strategyRules = VALIDATION_RULES.strategy_params[config.strategy];
    
    for (const [paramName, paramValue] of Object.entries(config.strategy_params)) {
      const rule = strategyRules[paramName];
      if (!rule) {
        result.addWarning(`strategy_params.${paramName}`, `Unbekannter Parameter für Strategie ${config.strategy}`);
        continue;
      }
      
      // Type validation
      const typeValidation = validateType(paramValue, rule.type, paramName);
      if (!typeValidation.valid) {
        result.addError(`strategy_params.${paramName}`, VALIDATION_ERRORS.INVALID_TYPE, typeValidation.message, paramValue);
        continue;
      }
      
      // Range validation for numeric types
      if ((rule.type === 'number' || rule.type === 'integer') && rule.min !== undefined && rule.max !== undefined) {
        const rangeValidation = validateRange(paramValue, rule.min, rule.max, paramName);
        if (!rangeValidation.valid) {
          result.addError(`strategy_params.${paramName}`, VALIDATION_ERRORS.OUT_OF_RANGE, rangeValidation.message, paramValue);
        }
      }
    }
    
    // Strategy-specific dependency validations
    const dependencyResult = validateStrategyDependencies(config.strategy, config.strategy_params);
    result.errors.push(...dependencyResult.errors);
    result.warnings.push(...dependencyResult.warnings);
  }
  
  // Cross-field validations
  if (config.mode === 'live' && config.capital && config.capital > 50000) {
    result.addWarning('capital', 'Hoher Kapitaleinsatz im Live-Modus - bitte mit Vorsicht verwenden');
  }
  
  return result;
};

// Real-time validation for form inputs
export const validateField = (fieldName, value, config = {}) => {
  const result = new ValidationResult();
  
  switch (fieldName) {
    case 'mode':
      if (value) {
        const validation = validateEnum(value, VALIDATION_RULES.modes, 'Modus');
        if (!validation.valid) {
          result.addError(fieldName, VALIDATION_ERRORS.INVALID_VALUE, validation.message, value);
        }
      }
      break;
      
    case 'strategy':
      if (value) {
        const validation = validateEnum(value, VALIDATION_RULES.strategies, 'Strategie');
        if (!validation.valid) {
          result.addError(fieldName, VALIDATION_ERRORS.INVALID_VALUE, validation.message, value);
        }
      }
      break;
      
    case 'symbol':
      if (value) {
        const formatValidation = validateSymbolFormat(value);
        if (!formatValidation.valid) {
          result.addError(fieldName, VALIDATION_ERRORS.FORMAT_ERROR, formatValidation.message, value);
        }
      }
      break;
      
    case 'capital':
      if (value !== undefined && value !== '') {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          result.addError(fieldName, VALIDATION_ERRORS.INVALID_TYPE, 'Kapital muss eine Zahl sein', value);
        } else {
          const rangeValidation = validateRange(numValue, VALIDATION_RULES.capital.min, VALIDATION_RULES.capital.max, 'Kapital');
          if (!rangeValidation.valid) {
            result.addError(fieldName, VALIDATION_ERRORS.OUT_OF_RANGE, rangeValidation.message, numValue);
          }
        }
      }
      break;
      
    case 'risk_per_trade':
      if (value !== undefined && value !== '') {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          result.addError(fieldName, VALIDATION_ERRORS.INVALID_TYPE, 'Risiko pro Trade muss eine Zahl sein', value);
        } else {
          const rangeValidation = validateRange(numValue, VALIDATION_RULES.risk_per_trade.min, VALIDATION_RULES.risk_per_trade.max, 'Risiko pro Trade');
          if (!rangeValidation.valid) {
            result.addError(fieldName, VALIDATION_ERRORS.OUT_OF_RANGE, rangeValidation.message, numValue);
          }
        }
      }
      break;
  }
  
  return result;
};

// Utility functions for UI
export const getValidationSeverity = (errors) => {
  if (errors.some(error => error.type === VALIDATION_ERRORS.REQUIRED_FIELD)) {
    return 'critical';
  }
  if (errors.some(error => error.type === VALIDATION_ERRORS.INVALID_TYPE || error.type === VALIDATION_ERRORS.OUT_OF_RANGE)) {
    return 'high';
  }
  if (errors.some(error => error.type === VALIDATION_ERRORS.DEPENDENCY_ERROR)) {
    return 'medium';
  }
  return 'low';
};

export const formatValidationMessage = (error) => {
  return `${error.message}${error.value !== null ? ` (Aktueller Wert: ${error.value})` : ''}`;
};

export default {
  validateTradingConfig,
  validateField,
  ValidationResult,
  VALIDATION_RULES,
  VALIDATION_ERRORS,
  getValidationSeverity,
  formatValidationMessage
};