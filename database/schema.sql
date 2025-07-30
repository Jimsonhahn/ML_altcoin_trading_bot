-- ML Altcoin Trading Bot Database Schema
-- Intelligente Lern-Infrastruktur für den Orchestrator
-- PostgreSQL/TimescaleDB Schema

-- Erstelle Extensions für TimescaleDB und weitere Features
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "ltree";

-- =====================================================
-- ORCHESTRATOR DECISIONS TABLE
-- Alle Entscheidungen des Strategy Orchestrators
-- =====================================================
CREATE TABLE orchestrator_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id UUID NOT NULL,
    decision_type VARCHAR(50) NOT NULL, -- 'strategy_allocation', 'risk_adjustment', 'market_regime_change', 'position_close', etc.
    
    -- Entscheidungskontext
    market_regime VARCHAR(20), -- 'bull', 'bear', 'volatile', 'crash', 'euphoria'
    volatility_level DECIMAL(10,4),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Strategien betroffen
    strategy_name VARCHAR(100),
    old_allocation DECIMAL(5,4),
    new_allocation DECIMAL(5,4),
    
    -- Trigger-Informationen
    trigger_source VARCHAR(100), -- 'volume_spike', 'technical_indicator', 'ml_signal', 'risk_limit'
    trigger_data JSONB,
    
    -- Ausführung
    decision_reasoning TEXT,
    execution_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'executed', 'failed', 'cancelled'
    execution_timestamp TIMESTAMPTZ,
    execution_error TEXT,
    
    -- Performance Tracking
    expected_impact DECIMAL(10,4),  -- Erwartete Auswirkung auf Portfolio
    actual_impact DECIMAL(10,4),    -- Tatsächliche Auswirkung (nach Messung)
    impact_measured_at TIMESTAMPTZ,
    
    -- Metadaten
    portfolio_value_before DECIMAL(15,2),
    portfolio_value_after DECIMAL(15,2),
    risk_score_before DECIMAL(5,4),
    risk_score_after DECIMAL(5,4),
    
    -- Indexing
    CONSTRAINT valid_allocation CHECK (
        (old_allocation IS NULL OR (old_allocation >= 0 AND old_allocation <= 1)) AND
        (new_allocation IS NULL OR (new_allocation >= 0 AND new_allocation <= 1))
    )
);

-- Erstelle Hypertable für Time-Series Daten
SELECT create_hypertable('orchestrator_decisions', 'timestamp', if_not_exists => TRUE);

-- =====================================================
-- STRATEGY PERFORMANCE TABLE
-- Detaillierte Trade-Performance mit vollem Kontext
-- =====================================================
CREATE TABLE strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Trade Identifikation
    trade_id UUID NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    session_id UUID NOT NULL,
    
    -- Asset & Market Info
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    
    -- Trade Details
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    trade_status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'cancelled'
    
    -- Timing
    entry_timestamp TIMESTAMPTZ NOT NULL,
    exit_timestamp TIMESTAMPTZ,
    duration_minutes INTEGER,
    
    -- Performance Metriken
    pnl_absolute DECIMAL(15,2),
    pnl_percentage DECIMAL(10,4),
    fees_paid DECIMAL(15,2),
    slippage DECIMAL(10,4),
    
    -- Risk Management
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    max_drawdown DECIMAL(10,4),
    risk_reward_ratio DECIMAL(10,2),
    position_size_usd DECIMAL(15,2),
    
    -- Market Context zur Entry-Zeit
    market_regime_at_entry VARCHAR(20),
    volatility_at_entry DECIMAL(10,4),
    volume_profile JSONB, -- 24h volume, avg volume, volume spike ratio
    
    -- Technical Indicators bei Entry
    technical_context JSONB, -- RSI, MACD, Bollinger Bands, etc.
    
    -- ML Insights
    ml_confidence DECIMAL(5,4),
    ml_features JSONB, -- Feature values used for decision
    ml_model_version VARCHAR(50),
    
    -- Strategy-specific Data
    strategy_parameters JSONB,
    signal_strength DECIMAL(5,4),
    correlation_with_other_trades DECIMAL(5,4),
    
    -- Risk & Portfolio Context
    portfolio_heat DECIMAL(5,4), -- % of portfolio at risk
    correlation_risk DECIMAL(5,4),
    portfolio_value_at_entry DECIMAL(15,2),
    
    -- Exit Reasoning
    exit_reason VARCHAR(50), -- 'take_profit', 'stop_loss', 'strategy_signal', 'risk_management', 'manual'
    exit_signal_strength DECIMAL(5,4),
    
    -- Post-Trade Analysis
    trade_quality_score DECIMAL(5,4), -- 0-1 score based on execution vs plan
    lessons_learned TEXT,
    
    -- Indexing
    CONSTRAINT valid_prices CHECK (entry_price > 0 AND (exit_price IS NULL OR exit_price > 0)),
    CONSTRAINT valid_quantity CHECK (quantity > 0)
);

-- Erstelle Hypertable
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

-- =====================================================
-- MARKET STATES TABLE
-- Snapshots des Marktzustands für Analyse
-- =====================================================
CREATE TABLE market_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Market Identification
    data_source VARCHAR(50) NOT NULL, -- 'binance', 'coinbase', 'coingecko'
    market_type VARCHAR(20) NOT NULL, -- 'spot', 'futures', 'options'
    
    -- Overall Market Metrics
    total_market_cap DECIMAL(20,2),
    btc_dominance DECIMAL(5,4),
    fear_greedy_index INTEGER CHECK (fear_greedy_index >= 0 AND fear_greedy_index <= 100),
    
    -- Regime Detection
    detected_regime VARCHAR(20) NOT NULL,
    regime_confidence DECIMAL(5,4),
    regime_duration_hours INTEGER,
    previous_regime VARCHAR(20),
    
    -- Volatility Analysis
    vix_crypto DECIMAL(10,4),
    realized_volatility_24h DECIMAL(10,4),
    implied_volatility DECIMAL(10,4),
    volatility_percentile DECIMAL(5,4), -- Compared to 90-day history
    
    -- Volume Analysis
    total_volume_24h DECIMAL(20,2),
    volume_ma_ratio DECIMAL(10,4), -- Current vs 30-day MA
    unusual_volume_detected BOOLEAN DEFAULT FALSE,
    volume_spike_threshold DECIMAL(10,4),
    
    -- Price Action
    major_support_levels DECIMAL[],
    major_resistance_levels DECIMAL[],
    trend_strength DECIMAL(5,4),
    trend_direction VARCHAR(10), -- 'up', 'down', 'sideways'
    
    -- Cross-Asset Correlations
    btc_correlation JSONB, -- Correlations with BTC for major assets
    traditional_markets_correlation JSONB, -- SPY, QQQ, Gold, etc.
    
    -- Sentiment Indicators
    social_sentiment_score DECIMAL(5,4), -- -1 to 1
    news_sentiment_score DECIMAL(5,4),
    funding_rates JSONB, -- Funding rates across exchanges
    
    -- Technical Indicators (Market-wide)
    rsi_composite DECIMAL(5,2),
    macd_signal VARCHAR(10), -- 'bullish', 'bearish', 'neutral'
    bollinger_position DECIMAL(5,4), -- Where price is within bands
    
    -- Risk Metrics
    systemic_risk_score DECIMAL(5,4),
    tail_risk_indicator DECIMAL(5,4),
    leverage_ratio DECIMAL(10,4),
    
    -- ML-derived insights
    anomaly_score DECIMAL(5,4), -- Detected anomalies
    predicted_next_regime VARCHAR(20),
    regime_change_probability DECIMAL(5,4)
);

-- Erstelle Hypertable
SELECT create_hypertable('market_states', 'timestamp', if_not_exists => TRUE);

-- =====================================================
-- ML INSIGHTS TABLE
-- Gefundene Muster und Erkenntnisse
-- =====================================================
CREATE TABLE ml_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Insight Classification
    insight_type VARCHAR(50) NOT NULL, -- 'pattern', 'correlation', 'anomaly', 'prediction'
    category VARCHAR(50) NOT NULL, -- 'market_structure', 'strategy_performance', 'risk_pattern'
    
    -- Pattern Details
    pattern_name VARCHAR(100) NOT NULL,
    pattern_description TEXT NOT NULL,
    
    -- Statistical Significance
    confidence_level DECIMAL(5,4) NOT NULL CHECK (confidence_level >= 0 AND confidence_level <= 1),
    statistical_significance DECIMAL(10,8), -- p-value
    sample_size INTEGER,
    
    -- Performance Metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    
    -- Pattern Context
    symbols_affected VARCHAR[] NOT NULL,
    strategies_affected VARCHAR[] NOT NULL,
    market_conditions JSONB, -- Conditions where pattern applies
    
    -- Time Context
    pattern_discovery_date TIMESTAMPTZ NOT NULL,
    lookback_period_days INTEGER NOT NULL,
    forward_testing_period_days INTEGER,
    
    -- Pattern Definition
    features_used JSONB NOT NULL, -- Feature names and importance scores
    thresholds JSONB, -- Threshold values for pattern detection
    pattern_rules TEXT, -- Human-readable rules
    
    -- Performance Tracking
    times_detected INTEGER DEFAULT 0,
    times_acted_upon INTEGER DEFAULT 0,
    successful_predictions INTEGER DEFAULT 0,
    failed_predictions INTEGER DEFAULT 0,
    
    -- Impact Assessment
    average_return_when_detected DECIMAL(10,4),
    max_return_observed DECIMAL(10,4),
    min_return_observed DECIMAL(10,4),
    volatility_during_pattern DECIMAL(10,4),
    
    -- Model Information
    ml_model_type VARCHAR(50), -- 'random_forest', 'neural_network', 'svm', etc.
    model_version VARCHAR(50),
    training_data_hash VARCHAR(64), -- Hash of training data for reproducibility
    
    -- Validation
    validation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'validated', 'invalidated'
    validation_notes TEXT,
    last_validation_date TIMESTAMPTZ,
    
    -- Business Impact
    estimated_edge DECIMAL(10,4), -- Expected edge in basis points
    implementation_complexity VARCHAR(20), -- 'low', 'medium', 'high'
    recommended_action TEXT,
    
    -- Lifecycle
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'under_review'
    deprecated_reason TEXT,
    replacement_insight_id UUID REFERENCES ml_insights(id)
);

-- Erstelle Hypertable
SELECT create_hypertable('ml_insights', 'timestamp', if_not_exists => TRUE);

-- =====================================================
-- STRATEGY COMBINATIONS TABLE
-- Synergien und Konflikte zwischen Strategien
-- =====================================================
CREATE TABLE strategy_combinations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Combination Identification
    combination_hash VARCHAR(64) NOT NULL, -- Hash of sorted strategy names
    strategies VARCHAR[] NOT NULL,
    combination_type VARCHAR(20) NOT NULL, -- 'synergy', 'conflict', 'neutral'
    
    -- Performance Metrics
    combined_sharpe_ratio DECIMAL(10,4),
    individual_sharpe_ratios DECIMAL[],
    synergy_score DECIMAL(5,4), -- -1 to 1, how much better/worse than sum of parts
    
    -- Risk Metrics
    combined_max_drawdown DECIMAL(10,4),
    correlation_coefficient DECIMAL(5,4),
    diversification_benefit DECIMAL(10,4),
    
    -- Timing Analysis
    simultaneous_trades_count INTEGER DEFAULT 0,
    conflicting_signals_count INTEGER DEFAULT 0,
    complementary_signals_count INTEGER DEFAULT 0,
    
    -- Market Conditions
    optimal_market_regime VARCHAR(20),
    worst_market_regime VARCHAR(20),
    volatility_sweet_spot DECIMAL(10,4), -- Optimal volatility range
    
    -- Allocation Insights
    optimal_weight_distribution DECIMAL[],
    tested_weight_combinations JSONB,
    best_performing_weights DECIMAL[],
    
    -- Performance Windows
    short_term_performance JSONB, -- 1D, 7D, 30D performance
    medium_term_performance JSONB, -- 90D, 180D performance  
    long_term_performance JSONB, -- 365D+ performance
    
    -- Risk-Adjusted Metrics
    sortino_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    maximum_adverse_excursion DECIMAL(10,4),
    maximum_favorable_excursion DECIMAL(10,4),
    
    -- Interaction Analysis
    interaction_strength DECIMAL(5,4), -- How much strategies affect each other
    interaction_type VARCHAR(50), -- 'amplifying', 'dampening', 'competing'
    interaction_lag_minutes INTEGER, -- Time delay in interaction
    
    -- Trading Costs Impact
    turnover_impact DECIMAL(10,4),
    slippage_impact DECIMAL(10,4),
    fees_impact DECIMAL(10,4),
    net_performance_after_costs DECIMAL(10,4),
    
    -- Machine Learning Insights
    predicted_future_performance DECIMAL(10,4),
    prediction_confidence DECIMAL(5,4),
    feature_importance JSONB,
    
    -- Validation & Testing
    backtest_start_date TIMESTAMPTZ NOT NULL,
    backtest_end_date TIMESTAMPTZ NOT NULL,
    out_of_sample_performance DECIMAL(10,4),
    walk_forward_results JSONB,
    
    -- Status & Lifecycle
    validation_status VARCHAR(20) DEFAULT 'pending',
    last_tested_timestamp TIMESTAMPTZ,
    performance_drift DECIMAL(10,4), -- Performance degradation over time
    
    -- Implementation Details
    implementation_complexity VARCHAR(20),
    resource_requirements JSONB,
    recommended_portfolio_percentage DECIMAL(5,4),
    
    -- Constraints
    CONSTRAINT valid_strategies CHECK (array_length(strategies, 1) >= 2),
    CONSTRAINT valid_synergy_score CHECK (synergy_score >= -1 AND synergy_score <= 1)
);

-- Erstelle Hypertable
SELECT create_hypertable('strategy_combinations', 'timestamp', if_not_exists => TRUE);

-- =====================================================
-- INDEXES FÜR PERFORMANCE OPTIMIERUNG
-- =====================================================

-- Orchestrator Decisions Indexes
CREATE INDEX idx_orchestrator_decisions_session_id ON orchestrator_decisions (session_id);
CREATE INDEX idx_orchestrator_decisions_strategy ON orchestrator_decisions (strategy_name);
CREATE INDEX idx_orchestrator_decisions_type ON orchestrator_decisions (decision_type);
CREATE INDEX idx_orchestrator_decisions_regime ON orchestrator_decisions (market_regime);
CREATE INDEX idx_orchestrator_decisions_status ON orchestrator_decisions (execution_status);
CREATE INDEX idx_orchestrator_decisions_trigger ON orchestrator_decisions USING GIN (trigger_data);

-- Strategy Performance Indexes
CREATE INDEX idx_strategy_performance_trade_id ON strategy_performance (trade_id);
CREATE INDEX idx_strategy_performance_strategy ON strategy_performance (strategy_name);
CREATE INDEX idx_strategy_performance_symbol ON strategy_performance (symbol);
CREATE INDEX idx_strategy_performance_status ON strategy_performance (trade_status);
CREATE INDEX idx_strategy_performance_pnl ON strategy_performance (pnl_percentage);
CREATE INDEX idx_strategy_performance_entry_time ON strategy_performance (entry_timestamp);
CREATE INDEX idx_strategy_performance_session ON strategy_performance (session_id);
CREATE INDEX idx_strategy_performance_ml_features ON strategy_performance USING GIN (ml_features);
CREATE INDEX idx_strategy_performance_technical ON strategy_performance USING GIN (technical_context);

-- Market States Indexes
CREATE INDEX idx_market_states_regime ON market_states (detected_regime);
CREATE INDEX idx_market_states_source ON market_states (data_source);
CREATE INDEX idx_market_states_volatility ON market_states (realized_volatility_24h);
CREATE INDEX idx_market_states_anomaly ON market_states (anomaly_score);
CREATE INDEX idx_market_states_sentiment ON market_states USING GIN (social_sentiment_score);

-- ML Insights Indexes
CREATE INDEX idx_ml_insights_type ON ml_insights (insight_type);
CREATE INDEX idx_ml_insights_pattern ON ml_insights (pattern_name);
CREATE INDEX idx_ml_insights_confidence ON ml_insights (confidence_level);
CREATE INDEX idx_ml_insights_symbols ON ml_insights USING GIN (symbols_affected);
CREATE INDEX idx_ml_insights_strategies ON ml_insights USING GIN (strategies_affected);
CREATE INDEX idx_ml_insights_status ON ml_insights (status);
CREATE INDEX idx_ml_insights_features ON ml_insights USING GIN (features_used);

-- Strategy Combinations Indexes
CREATE INDEX idx_strategy_combinations_hash ON strategy_combinations (combination_hash);
CREATE INDEX idx_strategy_combinations_strategies ON strategy_combinations USING GIN (strategies);
CREATE INDEX idx_strategy_combinations_type ON strategy_combinations (combination_type);
CREATE INDEX idx_strategy_combinations_synergy ON strategy_combinations (synergy_score);
CREATE INDEX idx_strategy_combinations_sharpe ON strategy_combinations (combined_sharpe_ratio);
CREATE INDEX idx_strategy_combinations_regime ON strategy_combinations (optimal_market_regime);

-- =====================================================
-- VIEWS FÜR HÄUFIGE ABFRAGEN
-- =====================================================

-- Real-time Strategy Performance Summary
CREATE VIEW strategy_performance_summary AS
SELECT 
    strategy_name,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE trade_status = 'closed') as closed_trades,
    AVG(pnl_percentage) FILTER (WHERE trade_status = 'closed') as avg_return,
    STDDEV(pnl_percentage) FILTER (WHERE trade_status = 'closed') as return_volatility,
    COUNT(*) FILTER (WHERE pnl_percentage > 0 AND trade_status = 'closed') as winning_trades,
    COUNT(*) FILTER (WHERE pnl_percentage <= 0 AND trade_status = 'closed') as losing_trades,
    AVG(duration_minutes) FILTER (WHERE trade_status = 'closed') as avg_duration_minutes,
    SUM(pnl_absolute) FILTER (WHERE trade_status = 'closed') as total_pnl,
    MAX(timestamp) as last_activity
FROM strategy_performance
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY strategy_name;

-- Current Market Regime Analysis
CREATE VIEW current_market_analysis AS
SELECT DISTINCT ON (data_source)
    data_source,
    detected_regime,
    regime_confidence,
    volatility_percentile,
    social_sentiment_score,
    systemic_risk_score,
    timestamp as last_update
FROM market_states
ORDER BY data_source, timestamp DESC;

-- Active ML Insights
CREATE VIEW active_ml_insights AS
SELECT 
    insight_type,
    pattern_name,
    confidence_level,
    symbols_affected,
    strategies_affected,
    average_return_when_detected,
    times_detected,
    successful_predictions,
    CASE 
        WHEN times_detected > 0 THEN (successful_predictions::DECIMAL / times_detected) 
        ELSE NULL 
    END as success_rate
FROM ml_insights
WHERE status = 'active' 
    AND validation_status = 'validated'
    AND confidence_level >= 0.7
ORDER BY confidence_level DESC, success_rate DESC NULLS LAST;

-- Strategy Synergy Matrix
CREATE VIEW strategy_synergy_matrix AS
SELECT 
    strategies[1] as strategy_1,
    strategies[2] as strategy_2,
    synergy_score,
    combined_sharpe_ratio,
    correlation_coefficient,
    optimal_market_regime,
    last_tested_timestamp
FROM strategy_combinations
WHERE array_length(strategies, 1) = 2
    AND validation_status = 'validated'
    AND last_tested_timestamp >= NOW() - INTERVAL '90 days'
ORDER BY synergy_score DESC;

-- =====================================================
-- DATENRETENTION POLICIES (TimescaleDB)
-- =====================================================

-- Behalte detaillierte Daten für 1 Jahr, dann komprimiere
SELECT add_retention_policy('orchestrator_decisions', INTERVAL '1 year');
SELECT add_retention_policy('strategy_performance', INTERVAL '2 years');
SELECT add_retention_policy('market_states', INTERVAL '1 year');
SELECT add_retention_policy('ml_insights', INTERVAL '3 years'); -- ML insights sind wertvoll
SELECT add_retention_policy('strategy_combinations', INTERVAL '2 years');

-- Komprimierung für bessere Storage-Effizienz
SELECT add_compression_policy('orchestrator_decisions', INTERVAL '7 days');
SELECT add_compression_policy('strategy_performance', INTERVAL '30 days');
SELECT add_compression_policy('market_states', INTERVAL '7 days');
SELECT add_compression_policy('ml_insights', INTERVAL '90 days');
SELECT add_compression_policy('strategy_combinations', INTERVAL '30 days');

-- =====================================================
-- TRIGGERS FÜR AUTOMATISCHE DATENQUALITÄT
-- =====================================================

-- Automatische Aktualisierung von ML Insights Performance
CREATE OR REPLACE FUNCTION update_ml_insight_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update performance counters when a trade is closed
    IF NEW.trade_status = 'closed' AND OLD.trade_status != 'closed' THEN
        -- Update all applicable ML insights
        UPDATE ml_insights 
        SET times_acted_upon = times_acted_upon + 1,
            successful_predictions = successful_predictions + 
                CASE WHEN NEW.pnl_percentage > 0 THEN 1 ELSE 0 END,
            failed_predictions = failed_predictions + 
                CASE WHEN NEW.pnl_percentage <= 0 THEN 1 ELSE 0 END
        WHERE NEW.strategy_name = ANY(strategies_affected)
            AND NEW.symbol = ANY(symbols_affected)
            AND status = 'active';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_ml_insight_performance
    AFTER UPDATE ON strategy_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_ml_insight_performance();

-- Automatische Berechnung von Trade-Dauer
CREATE OR REPLACE FUNCTION calculate_trade_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.exit_timestamp IS NOT NULL AND NEW.entry_timestamp IS NOT NULL THEN
        NEW.duration_minutes := EXTRACT(EPOCH FROM (NEW.exit_timestamp - NEW.entry_timestamp)) / 60;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_calculate_trade_duration
    BEFORE INSERT OR UPDATE ON strategy_performance
    FOR EACH ROW
    EXECUTE FUNCTION calculate_trade_duration();

COMMENT ON DATABASE postgres IS 'ML Altcoin Trading Bot - Intelligente Lern-Infrastruktur';