# Advanced Security and Risk Management System
# Integration in: core/security/ und core/risk/

import asyncio
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import yfinance as yf


class SecureKeyManager:
    """Hardware Security Module (HSM) kompatibles Key Management"""

    def __init__(self, master_key_path: str):
        self.master_key = self._load_master_key(master_key_path)
        self.key_rotation_interval = 86400  # 24 hours
        self.encrypted_keys = {}
        self.hsm_client = None  # For HSM integration

    def _load_master_key(self, path: str) -> bytes:
        """Lädt Master Key aus sicherem Storage"""
        # In Production: Use HSM or AWS KMS
        with open(path, 'rb') as f:
            return f.read()

    def encrypt_api_key(self, exchange: str, api_key: str, api_secret: str) -> Dict:
        """Verschlüsselt API Keys mit AES-256"""
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=exchange.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        f = Fernet(key)

        # Encrypt credentials
        encrypted_key = f.encrypt(api_key.encode())
        encrypted_secret = f.encrypt(api_secret.encode())

        return {
            'exchange': exchange,
            'encrypted_key': encrypted_key,
            'encrypted_secret': encrypted_secret,
            'timestamp': time.time()
        }

    async def get_decrypted_credentials(self, exchange: str) -> Tuple[str, str]:
        """Holt und entschlüsselt Credentials mit Audit Trail"""
        # Log access attempt
        await self._audit_log('credential_access', {
            'exchange': exchange,
            'timestamp': time.time(),
            'ip': self._get_caller_ip()
        })

        # Decrypt credentials
        creds = self.encrypted_keys[exchange]
        f = Fernet(self._derive_key(exchange))

        api_key = f.decrypt(creds['encrypted_key']).decode()
        api_secret = f.decrypt(creds['encrypted_secret']).decode()

        return api_key, api_secret


@dataclass
class RiskLimits:
    """Definiert Risiko-Limits pro Strategie/Asset"""
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    max_leverage: float
    var_limit: float  # Value at Risk
    concentration_limit: float


class AdvancedRiskManager:
    """Fortgeschrittenes Risikomanagement mit ML"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_limits = {}
        self.portfolio_var_model = None
        self.stress_test_scenarios = []
        self.real_time_exposure = {}

    async def calculate_portfolio_risk(self, positions: Dict) -> Dict:
        """Berechnet Portfolio-Risiko in Echtzeit"""
        # 1. Value at Risk (VaR) - Parametric, Historical, Monte Carlo
        var_parametric = self._calculate_parametric_var(positions)
        var_historical = self._calculate_historical_var(positions)
        var_monte_carlo = await self._calculate_monte_carlo_var(positions)

        # 2. Conditional VaR (CVaR) / Expected Shortfall
        cvar = self._calculate_cvar(positions)

        # 3. Greeks for derivatives
        greeks = self._calculate_greeks(positions)

        # 4. Correlation risk
        correlation_risk = self._calculate_correlation_risk(positions)

        # 5. Liquidity risk
        liquidity_risk = await self._calculate_liquidity_risk(positions)

        return {
            'var': {
                'parametric': var_parametric,
                'historical': var_historical,
                'monte_carlo': var_monte_carlo
            },
            'cvar': cvar,
            'greeks': greeks,
            'correlation_risk': correlation_risk,
            'liquidity_risk': liquidity_risk,
            'total_risk_score': self._calculate_risk_score(all_metrics)
        }

    def _calculate_monte_carlo_var(self, positions: Dict, simulations: int = 10000) -> float:
        """Monte Carlo VaR mit Copulas für Tail Dependencies"""
        # Get historical returns
        returns = self._get_historical_returns(positions)

        # Fit copula for dependencies
        from copulas.multivariate import GaussianMultivariate
        copula = GaussianMultivariate()
        copula.fit(returns)

        # Generate scenarios
        scenarios = copula.sample(simulations)

        # Calculate portfolio returns
        weights = np.array([pos['value'] for pos in positions.values()])
        portfolio_returns = scenarios @ weights

        # Calculate VaR at 95% confidence
        var_95 = np.percentile(portfolio_returns, 5)

        return abs(var_95)


class SmartOrderRouter:
    """Intelligentes Order Routing mit Slippage Minimierung"""

    def __init__(self):
        self.exchange_liquidity = {}
        self.historical_slippage = {}
        self.ml_router = self._build_routing_model()

    async def route_order(self, order: Dict) -> List[Dict]:
        """Teilt Order optimal auf mehrere Exchanges auf"""
        # 1. Liquidity Analysis
        liquidity = await self._analyze_liquidity(order['symbol'])

        # 2. Slippage Prediction
        predicted_slippage = self._predict_slippage(order, liquidity)

        # 3. Optimal Split using Dynamic Programming
        splits = self._optimize_order_split(order, liquidity, predicted_slippage)

        # 4. Time-Weighted Average Price (TWAP) / VWAP execution
        execution_schedule = self._create_execution_schedule(splits)

        return execution_schedule

    def _optimize_order_split(self, order: Dict, liquidity: Dict,
                              slippage: Dict) -> List[Dict]:
        """Optimiert Order-Aufteilung mit Reinforcement Learning"""
        state = self._create_state_vector(order, liquidity, slippage)

        # Use trained DQN model
        action = self.ml_router.predict(state)

        # Convert action to order splits
        splits = self._action_to_splits(action, order)

        return splits


class AnomalyDetector:
    """ML-basierte Anomalie-Erkennung für Sicherheit"""

    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.baseline_behavior = {}

    async def detect_anomalies(self, trading_data: pd.DataFrame) -> List[Dict]:
        """Erkennt anomale Trading-Patterns"""
        anomalies = []

        # 1. Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(trading_data)

        # 2. ML-based anomaly detection
        ml_anomalies = self._detect_ml_anomalies(trading_data)

        # 3. Behavioral analysis
        behavioral_anomalies = self._detect_behavioral_anomalies(trading_data)

        # 4. Network traffic analysis
        network_anomalies = await self._detect_network_anomalies()

        # Combine and prioritize
        all_anomalies = statistical_anomalies + ml_anomalies + \
                        behavioral_anomalies + network_anomalies

        return self._prioritize_anomalies(all_anomalies)


class ComplianceEngine:
    """Regulatory Compliance und Reporting"""

    def __init__(self, jurisdiction: str):
        self.jurisdiction = jurisdiction
        self.regulations = self._load_regulations(jurisdiction)
        self.audit_trail = []

    async def check_trade_compliance(self, trade: Dict) -> Tuple[bool, Optional[str]]:
        """Prüft Trade auf Compliance"""
        # 1. KYC/AML checks
        kyc_passed = await self._check_kyc_aml(trade)

        # 2. Position limits
        position_ok = self._check_position_limits(trade)

        # 3. Market manipulation checks
        manipulation_check = self._check_market_manipulation(trade)

        # 4. Wash trading detection
        wash_trading = self._detect_wash_trading(trade)

        if not all([kyc_passed, position_ok, manipulation_check, not wash_trading]):
            reason = self._get_rejection_reason(...)
            return False, reason

        return True, None

    async def generate_regulatory_report(self, period: str) -> Dict:
        """Generiert regulatorische Reports (MiFID II, etc.)"""
        report = {
            'transaction_report': await self._generate_transaction_report(period),
            'best_execution_report': await self._generate_best_execution_report(period),
            'risk_report': await self._generate_risk_report(period),
            'audit_trail': self._get_audit_trail(period)
        }

        # Sign report cryptographically
        report['signature'] = self._sign_report(report)

        return report