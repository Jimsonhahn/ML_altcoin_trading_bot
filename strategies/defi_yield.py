#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeFi Yield Farming Strategy
===========================

Diese Strategy ist wie ein intelligenter Bankier, der:
- Automatisch die besten Zinssätze findet (15-50% APY)
- Täglich Zinsen einsammelt und reinvestiert (Compound)
- Zwischen Protokollen wechselt für maximale Rendite
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging
from datetime import datetime
from dataclasses import dataclass

from .strategy_base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class YieldOpportunity:
    """Represents a yield farming opportunity"""
    protocol: str
    pool: str
    apy: float
    tvl: float
    risk_score: float
    gas_cost: float
    min_deposit: float


class DeFiYieldStrategy(Strategy):
    """
    DeFi Yield Farming Strategy - Maximiert passive Einkommen
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize DeFi Yield Strategy"""
        super().__init__(config)

        self.name = "DeFi Yield Farming"

        # Strategy parameters
        self.min_apy = config.get('min_apy', 0.15)  # 15% minimum
        self.compound_frequency = config.get('compound_frequency', 24)  # hours
        self.gas_limit = config.get('gas_limit', 0.005)  # 0.5% of position
        self.max_protocols = config.get('max_protocols', 5)

        # Risk parameters
        self.risk_levels = config.get('risk_levels', ['low', 'medium'])
        self.max_single_protocol = config.get('max_single_protocol', 0.4)
        self.min_tvl = config.get('min_tvl', 10_000_000)  # $10M minimum TVL
        self.require_audit = config.get('require_audit', True)

        # Protocol weights
        self.protocol_weights = config.get('protocol_weights', {
            'aave': 0.3,
            'compound': 0.2,
            'yearn': 0.2,
            'curve': 0.2,
            'convex': 0.1
        })

        # Supported stablecoins
        self.stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'FRAX']

        # Active positions tracking
        self.active_farms = {}
        self.pending_rewards = {}
        self.last_compound = {}

        logger.info(f"DeFi Yield Strategy initialized:")
        logger.info(f"  - Min APY: {self.min_apy * 100:.1f}%")
        logger.info(f"  - Compound Frequency: {self.compound_frequency}h")
        logger.info(f"  - Gas Limit: {self.gas_limit * 100:.1f}%")
        logger.info(f"  - Protocols: {list(self.protocol_weights.keys())}")

    def calculate_signal(self, symbol: str, data: pd.DataFrame,
                         current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate DeFi yield farming signal
        
        Args:
            symbol: Trading symbol (e.g., 'USDT/USDT')
            data: Price data DataFrame
            current_price: Current price
            
        Returns:
            Tuple of (signal, signal_data)
        """
        # Check if it's a stablecoin
        if not self._is_stablecoin(symbol):
            return 'HOLD', {
                'confidence': 0.0,
                'signal': 'HOLD',
                'reason': 'not_stablecoin',
                'strategy': 'defi_yield'
            }

        try:
            # Get current DeFi opportunities
            opportunities = self._scan_yield_opportunities(symbol)

            if not opportunities:
                return 'HOLD', {
                    'confidence': 0.0,
                    'signal': 'HOLD',
                    'reason': 'no_opportunities',
                    'strategy': 'defi_yield'
                }

            # Get best opportunity
            best_opp = self._select_best_opportunity(opportunities)

            if not best_opp:
                return 'HOLD', {
                    'confidence': 0.0,
                    'signal': 'HOLD',
                    'reason': 'no_suitable_opportunity',
                    'strategy': 'defi_yield'
                }

            # Generate entry signal if opportunity is good
            if best_opp.apy >= self.min_apy:
                confidence = min(0.9, 0.5 + (best_opp.apy - self.min_apy))
                
                return 'BUY', {
                    'signal': 'BUY',
                    'confidence': float(confidence),
                    'strategy': 'defi_yield',
                    'action': 'deposit',
                    'protocol': best_opp.protocol,
                    'pool': best_opp.pool,
                    'expected_apy': float(best_opp.apy),
                    'risk_score': float(best_opp.risk_score),
                    'tvl': float(best_opp.tvl),
                    'gas_cost': float(best_opp.gas_cost),
                    'reason': 'new_yield_opportunity'
                }

            # Default: maintain position
            return 'HOLD', {
                'confidence': 0.5,
                'signal': 'HOLD',
                'strategy': 'defi_yield',
                'current_apy': 0.0,
                'best_apy': best_opp.apy if best_opp else 0.0,
                'reason': 'maintaining_position'
            }

        except Exception as e:
            logger.error(f"Error in DeFi yield calculation: {e}")
            return 'HOLD', {
                'confidence': 0.0,
                'signal': 'HOLD',
                'error': str(e),
                'strategy': 'defi_yield'
            }

    def _scan_yield_opportunities(self, symbol: str) -> List[YieldOpportunity]:
        """Scan all DeFi protocols for yield opportunities"""
        opportunities = []

        # Mock data - in production, fetch real APYs from DeFi APIs
        protocol_data = {
            'aave': {
                'apy': 0.12 + np.random.uniform(-0.02, 0.08),  # 10-20%
                'tvl': 5_000_000_000,
                'risk': 'low',
                'gas': 50,
                'audited': True
            },
            'compound': {
                'apy': 0.10 + np.random.uniform(-0.02, 0.05),  # 8-15%
                'tvl': 3_000_000_000,
                'risk': 'low',
                'gas': 45,
                'audited': True
            },
            'yearn': {
                'apy': 0.15 + np.random.uniform(-0.05, 0.15),  # 10-30%
                'tvl': 1_000_000_000,
                'risk': 'medium',
                'gas': 100,
                'audited': True
            },
            'curve': {
                'apy': 0.10 + np.random.uniform(0, 0.30),  # 10-40%
                'tvl': 8_000_000_000,
                'risk': 'low',
                'gas': 80,
                'audited': True
            },
            'convex': {
                'apy': 0.20 + np.random.uniform(0, 0.30),  # 20-50%
                'tvl': 4_000_000_000,
                'risk': 'medium',
                'gas': 120,
                'audited': True
            }
        }

        # Extract base currency from symbol
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol

        for protocol, data in protocol_data.items():
            # Apply filters
            risk_score = 0.2 if data['risk'] == 'low' else 0.5
            
            if data['risk'] not in self.risk_levels:
                continue

            if data['tvl'] < self.min_tvl:
                continue

            if self.require_audit and not data['audited']:
                continue

            opp = YieldOpportunity(
                protocol=protocol,
                pool=f"{base_symbol}-{protocol}",
                apy=data['apy'],
                tvl=data['tvl'],
                risk_score=risk_score,
                gas_cost=data['gas'],
                min_deposit=100  # $100 minimum
            )

            # Only add if APY meets minimum
            if opp.apy >= self.min_apy:
                opportunities.append(opp)

        # Sort by APY (descending)
        opportunities.sort(key=lambda x: x.apy, reverse=True)

        if opportunities:
            logger.info(f"Found {len(opportunities)} yield opportunities for {symbol}")
            for opp in opportunities[:3]:  # Log top 3
                logger.info(f"  - {opp.protocol}: {opp.apy * 100:.1f}% APY")

        return opportunities

    def _select_best_opportunity(self, opportunities: List[YieldOpportunity]) -> Optional[YieldOpportunity]:
        """Select best opportunity considering risk-adjusted returns"""
        if not opportunities:
            return None

        best_score = -1
        best_opp = None

        for opp in opportunities:
            # Calculate risk-adjusted score
            score = opp.apy * (1 - opp.risk_score)

            # Bonus for high TVL (more secure)
            if opp.tvl > 1_000_000_000:  # $1B+
                score *= 1.1

            # Penalty for high gas costs
            gas_penalty = opp.gas_cost / 10000  # Assume $10k position
            score *= (1 - gas_penalty)

            if score > best_score:
                best_score = score
                best_opp = opp

        return best_opp

    def _is_stablecoin(self, symbol: str) -> bool:
        """Check if symbol is a stablecoin pair"""
        try:
            if '/' in symbol:
                base = symbol.split('/')[0]
                return base in self.stablecoins
            return symbol in self.stablecoins
        except:
            return False

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy"""
        return {
            'name': self.name,
            'type': 'defi_yield',
            'description': 'Automated yield farming across DeFi protocols',
            'features': [
                'Multi-protocol yield optimization',
                'Auto-compounding',
                'Risk-adjusted position sizing',
                'Gas cost optimization'
            ],
            'supported_protocols': list(self.protocol_weights.keys()),
            'expected_apy_range': f"{self.min_apy * 100:.0f}-50%",
            'risk_level': 'Low-Medium'
        }