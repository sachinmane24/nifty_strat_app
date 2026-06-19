"""
Risk Management Module for NIFTY Options Trading
Position sizing, portfolio heat, and capital protection
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from strategy_engine import Signal, StrategyType


@dataclass
class RiskParameters:
    """Risk management configuration"""
    max_portfolio_risk_pct: float = 0.02  # Max 2% portfolio risk per trade
    max_open_positions: int = 3  # Max concurrent positions
    max_correlated_positions: int = 2  # Max correlated positions
    max_strategy_concentration_pct: float = 0.50  # Max 50% in one strategy type
    max_daily_loss_pct: float = 0.03  # Max 3% daily loss
    max_drawdown_pct: float = 0.15  # Stop trading at 15% drawdown
    kelly_fraction: float = 0.25  # Use 25% of Kelly criterion
    min_risk_reward: float = 1.5  # Minimum 1.5:1 R/R
    
    # Strategy-specific limits
    max_short_vol_risk: float = 0.015  # Max 1.5% for short vol strategies
    max_long_vol_risk: float = 0.01  # Max 1% for long vol strategies
    max_directional_risk: float = 0.015  # Max 1.5% for directional


class RiskManager:
    """
    Institutional-grade risk management for NIFTY options trading.
    Implements position sizing, portfolio heat monitoring, and drawdown controls.
    """
    
    def __init__(self, risk_params: Optional[RiskParameters] = None):
        self.params = risk_params or RiskParameters()
        self.daily_pnl = 0
        self.daily_trades = 0
        self.current_drawdown = 0
        self.trading_enabled = True
        self.position_history = []
    
    def calculate_position_size(self, signal: Signal, capital: float) -> int:
        """
        Calculate optimal position size using Kelly Criterion + risk limits.
        
        Kelly Formula: f* = (p*b - q) / b
        Where p = win rate, b = avg win / avg loss, q = 1-p
        """
        if not self.trading_enabled:
            return 0
        
        # Get strategy-specific risk limit
        risk_limit = self._get_strategy_risk_limit(signal.strategy)
        max_risk_amount = capital * risk_limit
        
        # Estimate win rate based on strategy historical performance
        estimated_win_rate = self._estimate_win_rate(signal.strategy)
        
        # Estimate payoff ratio
        avg_win = signal.max_profit
        avg_loss = abs(signal.max_loss) if signal.max_loss < 0 else signal.max_loss
        payoff_ratio = avg_win / max(avg_loss, 1)
        
        # Kelly Criterion
        q = 1 - estimated_win_rate
        kelly = (estimated_win_rate * payoff_ratio - q) / max(payoff_ratio, 0.1)
        kelly = max(0, kelly)  # Kelly can't be negative
        
        # Fractional Kelly (conservative)
        fractional_kelly = kelly * self.params.kelly_fraction
        
        # Risk-based lot sizing
        if avg_loss > 0:
            lots_by_risk = int(max_risk_amount / avg_loss)
        else:
            lots_by_risk = 1
        
        # Kelly-based lot sizing
        kelly_amount = capital * fractional_kelly
        if signal.margin_required > 0:
            lots_by_kelly = int(kelly_amount / signal.margin_required)
        else:
            lots_by_kelly = 1
        
        # Take the more conservative of the two
        suggested_lots = min(lots_by_risk, lots_by_kelly, signal.suggested_lots)
        
        # Ensure at least 1 lot
        return max(1, suggested_lots)
    
    def validate_trade(self, signal: Signal, capital: float, open_positions: List[Dict]) -> Dict:
        """
        Validate if a trade should be taken based on risk rules.
        Returns validation result with approval status and reasons.
        """
        reasons = []
        approved = True
        
        # Check trading enabled (drawdown limit)
        if not self.trading_enabled:
            approved = False
            reasons.append("Trading halted due to drawdown limit")
        
        # Check max open positions
        if len(open_positions) >= self.params.max_open_positions:
            approved = False
            reasons.append(f"Max open positions ({self.params.max_open_positions}) reached")
        
        # Check risk/reward
        if signal.risk_reward < self.params.min_risk_reward:
            approved = False
            reasons.append(f"Risk/Reward {signal.risk_reward:.2f} below minimum {self.params.min_risk_reward}")
        
        # Check confidence threshold
        if signal.confidence < 0.6:
            approved = False
            reasons.append(f"Confidence {signal.confidence:.2f} below threshold 0.60")
        
        # Check strategy concentration
        strategy_count = sum(1 for p in open_positions if p.get('strategy') == signal.strategy.value)
        total_positions = len(open_positions) + 1
        if total_positions > 0:
            strategy_pct = strategy_count / total_positions
            if strategy_pct > self.params.max_strategy_concentration_pct:
                approved = False
                reasons.append(f"Strategy concentration would exceed {self.params.max_strategy_concentration_pct*100:.0f}%")
        
        # Check capital adequacy
        margin_required = signal.margin_required * signal.suggested_lots
        if margin_required > capital * 0.5:
            approved = False
            reasons.append(f"Margin required ({margin_required:,.0f}) exceeds 50% of capital")
        
        # Check daily loss limit
        if self.daily_pnl < -capital * self.params.max_daily_loss_pct:
            approved = False
            reasons.append("Daily loss limit reached")
        
        # Calculate position size if approved
        if approved:
            lots = self.calculate_position_size(signal, capital)
            if lots == 0:
                approved = False
                reasons.append("Position size calculated as 0 (risk limit)")
        else:
            lots = 0
        
        return {
            'approved': approved,
            'lots': lots,
            'reasons': reasons,
            'max_risk_amount': capital * self._get_strategy_risk_limit(signal.strategy),
            'margin_required': signal.margin_required * lots if lots > 0 else 0
        }
    
    def update_after_trade(self, pnl: float, capital: float):
        """Update risk metrics after trade close"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.position_history.append({
            'pnl': pnl,
            'capital': capital,
            'drawdown': self.current_drawdown
        })
        
        # Update drawdown
        peak = max(p['capital'] + p['pnl'] for p in self.position_history) if self.position_history else capital
        self.current_drawdown = (capital - peak) / peak if peak > 0 else 0
        
        # Check drawdown limit
        if abs(self.current_drawdown) > self.params.max_drawdown_pct:
            self.trading_enabled = False
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
        self.daily_trades = 0
    
    def get_portfolio_heat(self, open_positions: List[Dict], capital: float) -> Dict:
        """
        Calculate portfolio heat (total risk exposure).
        """
        total_margin = sum(p.get('margin', 0) for p in open_positions)
        total_risk = sum(abs(p.get('max_loss', 0)) for p in open_positions)
        
        return {
            'total_margin': total_margin,
            'margin_pct': total_margin / capital if capital > 0 else 0,
            'total_risk': total_risk,
            'risk_pct': total_risk / capital if capital > 0 else 0,
            'open_positions': len(open_positions),
            'available_margin': capital * 0.8 - total_margin,  # 80% of capital
            'heat_color': 'green' if total_risk / capital < 0.02 else 'yellow' if total_risk / capital < 0.04 else 'red'
        }
    
    def _get_strategy_risk_limit(self, strategy: StrategyType) -> float:
        """Get risk limit for specific strategy type"""
        if strategy in [StrategyType.THETA_HARVEST, StrategyType.QUIET_STRADDLE]:
            return self.params.max_short_vol_risk
        elif strategy == StrategyType.VOL_EXPANSION:
            return self.params.max_long_vol_risk
        elif strategy == StrategyType.DIRECTIONAL_MOMENTUM:
            return self.params.max_directional_risk
        else:
            return self.params.max_portfolio_risk_pct
    
    def _estimate_win_rate(self, strategy: StrategyType) -> float:
        """Estimate historical win rate for strategy type (from backtest research)"""
        win_rates = {
            StrategyType.THETA_HARVEST: 0.72,  # Iron condors have high win rate
            StrategyType.QUIET_STRADDLE: 0.65,  # Short straddles ~65% win rate
            StrategyType.VOL_EXPANSION: 0.45,  # Long vol lower win rate but higher payoff
            StrategyType.DIRECTIONAL_MOMENTUM: 0.55,  # Directional spreads
            StrategyType.EXPIRY_THETA: 0.68,  # Expiry theta capture
        }
        return win_rates.get(strategy, 0.55)
    
    def get_risk_report(self, capital: float, open_positions: List[Dict]) -> Dict:
        """Generate comprehensive risk report"""
        heat = self.get_portfolio_heat(open_positions, capital)
        
        return {
            'trading_enabled': self.trading_enabled,
            'current_drawdown_pct': self.current_drawdown * 100,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'portfolio_heat': heat,
            'drawdown_status': 'CRITICAL' if abs(self.current_drawdown) > 0.15 else 'WARNING' if abs(self.current_drawdown) > 0.08 else 'NORMAL',
            'recommendation': self._get_recommendation(capital, heat)
        }
    
    def _get_recommendation(self, capital: float, heat: Dict) -> str:
        """Generate trading recommendation based on risk state"""
        if not self.trading_enabled:
            return "STOP TRADING - Drawdown limit exceeded. Reduce size or take break."
        
        if heat['risk_pct'] > 0.04:
            return "REDUCE SIZE - Portfolio heat is high. Close some positions."
        
        if heat['risk_pct'] > 0.02:
            return "CAUTION - Monitor closely. No new directional positions."
        
        if self.daily_pnl < -capital * 0.01:
            return "DEFENSIVE - Daily loss accumulating. Reduce size by 50%."
        
        return "NORMAL - All systems green. Trade with standard sizing."
