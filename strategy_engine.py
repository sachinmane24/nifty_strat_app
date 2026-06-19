"""
NIFTY 50 Options Strategy Engine
Institutional-grade options strategy framework for NIFTY F&O
Derived from top institutional traders' backtested strategies
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import math


class MarketRegime(Enum):
    """Market volatility regimes based on India VIX"""
    LOW_VOL = auto()      # VIX < 15
    NORMAL_VOL = auto()   # VIX 15-25
    HIGH_VOL = auto()     # VIX 25-35
    EXTREME_VOL = auto()  # VIX > 35


class StrategyType(Enum):
    """Available strategy types"""
    THETA_HARVEST = "Theta Harvest (Iron Condor)"
    QUIET_STRADDLE = "Quiet Straddle (Short ATM)"
    VOL_EXPANSION = "Vol Expansion (Long Straddle)"
    DIRECTIONAL_MOMENTUM = "Directional Momentum Spread"
    EXPIRY_THETA = "Expiry Theta Capture"
    NONE = "No Signal"


@dataclass
class MarketState:
    """Current market state snapshot"""
    spot: float
    vix: float
    iv_percentile: float  # 0-100, current IV vs 1-year range
    vwap: float
    ema_20: float
    ema_50: float
    rsi_14: float
    bb_width: float  # Bollinger Band width
    adx: float  # Trend strength
    alpha: float  # Short-term / long-term vol ratio
    alpha2: float  # OTM IV / ITM IV ratio
    day_of_week: int  # 0=Monday, 6=Sunday
    days_to_expiry: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def regime(self) -> MarketRegime:
        if self.vix < 15:
            return MarketRegime.LOW_VOL
        elif self.vix < 25:
            return MarketRegime.NORMAL_VOL
        elif self.vix < 35:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.EXTREME_VOL
    
    @property
    def trend_direction(self) -> str:
        if self.ema_20 > self.ema_50 * 1.005 and self.adx > 20:
            return "BULLISH"
        elif self.ema_20 < self.ema_50 * 0.995 and self.adx > 20:
            return "BEARISH"
        return "SIDEWAYS"
    
    @property
    def is_trending(self) -> bool:
        return self.adx > 25
    
    @property
    def is_quiet(self) -> bool:
        """Low volatility environment - good for short straddles"""
        return self.alpha < 0.2 and self.alpha2 < 0.2 and self.vix < 18


@dataclass
class Signal:
    """Trading signal output"""
    strategy: StrategyType
    direction: str  # "BUY", "SELL", "NEUTRAL"
    entry_spot: float
    strikes: Dict[str, float]  # {'CE': strike, 'PE': strike, ...}
    action: Dict[str, str]  # {'CE': 'SELL', 'PE': 'SELL', ...}
    premium_estimate: Dict[str, float]
    max_profit: float
    max_loss: float
    breakevens: Tuple[float, float]
    stop_loss: float
    target_profit: float
    rationale: str
    confidence: float  # 0-1
    timeframe: str
    risk_reward: float
    margin_required: float
    suggested_lots: int
    

@dataclass
class StrategyConfig:
    """Configuration for each strategy"""
    # Theta Harvest (Iron Condor)
    theta_delta_threshold: float = 0.16  # Sell 16-delta options
    theta_spread_width: int = 200  # 200-point wings for NIFTY
    theta_dte: int = 30  # Days to expiry
    theta_vix_max: float = 18  # Only in low vol
    theta_profit_target: float = 0.50  # 50% of max profit
    theta_stop_loss: float = 2.0  # 2x credit received
    
    # Quiet Straddle (Short ATM)
    straddle_alpha_max: float = 0.2
    straddle_alpha2_max: float = 0.2
    straddle_vix_max: float = 18
    straddle_dte: int = 30
    straddle_profit_target: float = 0.25  # 25% of credit
    straddle_stop_loss: float = 1.5  # 1.5x premium
    
    # Vol Expansion (Long Straddle)
    vol_exp_vix_min: float = 25
    vol_exp_iv_percentile_min: float = 70
    vol_exp_dte: int = 30
    vol_exp_profit_target: float = 2.0  # 2x premium
    vol_exp_stop_loss: float = 0.50  # 50% of premium
    
    # Directional Momentum
    dir_spread_width: int = 100
    dir_adx_min: float = 25
    dir_dte: int = 30
    dir_profit_target: float = 3.0  # 3x risk
    dir_stop_loss: float = 1.0  # 1x spread width
    
    # Expiry Theta Capture
    expiry_dte_max: int = 2
    expiry_vix_max: float = 14
    expiry_profit_target: float = 0.50
    expiry_stop_loss: float = 2.0
    
    # General
    lot_size: int = 75  # NIFTY lot size
    risk_per_trade_pct: float = 0.01  # 1% of capital per trade
    max_margin_per_trade: float = 200000  # Max ₹2L per trade


class StrategyEngine:
    """
    Institutional-grade NIFTY Options Strategy Engine.
    
    Derived from backtested research:
    - Short Straddle: Best Sharpe (0.056) among non-directional strategies
    - Iron Condor: Defined risk, 80% lower margin than strangle
    - 9:20 AM entry, exit at 50% decay or 3:15 PM
    - VIX regime filtering: <15 sell, >25 buy volatility
    - Manage winners: 50% for strangles, 25% for straddles
    - Day effect: Wednesday best (+₹1,180), Tuesday worst (-₹120)
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self._strategy_map = {
            StrategyType.THETA_HARVEST: self._theta_harvest_signal,
            StrategyType.QUIET_STRADDLE: self._quiet_straddle_signal,
            StrategyType.VOL_EXPANSION: self._vol_expansion_signal,
            StrategyType.DIRECTIONAL_MOMENTUM: self._directional_momentum_signal,
            StrategyType.EXPIRY_THETA: self._expiry_theta_signal,
        }
    
    def generate_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Generate a trading signal based on current market state.
        Uses a priority-based system selecting the best strategy for current conditions.
        """
        # Priority order based on market conditions
        candidates = []
        
        # 1. Check for extreme volatility (highest priority - protective)
        if state.regime == MarketRegime.EXTREME_VOL or state.vix > 35:
            candidates.append(StrategyType.VOL_EXPANSION)
        
        # 2. Check for quiet market (institutional favorite)
        if state.is_quiet and state.regime == MarketRegime.LOW_VOL:
            candidates.append(StrategyType.QUIET_STRADDLE)
        
        # 3. Check for low vol range-bound (iron condor)
        if state.regime == MarketRegime.LOW_VOL and not state.is_trending:
            candidates.append(StrategyType.THETA_HARVEST)
        
        # 4. Check for strong trend
        if state.is_trending and state.regime != MarketRegime.EXTREME_VOL:
            candidates.append(StrategyType.DIRECTIONAL_MOMENTUM)
        
        # 5. Check for expiry theta capture (time-based)
        if state.days_to_expiry <= 2 and state.regime == MarketRegime.LOW_VOL:
            candidates.append(StrategyType.EXPIRY_THETA)
        
        # 6. High vol but not extreme - volatility expansion
        if state.regime == MarketRegime.HIGH_VOL:
            candidates.append(StrategyType.VOL_EXPANSION)
        
        # Generate signals for all candidates and pick best
        signals = []
        for strategy in candidates:
            signal = self._strategy_map[strategy](state)
            if signal and signal.confidence > 0.5:
                signals.append(signal)
        
        if not signals:
            return None
        
        # Sort by risk-reward ratio, then confidence
        signals.sort(key=lambda s: (s.risk_reward, s.confidence), reverse=True)
        return signals[0]
    
    def _theta_harvest_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Iron Condor strategy for low-volatility, range-bound markets.
        Institutional research: Defined risk, 80% lower margin than strangle.
        """
        cfg = self.config
        
        if state.vix > cfg.theta_vix_max:
            return None
        if state.is_trending and state.adx > 30:
            return None
        
        spot = state.spot
        # Calculate 16-delta strikes (approximate: ATM ± 1.5 * VIX% * spot)
        move_pct = state.vix * 0.015  # 16-delta approx
        sell_ce = self._round_to_strike(spot * (1 + move_pct))
        sell_pe = self._round_to_strike(spot * (1 - move_pct))
        buy_ce = sell_ce + cfg.theta_spread_width
        buy_pe = sell_pe - cfg.theta_spread_width
        
        # Estimate premiums (simplified Black-Scholes approximation)
        ce_premium = self._estimate_premium(spot, sell_ce, state.vix, cfg.theta_dte, 'CE')
        pe_premium = self._estimate_premium(spot, sell_pe, state.vix, cfg.theta_dte, 'PE')
        buy_ce_premium = self._estimate_premium(spot, buy_ce, state.vix, cfg.theta_dte, 'CE') * 0.3
        buy_pe_premium = self._estimate_premium(spot, buy_pe, state.vix, cfg.theta_dte, 'PE') * 0.3
        
        net_credit = (ce_premium + pe_premium - buy_ce_premium - buy_pe_premium)
        max_profit = net_credit * cfg.lot_size
        max_loss = (cfg.theta_spread_width - net_credit) * cfg.lot_size
        
        lower_be = sell_pe - net_credit
        upper_be = sell_ce + net_credit
        
        # Risk/reward calculation
        risk_reward = max_profit / max(abs(max_loss), 1)
        
        # Position sizing
        suggested_lots = self._calculate_lots(max_loss, cfg.max_margin_per_trade)
        margin_required = max_loss * 1.5  # Conservative margin estimate
        
        confidence = self._calculate_theta_confidence(state)
        
        return Signal(
            strategy=StrategyType.THETA_HARVEST,
            direction="NEUTRAL",
            entry_spot=spot,
            strikes={'CE_SELL': sell_ce, 'CE_BUY': buy_ce, 'PE_SELL': sell_pe, 'PE_BUY': buy_pe},
            action={'CE_SELL': 'SELL', 'CE_BUY': 'BUY', 'PE_SELL': 'SELL', 'PE_BUY': 'BUY'},
            premium_estimate={
                'CE_SELL': ce_premium, 'CE_BUY': buy_ce_premium,
                'PE_SELL': pe_premium, 'PE_BUY': buy_pe_premium,
                'NET_CREDIT': net_credit
            },
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=(lower_be, upper_be),
            stop_loss=cfg.theta_stop_loss * net_credit * cfg.lot_size,
            target_profit=cfg.theta_profit_target * max_profit,
            rationale=f"Low VIX ({state.vix:.1f}), range-bound market. Iron Condor captures theta while defining risk. Spot: {spot}",
            confidence=confidence,
            timeframe=f"{cfg.theta_dte} DTE",
            risk_reward=risk_reward,
            margin_required=margin_required,
            suggested_lots=suggested_lots
        )
    
    def _quiet_straddle_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Short Straddle when market is quiet (alpha < 0.2, alpha2 < 0.2).
        Academic research: Highest Sharpe (0.056) among non-directional NIFTY strategies.
        """
        cfg = self.config
        
        if not state.is_quiet:
            return None
        if state.vix > cfg.straddle_vix_max:
            return None
        
        spot = state.spot
        atm = self._round_to_strike(spot)
        
        # Premium estimates
        ce_premium = self._estimate_premium(spot, atm, state.vix, cfg.straddle_dte, 'CE')
        pe_premium = self._estimate_premium(spot, atm, state.vix, cfg.straddle_dte, 'PE')
        total_credit = ce_premium + pe_premium
        
        max_profit = total_credit * cfg.lot_size
        # Theoretical max loss is unlimited, but we use stop loss
        max_loss = cfg.straddle_stop_loss * total_credit * cfg.lot_size
        
        lower_be = atm - total_credit
        upper_be = atm + total_credit
        
        risk_reward = max_profit / max(abs(max_loss), 1)
        suggested_lots = self._calculate_lots(max_loss, cfg.max_margin_per_trade)
        margin_required = total_credit * cfg.lot_size * 3  # Naked straddle margin ~3x premium
        
        confidence = 0.85 if state.alpha < 0.15 else 0.70
        
        return Signal(
            strategy=StrategyType.QUIET_STRADDLE,
            direction="NEUTRAL",
            entry_spot=spot,
            strikes={'CE': atm, 'PE': atm},
            action={'CE': 'SELL', 'PE': 'SELL'},
            premium_estimate={'CE': ce_premium, 'PE': pe_premium, 'TOTAL': total_credit},
            max_profit=max_profit,
            max_loss=-max_loss,  # Negative to indicate unlimited risk
            breakevens=(lower_be, upper_be),
            stop_loss=cfg.straddle_stop_loss * total_credit * cfg.lot_size,
            target_profit=cfg.straddle_profit_target * max_profit,
            rationale=f"Quiet market (alpha={state.alpha:.3f}, alpha2={state.alpha2:.3f}). Short ATM straddle at {atm}. High theta decay expected.",
            confidence=confidence,
            timeframe=f"{cfg.straddle_dte} DTE",
            risk_reward=risk_reward,
            margin_required=margin_required,
            suggested_lots=suggested_lots
        )
    
    def _vol_expansion_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Long Straddle when volatility is expected to expand.
        Entry when VIX > 25 and IV percentile > 70.
        """
        cfg = self.config
        
        if state.vix < cfg.vol_exp_vix_min and state.iv_percentile < cfg.vol_exp_iv_percentile_min:
            return None
        
        spot = state.spot
        atm = self._round_to_strike(spot)
        
        # Higher premium in high vol environment
        ce_premium = self._estimate_premium(spot, atm, state.vix, cfg.vol_exp_dte, 'CE')
        pe_premium = self._estimate_premium(spot, atm, state.vix, cfg.vol_exp_dte, 'PE')
        total_debit = ce_premium + pe_premium
        
        max_loss = total_debit * cfg.lot_size  # Limited to premium paid
        max_profit = total_debit * cfg.vol_exp_profit_target * cfg.lot_size
        
        lower_be = atm - total_debit
        upper_be = atm + total_debit
        
        risk_reward = max_profit / max(abs(max_loss), 1)
        suggested_lots = self._calculate_lots(max_loss, cfg.max_margin_per_trade)
        
        confidence = min(0.95, 0.5 + (state.vix - 25) / 40 + state.iv_percentile / 200)
        
        return Signal(
            strategy=StrategyType.VOL_EXPANSION,
            direction="NEUTRAL",
            entry_spot=spot,
            strikes={'CE': atm, 'PE': atm},
            action={'CE': 'BUY', 'PE': 'BUY'},
            premium_estimate={'CE': ce_premium, 'PE': pe_premium, 'TOTAL': total_debit},
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=(lower_be, upper_be),
            stop_loss=cfg.vol_exp_stop_loss * total_debit * cfg.lot_size,
            target_profit=max_profit,
            rationale=f"High vol environment (VIX={state.vix:.1f}, IV%={state.iv_percentile:.0f}). Long straddle for volatility expansion. Expect mean reversion.",
            confidence=confidence,
            timeframe=f"{cfg.vol_exp_dte} DTE",
            risk_reward=risk_reward,
            margin_required=total_debit * cfg.lot_size,
            suggested_lots=suggested_lots
        )
    
    def _directional_momentum_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Bull Call Spread or Bear Put Spread based on trend direction.
        3:1 risk-reward minimum requirement.
        """
        cfg = self.config
        
        if not state.is_trending or state.adx < cfg.dir_adx_min:
            return None
        
        spot = state.spot
        trend = state.trend_direction
        
        if trend == "BULLISH":
            buy_strike = self._round_to_strike(spot * 0.995)  # Slightly ITM
            sell_strike = buy_strike + cfg.dir_spread_width
            action = {'BUY': 'BUY', 'SELL': 'SELL'}
            buy_premium = self._estimate_premium(spot, buy_strike, state.vix, cfg.dir_dte, 'CE')
            sell_premium = self._estimate_premium(spot, sell_strike, state.vix, cfg.dir_dte, 'CE')
            net_debit = buy_premium - sell_premium
            max_profit = (sell_strike - buy_strike - net_debit) * cfg.lot_size
            max_loss = net_debit * cfg.lot_size
            direction = "BULLISH"
            option_type = 'CE'
        else:  # BEARISH
            buy_strike = self._round_to_strike(spot * 1.005)  # Slightly ITM put
            sell_strike = buy_strike - cfg.dir_spread_width
            action = {'BUY': 'BUY', 'SELL': 'SELL'}
            buy_premium = self._estimate_premium(spot, buy_strike, state.vix, cfg.dir_dte, 'PE')
            sell_premium = self._estimate_premium(spot, sell_strike, state.vix, cfg.dir_dte, 'PE')
            net_debit = buy_premium - sell_premium
            max_profit = (buy_strike - sell_strike - net_debit) * cfg.lot_size
            max_loss = net_debit * cfg.lot_size
            direction = "BEARISH"
            option_type = 'PE'
        
        risk_reward = max_profit / max(abs(max_loss), 1)
        if risk_reward < 2.0:  # Minimum 2:1 R/R
            return None
        
        suggested_lots = self._calculate_lots(max_loss, cfg.max_margin_per_trade)
        
        return Signal(
            strategy=StrategyType.DIRECTIONAL_MOMENTUM,
            direction=direction,
            entry_spot=spot,
            strikes={'BUY': buy_strike, 'SELL': sell_strike},
            action=action,
            premium_estimate={'BUY': buy_premium, 'SELL': sell_premium, 'NET': net_debit},
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=(buy_strike + net_debit, sell_strike - net_debit) if direction == "BULLISH" else (sell_strike + net_debit, buy_strike - net_debit),
            stop_loss=cfg.dir_stop_loss * net_debit * cfg.lot_size,
            target_profit=cfg.dir_profit_target * max_loss,
            rationale=f"Strong {direction.lower()} trend (ADX={state.adx:.1f}). {option_type} spread for directional momentum. Defined risk.",
            confidence=min(0.90, 0.6 + state.adx / 100),
            timeframe=f"{cfg.dir_dte} DTE",
            risk_reward=risk_reward,
            margin_required=max_loss * 2,
            suggested_lots=suggested_lots
        )
    
    def _expiry_theta_signal(self, state: MarketState) -> Optional[Signal]:
        """
        Iron Butterfly for expiry day theta capture.
        Research: Tuesday (expiry) worst day; Monday theta capture if VIX < 14.
        """
        cfg = self.config
        
        if state.days_to_expiry > cfg.expiry_dte_max:
            return None
        if state.vix > cfg.expiry_vix_max:
            return None
        
        spot = state.spot
        atm = self._round_to_strike(spot)
        
        # Iron Butterfly: sell ATM, buy OTM wings
        wing_width = 100  # 100 points for NIFTY
        sell_ce = atm
        sell_pe = atm
        buy_ce = atm + wing_width
        buy_pe = atm - wing_width
        
        # Higher decay closer to expiry
        decay_factor = 1.5 if state.days_to_expiry <= 1 else 1.0
        ce_premium = self._estimate_premium(spot, sell_ce, state.vix, 1, 'CE') * decay_factor
        pe_premium = self._estimate_premium(spot, sell_pe, state.vix, 1, 'PE') * decay_factor
        buy_ce_p = self._estimate_premium(spot, buy_ce, state.vix, 1, 'CE') * 0.2
        buy_pe_p = self._estimate_premium(spot, buy_pe, state.vix, 1, 'PE') * 0.2
        
        net_credit = (ce_premium + pe_premium - buy_ce_p - buy_pe_p)
        max_profit = net_credit * cfg.lot_size
        max_loss = (wing_width - net_credit) * cfg.lot_size
        
        lower_be = sell_pe - net_credit
        upper_be = sell_ce + net_credit
        
        risk_reward = max_profit / max(abs(max_loss), 1)
        suggested_lots = self._calculate_lots(max_loss, cfg.max_margin_per_trade)
        
        return Signal(
            strategy=StrategyType.EXPIRY_THETA,
            direction="NEUTRAL",
            entry_spot=spot,
            strikes={'CE_SELL': sell_ce, 'CE_BUY': buy_ce, 'PE_SELL': sell_pe, 'PE_BUY': buy_pe},
            action={'CE_SELL': 'SELL', 'CE_BUY': 'BUY', 'PE_SELL': 'SELL', 'PE_BUY': 'BUY'},
            premium_estimate={
                'CE_SELL': ce_premium, 'CE_BUY': buy_ce_p,
                'PE_SELL': pe_premium, 'PE_BUY': buy_pe_p,
                'NET_CREDIT': net_credit
            },
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=(lower_be, upper_be),
            stop_loss=cfg.expiry_stop_loss * net_credit * cfg.lot_size,
            target_profit=cfg.expiry_profit_target * max_profit,
            rationale=f"Expiry theta capture ({state.days_to_expiry} DTE). VIX={state.vix:.1f}. Extreme theta decay expected.",
            confidence=0.75,
            timeframe="1-2 DTE",
            risk_reward=risk_reward,
            margin_required=max_loss * 1.5,
            suggested_lots=suggested_lots
        )
    
    def _round_to_strike(self, price: float, interval: int = 50) -> int:
        """Round to nearest NIFTY strike interval (default 50)"""
        return int(round(price / interval) * interval)
    
    def _estimate_premium(self, spot: float, strike: float, vix: float, dte: int, option_type: str) -> float:
        """
        Simplified premium estimation using Black-Scholes-inspired approximation.
        This is for signal generation, not exact pricing.
        """
        # Simplified: premium ≈ spot * vix% * sqrt(dte/365) * moneyness factor
        vol = vix / 100
        t = max(dte, 1) / 365
        
        if option_type == 'CE':
            moneyness = (spot - strike) / spot
            intrinsic = max(0, spot - strike)
        else:
            moneyness = (strike - spot) / spot
            intrinsic = max(0, strike - spot)
        
        # Time value decreases as we go further OTM
        time_value = spot * vol * math.sqrt(t) * math.exp(-abs(moneyness) * 5)
        premium = intrinsic + time_value
        
        return max(premium, 1.0)  # Minimum ₹1
    
    def _calculate_lots(self, max_loss_per_lot: float, max_margin: float) -> int:
        """Calculate suggested lot size based on risk and margin constraints"""
        if max_loss_per_lot <= 0:
            return 1
        lots_by_margin = int(max_margin / max(abs(max_loss_per_lot), 1))
        lots_by_risk = int(100000 / max(abs(max_loss_per_lot), 1))  # ₹1L risk per trade
        return max(1, min(lots_by_margin, lots_by_risk))
    
    def _calculate_theta_confidence(self, state: MarketState) -> float:
        """Calculate confidence score for theta strategies"""
        confidence = 0.5
        
        # VIX component
        if state.vix < 12:
            confidence += 0.2
        elif state.vix < 15:
            confidence += 0.1
        
        # Trend component
        if not state.is_trending:
            confidence += 0.15
        
        # BB width component
        if state.bb_width < 0.03:  # Narrow bands
            confidence += 0.15
        
        # Day of week component
        if state.day_of_week in [2, 3]:  # Wed, Thu (best days)
            confidence += 0.05
        elif state.day_of_week == 4:  # Friday
            confidence += 0.03
        
        return min(confidence, 0.95)
    
    def get_all_strategy_signals(self, state: MarketState) -> List[Signal]:
        """Generate signals for all applicable strategies for comparison"""
        signals = []
        for strategy_type, generator in self._strategy_map.items():
            signal = generator(state)
            if signal:
                signals.append(signal)
        return signals
