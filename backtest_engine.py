"""
Backtesting Engine for NIFTY Options Strategies
Simulates historical performance with realistic assumptions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from strategy_engine import StrategyEngine, MarketState, Signal, StrategyType, StrategyConfig


@dataclass
class TradeResult:
    """Result of a single trade"""
    entry_date: datetime
    exit_date: datetime
    strategy: StrategyType
    entry_spot: float
    exit_spot: float
    pnl: float
    max_pnl: float
    max_drawdown: float
    exit_reason: str  # 'TARGET', 'STOP_LOSS', 'EXPIRY', 'REGIME_CHANGE'
    lots: int
    margin_used: float
    

@dataclass
class BacktestResults:
    """Complete backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    equity_curve: pd.DataFrame
    trade_log: List[TradeResult]
    monthly_returns: pd.Series
    strategy_breakdown: Dict[str, Dict]
    
    @property
    def return_pct(self) -> float:
        if len(self.equity_curve) > 0:
            return (self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0] - 1) * 100
        return 0


class BacktestEngine:
    """
    Backtesting engine that simulates strategy performance on historical NIFTY data.
    Uses realistic assumptions for slippage, brokerage, and execution.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.engine = StrategyEngine(config)
        
        # Realistic assumptions for Indian markets
        self.brokerage_per_lot = 40  # â‚¹40 per lot (Zerodha-like)
        self.stt_per_lot = 20  # Securities Transaction Tax
        self.transaction_charges = 10  # Exchange charges
        self.gst_pct = 0.18  # 18% GST on brokerage + charges
        self.slippage = 0.0005  # 0.05% slippage
        
    def run_backtest(self, 
                     historical_data: pd.DataFrame,
                     vix_data: pd.DataFrame,
                     initial_capital: float = 500000,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     forced_strategy: Optional[StrategyType] = None) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            historical_data: DataFrame with columns [date, open, high, low, close, volume]
            vix_data: DataFrame with columns [date, vix]
            initial_capital: Starting capital
        """
        # Prepare data
        data = historical_data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
        
        # Merge VIX data
        if vix_data is not None and len(vix_data) > 0:
            vix_data = vix_data.copy()
            vix_data['date'] = pd.to_datetime(vix_data['date'])
            data = data.merge(vix_data[['date', 'vix']], on='date', how='left')
            data['vix'] = data['vix'].ffill().fillna(15)
        else:
            data['vix'] = 15  # Default VIX
        
        # Calculate technical indicators
        data = self._calculate_indicators(data)
        
        # Simulate trading
        capital = initial_capital
        equity = [capital]
        dates = [data['date'].iloc[0]]
        trade_log = []
        active_signal = None
        active_trade = None
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_date = row['date']
            
            # Check if active trade needs to be closed
            if active_trade is not None:
                result = self._check_trade_exit(active_trade, active_signal, row, capital)
                if result is not None:
                    # Close trade
                    pnl_after_costs = result.pnl - self._calculate_costs(active_signal)
                    capital += pnl_after_costs
                    trade_log.append(result)
                    active_trade = None
                    active_signal = None
            
            # Generate new signal if no active trade
            if active_trade is None:
                state = self._create_market_state(row, i)
                if forced_strategy is not None:
                    signal = self.engine._strategy_map[forced_strategy](state)
                else:
                    signal = self.engine.generate_signal(state)
                
                if signal is not None and signal.confidence > 0.6:
                    # Check if we have enough capital
                    margin_needed = signal.margin_required * signal.suggested_lots
                    if capital >= margin_needed * 0.5:  # Need 50% of margin
                        active_signal = signal
                        active_trade = {
                            'entry_date': current_date,
                            'entry_spot': row['close'],
                            'entry_price': row['close'],
                            'signal': signal,
                            'max_pnl': 0,
                            'max_dd': 0,
                            'lots': signal.suggested_lots
                        }
            
            equity.append(capital)
            dates.append(current_date)
        
        # Close any open trade at the end
        if active_trade is not None:
            last_row = data.iloc[-1]
            result = self._force_close_trade(active_trade, active_signal, last_row, capital)
            if result:
                pnl_after_costs = result.pnl - self._calculate_costs(active_signal)
                capital += pnl_after_costs
                trade_log.append(result)
        
        # Build results
        equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity
        })
        
        return self._calculate_metrics(trade_log, equity_curve, initial_capital)
    
    def run_strategy_comparison(self,
                                historical_data: pd.DataFrame,
                                vix_data: pd.DataFrame,
                                initial_capital: float = 500000,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, BacktestResults]:
        """Run each strategy independently over the same historical window."""
        results = {}
        for strategy in [
            StrategyType.THETA_HARVEST,
            StrategyType.QUIET_STRADDLE,
            StrategyType.VOL_EXPANSION,
            StrategyType.DIRECTIONAL_MOMENTUM,
            StrategyType.EXPIRY_THETA,
        ]:
            results[strategy.value] = self.run_backtest(
                historical_data,
                vix_data,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                forced_strategy=strategy,
            )
        return results
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for market state"""
        # EMAs
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_mid'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * bb_std
        data['bb_lower'] = data['bb_mid'] - 2 * bb_std
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_mid']
        
        # ADX (simplified)
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(window=14).mean()
        data['adx'] = 20 + (data['atr'] / data['close'] * 100) * 2  # Simplified ADX proxy
        
        # VWAP (daily reset simulation)
        data['vwap'] = data['close']  # Simplified
        
        # IV Percentile (simplified)
        data['iv_percentile'] = data['vix'].rolling(window=252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )
        
        # Alpha (vol ratio) - simplified
        data['alpha'] = 0.15 + np.random.normal(0, 0.05, len(data))  # Simulated
        data['alpha'] = data['alpha'].clip(0.05, 0.5)
        data['alpha2'] = 0.15 + np.random.normal(0, 0.05, len(data))  # Simulated
        data['alpha2'] = data['alpha2'].clip(0.05, 0.5)
        
        return data
    
    def _create_market_state(self, row: pd.Series, index: int) -> MarketState:
        """Create MarketState from data row"""
        return MarketState(
            spot=row['close'],
            vix=row.get('vix', 15),
            iv_percentile=row.get('iv_percentile', 50),
            vwap=row.get('vwap', row['close']),
            ema_20=row.get('ema_20', row['close']),
            ema_50=row.get('ema_50', row['close']),
            rsi_14=row.get('rsi_14', 50),
            bb_width=row.get('bb_width', 0.03),
            adx=row.get('adx', 15),
            alpha=row.get('alpha', 0.15),
            alpha2=row.get('alpha2', 0.15),
            day_of_week=pd.to_datetime(row['date']).weekday(),
            days_to_expiry=7 - (pd.to_datetime(row['date']).weekday() % 7),  # Weekly expiry
            timestamp=pd.to_datetime(row['date'])
        )
    
    def _check_trade_exit(self, trade: Dict, signal: Signal, row: pd.Series, capital: float) -> Optional[TradeResult]:
        """Check if trade should be exited based on rules"""
        entry_spot = trade['entry_spot']
        current_spot = row['close']
        lots = trade['lots']
        
        days_held = (pd.to_datetime(row['date']) - pd.to_datetime(trade['entry_date'])).days
        
        # Calculate unrealized P&L based on strategy type
        pnl = self._calculate_unrealized_pnl(signal, entry_spot, current_spot, lots, days_held)
        
        # Track max P&L and drawdown
        trade['max_pnl'] = max(trade['max_pnl'], pnl)
        trade['max_dd'] = min(trade['max_dd'], pnl)
        
        # Check target
        if pnl >= signal.target_profit:
            return TradeResult(
                entry_date=trade['entry_date'],
                exit_date=row['date'],
                strategy=signal.strategy,
                entry_spot=entry_spot,
                exit_spot=current_spot,
                pnl=pnl,
                max_pnl=trade['max_pnl'],
                max_drawdown=trade['max_dd'],
                exit_reason='TARGET',
                lots=lots,
                margin_used=signal.margin_required
            )
        
        # Check stop loss
        if pnl <= -signal.stop_loss:
            return TradeResult(
                entry_date=trade['entry_date'],
                exit_date=row['date'],
                strategy=signal.strategy,
                entry_spot=entry_spot,
                exit_spot=current_spot,
                pnl=pnl,
                max_pnl=trade['max_pnl'],
                max_drawdown=trade['max_dd'],
                exit_reason='STOP_LOSS',
                lots=lots,
                margin_used=signal.margin_required
            )
        
        # Check regime change (for short vol strategies)
        if signal.strategy in [StrategyType.THETA_HARVEST, StrategyType.QUIET_STRADDLE]:
            if row.get('vix', 15) > 25:  # VIX spike - exit
                return TradeResult(
                    entry_date=trade['entry_date'],
                    exit_date=row['date'],
                    strategy=signal.strategy,
                    entry_spot=entry_spot,
                    exit_spot=current_spot,
                    pnl=pnl,
                    max_pnl=trade['max_pnl'],
                    max_drawdown=trade['max_dd'],
                    exit_reason='REGIME_CHANGE',
                    lots=lots,
                    margin_used=signal.margin_required
                )
        
        # Time-based exit (DTE expired)
        if days_held >= 30:
            return TradeResult(
                entry_date=trade['entry_date'],
                exit_date=row['date'],
                strategy=signal.strategy,
                entry_spot=entry_spot,
                exit_spot=current_spot,
                pnl=pnl,
                max_pnl=trade['max_pnl'],
                max_drawdown=trade['max_dd'],
                exit_reason='EXPIRY',
                lots=lots,
                margin_used=signal.margin_required
            )
        
        return None
    
    def _calculate_unrealized_pnl(self, signal: Signal, entry_spot: float, current_spot: float, lots: int, days_held: int = 0) -> float:
        """Calculate unrealized P&L for a strategy"""
        cfg = self.config
        
        if signal.strategy == StrategyType.THETA_HARVEST:
            # Iron Condor: profit if within breakevens
            lower, upper = signal.breakevens
            if lower <= current_spot <= upper:
                # Decay profit based on holding time (simplified daily model)
                decay_capture = min(max(days_held, 0) / max(self.config.theta_dte, 1), 1.0)
                return signal.max_profit * decay_capture * lots
            elif current_spot < lower:
                loss = (lower - current_spot) * lots
                return -min(loss, abs(signal.max_loss))
            else:
                loss = (current_spot - upper) * lots
                return -min(loss, abs(signal.max_loss))
        
        elif signal.strategy == StrategyType.QUIET_STRADDLE:
            # Short Straddle: profit if near ATM
            atm = list(signal.strikes.values())[0]
            move = abs(current_spot - atm)
            total_credit = signal.premium_estimate.get('TOTAL', 100)
            if move < total_credit:
                return (total_credit - move) * lots * cfg.lot_size
            else:
                return -(move - total_credit) * lots * cfg.lot_size
        
        elif signal.strategy == StrategyType.VOL_EXPANSION:
            # Long Straddle: profit on large moves
            atm = list(signal.strikes.values())[0]
            move = abs(current_spot - atm)
            total_debit = signal.premium_estimate.get('TOTAL', 100)
            return (move - total_debit) * lots * cfg.lot_size
        
        elif signal.strategy == StrategyType.DIRECTIONAL_MOMENTUM:
            # Bull/Bear spread
            buy_strike = signal.strikes['BUY']
            sell_strike = signal.strikes['SELL']
            if signal.direction == 'BULLISH':
                pnl = (current_spot - buy_strike) * lots * cfg.lot_size
                pnl = max(0, min(pnl, (sell_strike - buy_strike) * lots * cfg.lot_size))
            else:
                pnl = (buy_strike - current_spot) * lots * cfg.lot_size
                pnl = max(0, min(pnl, (buy_strike - sell_strike) * lots * cfg.lot_size))
            return pnl - signal.premium_estimate.get('NET', 0) * lots * cfg.lot_size
        
        elif signal.strategy == StrategyType.EXPIRY_THETA:
            # Iron Butterfly
            lower, upper = signal.breakevens
            if lower <= current_spot <= upper:
                decay_capture = min(max(days_held, 0) / max(self.config.expiry_dte_max, 1), 1.0)
                return signal.max_profit * decay_capture * lots
            elif current_spot < lower:
                return -(lower - current_spot) * lots
            else:
                return -(current_spot - upper) * lots
        
        return 0
    
    def _force_close_trade(self, trade: Dict, signal: Signal, row: pd.Series, capital: float) -> Optional[TradeResult]:
        """Force close trade at end of backtest period"""
        days_held = (pd.to_datetime(row['date']) - pd.to_datetime(trade['entry_date'])).days
        pnl = self._calculate_unrealized_pnl(signal, trade['entry_spot'], row['close'], trade['lots'], days_held)
        return TradeResult(
            entry_date=trade['entry_date'],
            exit_date=row['date'],
            strategy=signal.strategy,
            entry_spot=trade['entry_spot'],
            exit_spot=row['close'],
            pnl=pnl,
            max_pnl=trade['max_pnl'],
            max_drawdown=trade['max_dd'],
            exit_reason='EXPIRY',
            lots=trade['lots'],
            margin_used=signal.margin_required
        )
    
    def _calculate_costs(self, signal: Signal) -> float:
        """Calculate transaction costs for a trade"""
        lots = signal.suggested_lots
        legs = len(signal.action)
        
        # Entry + Exit costs
        brokerage = self.brokerage_per_lot * lots * 2  # Entry + exit
        stt = self.stt_per_lot * lots * 2
        trans = self.transaction_charges * lots * 2
        
        subtotal = brokerage + stt + trans
        gst = subtotal * self.gst_pct
        
        # Slippage on premium
        total_premium = sum(signal.premium_estimate.values()) * lots * self.config.lot_size * self.slippage * 2
        
        return brokerage + stt + trans + gst + total_premium
    
    def _calculate_metrics(self, trade_log: List[TradeResult], equity_curve: pd.DataFrame, initial_capital: float) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        if not trade_log:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_pnl=0, avg_win=0, avg_loss=0,
                profit_factor=0, max_drawdown=0, max_drawdown_pct=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                equity_curve=equity_curve, trade_log=[], monthly_returns=pd.Series(),
                strategy_breakdown={}
            )
        
        pnls = [t.pnl for t in trade_log]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        total_trades = len(trade_log)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        # Calculate drawdown from equity curve
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = equity_curve['equity'] - equity_curve['peak']
        equity_curve['drawdown_pct'] = equity_curve['drawdown'] / equity_curve['peak'] * 100
        
        max_drawdown = equity_curve['drawdown'].min()
        max_drawdown_pct = equity_curve['drawdown_pct'].min()
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
        daily_returns = equity_curve['returns']
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation only)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
        else:
            sortino = 0
        
        # Calmar ratio
        max_dd_abs = abs(max_drawdown)
        if max_dd_abs > 0:
            calmar = (total_pnl / initial_capital) / (max_dd_abs / initial_capital)
        else:
            calmar = 0
        
        # Monthly returns
        equity_curve['month'] = pd.to_datetime(equity_curve['date']).dt.to_period('M')
        monthly = equity_curve.groupby('month')['equity'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        
        # Strategy breakdown
        strategy_stats = {}
        for strategy in StrategyType:
            strategy_trades = [t for t in trade_log if t.strategy == strategy]
            if strategy_trades:
                strategy_pnls = [t.pnl for t in strategy_trades]
                strategy_stats[strategy.value] = {
                    'trades': len(strategy_trades),
                    'win_rate': len([p for p in strategy_pnls if p > 0]) / len(strategy_pnls),
                    'total_pnl': sum(strategy_pnls),
                    'avg_pnl': sum(strategy_pnls) / len(strategy_pnls)
                }
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            equity_curve=equity_curve,
            trade_log=trade_log,
            monthly_returns=monthly,
            strategy_breakdown=strategy_stats
        )
    
    def generate_simulated_data(self, 
                                days: int = 500,
                                start_price: float = 23500,
                                volatility: float = 0.015) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate simulated NIFTY-like historical data for backtesting.
        Uses geometric Brownian motion with regime switching.
        """
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Regime switching: low vol (60%), high vol (20%), normal (20%)
        regimes = np.random.choice([0, 1, 2], size=days, p=[0.6, 0.2, 0.2])
        vol_multipliers = np.where(regimes == 0, 0.7, np.where(regimes == 1, 2.0, 1.0))
        
        # Price generation with drift
        returns = np.random.normal(0.0003, volatility * vol_multipliers, days)
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        daily_vol = volatility * vol_multipliers * prices
        highs = prices + np.abs(np.random.normal(0, daily_vol * 0.5, days))
        lows = prices - np.abs(np.random.normal(0, daily_vol * 0.5, days))
        opens = prices + np.random.normal(0, daily_vol * 0.3, days)
        volumes = np.random.randint(5000000, 15000000, days)
        
        # VIX generation (correlated with volatility)
        vix_base = 15 + (vol_multipliers - 0.7) * 10 + np.random.normal(0, 2, days)
        vix = np.clip(vix_base, 10, 50)
        
        nifty_data = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        vix_data = pd.DataFrame({
            'date': dates,
            'vix': vix
        })
        
        return nifty_data, vix_data



