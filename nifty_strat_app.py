"""
NIFTY 50 Options Strategy Signal App
Main Streamlit application for institutional-grade NIFTY F&O trading signals

Usage:
    streamlit run nifty_strat_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_engine import StrategyEngine, MarketState, Signal, StrategyType, StrategyConfig
from backtest_engine import BacktestEngine, BacktestResults
from risk_manager import RiskManager, RiskParameters
from data_fetcher import NiftyDataFetcher

# Page configuration
st.set_page_config(
    page_title="NIFTY 50 Options Strategy Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5a7a9a;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .signal-card {
        background: #f0f7ff;
        border-left: 4px solid #1f4e79;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .bullish { color: #00c853; font-weight: bold; }
    .bearish { color: #ff1744; font-weight: bold; }
    .neutral { color: #ff9100; font-weight: bold; }
    .profit { color: #00c853; }
    .loss { color: #ff1744; }
    .strategy-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .strategy-theta { background: #e3f2fd; color: #1565c0; }
    .strategy-vol { background: #fce4ec; color: #c62828; }
    .strategy-dir { background: #e8f5e9; color: #2e7d32; }
    .strategy-expiry { background: #fff3e0; color: #ef6c00; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'engine' not in st.session_state:
        st.session_state.engine = StrategyEngine()
    if 'backtest_engine' not in st.session_state:
        st.session_state.backtest_engine = BacktestEngine()
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager()
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = NiftyDataFetcher()
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'capital' not in st.session_state:
        st.session_state.capital = 500000
    if 'open_positions' not in st.session_state:
        st.session_state.open_positions = []


def get_strategy_badge_class(strategy: StrategyType) -> str:
    """Get CSS class for strategy badge"""
    if strategy in [StrategyType.THETA_HARVEST, StrategyType.QUIET_STRADDLE]:
        return 'strategy-theta'
    elif strategy == StrategyType.VOL_EXPANSION:
        return 'strategy-vol'
    elif strategy == StrategyType.DIRECTIONAL_MOMENTUM:
        return 'strategy-dir'
    else:
        return 'strategy-expiry'


def render_header():
    """Render the app header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📈 NIFTY 50 Options Strategy Terminal</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Institutional-Grade F&O Signal Generation & Backtesting</p>', unsafe_allow_html=True)
    with col2:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))
        st.metric("Market Status", "CLOSED" if datetime.now().hour < 9 or datetime.now().hour > 15 else "OPEN")


def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Capital
    st.session_state.capital = st.sidebar.number_input(
        "Trading Capital (₹)",
        min_value=100000,
        max_value=10000000,
        value=st.session_state.capital,
        step=50000
    )
    
    st.sidebar.markdown("---")
    
    # Strategy Configuration
    st.sidebar.markdown("### Strategy Settings")
    
    with st.sidebar.expander("Theta Harvest (Iron Condor)", expanded=False):
        theta_vix = st.slider("Max VIX", 10, 25, 18)
        theta_spread = st.slider("Spread Width", 100, 500, 200, 50)
        theta_dte = st.slider("Days to Expiry", 15, 60, 30)
        theta_profit = st.slider("Profit Target (%)", 25, 75, 50)
    
    with st.sidebar.expander("Quiet Straddle", expanded=False):
        straddle_vix = st.slider("Max VIX for Straddle", 10, 25, 18)
        straddle_sl = st.slider("Stop Loss (x Premium)", 1.0, 3.0, 1.5, 0.1)
        straddle_profit = st.slider("Straddle Profit Target (%)", 15, 50, 25)
    
    with st.sidebar.expander("Vol Expansion", expanded=False):
        vol_vix = st.slider("Min VIX", 15, 40, 25)
        vol_iv = st.slider("Min IV Percentile", 50, 100, 70)
    
    with st.sidebar.expander("Risk Management", expanded=False):
        max_risk = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        max_dd = st.slider("Max Drawdown (%)", 5, 30, 15)
        max_positions = st.slider("Max Open Positions", 1, 10, 3)
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Live Signals", "📉 Backtesting", "📋 Trade Log", "📚 Strategy Guide"]
    )
    
    return page


def render_live_signals():
    """Render the live signals page"""
    st.markdown("## 📊 Live Market Signals")
    
    # Fetch current market data
    with st.spinner("Fetching market data..."):
        market_data = st.session_state.data_fetcher.fetch_latest_market_state()
    
    if market_data is None:
        st.error("Unable to fetch live market data. Using simulated data for demonstration.")
        # Generate simulated market state
        market_data = {
            'spot': 23500 + np.random.normal(0, 100),
            'vix': 15 + np.random.normal(0, 2),
            'iv_percentile': 45 + np.random.normal(0, 10),
            'vwap': 23500,
            'ema_20': 23450 + np.random.normal(0, 50),
            'ema_50': 23400 + np.random.normal(0, 50),
            'rsi_14': 50 + np.random.normal(0, 10),
            'bb_width': 0.025 + np.random.normal(0, 0.005),
            'adx': 20 + np.random.normal(0, 5),
            'alpha': 0.15 + np.random.normal(0, 0.03),
            'alpha2': 0.15 + np.random.normal(0, 0.03),
            'day_of_week': datetime.now().weekday(),
            'days_to_expiry': 7 - datetime.now().weekday(),
            'timestamp': datetime.now()
        }
    
    # Create MarketState
    state = MarketState(**market_data)
    
    # Market State Display
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("NIFTY Spot", f"₹{state.spot:,.0f}")
    with col2:
        vix_color = "🔴" if state.vix > 25 else "🟡" if state.vix > 15 else "🟢"
        st.metric("India VIX", f"{vix_color} {state.vix:.2f}")
    with col3:
        st.metric("Trend", state.trend_direction)
    with col4:
        st.metric("ADX", f"{state.adx:.1f}")
    with col5:
        st.metric("DTE", state.days_to_expiry)
    
    # Market Regime Banner
    regime_colors = {
        'LOW_VOL': '#4caf50',
        'NORMAL_VOL': '#ff9800',
        'HIGH_VOL': '#f44336',
        'EXTREME_VOL': '#9c27b0'
    }
    regime = state.regime.name
    st.markdown(f"""
    <div style="background-color: {regime_colors.get(regime, '#666')}; 
                color: white; 
                padding: 10px; 
                border-radius: 10px; 
                text-align: center; 
                font-weight: bold; 
                font-size: 1.2rem;
                margin: 10px 0;">
        Current Market Regime: {regime.replace('_', ' ')}
    </div>
    """, unsafe_allow_html=True)
    
    # Generate primary signal
    signal = st.session_state.engine.generate_signal(state)
    
    # Risk validation
    risk_result = st.session_state.risk_manager.validate_trade(
        signal, st.session_state.capital, st.session_state.open_positions
    ) if signal else None
    
    # Display Primary Signal
    if signal and risk_result and risk_result['approved']:
        st.markdown("### 🎯 Primary Signal")
        
        badge_class = get_strategy_badge_class(signal.strategy)
        direction_class = 'bullish' if signal.direction == 'BULLISH' else 'bearish' if signal.direction == 'BEARISH' else 'neutral'
        
        st.markdown(f"""
        <div class="signal-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span class="strategy-badge {badge_class}">{signal.strategy.value}</span>
                <span class="{direction_class}">{signal.direction}</span>
                <span style="color: #666;">Confidence: {signal.confidence:.0%}</span>
            </div>
            <p><strong>Rationale:</strong> {signal.rationale}</p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                <div><strong>Max Profit:</strong> <span class="profit">₹{signal.max_profit:,.0f}</span></div>
                <div><strong>Max Loss:</strong> <span class="loss">₹{abs(signal.max_loss):,.0f}</span></div>
                <div><strong>R/R:</strong> {signal.risk_reward:.2f}</div>
                <div><strong>Suggested Lots:</strong> {signal.suggested_lots}</div>
            </div>
            <div style="margin-top: 1rem;">
                <strong>Strikes:</strong> {', '.join([f"{k}={v}" for k, v in signal.strikes.items()])}
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Actions:</strong> {', '.join([f"{k}={v}" for k, v in signal.action.items()])}
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Breakevens:</strong> {signal.breakevens[0]:,.0f} - {signal.breakevens[1]:,.0f}
            </div>
            <div style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                <strong>Target:</strong> ₹{signal.target_profit:,.0f} | 
                <strong>Stop Loss:</strong> ₹{signal.stop_loss:,.0f} | 
                <strong>Margin:</strong> ₹{signal.margin_required:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk validation output
        st.markdown("### ✅ Risk Validation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Approved", "YES" if risk_result['approved'] else "NO")
        with col2:
            st.metric("Position Size", f"{risk_result['lots']} lots")
        with col3:
            st.metric("Margin Required", f"₹{risk_result['margin_required']:,.0f}")
        
    elif signal and risk_result and not risk_result['approved']:
        st.warning("Signal generated but **NOT APPROVED** by risk management:")
        for reason in risk_result['reasons']:
            st.markdown(f"- {reason}")
    else:
        st.info("No high-confidence signal detected for current market conditions. This is normal - our system only triggers when conditions are favorable.")
    
    # All Strategy Comparison
    st.markdown("---")
    st.markdown("### 📊 All Strategy Comparison")
    
    all_signals = st.session_state.engine.get_all_strategy_signals(state)
    if all_signals:
        signal_data = []
        for s in all_signals:
            signal_data.append({
                'Strategy': s.strategy.value,
                'Direction': s.direction,
                'Confidence': f"{s.confidence:.0%}",
                'Max Profit': f"₹{s.max_profit:,.0f}",
                'Max Loss': f"₹{abs(s.max_loss):,.0f}",
                'R/R': f"{s.risk_reward:.2f}",
                'Lots': s.suggested_lots,
                'Status': '✅' if s.confidence > 0.6 else '⚠️'
            })
        
        df_signals = pd.DataFrame(signal_data)
        st.dataframe(df_signals, use_container_width=True, hide_index=True)
    
    # Risk Dashboard
    st.markdown("---")
    st.markdown("### 🛡️ Risk Dashboard")
    
    risk_report = st.session_state.risk_manager.get_risk_report(
        st.session_state.capital, st.session_state.open_positions
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_color = "🟢" if risk_report['trading_enabled'] else "🔴"
        st.metric("Trading Status", f"{status_color} {'ON' if risk_report['trading_enabled'] else 'OFF'}")
    with col2:
        st.metric("Current Drawdown", f"{risk_report['current_drawdown_pct']:.1f}%")
    with col3:
        st.metric("Daily P&L", f"₹{risk_report['daily_pnl']:,.0f}")
    with col4:
        st.metric("Daily Trades", risk_report['daily_trades'])
    
    st.info(f"💡 **Recommendation:** {risk_report['recommendation']}")
    
    # Portfolio Heat
    heat = risk_report['portfolio_heat']
    st.markdown(f"""
    <div style="background-color: {'#ffebee' if heat['heat_color'] == 'red' else '#fff8e1' if heat['heat_color'] == 'yellow' else '#e8f5e9'}; 
                padding: 15px; 
                border-radius: 10px; 
                margin: 10px 0;">
        <strong>Portfolio Heat:</strong> Risk = {heat['risk_pct']:.1%} | Margin = {heat['margin_pct']:.1%} | Open = {heat['open_positions']}
    </div>
    """, unsafe_allow_html=True)


def render_backtesting():
    """Render the backtesting page"""
    st.markdown("## 📉 Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_period = st.selectbox(
            "Backtest Period",
            ["1 Year", "2 Years", "3 Years", "5 Years"],
            index=0
        )
    with col2:
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=100000,
            max_value=5000000,
            value=500000,
            step=50000
        )
    
    period_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes."):
            # Fetch data
            fetcher = st.session_state.data_fetcher
            
            try:
                nifty_data = fetcher.fetch_historical_data(period=period_map[backtest_period])
                vix_data = fetcher.fetch_vix_data(period=period_map[backtest_period])
            except Exception as e:
                st.warning(f"Using simulated data for backtest: {e}")
                nifty_data, vix_data = st.session_state.backtest_engine.generate_simulated_data(days=500)
            
            # Run backtest
            results = st.session_state.backtest_engine.run_backtest(
                nifty_data, vix_data, initial_capital
            )
            
            st.session_state.backtest_results = results
        
        st.success("Backtest complete!")
    
    # Display results if available
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        # Key Metrics
        st.markdown("### 📈 Performance Metrics")
        
        cols = st.columns(4)
        metrics = [
            ("Total Return", f"{results.return_pct:.1f}%", results.return_pct > 0),
            ("Win Rate", f"{results.win_rate:.1%}", results.win_rate > 0.5),
            ("Sharpe Ratio", f"{results.sharpe_ratio:.2f}", results.sharpe_ratio > 1),
            ("Max Drawdown", f"{results.max_drawdown_pct:.1f}%", results.max_drawdown_pct > -15)
        ]
        
        for col, (name, value, good) in zip(cols, metrics):
            with col:
                st.metric(name, value)
        
        cols2 = st.columns(4)
        metrics2 = [
            ("Total Trades", results.total_trades),
            ("Profit Factor", f"{results.profit_factor:.2f}"),
            ("Avg Win", f"₹{results.avg_win:,.0f}"),
            ("Avg Loss", f"₹{abs(results.avg_loss):,.0f}")
        ]
        
        for col, (name, value) in zip(cols2, metrics2):
            with col:
                st.metric(name, value)
        
        # Equity Curve
        st.markdown("### 📊 Equity Curve")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(results.equity_curve['date'], results.equity_curve['equity'], color='#1f4e79', linewidth=1.5)
        ax.fill_between(results.equity_curve['date'], results.equity_curve['equity'], 
                        alpha=0.3, color='#1f4e79')
        ax.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Drawdown Chart
        st.markdown("### 📉 Drawdown Analysis")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.fill_between(results.equity_curve['date'], results.equity_curve['drawdown_pct'], 
                         0, color='red', alpha=0.3)
        ax2.plot(results.equity_curve['date'], results.equity_curve['drawdown_pct'], 
                 color='red', linewidth=1)
        ax2.set_title('Drawdown %', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Strategy Breakdown
        if results.strategy_breakdown:
            st.markdown("### 🎯 Strategy Breakdown")
            strategy_df = pd.DataFrame(results.strategy_breakdown).T
            st.dataframe(strategy_df, use_container_width=True)
        
        # Monthly Returns
        if len(results.monthly_returns) > 0:
            st.markdown("### 📅 Monthly Returns")
            fig3, ax3 = plt.subplots(figsize=(12, 4))
            colors = ['green' if x > 0 else 'red' for x in results.monthly_returns.values]
            ax3.bar(range(len(results.monthly_returns)), results.monthly_returns.values, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            ax3.set_title('Monthly Returns %', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Return %')
            ax3.set_xticks(range(0, len(results.monthly_returns), max(1, len(results.monthly_returns)//6)))
            ax3.set_xticklabels([str(results.monthly_returns.index[i]) for i in range(0, len(results.monthly_returns), max(1, len(results.monthly_returns)//6))], rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
        
        # Trade Distribution
        if results.trade_log:
            st.markdown("### 📊 Trade Distribution")
            pnls = [t.pnl for t in results.trade_log]
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.hist(pnls, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_title('P&L Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('P&L (₹)')
            ax4.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig4)
    else:
        st.info("Click 'Run Backtest' to see results. Use the sidebar to configure strategy parameters.")


def render_trade_log():
    """Render the trade log page"""
    st.markdown("## 📋 Trade Log & Journal")
    
    if st.session_state.backtest_results and st.session_state.backtest_results.trade_log:
        trades = st.session_state.backtest_results.trade_log
        
        # Summary stats
        total_pnl = sum(t.pnl for t in trades)
        avg_holding = np.mean([(t.exit_date - t.entry_date).days for t in trades])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
        with col3:
            st.metric("Avg Hold", f"{avg_holding:.1f} days")
        with col4:
            win_rate = len([t for t in trades if t.pnl > 0]) / len(trades)
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        # Trade table
        trade_data = []
        for t in trades:
            trade_data.append({
                'Entry': t.entry_date.strftime('%Y-%m-%d'),
                'Exit': t.exit_date.strftime('%Y-%m-%d'),
                'Strategy': t.strategy.value,
                'Entry Spot': f"{t.entry_spot:,.0f}",
                'Exit Spot': f"{t.exit_spot:,.0f}",
                'P&L': f"₹{t.pnl:,.0f}",
                'Exit Reason': t.exit_reason,
                'Lots': t.lots
            })
        
        df = pd.DataFrame(trade_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log (CSV)",
            data=csv,
            file_name=f"nifty_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Run a backtest to generate trade log data.")


def render_strategy_guide():
    """Render the strategy guide page"""
    st.markdown("## 📚 Strategy Guide")
    
    st.markdown("""
    ### 🏛️ Institutional Strategy Framework
    
    This system is built on rigorous backtesting research from institutional traders and academic studies on NIFTY 50 options.
    
    ---
    
    #### 1. Theta Harvest (Iron Condor)
    **Best For:** Low volatility, range-bound markets (VIX < 15)
    
    **Setup:**
    - Sell OTM Call & Put (16-delta)
    - Buy further OTM protection (wider wings)
    - Target 30 DTE, manage at 50% profit
    
    **Research:** Defined risk reduces margin by 80% vs naked strangles. 
    Average win rate ~72% in low vol environments.
    
    **Risk:** Limited to spread width minus credit. Exit if VIX > 20.
    
    ---
    
    #### 2. Quiet Straddle (Short ATM)
    **Best For:** Ultra-quiet markets (alpha < 0.2, alpha2 < 0.2)
    
    **Setup:**
    - Sell ATM Call + Put simultaneously
    - Entry when volatility is compressed
    - Target 30 DTE, manage at 25% profit
    
    **Research:** Highest Sharpe ratio (0.056) among non-directional NIFTY strategies.
    Average monthly return 2.5%, win rate ~65%.
    
    **Risk:** Unlimited risk. Strict 1.5x premium stop loss required.
    
    ---
    
    #### 3. Vol Expansion (Long Straddle)
    **Best For:** High volatility environments (VIX > 25, IV percentile > 70)
    
    **Setup:**
    - Buy ATM Call + Put
    - Entry when volatility is expected to expand
    - Target 2x premium, stop at 50% loss
    
    **Research:** Mean reversion in VIX after spikes. Average payoff 2-3x when timed correctly.
    
    **Risk:** Premium decay if vol doesn't expand. Time is enemy.
    
    ---
    
    #### 4. Directional Momentum Spread
    **Best For:** Strong trending markets (ADX > 25)
    
    **Setup:**
    - Bull Call Spread (uptrend) or Bear Put Spread (downtrend)
    - 100-point spread width for NIFTY
    - Minimum 3:1 risk-reward
    
    **Research:** Trend-following with defined risk. Better than naked options in strong trends.
    
    **Risk:** Loss limited to net debit. Worse in choppy markets.
    
    ---
    
    #### 5. Expiry Theta Capture
    **Best For:** 1-2 DTE, low VIX (< 14)
    
    **Setup:**
    - Iron Butterfly: Sell ATM, buy OTM wings
    - Extreme theta decay in final hours
    - Avoid Tuesday (expiry day) gamma risk
    
    **Research:** Wednesday post-expiry best (+₹1,180 avg), Tuesday worst (-₹120 avg).
    
    **Risk:** Gamma risk near expiry. Use defined risk only.
    
    ---
    
    ### ⚠️ Risk Management Rules
    
    1. **Max 1% risk per trade** on total capital
    2. **Max 3 open positions** simultaneously
    3. **Stop trading at 15% drawdown**
    4. **Never hold short straddles through VIX > 25**
    5. **Close 50% of strangle profit, 25% of straddle profit**
    6. **Skip Tuesday expiry** for new straddle positions
    
    ---
    
    ### 📊 Key Research Sources
    
    - Academic study: "Writing Non-Directional Option Strategies on NIFTY 50" (5-year backtest, 60 months)
    - TastyTrade research: 21 million strategy backtests across SPY/indices
    - India VIX research: VIX-Filtered Long Sharpe 1.02 vs passive 0.54
    - 9:20 AM Straddle backtest: 248 trades, 60.9% win rate, Sharpe 1.42
    - Zerodha Varsity: Iron Condor margin analysis
    
    ---
    
    *Disclaimer: This system is for educational and research purposes. 
    Past performance does not guarantee future results. Options trading 
    carries substantial risk of loss. Consult a SEBI-registered advisor 
    before trading.*
    """)


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    page = render_sidebar()
    
    if page == "📊 Live Signals":
        render_live_signals()
    elif page == "📉 Backtesting":
        render_backtesting()
    elif page == "📋 Trade Log":
        render_trade_log()
    elif page == "📚 Strategy Guide":
        render_strategy_guide()


if __name__ == "__main__":
    main()
