"""
FastHTML web app for the NIFTY options strategy terminal.

Run:
    uvicorn app_fast:app --reload --host 127.0.0.1 --port 8000
"""

from datetime import datetime
from html import escape
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from fasthtml.common import *

from backtest_engine import BacktestEngine, BacktestResults
from data_fetcher import DHAN_AVAILABLE, NiftyDataFetcher
from risk_manager import RiskManager
from strategy_engine import MarketState, Signal, StrategyEngine, StrategyType

CSS = """
:root {
  color-scheme: dark;
  --bg: #08111f;
  --panel: rgba(16, 27, 45, .84);
  --panel-2: #101b2d;
  --border: rgba(148, 163, 184, .24);
  --muted: #8da0ba;
  --text: #eef5ff;
  --accent: #2dd4bf;
  --accent-2: #f4c95d;
  --danger: #fb7185;
  --good: #22c55e;
  --blue: #60a5fa;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background:
    radial-gradient(circle at 14% 10%, rgba(45, 212, 191, .18), transparent 28rem),
    radial-gradient(circle at 85% 0%, rgba(96, 165, 250, .16), transparent 30rem),
    linear-gradient(145deg, #08111f 0%, #0b1424 42%, #111827 100%);
}
a { color: inherit; text-decoration: none; }
.app-shell { width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 22px 0 48px; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
.brand { display: flex; flex-direction: column; gap: 4px; }
.brand h1 { margin: 0; font-size: clamp(1.55rem, 2.8vw, 2.55rem); letter-spacing: 0; }
.brand p { margin: 0; color: var(--muted); font-size: .98rem; }
.nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.nav a { padding: 10px 13px; border: 1px solid var(--border); border-radius: 8px; color: #cbd5e1; background: rgba(15, 23, 42, .64); }
.nav a.active, .nav a:hover { border-color: rgba(45, 212, 191, .62); color: white; background: rgba(45, 212, 191, .12); }
.hero { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr); gap: 16px; align-items: stretch; margin-bottom: 16px; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 20px 70px rgba(0,0,0,.22); }
.panel.pad { padding: 18px; }
.status-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.pill { display: inline-flex; align-items: center; gap: 7px; min-height: 32px; padding: 7px 10px; border-radius: 999px; border: 1px solid var(--border); color: #cbd5e1; background: rgba(15, 23, 42, .5); font-size: .86rem; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); }
.dot.good { background: var(--good); } .dot.warn { background: var(--accent-2); } .dot.bad { background: var(--danger); }
.grid { display: grid; gap: 12px; }
.metrics { grid-template-columns: repeat(5, minmax(130px, 1fr)); margin-bottom: 16px; }
.metric { padding: 14px; border: 1px solid var(--border); border-radius: 8px; background: rgba(15, 23, 42, .58); }
.metric span { display: block; color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric strong { display: block; margin-top: 7px; font-size: clamp(1.2rem, 2vw, 1.75rem); letter-spacing: 0; }
.section-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.section-title h2, .section-title h3 { margin: 0; font-size: 1.02rem; }
.signal { display: grid; grid-template-columns: minmax(0, 1fr) 220px; gap: 14px; padding: 16px; border-radius: 8px; border: 1px solid rgba(45, 212, 191, .34); background: linear-gradient(135deg, rgba(45, 212, 191, .13), rgba(96, 165, 250, .06)); }
.signal h2 { margin: 9px 0 8px; font-size: clamp(1.35rem, 2.3vw, 2rem); }
.signal p { margin: 0; color: #cbd5e1; line-height: 1.55; }
.badge { display: inline-flex; width: fit-content; padding: 6px 9px; border-radius: 999px; background: rgba(45, 212, 191, .13); color: #99f6e4; border: 1px solid rgba(45, 212, 191, .38); font-size: .78rem; font-weight: 700; }
.kv { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }
.kv div { padding: 10px; border-radius: 8px; background: rgba(8, 17, 31, .58); border: 1px solid var(--border); }
.kv span { display: block; color: var(--muted); font-size: .75rem; }
.kv strong { display: block; margin-top: 4px; }
.two { grid-template-columns: minmax(0, 1fr) minmax(320px, .75fr); }
table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }
th, td { padding: 11px 12px; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, .14); font-size: .9rem; }
th { color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; background: rgba(15, 23, 42, .74); }
tr:last-child td { border-bottom: 0; }
.chart { width: 100%; height: 280px; display: block; }
.form-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }
label { display: grid; gap: 6px; color: var(--muted); font-size: .82rem; }
input, select { min-width: 150px; height: 40px; border-radius: 8px; border: 1px solid var(--border); background: #0b1220; color: var(--text); padding: 0 10px; }
button, .button { height: 40px; border: 0; border-radius: 8px; padding: 0 14px; color: #04111d; background: var(--accent); font-weight: 800; cursor: pointer; }
.note { color: var(--muted); line-height: 1.6; }
.list { display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }
.list li { padding: 12px; border-radius: 8px; border: 1px solid var(--border); background: rgba(15, 23, 42, .5); }
.warning { border-color: rgba(244, 201, 93, .45); background: rgba(244, 201, 93, .09); color: #fde68a; }
@media (max-width: 980px) { .hero, .two { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .signal { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .app-shell { width: min(100% - 20px, 1440px); padding-top: 14px; } .topbar { align-items: flex-start; flex-direction: column; } .nav { justify-content: flex-start; } .metrics { grid-template-columns: 1fr; } th, td { padding: 9px 8px; font-size: .82rem; } }
"""

app, rt = fast_app(hdrs=(Style(CSS),), title="NIFTY Strategy Terminal")

BROKER_SETTINGS = {
    "provider": "auto",
    "dhan_client_id": "",
    "dhan_access_token": "",
    "dhan_nifty_security_id": "13",
    "dhan_vix_security_id": "21",
    "dhan_index_segment": "IDX_I",
    "dhan_index_type": "INDEX",
}


def get_fetcher() -> NiftyDataFetcher:
    NiftyDataFetcher.DHAN_NIFTY_SECURITY_ID = BROKER_SETTINGS["dhan_nifty_security_id"] or "13"
    NiftyDataFetcher.DHAN_VIX_SECURITY_ID = BROKER_SETTINGS["dhan_vix_security_id"] or "21"
    NiftyDataFetcher.DHAN_INDEX_SEGMENT = BROKER_SETTINGS["dhan_index_segment"] or "IDX_I"
    NiftyDataFetcher.DHAN_INDEX_TYPE = BROKER_SETTINGS["dhan_index_type"] or "INDEX"
    return NiftyDataFetcher(
        provider=BROKER_SETTINGS["provider"],
        dhan_client_id=BROKER_SETTINGS["dhan_client_id"],
        dhan_access_token=BROKER_SETTINGS["dhan_access_token"],
    )


def money(value: float) -> str:
    return f"Rs. {value:,.0f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def current_status() -> tuple[str, str]:
    now = datetime.now()
    market_open = (now.weekday() < 5 and ((now.hour, now.minute) >= (9, 15)) and ((now.hour, now.minute) <= (15, 30)))
    return ("OPEN", "good") if market_open else ("CLOSED", "warn")


def nav(active: str):
    items = [("/", "Live"), ("/broker", "Broker"), ("/backtest", "Backtest"), ("/trades", "Trade Log"), ("/guide", "Strategy Guide")]
    return Nav(*[A(label, href=href, cls="active" if active == label else "") for href, label in items], cls="nav")


def shell(active: str, *content):
    status, dot = current_status()
    return Div(
        Div(
            Div(H1("NIFTY 50 Options Strategy Terminal"), P("Broker-aware signals, regime logic, and backtesting for Indian F&O."), cls="brand"),
            nav(active),
            cls="topbar",
        ),
        *content,
        cls="app-shell",
    )


def metric(label: str, value: str):
    return Div(Span(label), Strong(value), cls="metric")


def raw_svg_line(values: Sequence[float], color: str = "#2dd4bf", fill: str = "rgba(45,212,191,.14)"):
    if len(values) < 2:
        return Div(P("Not enough data to chart yet.", cls="note"), cls="chart")
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return Div(P("Not enough data to chart yet.", cls="note"), cls="chart")
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = hi - lo if hi != lo else 1
    width, height, pad = 900, 280, 22
    points = []
    for i, val in enumerate(vals):
        x = pad + i * (width - pad * 2) / max(len(vals) - 1, 1)
        y = height - pad - ((val - lo) / span) * (height - pad * 2)
        points.append(f"{x:.1f},{y:.1f}")
    area = " ".join([f"{pad},{height-pad}"] + points + [f"{width-pad},{height-pad}"])
    line = " ".join(points)
    svg = f"""
    <svg class='chart' viewBox='0 0 {width} {height}' role='img' aria-label='Chart'>
      <rect x='0' y='0' width='{width}' height='{height}' rx='8' fill='rgba(8,17,31,.52)' />
      <polyline points='{area}' fill='{fill}' stroke='none'></polyline>
      <polyline points='{line}' fill='none' stroke='{color}' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'></polyline>
      <line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='rgba(148,163,184,.25)' />
      <text x='{pad}' y='22' fill='#8da0ba' font-size='13'>{escape(f'{hi:,.0f}')}</text>
      <text x='{pad}' y='{height-8}' fill='#8da0ba' font-size='13'>{escape(f'{lo:,.0f}')}</text>
    </svg>
    """
    return NotStr(svg)


def simple_table(headers: Sequence[str], rows: Iterable[Sequence[str]]):
    return Table(
        Thead(Tr(*[Th(h) for h in headers])),
        Tbody(*[Tr(*[Td(NotStr(str(cell)) if str(cell).startswith("<") else str(cell)) for cell in row]) for row in rows]),
    )


def signal_rows(signal: Signal):
    return [
        ("Strategy", signal.strategy.value),
        ("Direction", signal.direction),
        ("Confidence", f"{signal.confidence:.0%}"),
        ("Risk/Reward", f"{signal.risk_reward:.2f}"),
        ("Max Profit", money(signal.max_profit)),
        ("Max Loss", money(abs(signal.max_loss))),
        ("Stop", money(signal.stop_loss)),
        ("Target", money(signal.target_profit)),
    ]


def signal_panel(signal: Signal | None, risk: dict | None):
    if not signal:
        return Div(
            Div(Span("No Trade", cls="badge"), H2("No high-confidence setup right now"), P("The engine is waiting for a cleaner volatility, trend, or expiry condition.")),
            cls="signal",
        )

    risk_note = "Approved" if risk and risk.get("approved") else "Risk blocked"
    return Div(
        Div(
            Span(risk_note, cls="badge"),
            H2(signal.strategy.value),
            P(signal.rationale),
            Div(
                *[Div(Span(k), Strong(v)) for k, v in signal_rows(signal)[:4]],
                cls="kv",
            ),
        ),
        Div(
            H3("Execution"),
            simple_table(["Leg", "Action", "Strike", "Premium"], [
                (leg, action, f"{signal.strikes.get(leg, 0):,.0f}", money(signal.premium_estimate.get(leg, 0)))
                for leg, action in signal.action.items()
            ]),
        ),
        cls="signal",
    )


def live_data():
    fetcher = get_fetcher()
    engine = StrategyEngine()
    risk_manager = RiskManager()
    market = fetcher.fetch_latest_market_state()
    simulated = False
    if market is None:
        simulated = True
        market = {
            "spot": 23500 + np.random.normal(0, 100),
            "vix": 15 + np.random.normal(0, 2),
            "iv_percentile": 45 + np.random.normal(0, 10),
            "vwap": 23500,
            "ema_20": 23450 + np.random.normal(0, 50),
            "ema_50": 23400 + np.random.normal(0, 50),
            "rsi_14": 50 + np.random.normal(0, 10),
            "bb_width": 0.025 + np.random.normal(0, 0.005),
            "adx": 20 + np.random.normal(0, 5),
            "alpha": 0.15,
            "alpha2": 0.15,
            "day_of_week": datetime.now().weekday(),
            "days_to_expiry": 7 - datetime.now().weekday(),
            "timestamp": datetime.now(),
        }
    state = MarketState(**market)
    signal = engine.generate_signal(state)
    risk = risk_manager.validate_trade(signal, 500000, []) if signal else None
    all_signals = engine.get_all_strategy_signals(state)
    risk_report = risk_manager.get_risk_report(500000, [])
    return fetcher, state, signal, risk, all_signals, risk_report, simulated


@rt("/")
def get():
    fetcher, state, signal, risk, all_signals, risk_report, simulated = live_data()
    status, dot = current_status()
    source_dot = "good" if fetcher.broker_enabled else "warn"
    return shell(
        "Live",
        Div(
            Div(
                H2("Live Signal Desk"),
                P("One screen for current regime, preferred strategy, execution legs, and portfolio heat.", cls="note"),
                Div(
                    Span(Span(cls=f"dot {dot}"), f"Market {status}", cls="pill"),
                    Span(Span(cls=f"dot {source_dot}"), f"Data: {fetcher.provider_label()}", cls="pill"),
                    Span(Span(cls="dot warn" if simulated else "dot good"), "Fallback simulation" if simulated else "Broker/live data attempted", cls="pill"),
                    Span(datetime.now().strftime("%d %b %Y, %H:%M"), cls="pill"),
                    cls="status-row",
                ),
                cls="panel pad",
            ),
            Div(
                H3("Risk State"),
                Div(
                    Div(Span("Trading"), Strong("ON" if risk_report["trading_enabled"] else "OFF")),
                    Div(Span("Daily P&L"), Strong(money(risk_report["daily_pnl"]))),
                    Div(Span("Drawdown"), Strong(pct(risk_report["current_drawdown_pct"]))),
                    Div(Span("Open Risk"), Strong(pct(risk_report["portfolio_heat"]["risk_pct"] * 100))),
                    cls="kv",
                ),
                P(risk_report["recommendation"], cls="note"),
                cls="panel pad",
            ),
            cls="hero",
        ),
        Div(
            metric("NIFTY Spot", money(state.spot)),
            metric("India VIX", f"{state.vix:.2f}"),
            metric("Regime", state.regime.name.replace("_", " ")),
            metric("Trend", state.trend_direction),
            metric("DTE", str(state.days_to_expiry)),
            cls="grid metrics",
        ),
        Div(signal_panel(signal, risk), cls="panel pad"),
        Div(
            Div(H3("Strategy Comparison"), cls="section-title"),
            simple_table(["Strategy", "Direction", "Confidence", "Risk/Reward", "Max Profit", "Max Loss"], [
                (s.strategy.value, s.direction, f"{s.confidence:.0%}", f"{s.risk_reward:.2f}", money(s.max_profit), money(abs(s.max_loss)))
                for s in all_signals
            ] or [("No candidate", "-", "-", "-", "-", "-")]),
            cls="panel pad",
        ),
    )



def broker_form(message: str = ""):
    provider = BROKER_SETTINGS["provider"]
    token_value = BROKER_SETTINGS["dhan_access_token"]
    masked = "Saved in memory" if token_value else "Not set"
    sdk_status = "Installed" if DHAN_AVAILABLE else "Not installed - run pip install -r requirements.txt"
    return Div(
        Div(
            H2("Broker Settings"),
            P("Enter Dhan credentials here and save. The app will use them immediately without PowerShell environment variables.", cls="note"),
            cls="section-title",
        ),
        Div(
            Span(Span(cls="dot good" if DHAN_AVAILABLE else "dot warn"), f"Dhan SDK: {sdk_status}", cls="pill"),
            Span(Span(cls="dot good" if BROKER_SETTINGS["dhan_client_id"] else "dot warn"), f"Client ID: {'Set' if BROKER_SETTINGS['dhan_client_id'] else 'Not set'}", cls="pill"),
            Span(Span(cls="dot good" if token_value else "dot warn"), f"Access token: {masked}", cls="pill"),
            cls="status-row",
        ),
        P(message, cls="note") if message else "",
        Form(
            Div(
                Label("Provider", Select(
                    Option("Auto", value="auto", selected=(provider == "auto")),
                    Option("Dhan", value="dhan", selected=(provider == "dhan")),
                    Option("yfinance", value="yfinance", selected=(provider == "yfinance")),
                    Option("Simulated", value="simulated", selected=(provider == "simulated")),
                    name="provider",
                )),
                Label("Dhan Client ID", Input(type="text", name="dhan_client_id", value=BROKER_SETTINGS["dhan_client_id"], placeholder="1100000000")),
                Label("Dhan Access Token", Input(type="password", name="dhan_access_token", value="", placeholder="Paste token to update")),
                cls="form-row",
            ),
            Div(
                Label("NIFTY Security ID", Input(type="text", name="dhan_nifty_security_id", value=BROKER_SETTINGS["dhan_nifty_security_id"])),
                Label("VIX Security ID", Input(type="text", name="dhan_vix_security_id", value=BROKER_SETTINGS["dhan_vix_security_id"])),
                Label("Exchange Segment", Input(type="text", name="dhan_index_segment", value=BROKER_SETTINGS["dhan_index_segment"])),
                Label("Instrument Type", Input(type="text", name="dhan_index_type", value=BROKER_SETTINGS["dhan_index_type"])),
                Button("Save & Test Dhan", type="submit"),
                cls="form-row",
            ),
            action="/broker",
            method="post",
        ),
        Div(A("Open Live Signals", href="/", cls="button"), cls="status-row"),
        P("Credentials are kept only in this running Python server's memory. Restarting the server clears them.", cls="note"),
        cls="panel pad",
    )


@rt("/broker")
def get():
    return shell("Broker", broker_form())


@rt("/broker")
def post(provider: str = "auto", dhan_client_id: str = "", dhan_access_token: str = "", dhan_nifty_security_id: str = "13", dhan_vix_security_id: str = "21", dhan_index_segment: str = "IDX_I", dhan_index_type: str = "INDEX"):
    BROKER_SETTINGS["provider"] = provider or "auto"
    BROKER_SETTINGS["dhan_client_id"] = dhan_client_id.strip()
    if dhan_access_token.strip():
        BROKER_SETTINGS["dhan_access_token"] = dhan_access_token.strip()
    BROKER_SETTINGS["dhan_nifty_security_id"] = dhan_nifty_security_id.strip() or "13"
    BROKER_SETTINGS["dhan_vix_security_id"] = dhan_vix_security_id.strip() or "21"
    BROKER_SETTINGS["dhan_index_segment"] = dhan_index_segment.strip() or "IDX_I"
    BROKER_SETTINGS["dhan_index_type"] = dhan_index_type.strip() or "INDEX"

    if BROKER_SETTINGS["provider"] in {"dhan", "auto"} and BROKER_SETTINGS["dhan_client_id"] and BROKER_SETTINGS["dhan_access_token"]:
        fetcher = get_fetcher()
        spot = fetcher.fetch_current_spot()
        if spot:
            message = f"Saved and connected to Dhan. Latest NIFTY spot: {money(spot)}. Open Live Signals to refresh the dashboard."
        else:
            latest_close = fetcher.fetch_dhan_latest_close()
            if latest_close:
                close, close_date = latest_close
                message = f"Saved and connected to Dhan historical data. Live LTP was unavailable, likely because the market is closed. Latest NIFTY close: {money(close)} on {close_date.strftime('%d %b %Y')}. Open Live Signals to refresh the dashboard."
            else:
                detail = f" Detail: {fetcher.last_error}" if getattr(fetcher, "last_error", "") else ""
                message = "Saved, but Dhan did not return live spot or historical close. Check the token, market-data access, and security id; Live will fall back to simulated data until Dhan responds." + detail
    elif BROKER_SETTINGS["provider"] == "simulated":
        message = "Saved. Provider is Simulated, so Live will use generated market data."
    elif BROKER_SETTINGS["provider"] == "yfinance":
        message = "Saved. Provider is yfinance, so Live will use Yahoo data when available."
    else:
        message = "Saved. Add Dhan Client ID and Access Token, then save again to test the connection."

    return shell("Broker", broker_form(message))
def load_backtest_data(period: str):
    fetcher = get_fetcher()
    nifty = fetcher.fetch_historical_data(period=period)
    vix = fetcher.fetch_vix_data(period=period)
    return nifty, vix


def run_backtest(period: str, capital: int) -> BacktestResults:
    nifty, vix = load_backtest_data(period)
    return BacktestEngine().run_backtest(nifty, vix, initial_capital=capital)


def run_strategy_comparison(period: str, capital: int):
    nifty, vix = load_backtest_data(period)
    return BacktestEngine().run_strategy_comparison(nifty, vix, initial_capital=capital)


@rt("/backtest")
def get(period: str = "1y", capital: int = 500000, run: str = "", compare: str = ""):
    capital = int(capital)
    results = run_backtest(period, capital) if run else None
    comparison = run_strategy_comparison(period, capital) if compare else None
    period_options = [("6mo", "6 Months"), ("1y", "1 Year"), ("2y", "2 Years"), ("5y", "5 Years")]
    return shell(
        "Backtest",
        Div(
            Div(H2("Backtesting Lab"), P("Run the existing strategy engine against Dhan/yfinance history, with simulation as fallback.", cls="note"), cls="section-title"),
            Form(
                Div(
                    Label("Period", Select(*[Option(label, value=value, selected=(value == period)) for value, label in period_options], name="period")),
                    Label("Initial Capital", Input(type="number", name="capital", value=str(capital), min="100000", step="50000")),
                    Button("Run Adaptive", type="submit", name="run", value="1"),
                    Button("Compare Strategies", type="submit", name="compare", value="1"),
                    cls="form-row",
                ),
                action="/backtest",
                method="get",
            ),
            cls="panel pad",
        ),
        comparison_results_panel(comparison) if comparison else "",
        backtest_results_panel(results, capital) if results else (Div(P("Choose a period and run the adaptive selector or compare every strategy independently.", cls="note"), cls="panel pad") if not comparison else ""),
    )


def comparison_results_panel(comparison):
    rows = []
    for strategy, results in comparison.items():
        rows.append((
            strategy,
            str(results.total_trades),
            pct(results.return_pct),
            f"{results.win_rate:.1%}",
            f"{results.sharpe_ratio:.2f}",
            pct(results.max_drawdown_pct),
            money(results.total_pnl),
        ))
    return Div(
        Div(H3("Strategy Comparison"), cls="section-title"),
        P("Each row forces one strategy over the same period. A strategy with zero trades had no qualifying setup under its own rules. Premiums and exits are still modeled, not true option-chain fills, so use this for relative diagnostics rather than production P&L.", cls="note warning"),
        simple_table(["Strategy", "Trades", "Return", "Win Rate", "Sharpe", "Max Drawdown", "Total P&L"], rows),
        cls="panel pad",
    )


def backtest_results_panel(results: BacktestResults, capital: int):
    equity = results.equity_curve["equity"].tolist() if len(results.equity_curve) else []
    drawdown = results.equity_curve.get("drawdown_pct", pd.Series(dtype=float)).tolist() if len(results.equity_curve) else []
    trade_rows = [
        (t.entry_date.strftime("%Y-%m-%d"), t.exit_date.strftime("%Y-%m-%d"), t.strategy.value, money(t.pnl), t.exit_reason, str(t.lots))
        for t in results.trade_log[:25]
    ]
    mix_rows = []
    for name, stats in results.strategy_breakdown.items():
        share = stats["trades"] / results.total_trades if results.total_trades else 0
        mix_rows.append((name, str(stats["trades"]), f"{share:.0%}", f"{stats['win_rate']:.1%}", money(stats["total_pnl"]), money(stats["avg_pnl"])))
    concentration = max((row[2] for row in mix_rows), default="0%")
    note = "This is an adaptive-selector backtest: it chooses one strategy at a time based on the market state. It is not a separate side-by-side backtest of every strategy."
    if results.total_trades and any(float(row[2].strip('%')) >= 75 for row in mix_rows):
        note += " The selector is highly concentrated, so inspect VIX/regime assumptions before treating the result as robust."
    return Div(
        Div(
            metric("Total Return", pct(results.return_pct)),
            metric("Win Rate", f"{results.win_rate:.1%}"),
            metric("Sharpe", f"{results.sharpe_ratio:.2f}"),
            metric("Max Drawdown", pct(results.max_drawdown_pct)),
            metric("Trades", str(results.total_trades)),
            cls="grid metrics",
        ),
        Div(P(note, cls="note warning"), cls="panel pad"),
        Div(
            Div(Div(H3("Equity Curve"), cls="section-title"), raw_svg_line(equity, "#2dd4bf", "rgba(45,212,191,.16)"), cls="panel pad"),
            Div(Div(H3("Drawdown"), cls="section-title"), raw_svg_line(drawdown, "#fb7185", "rgba(251,113,133,.14)"), cls="panel pad"),
            cls="grid two",
        ),
        Div(
            Div(H3("Adaptive Strategy Mix"), cls="section-title"),
            simple_table(["Strategy", "Trades", "Share", "Win Rate", "Total P&L", "Avg P&L"], mix_rows or [("No trades", "-", "-", "-", "-", "-")]),
            cls="panel pad",
        ),
        Div(
            Div(H3("Recent Trades"), cls="section-title"),
            simple_table(["Entry", "Exit", "Strategy", "P&L", "Reason", "Lots"], trade_rows or [("No trades", "-", "-", "-", "-", "-")]),
            cls="panel pad",
        ),
    )


@rt("/trades")
def get():
    results = run_backtest("1y", 500000)
    rows = [
        (t.entry_date.strftime("%Y-%m-%d"), t.exit_date.strftime("%Y-%m-%d"), t.strategy.value, f"{t.entry_spot:,.0f}", f"{t.exit_spot:,.0f}", money(t.pnl), t.exit_reason, str(t.lots))
        for t in results.trade_log
    ]
    return shell(
        "Trade Log",
        Div(
            Div(H2("Trade Log"), P("Generated from the latest 1Y backtest run.", cls="note"), cls="section-title"),
            Div(
                metric("Trades", str(results.total_trades)),
                metric("Total P&L", money(results.total_pnl)),
                metric("Avg P&L", money(results.avg_pnl)),
                metric("Win Rate", f"{results.win_rate:.1%}"),
                metric("Profit Factor", f"{results.profit_factor:.2f}" if np.isfinite(results.profit_factor) else "Inf"),
                cls="grid metrics",
            ),
            simple_table(["Entry", "Exit", "Strategy", "Entry Spot", "Exit Spot", "P&L", "Reason", "Lots"], rows or [("No trades", "-", "-", "-", "-", "-", "-", "-")]),
            cls="panel pad",
        ),
    )


@rt("/guide")
def get():
    strategies = [
        ("Theta Harvest", "Iron condor for low VIX, range-bound markets. Defined risk with premium decay as the edge."),
        ("Quiet Straddle", "Short ATM straddle only during compressed volatility. Requires strict stop discipline."),
        ("Vol Expansion", "Long straddle when VIX and IV percentile imply a large move may be developing."),
        ("Directional Momentum", "Bull call or bear put spreads when ADX confirms trend strength."),
        ("Expiry Theta", "Defined-risk butterfly for 1-2 DTE low-volatility windows."),
    ]
    return shell(
        "Strategy Guide",
        Div(
            H2("Strategy Guide"),
            P("The UI is new, but the strategy engine is the same regime-aware framework from the Streamlit app.", cls="note"),
            Ul(*[Li(Strong(name), P(desc, cls="note")) for name, desc in strategies], cls="list"),
            Div(
                Strong("Dhan setup"),
                P("Open the Broker tab, paste your Dhan Client ID and Access Token, choose Dhan, and save.", cls="note"),
                cls="panel pad warning",
            ),
            cls="panel pad",
        ),
    )


if __name__ == "__main__":
    serve()














