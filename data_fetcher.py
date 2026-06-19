"""
Data Fetcher for NIFTY 50
Handles real-time and historical data retrieval.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

try:
    from dhanhq import DhanContext, dhanhq
    DHAN_AVAILABLE = True
except ImportError:
    dhanhq = None
    DHAN_AVAILABLE = False

warnings.filterwarnings("ignore")

def _add_no_proxy_host(host: str):
    """Ensure requests bypasses broken local proxy settings for a specific host."""
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "")
        hosts = [h.strip() for h in current.split(",") if h.strip()]
        if host not in hosts:
            hosts.append(host)
            os.environ[key] = ",".join(hosts)



class NiftyDataFetcher:
    """
    Fetches NIFTY 50 spot and India VIX data.

    Provider priority:
    1. DhanHQ when DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are set
    2. yfinance when NIFTY_DATA_PROVIDER=yfinance
    3. simulated data fallback
    """

    NIFTY_TICKER = "^NSEI"
    VIX_TICKER = "^INDIAVIX"

    # Dhan index identifiers. Override with env vars if Dhan changes ids.
    DHAN_NIFTY_SECURITY_ID = os.getenv("DHAN_NIFTY_SECURITY_ID", "13")
    DHAN_VIX_SECURITY_ID = os.getenv("DHAN_VIX_SECURITY_ID", "21")
    DHAN_INDEX_SEGMENT = os.getenv("DHAN_INDEX_SEGMENT", "IDX_I")
    DHAN_INDEX_TYPE = os.getenv("DHAN_INDEX_TYPE", "INDEX")

    def __init__(self, provider: Optional[str] = None, dhan_client_id: Optional[str] = None, dhan_access_token: Optional[str] = None):
        self._cache = {}
        self._cache_time = None
        self.provider = (provider or os.getenv("NIFTY_DATA_PROVIDER", "auto")).lower()
        self.dhan_client_id = dhan_client_id or os.getenv("DHAN_CLIENT_ID")
        self.dhan_access_token = dhan_access_token or os.getenv("DHAN_ACCESS_TOKEN")
        self.last_error = ""
        self.last_response = ""
        self.dhan = self._init_dhan()

    @property
    def dhan_enabled(self) -> bool:
        """Return True when DhanHQ is installed and credentials are configured."""
        return self.dhan is not None

    @property
    def broker_enabled(self) -> bool:
        return self.dhan_enabled

    def provider_label(self) -> str:
        if self._should_use_dhan():
            return "DhanHQ"
        if self._should_use_yfinance():
            return "yfinance"
        return "Simulated"

    def _init_dhan(self):
        """Create a DhanHQ client from environment credentials when available."""
        if not DHAN_AVAILABLE:
            return None
        _add_no_proxy_host("api.dhan.co")
        _add_no_proxy_host("images.dhan.co")
        if not self.dhan_client_id or not self.dhan_access_token:
            return None
        try:
            context = DhanContext(self.dhan_client_id, self.dhan_access_token)
            return dhanhq(context)
        except TypeError:
            return dhanhq(self.dhan_client_id, self.dhan_access_token)

    def _should_use_dhan(self) -> bool:
        return self.provider in {"auto", "dhan", "dhanhq"} and self.dhan_enabled

    def _should_use_yfinance(self) -> bool:
        return self.provider in {"yfinance", "yf"} and YFINANCE_AVAILABLE

    def _period_to_dates(self, period: str) -> Tuple[datetime, datetime]:
        """Map yfinance-style periods to broker date ranges."""
        end = datetime.now()
        period_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 31,
            "3mo": 93,
            "6mo": 186,
            "1y": 366,
            "2y": 732,
            "5y": 1830,
            "max": 1830,
        }
        return end - timedelta(days=period_days.get(period, 366)), end

    def _fmt_date(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%d")

    def _extract_dhan_rows(self, response) -> pd.DataFrame:
        """Normalize common DhanHQ response shapes into an OHLC dataframe."""
        self.last_response = str(response)[:1000]
        if isinstance(response, dict) and response.get("status") == "failure":
            self.last_error = str(response.get("remarks") or response.get("message") or response)
            return pd.DataFrame()

        payload = response.get("data", response) if isinstance(response, dict) else response
        if payload is None or isinstance(payload, (str, int, float)):
            if isinstance(response, dict):
                self.last_error = str(response.get("remarks") or response.get("message") or response)
            else:
                self.last_error = f"Empty/non-tabular response: {response!r}"
            return pd.DataFrame()
        if isinstance(payload, dict) and payload.get("status") == "failure":
            self.last_error = str(payload.get("remarks") or payload.get("message") or payload)
            return pd.DataFrame()

        if isinstance(payload, dict) and all(k in payload for k in ["open", "high", "low", "close"]):
            rows = payload
            dates = rows.get("timestamp") or rows.get("start_Time") or rows.get("date") or []
            df = pd.DataFrame({
                "date": dates,
                "open": rows.get("open", []),
                "high": rows.get("high", []),
                "low": rows.get("low", []),
                "close": rows.get("close", []),
                "volume": rows.get("volume", [0] * len(rows.get("close", []))),
            })
        else:
            try:
                df = pd.DataFrame(payload)
            except Exception as e:
                self.last_error = f"Could not parse Dhan response: {e}; response={str(response)[:300]}"
                return pd.DataFrame()

        if df.empty:
            self.last_error = f"Dhan returned no rows; response={str(response)[:300]}"
            return df

        rename_map = {
            "start_Time": "date",
            "start_time": "date",
            "timestamp": "date",
            "trading_symbol": "symbol",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "date" not in df.columns:
            df["date"] = pd.date_range(end=datetime.now(), periods=len(df), freq="B")
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce").fillna(pd.to_datetime(df["date"], errors="coerce"))
        df["date"] = df["date"].dt.tz_localize(None)
        if "volume" not in df.columns:
            df["volume"] = 0
        return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    def fetch_current_spot(self) -> Optional[float]:
        """Fetch current NIFTY spot price."""
        if self._should_use_dhan():
            try:
                securities = {"NSE_IDX": [int(self.DHAN_NIFTY_SECURITY_ID)]}
                if hasattr(self.dhan, "ltp_data"):
                    quote = self.dhan.ltp_data(securities=securities)
                elif hasattr(self.dhan, "ticker_data"):
                    quote = self.dhan.ticker_data(securities=securities)
                else:
                    quote = None
                self.last_response = str(quote)[:1000]
                price = self._extract_dhan_ltp(quote, self.DHAN_NIFTY_SECURITY_ID)
                if price is not None:
                    return price
            except Exception as e:
                self.last_error = f"Dhan live LTP error: {e}"
                print(f"Error fetching Dhan spot: {e}")

        if not self._should_use_yfinance():
            return None
        try:
            ticker = yf.Ticker(self.NIFTY_TICKER)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"Error fetching spot: {e}")
        return None

    def _extract_dhan_ltp(self, quote, security_id: str) -> Optional[float]:
        if not quote:
            return None
        payload = quote.get("data", quote) if isinstance(quote, dict) else quote
        candidates = []
        if isinstance(payload, dict):
            candidates.extend([payload])
            candidates.extend(v for v in payload.values() if isinstance(v, dict))
            nested = payload.get("NSE_IDX")
            if isinstance(nested, dict):
                candidates.extend(v for v in nested.values() if isinstance(v, dict))
                if security_id in nested:
                    candidates.append(nested[security_id])
                if int(security_id) in nested:
                    candidates.append(nested[int(security_id)])
        for item in candidates:
            for key in ["last_price", "ltp", "LTP", "close"]:
                if isinstance(item, dict) and key in item:
                    return float(item[key])
        return None

    def fetch_dhan_latest_close(self) -> Optional[Tuple[float, datetime]]:
        """Fetch the latest NIFTY close from Dhan historical data without simulation fallback."""
        if not self._should_use_dhan() or not hasattr(self.dhan, "historical_daily_data"):
            return None
        try:
            start, end = self._period_to_dates("1mo")
            end = end + timedelta(days=1)
            response = self.dhan.historical_daily_data(
                security_id=self.DHAN_NIFTY_SECURITY_ID,
                exchange_segment=self.DHAN_INDEX_SEGMENT,
                instrument_type=self.DHAN_INDEX_TYPE,
                from_date=self._fmt_date(start),
                to_date=self._fmt_date(end),
            )
            df = self._extract_dhan_rows(response)
            if df.empty:
                return None
            row = df.sort_values("date").iloc[-1]
            return float(row["close"]), row["date"].to_pydatetime()
        except Exception as e:
            self.last_error = f"Dhan historical close error: {e}"
            print(f"Error fetching Dhan latest close: {e}")
            return None
    def fetch_historical_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical NIFTY data.

        Args:
            period: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            interval: yfinance interval or Dhan daily/intraday interval
        """
        if self._should_use_dhan():
            try:
                start, end = self._period_to_dates(period)
                if interval == "1d":
                    end = end + timedelta(days=1)
                if interval == "1d" and hasattr(self.dhan, "historical_daily_data"):
                    response = self.dhan.historical_daily_data(
                        security_id=self.DHAN_NIFTY_SECURITY_ID,
                        exchange_segment=self.DHAN_INDEX_SEGMENT,
                        instrument_type=self.DHAN_INDEX_TYPE,
                        from_date=self._fmt_date(start),
                        to_date=self._fmt_date(end),
                    )
                elif hasattr(self.dhan, "intraday_minute_data"):
                    response = self.dhan.intraday_minute_data(
                        security_id=self.DHAN_NIFTY_SECURITY_ID,
                        exchange_segment=self.DHAN_INDEX_SEGMENT,
                        instrument_type=self.DHAN_INDEX_TYPE,
                        from_date=self._fmt_date(start),
                        to_date=self._fmt_date(end),
                        interval=1,
                    )
                else:
                    response = None
                df = self._extract_dhan_rows(response)
                if not df.empty:
                    return df
            except Exception as e:
                self.last_error = f"Dhan historical data error: {e}"
                print(f"Error fetching Dhan historical data: {e}")
                if self.provider in {"dhan", "dhanhq"}:
                    return self._generate_simulated_data(days=252)

        if not self._should_use_yfinance():
            print("External market data disabled or unavailable. Using simulated data.")
            return self._generate_simulated_data(days=252)

        try:
            ticker = yf.Ticker(self.NIFTY_TICKER)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                raise ValueError("No data returned")

            data = data.reset_index()
            data.columns = [c.lower().replace(" ", "_") for c in data.columns]
            data = data.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })

            if "date" not in data.columns:
                date_col = [c for c in data.columns if "date" in c or "datetime" in c]
                if date_col:
                    data = data.rename(columns={date_col[0]: "date"})

            data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
            return data[["date", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self._generate_simulated_data(days=252)

    def fetch_vix_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch India VIX data, or simulate it from NIFTY realized volatility."""
        if self._should_use_dhan():
            try:
                start, end = self._period_to_dates(period)
                if interval == "1d":
                    end = end + timedelta(days=1)
                response = self.dhan.historical_daily_data(
                    security_id=self.DHAN_VIX_SECURITY_ID,
                    exchange_segment=self.DHAN_INDEX_SEGMENT,
                    instrument_type=self.DHAN_INDEX_TYPE,
                    from_date=self._fmt_date(start),
                    to_date=self._fmt_date(end),
                )
                df = self._extract_dhan_rows(response)
                if not df.empty:
                    return df.rename(columns={"close": "vix"})[["date", "vix"]]
            except Exception as e:
                print(f"Error fetching Dhan VIX data: {e}")
                if self.provider in {"dhan", "dhanhq"}:
                    nifty = self.fetch_historical_data(period=period)
                    return self._simulate_vix_from_nifty(nifty)

        if not self._should_use_yfinance():
            nifty = self._generate_simulated_data(days=252)
            return self._simulate_vix_from_nifty(nifty)

        try:
            ticker = yf.Ticker(self.VIX_TICKER)
            data = ticker.history(period=period)

            if not data.empty:
                data = data.reset_index()
                data.columns = [c.lower().replace(" ", "_") for c in data.columns]
                data = data.rename(columns={"close": "vix"})
                data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
                return data[["date", "vix"]]
        except Exception:
            pass

        nifty = self.fetch_historical_data(period=period)
        return self._simulate_vix_from_nifty(nifty)

    def fetch_latest_market_state(self) -> Optional[dict]:
        """Fetch latest market state for signal generation."""
        try:
            hist = self.fetch_historical_data(period="3mo", interval="1d")
            if hist.empty or len(hist) < 50:
                return None

            spot = self.fetch_current_spot() or hist["close"].iloc[-1]

            ema_20 = hist["close"].ewm(span=20).mean().iloc[-1]
            ema_50 = hist["close"].ewm(span=50).mean().iloc[-1]

            delta = hist["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            bb_mid = hist["close"].rolling(window=20).mean().iloc[-1]
            bb_std = hist["close"].rolling(window=20).std().iloc[-1]
            bb_width = (2 * bb_std * 2) / bb_mid

            tr = np.maximum(
                hist["high"] - hist["low"],
                np.maximum(
                    abs(hist["high"] - hist["close"].shift(1)),
                    abs(hist["low"] - hist["close"].shift(1)),
                ),
            )
            atr = tr.rolling(window=14).mean().iloc[-1]
            adx = 20 + (atr / spot * 100) * 2

            vix_data = self.fetch_vix_data(period="3mo")
            vix = vix_data["vix"].iloc[-1] if not vix_data.empty else 15

            if not vix_data.empty and vix_data["vix"].max() != vix_data["vix"].min():
                iv_pct = (vix - vix_data["vix"].min()) / (vix_data["vix"].max() - vix_data["vix"].min()) * 100
            else:
                iv_pct = 50

            return {
                "spot": float(spot),
                "vix": float(vix),
                "iv_percentile": float(iv_pct),
                "vwap": float(spot),
                "ema_20": float(ema_20),
                "ema_50": float(ema_50),
                "rsi_14": float(rsi),
                "bb_width": float(bb_width),
                "adx": float(adx),
                "alpha": 0.15,
                "alpha2": 0.15,
                "day_of_week": datetime.now().weekday(),
                "days_to_expiry": self._days_to_expiry(),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            print(f"Error fetching market state: {e}")
            return None

    def _simulate_vix_from_nifty(self, nifty_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate VIX from NIFTY realized volatility with mean reversion."""
        log_returns = np.log(nifty_data["close"] / nifty_data["close"].shift(1))
        realized_vol = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
        iv_premium = 2 + np.random.normal(0, 1, len(nifty_data))
        vix = (realized_vol + iv_premium).fillna(15).clip(10, 50)
        return pd.DataFrame({"date": nifty_data["date"], "vix": vix})

    def _generate_simulated_data(self, days: int = 252) -> pd.DataFrame:
        """Generate simulated NIFTY data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        returns = np.random.normal(0.0003, 0.015, days)
        prices = 23500 * np.exp(np.cumsum(returns))
        daily_vol = 0.015 * prices
        highs = prices + np.abs(np.random.normal(0, daily_vol * 0.5, days))
        lows = prices - np.abs(np.random.normal(0, daily_vol * 0.5, days))
        opens = prices + np.random.normal(0, daily_vol * 0.3, days)
        volumes = np.random.randint(5000000, 15000000, days)

        return pd.DataFrame({
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        })

    def _days_to_expiry(self) -> int:
        """Calculate days to next weekly expiry. Adjust this if exchange rules change."""
        today = datetime.now()
        days_until_tuesday = (1 - today.weekday()) % 7
        if days_until_tuesday == 0:
            days_until_tuesday = 7
        return days_until_tuesday













