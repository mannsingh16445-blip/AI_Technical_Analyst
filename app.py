"""EMA200 Breakout + Low-Volume Retest + CCI(20) Momentum Scanner

Streamlit stock-discovery scanner. It reports candidates for manual chart review;
it does not issue buy/sell instructions or calculate entries/exits.

Core conditions:
1) Recent clean bullish candle closes/crosses above EMA200.
2) Price subsequently retests EMA200 on low volume while EMA9 and EMA21 are
   both below EMA200, rising, and closing the distance to EMA200.
3) CCI(20) is above +100 for the last N completed daily sessions (default 5).
4) Latest completed close exceeds each of the prior 5 closes and the breakout
   candle's close; latest close must also remain above current EMA200.
"""

import datetime as dt
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="EMA200 Retest + CCI Scanner",
    page_icon="📈",
    layout="wide",
)

# --------------------------- Constants ---------------------------
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
]

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}


def _valid_symbols(values):
    out = []
    for value in values:
        s = str(value).strip().upper()
        if not s or s in {"NAN", "NONE", "NULL", "SYMBOL", "TICKER"}:
            continue
        out.append(s)
    return list(dict.fromkeys(out))


def _to_yahoo_ticker(symbol, default_exchange="NSE"):
    s = str(symbol).strip().upper()
    if not s:
        return ""
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if default_exchange == "BSE":
        return f"{s}.BO"
    return f"{s}.NS"


@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_equity():
    urls = [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    for url in urls:
        try:
            session = requests.Session()
            session.headers.update(UA_HEADERS)
            try:
                session.get("https://www.nseindia.com/", timeout=12)
            except Exception:
                pass
            resp = session.get(url, timeout=25)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [str(c).strip().upper() for c in df.columns]
            if "SYMBOL" not in df.columns:
                continue
            if "SERIES" in df.columns:
                # Include the common listed equity series; exclude funds/debt where possible.
                series = df["SERIES"].astype(str).str.strip().str.upper()
                keep = series.isin(["EQ", "BE", "BZ", "SM", "ST", "E2", "E1"])
                if keep.sum() >= 500:
                    df = df.loc[keep]
            symbols = _valid_symbols(df["SYMBOL"].tolist())
            if len(symbols) >= 500:
                return [f"{s}.NS" for s in symbols]
        except Exception:
            continue
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():
    urls = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/nifty500_symbols.csv",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=UA_HEADERS, timeout=25)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [str(c).strip().upper() for c in df.columns]
            col = "SYMBOL" if "SYMBOL" in df.columns else None
            if not col:
                continue
            symbols = _valid_symbols(df[col].tolist())
            if len(symbols) >= 400:
                return [f"{s}.NS" for s in sorted(symbols)[:500]]
        except Exception:
            continue
    return []


@st.cache_data(ttl=21600, show_spinner=False)
def load_bse_active_bhavcopy():
    """Build an active BSE universe from the most recent accessible daily bhavcopy.

    This is a recent-traded BSE universe, not a guarantee of every BSE listing.
    Users can upload/paste a broader BSE scrip master for full-universe scans.
    """
    today = dt.date.today()
    for back in range(0, 12):
        day = today - dt.timedelta(days=back)
        stamp = day.strftime("%Y%m%d")
        url = (
            "https://www.bseindia.com/download/BhavCopy/Equity/"
            f"BhavCopy_BSE_CM_0_0_0_{stamp}_F_0000.CSV"
        )
        try:
            resp = requests.get(url, headers=UA_HEADERS, timeout=18)
            if resp.status_code != 200 or len(resp.content) < 1000:
                continue
            df = pd.read_csv(StringIO(resp.content.decode("utf-8", errors="ignore")))
            df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
            code_col = next((c for c in ["SCRIP_CD", "SCRIPCODE", "SC_CODE", "SCRIP_CODE", "SECURITY_CODE"] if c in df.columns), None)
            if code_col is None:
                # Some BSE files use UDiFF instrument labels; only accept an obvious code field.
                code_col = next((c for c in df.columns if "SCRIP" in c and "CD" in c), None)
            if code_col is None:
                continue
            codes = pd.to_numeric(df[code_col], errors="coerce").dropna().astype("int64").astype(str)
            codes = codes[codes.str.len().between(5, 6)]
            if len(codes) >= 250:
                return [f"{code}.BO" for code in sorted(codes.drop_duplicates().tolist())]
        except Exception:
            continue
    return []


def parse_manual_symbols(text, exchange):
    raw = str(text or "").replace("\n", ",").replace(";", ",").split(",")
    symbols = _valid_symbols(raw)
    return [_to_yahoo_ticker(s, exchange) for s in symbols]


def parse_uploaded_universe(uploaded):
    if uploaded is None:
        return []
    try:
        df = pd.read_csv(uploaded)
        df.columns = [str(c).strip().upper() for c in df.columns]
        sym_col = next((c for c in ["YAHOO_TICKER", "TICKER", "SYMBOL", "SCRIPCODE", "SCRIP_CODE"] if c in df.columns), None)
        if sym_col is None:
            return []
        exchange_col = next((c for c in ["EXCHANGE", "MARKET"] if c in df.columns), None)
        tickers = []
        for _, row in df.iterrows():
            exch = str(row[exchange_col]).strip().upper() if exchange_col else "NSE"
            exch = "BSE" if exch in {"BSE", "BO", "BOMBAY"} else "NSE"
            tickers.append(_to_yahoo_ticker(row[sym_col], exch))
        return list(dict.fromkeys(t for t in tickers if t))
    except Exception:
        return []


# --------------------------- Price data ---------------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_batch_cached(tickers_tuple, period="3y"):
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="ticker",
            timeout=25,
        )
    except Exception:
        return pd.DataFrame()


def extract_one_ticker(downloaded, ticker, batch):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy() if len(batch) == 1 else pd.DataFrame()
    try:
        lvl0 = downloaded.columns.get_level_values(0)
        lvl1 = downloaded.columns.get_level_values(1)
        if ticker in lvl0:
            frame = downloaded[ticker].copy()
        elif ticker in lvl1:
            frame = downloaded.xs(ticker, axis=1, level=1).copy()
        elif len(batch) == 1:
            frame = downloaded.copy()
            frame.columns = frame.columns.get_level_values(-1)
        else:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(-1)
        return frame
    except Exception:
        return pd.DataFrame()


# --------------------------- Strategy logic ---------------------------
def calculate_indicators(df):
    d = df.copy()
    d.columns = [str(c).title() for c in d.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in d.columns for c in required):
        return pd.DataFrame()
    d = d[required].apply(pd.to_numeric, errors="coerce").dropna(subset=required)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if len(d) < 230:
        return pd.DataFrame()

    close = d["Close"]
    d["EMA9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    d["EMA21"] = close.ewm(span=21, adjust=False, min_periods=21).mean()
    d["EMA200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    d["VOL_SMA20"] = d["Volume"].rolling(20, min_periods=20).mean()

    tp = (d["High"] + d["Low"] + close) / 3.0
    tp_mean = tp.rolling(20, min_periods=20).mean()
    mean_dev = tp.rolling(20, min_periods=20).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    d["CCI20"] = (tp - tp_mean) / (0.015 * mean_dev.replace(0, np.nan))
    d["EMA9_DIST200_PCT"] = (d["EMA200"] - d["EMA9"]) / d["EMA200"] * 100.0
    d["EMA21_DIST200_PCT"] = (d["EMA200"] - d["EMA21"]) / d["EMA200"] * 100.0
    d["EMA9_RISE"] = d["EMA9"] > d["EMA9"].shift(3)
    d["EMA21_RISE"] = d["EMA21"] > d["EMA21"].shift(3)
    d["EMA9_APPROACH"] = d["EMA9_DIST200_PCT"] < d["EMA9_DIST200_PCT"].shift(5)
    d["EMA21_APPROACH"] = d["EMA21_DIST200_PCT"] < d["EMA21_DIST200_PCT"].shift(5)

    candle_range = (d["High"] - d["Low"]).replace(0, np.nan)
    body = (d["Close"] - d["Open"]).abs()
    d["WICK_PCT"] = ((candle_range - body).clip(lower=0) / candle_range * 100.0)
    d["GREEN"] = d["Close"] > d["Open"]
    d["CROSS_UP_200"] = (d["Close"] > d["EMA200"]) & (d["Close"].shift(1) <= d["EMA200"].shift(1))
    return d


def evaluate_stock(ticker, raw_df, cfg):
    d = calculate_indicators(raw_df)
    if d.empty or len(d) < max(230, cfg["cci_days"] + 8):
        return {"Ticker": ticker, "Qualified": False, "Pass Count": 0, "Diagnostic": "Insufficient/invalid daily history"}

    d = d.dropna(subset=["EMA200", "CCI20", "VOL_SMA20"]).copy()
    if len(d) < max(10, cfg["cci_days"] + 2):
        return {"Ticker": ticker, "Qualified": False, "Pass Count": 0, "Diagnostic": "Insufficient indicators"}

    last_i = len(d) - 1
    latest = d.iloc[last_i]
    prior5 = d.iloc[max(0, last_i - 5):last_i]
    if len(prior5) < 5:
        return {"Ticker": ticker, "Qualified": False, "Pass Count": 0, "Diagnostic": "Need 5 prior closes"}

    # Rule 1: locate a recent clean bullish breakout candle crossing EMA200 from below.
    breakout = None
    search_start = max(1, last_i - cfg["breakout_lookback"])
    for i in range(last_i - 1, search_start - 1, -1):
        r = d.iloc[i]
        if (
            bool(r["GREEN"])
            and float(r["Close"]) > float(r["EMA200"])
            and float(r["WICK_PCT"]) <= cfg["max_wick_pct"]
            and (not cfg["require_cross_from_below"] or bool(r["CROSS_UP_200"]))
        ):
            breakout = (i, r)
            break

    condition1 = breakout is not None
    breakout_i = breakout[0] if breakout else None
    breakout_row = breakout[1] if breakout else None

    # Rule 2: after breakout, identify a low-volume retest while EMA9/21 approach 200 from below.
    retest = None
    if breakout is not None:
        # Avoid using the latest candle as a retest; it must be followed by confirmation.
        for i in range(last_i - 1, breakout_i, -1):
            r = d.iloc[i]
            e200 = float(r["EMA200"])
            if not np.isfinite(e200) or e200 <= 0:
                continue
            low_distance = abs(float(r["Low"]) - e200) / e200 * 100.0
            close_floor_ok = float(r["Close"]) >= e200 * (1.0 - cfg["retest_tolerance_pct"] / 100.0)
            touches_zone = (
                float(r["Low"]) <= e200 * (1.0 + cfg["retest_tolerance_pct"] / 100.0)
                and low_distance <= cfg["retest_tolerance_pct"]
                and close_floor_ok
            )
            vol_sma = float(r["VOL_SMA20"])
            low_volume = (
                np.isfinite(vol_sma) and vol_sma > 0
                and float(r["Volume"]) <= vol_sma * cfg["retest_volume_max_ratio"]
                and float(r["Volume"]) < float(breakout_row["Volume"])
            )
            both_below = float(r["EMA9"]) < e200 and float(r["EMA21"]) < e200
            approaching = (
                bool(r["EMA9_RISE"]) and bool(r["EMA21_RISE"])
                and bool(r["EMA9_APPROACH"]) and bool(r["EMA21_APPROACH"])
            )
            if touches_zone and low_volume and both_below and approaching:
                retest = (i, r)
                break

    condition2 = retest is not None

    # Rule 3: CCI has remained above +100 for the last N completed daily bars.
    cci_slice = d["CCI20"].iloc[-cfg["cci_days"]:]
    condition3 = len(cci_slice) == cfg["cci_days"] and bool((cci_slice > cfg["cci_threshold"]).all())

    # Rule 4: latest completed close exceeds each of prior five closes and the breakout close.
    close_above_prior5 = bool(float(latest["Close"]) > float(prior5["Close"].max()))
    close_above_breakout = bool(breakout is not None and float(latest["Close"]) > float(breakout_row["Close"]))
    above_current_ema200 = bool(float(latest["Close"]) > float(latest["EMA200"]))
    condition4 = close_above_prior5 and close_above_breakout and above_current_ema200

    conditions = [condition1, condition2, condition3, condition4]
    passed = int(sum(conditions))
    out = {
        "Ticker": ticker,
        "Qualified": all(conditions),
        "Pass Count": passed,
        "Condition 1 — Clean EMA200 breakout": "PASS" if condition1 else "FAIL",
        "Condition 2 — Low-volume retest + EMAs approaching": "PASS" if condition2 else "FAIL",
        f"Condition 3 — CCI > {cfg['cci_threshold']} for {cfg['cci_days']} days": "PASS" if condition3 else "FAIL",
        "Condition 4 — Close above prior 5 closes + breakout close": "PASS" if condition4 else "FAIL",
        "Latest Date": pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d"),
        "Latest Close": float(latest["Close"]),
        "EMA9": float(latest["EMA9"]),
        "EMA21": float(latest["EMA21"]),
        "EMA200": float(latest["EMA200"]),
        "CCI20": float(latest["CCI20"]),
        "CCI Days Above Threshold": int((cci_slice > cfg["cci_threshold"]).sum()),
        "Close > Prior 5-Close High (%)": (float(latest["Close"]) / float(prior5["Close"].max()) - 1) * 100,
        "Volume / SMA20 (latest)": float(latest["Volume"] / latest["VOL_SMA20"]) if latest["VOL_SMA20"] > 0 else np.nan,
        "Breakout Date": pd.Timestamp(breakout_row.name).strftime("%Y-%m-%d") if breakout else "",
        "Breakout Close": float(breakout_row["Close"]) if breakout else np.nan,
        "Breakout Wick %": float(breakout_row["WICK_PCT"]) if breakout else np.nan,
        "Breakout Volume": float(breakout_row["Volume"]) if breakout else np.nan,
        "Retest Date": pd.Timestamp(retest[1].name).strftime("%Y-%m-%d") if retest else "",
        "Retest Close": float(retest[1]["Close"]) if retest else np.nan,
        "Retest Low": float(retest[1]["Low"]) if retest else np.nan,
        "Retest EMA200": float(retest[1]["EMA200"]) if retest else np.nan,
        "Retest Vol / SMA20": float(retest[1]["Volume"] / retest[1]["VOL_SMA20"]) if retest and retest[1]["VOL_SMA20"] > 0 else np.nan,
        "Retest EMA9 Below EMA200": bool(retest[1]["EMA9"] < retest[1]["EMA200"]) if retest else False,
        "Retest EMA21 Below EMA200": bool(retest[1]["EMA21"] < retest[1]["EMA200"]) if retest else False,
        "Close > Breakout Close (%)": (float(latest["Close"]) / float(breakout_row["Close"]) - 1) * 100 if breakout else np.nan,
    }
    if out["Qualified"]:
        out["Diagnostic"] = "Candidate for manual chart review — all four rules pass"
    else:
        out["Diagnostic"] = f"{passed}/4 conditions pass"
    return out


# --------------------------- User Interface ---------------------------
st.title("📈 EMA200 Breakout → Low-Volume Retest Scanner")
st.caption("Daily stock-discovery scanner based on your four conditions. No automatic entry, stop-loss, target, or exit is generated.")

with st.expander("Scanner rules implemented", expanded=True):
    st.markdown(
        """
        1. **Breakout candle:** a green candle closes above EMA200; total wick is configurable. By default, the close must cross EMA200 from below.
        2. **Retest:** after breakout, a candle returns near EMA200 on volume below its 20-day average threshold and below breakout-candle volume. At that retest, EMA9 and EMA21 must still be below EMA200, rising, and moving closer to EMA200.
        3. **CCI(20):** above +100 on each of the last five completed daily candles by default.
        4. **Price confirmation:** latest completed close is above each of the prior five closes, above the breakout candle close, and above current EMA200.
        """
    )

st.sidebar.header("1. Stock universe")
universe = st.sidebar.selectbox(
    "Choose stocks to scan",
    [
        "Nifty 50",
        "Nifty 500",
        "Full NSE Equity",
        "BSE Active (recent bhavcopy)",
        "NSE + BSE Active",
        "Manual Symbols",
        "Upload a symbol CSV",
    ],
)
manual_exchange = "NSE"
manual_text = "RELIANCE, BHARTIARTL, SBIN"
uploaded_file = None
if universe == "Manual Symbols":
    manual_exchange = st.sidebar.selectbox("Default exchange for symbols without suffix", ["NSE", "BSE"])
    manual_text = st.sidebar.text_area(
        "Symbols (comma/newline separated)",
        value="RELIANCE, BHARTIARTL, SBIN",
        help="Use .NS or .BO to override the default exchange. For BSE, numeric scrip codes such as 500325 are supported with the BSE default.",
    )
elif universe == "Upload a symbol CSV":
    uploaded_file = st.sidebar.file_uploader("CSV with Symbol/Ticker column (optional Exchange column)", type=["csv"])

st.sidebar.header("2. Pattern thresholds")
breakout_lookback = st.sidebar.slider("Search breakout in prior N sessions", 5, 60, 30, 1)
max_wick_pct = st.sidebar.slider("Maximum total wick (% of candle range)", 5, 60, 25, 5)
require_cross_from_below = st.sidebar.checkbox("Require breakout candle to cross EMA200 from below", value=True)
retest_tolerance_pct = st.sidebar.slider("Retest distance from EMA200 (%)", 0.25, 3.0, 1.0, 0.25)
retest_volume_max_ratio = st.sidebar.slider("Maximum retest volume / SMA20", 0.25, 1.25, 0.80, 0.05)
cci_days = st.sidebar.slider("Consecutive recent sessions CCI must stay above threshold", 3, 10, 5, 1)
cci_threshold = st.sidebar.slider("CCI threshold", 50, 200, 100, 10)

st.sidebar.header("3. Scan settings")
period = st.sidebar.selectbox("Daily history", ["2y", "3y", "5y"], index=1)
batch_size = st.sidebar.slider("Download batch size", 10, 80, 40, 10)
max_stocks = st.sidebar.number_input("Maximum stocks (0 = all loaded stocks)", min_value=0, max_value=10000, value=500, step=100)
show_near_misses = st.sidebar.checkbox("Show near-misses (1–3 conditions passed)", value=True)

cfg = {
    "breakout_lookback": int(breakout_lookback),
    "max_wick_pct": float(max_wick_pct),
    "require_cross_from_below": bool(require_cross_from_below),
    "retest_tolerance_pct": float(retest_tolerance_pct),
    "retest_volume_max_ratio": float(retest_volume_max_ratio),
    "cci_days": int(cci_days),
    "cci_threshold": float(cci_threshold),
}

if universe == "Nifty 50":
    tickers = [f"{s}.NS" for s in NIFTY50]
elif universe == "Nifty 500":
    with st.spinner("Loading Nifty 500 constituents..."):
        tickers = load_nifty500()
elif universe == "Full NSE Equity":
    with st.spinner("Loading NSE equity list..."):
        tickers = load_nse_equity()
elif universe == "BSE Active (recent bhavcopy)":
    with st.spinner("Loading BSE active universe from recent bhavcopy..."):
        tickers = load_bse_active_bhavcopy()
elif universe == "NSE + BSE Active":
    with st.spinner("Loading NSE and BSE active universes..."):
        tickers = load_nse_equity() + load_bse_active_bhavcopy()
elif universe == "Manual Symbols":
    tickers = parse_manual_symbols(manual_text, manual_exchange)
else:
    tickers = parse_uploaded_universe(uploaded_file)

tickers = list(dict.fromkeys(t for t in tickers if t))
if max_stocks and len(tickers) > int(max_stocks):
    tickers = tickers[: int(max_stocks)]

col_a, col_b, col_c = st.columns(3)
col_a.metric("Universe", universe)
col_b.metric("Stocks queued", f"{len(tickers):,}")
col_c.metric("History", period)

if universe == "BSE Active (recent bhavcopy)" or universe == "NSE + BSE Active":
    st.info("BSE universe is built from the latest accessible daily bhavcopy, so it represents recent traded securities rather than a guaranteed exhaustive list of every BSE listing. For a complete custom BSE list, upload a current BSE scrip-master CSV.")

if not tickers:
    st.warning("No symbols could be loaded. Try again, use Manual Symbols, or upload your exchange symbol CSV.")

run = st.button("🔎 Scan stocks", type="primary", disabled=not tickers)
if run:
    all_rows = []
    bar = st.progress(0)
    status = st.empty()
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(tickers), batch_size), start=1):
        batch = tickers[start : start + batch_size]
        status.write(f"Downloading and checking batch {batch_no}/{n_batches} ({len(batch)} stocks)...")
        downloaded = download_batch_cached(tuple(batch), period=period)
        for ticker in batch:
            frame = extract_one_ticker(downloaded, ticker, batch)
            if frame.empty:
                all_rows.append({"Ticker": ticker, "Qualified": False, "Pass Count": 0, "Diagnostic": "No Yahoo Finance OHLCV data returned"})
                continue
            try:
                all_rows.append(evaluate_stock(ticker, frame, cfg))
            except Exception as exc:
                all_rows.append({"Ticker": ticker, "Qualified": False, "Pass Count": 0, "Diagnostic": f"Analysis error: {type(exc).__name__}"})
        bar.progress(min(batch_no / max(1, n_batches), 1.0))
        time.sleep(0.15)

    bar.empty()
    status.empty()
    results = pd.DataFrame(all_rows)
    if results.empty:
        st.error("The scan returned no analyzable symbols. Check data-provider availability and ticker formats.")
    else:
        qualified = results[results.get("Qualified", False) == True].copy()  # noqa: E712
        if not qualified.empty:
            qualified = qualified.sort_values(
                ["Close > Prior 5-Close High (%)", "Close > Breakout Close (%)", "CCI20"],
                ascending=False,
                na_position="last",
            )
            st.success(f"Found {len(qualified)} candidate(s) passing all four conditions.")
            st.caption("These are candidates for manual chart review, not trading instructions.")
            preferred = [
                "Ticker", "Latest Date", "Latest Close", "Breakout Date", "Breakout Close",
                "Breakout Wick %", "Retest Date", "Retest Vol / SMA20", "EMA9", "EMA21",
                "EMA200", "CCI20", "CCI Days Above Threshold", "Close > Prior 5-Close High (%)",
                "Close > Breakout Close (%)", "Condition 1 — Clean EMA200 breakout",
                "Condition 2 — Low-volume retest + EMAs approaching",
                f"Condition 3 — CCI > {cfg['cci_threshold']} for {cfg['cci_days']} days",
                "Condition 4 — Close above prior 5 closes + breakout close",
            ]
            show_cols = [c for c in preferred if c in qualified.columns]
            st.dataframe(qualified[show_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download qualified candidates CSV",
                data=qualified.to_csv(index=False).encode("utf-8"),
                file_name="ema200_retest_cci100_candidates.csv",
                mime="text/csv",
            )
        else:
            st.warning("No stock passed all four rules in this scan. See near-misses to identify which condition is too restrictive.")

        if show_near_misses:
            misses = results[results.get("Qualified", False) != True].copy()  # noqa: E712
            if not misses.empty:
                misses = misses.sort_values(["Pass Count", "Ticker"], ascending=[False, True]).head(150)
                st.subheader("Near-misses / scan diagnostics")
                st.caption("These stocks do not satisfy every rule. The condition columns help explain why they were excluded.")
                diag_cols = [
                    "Ticker", "Pass Count", "Diagnostic", "Latest Date", "Latest Close",
                    "Breakout Date", "Retest Date", "CCI20", "CCI Days Above Threshold",
                    "Condition 1 — Clean EMA200 breakout", "Condition 2 — Low-volume retest + EMAs approaching",
                    f"Condition 3 — CCI > {cfg['cci_threshold']} for {cfg['cci_days']} days",
                    "Condition 4 — Close above prior 5 closes + breakout close",
                ]
                diag_cols = [c for c in diag_cols if c in misses.columns]
                st.dataframe(misses[diag_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download scan diagnostics CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="ema200_retest_cci100_scan_diagnostics.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Interpretation note: data is daily OHLCV from Yahoo Finance and can be delayed, adjusted differently, or incomplete for some securities. Review each candidate on your charting platform. The scanner does not place trades or make discretionary decisions.")
