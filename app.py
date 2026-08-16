import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
import os
import time

from io import StringIO

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MOBILE + DESKTOP CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }


    /* Mobile */

    @media (max-width: 768px) {

        .block-container {
            padding-top: 0.5rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        h2 {
            font-size: 1.3rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        [data-testid="stMetric"] {
            padding: 0.3rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }

        .stButton button {
            width: 100%;
            min-height: 2.6rem;
        }

        input {
            font-size: 16px !important;
        }

        .desktop-only {
            display: none !important;
        }

        .mobile-only {
            display: block !important;
        }

        .stock-card {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .stock-name {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .stock-score {
            font-size: 1rem;
            font-weight: 700;
            margin-top: 5px;
            margin-bottom: 8px;
        }

        .stock-line {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.9rem;
        }

    }

    .mobile-only {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = None

try:
    OPENAI_API_KEY = st.secrets.get(
        "OPENAI_API_KEY",
        None
    )
except Exception:
    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY"
    )

if (
    OPENAI_API_KEY
    and OpenAI is not None
):

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    client = None


# ============================================================
# APP TITLE
# ============================================================

st.title(
    "📈 AI Technical Analyst"
)

st.caption(
    "NSE Technical Analysis • Smart Scanner • AI Assistant"
)


# ============================================================
# NIFTY 50
# ============================================================

NIFTY50 = [

    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO"

]


# ============================================================
# COMMON HTTP HEADERS
# ============================================================

NSE_HEADERS = {

    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36",

    "Accept":
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,text/csv;q=0.8,*/*;q=0.7",

    "Accept-Language":
        "en-US,en;q=0.9",

    "Referer":
        "https://www.nseindia.com/"

}


# ============================================================
# GENERIC CSV LOADER
# ============================================================

def download_csv(
    url,
    headers=None,
    timeout=30
):

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout
        )

        if response.status_code != 200:
            return None

        if not response.text.strip():
            return None

        return pd.read_csv(
            StringIO(response.text)
        )

    except Exception:

        return None


# ============================================================
# FULL NSE EQUITY LIST
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_full_nse():

    urls = [

        # Current NSE equity securities file
        "https://nsearchives.nseindia.com/"
        "content/equities/EQUITY_L.csv",

        # Archive fallback
        "https://archives.nseindia.com/"
        "content/equities/EQUITY_L.csv"

    ]

    for url in urls:

        df = download_csv(
            url,
            NSE_HEADERS
        )

        if df is None:
            continue

        df.columns = [
            str(c).strip().upper()
            for c in df.columns
        ]

        if "SYMBOL" not in df.columns:
            continue

        symbols = (
            df["SYMBOL"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Equity series only where available
        if "SERIES" in df.columns:

            series = (
                df["SERIES"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            eq_symbols = symbols[
                series == "EQ"
            ]

            if len(eq_symbols) > 1000:

                symbols = eq_symbols

        symbols = symbols[
            ~symbols.isin(
                [
                    "",
                    "NAN",
                    "NONE"
                ]
            )
        ]

        symbols = sorted(
            symbols
            .drop_duplicates()
            .tolist()
        )

        if len(symbols) > 1000:

            return symbols

    return []


# ============================================================
# NIFTY 500
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_nifty500():

    urls = [

        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://archives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"

    ]

    for url in urls:

        headers = dict(
            NSE_HEADERS
        )

        headers[
            "Referer"
        ] = (
            "https://www.niftyindices.com/"
        )

        df = download_csv(
            url,
            headers
        )

        if df is None:
            continue

        df.columns = [
            str(c).strip().upper()
            for c in df.columns
        ]

        if "SYMBOL" not in df.columns:
            continue

        symbols = (
            df["SYMBOL"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        symbols = symbols[
            ~symbols.isin(
                [
                    "",
                    "NAN",
                    "NONE"
                ]
            )
        ]

        symbols = sorted(
            symbols
            .drop_duplicates()
            .tolist()
        )

        if len(symbols) >= 450:

            return symbols

    return []


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_stock_data(
    symbol,
    period="1y"
):

    ticker = (
        symbol.upper().strip()
        + ".NS"
    )

    try:

        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if data is None:
            return pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        # Handle MultiIndex
        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = [
                column[0]
                for column
                in data.columns
            ]

        data.columns = [
            str(c).strip()
            for c in data.columns
        ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            c
            for c in required
            if c not in data.columns
        ]

        if missing:

            return pd.DataFrame()

        data = data[
            required
        ].copy()

        data = data.dropna(
            subset=required
        )

        return data

    except Exception:

        return pd.DataFrame()


# ============================================================
# BATCH DOWNLOAD
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def get_batch_data(
    symbols,
    period="1y"
):

    result = {}

    symbols = list(
        dict.fromkeys(symbols)
    )

    for symbol in symbols:

        data = get_stock_data(
            symbol,
            period
        )

        if not data.empty:

            result[
                symbol
            ] = data

    return result


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(
    data
):

    data = data.copy()

    if data.empty:
        return data

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    data["SMA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["SMA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["SMA200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = (
        data["Close"]
        .diff()
    )

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    data["RSI14"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema12
        -
        ema26
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["MACD_HIST"] = (
        data["MACD"]
        -
        data["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    data["AVG_VOLUME20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["VOLUME_RATIO"] = (
        data["Volume"]
        /
        data["AVG_VOLUME20"]
    )

    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    data["TURNOVER"] = (
        data["Close"]
        *
        data["Volume"]
    )

    data["AVG_TURNOVER20"] = (
        data["TURNOVER"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # DONCHIAN
    # --------------------------------------------------------

    data["DONCHIAN_UPPER"] = (
        data["High"]
        .shift(1)
        .rolling(3)
        .max()
    )

    data["DONCHIAN_LOWER"] = (
        data["Low"]
        .shift(1)
        .rolling(3)
        .min()
    )

    return data


# ============================================================
# BREAKOUT ANALYSIS
# ============================================================

def analyze_stock(
    data
):

    if len(data) < 210:

        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    n_minus_2 = data.iloc[-3]

    values = [

        latest["Close"],
        latest["SMA200"],
        latest["RSI14"],
        latest["VOLUME_RATIO"],
        latest["DONCHIAN_UPPER"],
        latest["DONCHIAN_LOWER"]

    ]

    if any(
        pd.isna(v)
        for v in values
    ):

        return None

    # --------------------------------------------------------
    # C1
    # --------------------------------------------------------

    c1 = (
        latest["Close"]
        >
        previous["High"]
    )

    # --------------------------------------------------------
    # C2
    # --------------------------------------------------------

    c2 = (
        previous["High"]
        <
        n_minus_2["High"]
    )

    # --------------------------------------------------------
    # C3
    # --------------------------------------------------------

    c3 = (
        latest["Close"]
        <
        latest["DONCHIAN_UPPER"]
    )

    # --------------------------------------------------------
    # C4
    # --------------------------------------------------------

    c4 = (
        latest["Low"]
        >
        latest["DONCHIAN_LOWER"]
    )

    # --------------------------------------------------------
    # C5
    # --------------------------------------------------------

    c5 = (
        latest["Close"]
        >
        latest["SMA200"]
    )

    # --------------------------------------------------------
    # Confirmations
    # --------------------------------------------------------

    volume_confirm = (
        latest["VOLUME_RATIO"]
        >= 1.5
    )

    rsi_confirm = (
        latest["RSI14"]
        > 50
    )

    macd_confirm = (
        latest["MACD"]
        >
        latest["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0

    if c1:
        score += 2

    if c2:
        score += 1

    if c3:
        score += 1

    if c4:
        score += 1

    if c5:
        score += 2

    if volume_confirm:
        score += 1

    if rsi_confirm:
        score += 1

    if macd_confirm:
        score += 1

    return {

        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,

        "Volume Confirm":
            volume_confirm,

        "RSI Confirm":
            rsi_confirm,

        "MACD Confirm":
            macd_confirm,

        "Score":
            score,

        "Close":
            float(
                latest["Close"]
            ),

        "SMA20":
            float(
                latest["SMA20"]
            ),

        "SMA50":
            float(
                latest["SMA50"]
            ),

        "SMA200":
            float(
                latest["SMA200"]
            ),

        "RSI":
            float(
                latest["RSI14"]
            ),

        "MACD":
            float(
                latest["MACD"]
            ),

        "MACD Signal":
            float(
                latest["MACD_SIGNAL"]
            ),

        "Volume Ratio":
            float(
                latest["VOLUME_RATIO"]
            ),

        "Donchian Upper":
            float(
                latest["DONCHIAN_UPPER"]
            ),

        "Donchian Lower":
            float(
                latest["DONCHIAN_LOWER"]
            )

    }


# ============================================================
# TECHNICAL CHART FUNCTION
# ============================================================

def create_chart(
    data,
    symbol
):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(

            x=data.index,

            open=data["Open"],

            high=data["High"],

            low=data["Low"],

            close=data["Close"],

            name=symbol

        )
    )

    fig.add_trace(
        go.Scatter(

            x=data.index,

            y=data["SMA20"],

            mode="lines",

            name="SMA 20"

        )
    )

    fig.add_trace(
        go.Scatter(

            x=data.index,

            y=data["SMA50"],

            mode="lines",

            name="SMA 50"

        )
    )

    fig.add_trace(
        go.Scatter(

            x=data.index,

            y=data["SMA200"],

            mode="lines",

            name="SMA 200"

        )
    )

    fig.add_trace(
        go.Scatter(

            x=data.index,

            y=data[
                "DONCHIAN_UPPER"
            ],

            mode="lines",

            name="Donchian Upper"

        )
    )

    fig.add_trace(
        go.Scatter(

            x=data.index,

            y=data[
                "DONCHIAN_LOWER"
            ],

            mode="lines",

            name="Donchian Lower"

        )
    )

    fig.update_layout(

        title=(
            f"{symbol} — "
            "Technical Analysis"
        ),

        height=600,

        xaxis_rangeslider_visible=False,

        hovermode="x unified",

        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )

    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Control Panel"
)

module = st.sidebar.radio(

    "Select Module",

    [
        "📊 Technical Chart",
        "🚀 Smart Scanner",
        "🤖 AI Assistant"
    ]

)


# ============================================================
# TECHNICAL CHART
# ============================================================

if module == "📊 Technical Chart":

    st.header(
        "📊 Technical Chart"
    )

    symbol = st.sidebar.text_input(
        "NSE Symbol",
        value="RELIANCE"
    )

    symbol = (
        symbol
        .strip()
        .upper()
    )

    period = st.sidebar.selectbox(

        "Chart Period",

        [
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y"
        ],

        index=2

    )

    if st.sidebar.button(
        "📈 Load Chart",
        type="primary"
    ):

        with st.spinner(
            f"Loading {symbol}..."
        ):

            data = get_stock_data(
                symbol,
                period
            )

        if data.empty:

            st.error(
                f"""
                Could not retrieve data for
                **{symbol}.NS**

                Please check the NSE symbol.
                """
            )

        else:

            data = calculate_indicators(
                data
            )

            fig = create_chart(
                data,
                symbol
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            latest = data.iloc[-1]

            # Desktop/mobile metrics
            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(
                "Close",
                f"₹{latest['Close']:.2f}"
            )

            c2.metric(
                "SMA 20",
                f"₹{latest['SMA20']:.2f}"
            )

            c3.metric(
                "SMA 50",
                f"₹{latest['SMA50']:.2f}"
            )

            c4.metric(
                "SMA 200",
                f"₹{latest['SMA200']:.2f}"
            )

            c5.metric(
                "RSI",
                f"{latest['RSI14']:.1f}"
            )

            analysis = analyze_stock(
                data
            )

            if analysis:

                score = analysis[
                    "Score"
                ]

                if score >= 8:

                    st.success(
                        f"🔥 Strong setup — "
                        f"Technical Score "
                        f"{score}/10"
                    )

                elif score >= 5:

                    st.warning(
                        f"⚠️ Moderate setup — "
                        f"Technical Score "
                        f"{score}/10"
                    )

                else:

                    st.info(
                        f"Technical Score "
                        f"{score}/10"
                    )


# ============================================================
# SMART SCANNER
# ============================================================

elif module == "🚀 Smart Scanner":

    st.header(
        "🚀 Smart NSE Breakout Scanner"
    )

    st.write(
        """
        Two-stage technical scanner for
        Nifty 50, Nifty 500 and Full NSE.
        """
    )

    # --------------------------------------------------------
    # LOAD UNIVERSES
    # --------------------------------------------------------

    with st.spinner(
        "Loading NSE universes..."
    ):

        nifty500 = load_nifty500()

        full_nse = load_full_nse()

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    universe = st.sidebar.selectbox(

        "Stock Universe",

        [
            "Nifty 50",
            "Nifty 500",
            "Full NSE"
        ]

    )

    if universe == "Nifty 50":

        stocks = NIFTY50

    elif universe == "Nifty 500":

        stocks = nifty500

    else:

        stocks = full_nse

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.write(
        f"🟢 Nifty 50: "
        f"**{len(NIFTY50)}**"
    )

    st.sidebar.write(
        f"🔵 Nifty 500: "
        f"**{len(nifty500)}**"
    )

    st.sidebar.write(
        f"🟣 Full NSE: "
        f"**{len(full_nse)}**"
    )

    if len(stocks) == 0:

        st.error(
            f"""
            The **{universe}** universe
            could not be loaded.

            Try again after refreshing
            the Streamlit application.
            """
        )

        st.stop()

    st.info(
        f"Selected universe: "
        f"**{universe}** — "
        f"**{len(stocks)} stocks**"
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stage 1 Filters"
    )

    min_price = st.sidebar.number_input(
        "Minimum Price ₹",
        min_value=0.0,
        value=20.0,
        step=5.0
    )

    min_volume = st.sidebar.number_input(
        "Minimum Avg Volume",
        min_value=0,
        value=100000,
        step=50000
    )

    min_turnover = st.sidebar.number_input(
        "Minimum Avg Turnover ₹ Cr",
        min_value=0.0,
        value=1.0,
        step=0.5
    )

    st.sidebar.subheader(
        "Stage 2 Filters"
    )

    min_score = st.sidebar.slider(
        "Minimum Technical Score",
        0,
        10,
        5
    )

    require_volume = st.sidebar.checkbox(
        "Require Volume Confirmation",
        False
    )

    require_rsi = st.sidebar.checkbox(
        "Require RSI > 50",
        False
    )

    require_macd = st.sidebar.checkbox(
        "Require MACD > Signal",
        False
    )

    # IMPORTANT:
    # Full NSE can contain thousands of securities.
    # Limiting scan size protects Yahoo Finance
    # and Streamlit Cloud from excessive requests.

    max_scan = st.sidebar.selectbox(

        "Maximum stocks to scan",

        [
            50,
            100,
            250,
            500,
            1000,
            "All"
        ],

        index=3

    )

    if max_scan != "All":

        scan_stocks = stocks[
            :int(max_scan)
        ]

    else:

        scan_stocks = stocks

    st.sidebar.info(
        f"Scanner will process "
        f"**{len(scan_stocks)} stocks**"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run_scanner = st.sidebar.button(
        "🚀 RUN SCANNER",
        type="primary"
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    with st.expander(
        "📋 Scanner Conditions"
    ):

        st.markdown(
            """
### C1
Recent Close > Previous Day High

### C2
Previous Day High < High of N-2

### C3
Recent Close < 3-Day Donchian Upper

### C4
Recent Low > 3-Day Donchian Lower

### C5
Recent Close > 200 SMA

### Confirmations

Volume ≥ 1.5 × 20-day average

RSI > 50

MACD > Signal

Maximum technical score = 10
            """
        )

    # ========================================================
    # RUN SCANNER
    # ========================================================

    if run_scanner:

        progress = st.progress(
            0,
            text="Starting scanner..."
        )

        results = []

        total = len(
            scan_stocks
        )

        for i, symbol in enumerate(
            scan_stocks
        ):

            try:

                data = get_stock_data(
                    symbol,
                    "1y"
                )

                if data.empty:

                    continue

                data = calculate_indicators(
                    data
                )

                if len(data) < 210:

                    continue

                latest = data.iloc[-1]

                # ------------------------------------------------
                # STAGE 1
                # ------------------------------------------------

                if (
                    latest["Close"]
                    < min_price
                ):

                    continue

                if (
                    latest[
                        "AVG_VOLUME20"
                    ]
                    < min_volume
                ):

                    continue

                avg_turnover_cr = (
                    latest[
                        "AVG_TURNOVER20"
                    ]
                    /
                    10000000
                )

                if (
                    avg_turnover_cr
                    < min_turnover
                ):

                    continue

                if (
                    latest["Close"]
                    <= latest["SMA200"]
                ):

                    continue

                # ------------------------------------------------
                # STAGE 2
                # ------------------------------------------------

                analysis = analyze_stock(
                    data
                )

                if analysis is None:

                    continue

                if (
                    analysis["Score"]
                    < min_score
                ):

                    continue

                if (
                    require_volume
                    and not analysis[
                        "Volume Confirm"
                    ]
                ):

                    continue

                if (
                    require_rsi
                    and not analysis[
                        "RSI Confirm"
                    ]
                ):

                    continue

                if (
                    require_macd
                    and not analysis[
                        "MACD Confirm"
                    ]
                ):

                    continue

                results.append(

                    {

                        "Stock":
                            symbol,

                        "Close":
                            round(
                                analysis["Close"],
                                2
                            ),

                        "SMA200":
                            round(
                                analysis["SMA200"],
                                2
                            ),

                        "RSI":
                            round(
                                analysis["RSI"],
                                1
                            ),

                        "Volume":
                            round(
                                analysis[
                                    "Volume Ratio"
                                ],
                                2
                            ),

                        "C1":
                            "✓"
                            if analysis["C1"]
                            else "✗",

                        "C2":
                            "✓"
                            if analysis["C2"]
                            else "✗",

                        "C3":
                            "✓"
                            if analysis["C3"]
                            else "✗",

                        "C4":
                            "✓"
                            if analysis["C4"]
                            else "✗",

                        "C5":
                            "✓"
                            if analysis["C5"]
                            else "✗",

                        "Technical Score":
                            analysis["Score"]

                    }

                )

            except Exception:

                continue

            progress.progress(

                int(
                    (
                        i + 1
                    )
                    /
                    total
                    *
                    100
                ),

                text=(
                    f"Scanning "
                    f"{i + 1}/{total}"
                )

            )

        progress.empty()

        # ====================================================
        # RESULTS
        # ====================================================

        if not results:

            st.warning(
                """
                No stocks passed the selected
                conditions.

                Try lowering the minimum score
                or liquidity filters.
                """
            )

        else:

            results_df = pd.DataFrame(
                results
            )

            results_df = (
                results_df
                .sort_values(
                    "Technical Score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            st.success(
                f"🎯 "
                f"{len(results_df)} stocks "
                f"passed the scanner."
            )

            # ------------------------------------------------
            # SELECT STOCK
            # ------------------------------------------------

            st.subheader(
                "🔍 Select Stock for Analysis"
            )

            selected_symbol = st.selectbox(

                "Choose a scanned stock",

                results_df[
                    "Stock"
                ].tolist()

            )

            # ------------------------------------------------
            # DESKTOP TABLE
            # ------------------------------------------------

            st.subheader(
                "🏆 Top Breakout Candidates"
            )

            top = results_df.head(20)

            st.markdown(
                '<div class="desktop-only">',
                unsafe_allow_html=True
            )

            st.dataframe(

                top,

                width="stretch",

                hide_index=True

            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # MOBILE CARDS
            # ------------------------------------------------

            st.markdown(
                '<div class="mobile-only">',
                unsafe_allow_html=True
            )

            for _, row in top.iterrows():

                score = int(
                    row[
                        "Technical Score"
                    ]
                )

                if score >= 8:

                    badge = (
                        "🔥 STRONG"
                    )

                elif score >= 6:

                    badge = (
                        "🟢 POSITIVE"
                    )

                else:

                    badge = (
                        "🟡 WATCH"
                    )

                st.markdown(

                    f"""
                    <div class="stock-card">

                        <div class="stock-name">
                            📈 {row['Stock']}
                        </div>

                        <div class="stock-score">
                            {badge} —
                            {score}/10
                        </div>

                        <div class="stock-line">
                            <span>Close</span>
                            <b>
                                ₹{row['Close']:.2f}
                            </b>
                        </div>

                        <div class="stock-line">
                            <span>200 SMA</span>
                            <b>
                                ₹{row['SMA200']:.2f}
                            </b>
                        </div>

                        <div class="stock-line">
                            <span>RSI</span>
                            <b>
                                {row['RSI']:.1f}
                            </b>
                        </div>

                        <div class="stock-line">
                            <span>Volume</span>
                            <b>
                                {row['Volume']:.2f}x
                            </b>
                        </div>

                    </div>
                    """,

                    unsafe_allow_html=True

                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # =================================================
            # SELECTED STOCK
            # =================================================

            selected_data = get_stock_data(
                selected_symbol,
                "1y"
            )

            if not selected_data.empty:

                selected_data = (
                    calculate_indicators(
                        selected_data
                    )
                )

                selected_analysis = (
                    analyze_stock(
                        selected_data
                    )
                )

                if selected_analysis:

                    st.divider()

                    st.subheader(
                        f"📊 {selected_symbol}"
                    )

                    latest = (
                        selected_data.iloc[-1]
                    )

                    # -----------------------------------------
                    # METRICS
                    # -----------------------------------------

                    a, b, c, d = (
                        st.columns(4)
                    )

                    a.metric(
                        "Close",
                        f"₹{latest['Close']:.2f}"
                    )

                    b.metric(
                        "RSI",
                        f"{latest['RSI14']:.1f}"
                    )

                    c.metric(
                        "Volume",
                        f"{latest['VOLUME_RATIO']:.2f}x"
                    )

                    d.metric(
                        "Score",
                        f"{selected_analysis['Score']}/10"
                    )

                    # -----------------------------------------
                    # CHART
                    # -----------------------------------------

                    selected_fig = (
                        create_chart(
                            selected_data,
                            selected_symbol
                        )
                    )

                    st.plotly_chart(
                        selected_fig,
                        width="stretch"
                    )

                    # -----------------------------------------
                    # CONDITIONS
                    # -----------------------------------------

                    st.subheader(
                        "🎯 Breakout Conditions"
                    )

                    conditions = pd.DataFrame(

                        {

                            "Condition": [

                                "C1 Close > Previous High",

                                "C2 Previous High < N-2 High",

                                "C3 Close < Donchian Upper",

                                "C4 Low > Donchian Lower",

                                "C5 Close > 200 SMA",

                                "Volume Confirmation",

                                "RSI Confirmation",

                                "MACD Confirmation"

                            ],

                            "Status": [

                                "✓"
                                if selected_analysis[
                                    "C1"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "C2"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "C3"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "C4"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "C5"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "Volume Confirm"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "RSI Confirm"
                                ]
                                else "✗",

                                "✓"
                                if selected_analysis[
                                    "MACD Confirm"
                                ]
                                else "✗"

                            ]

                        }

                    )

                    st.dataframe(
                        conditions,
                        width="stretch",
                        hide_index=True
                    )

                    # -----------------------------------------
                    # AI ANALYSIS
                    # -----------------------------------------

                    st.subheader(
                        "🤖 AI Technical Analysis"
                    )

                    if st.button(
                        "🤖 Analyse This Stock",
                        type="primary"
                    ):

                        if client is None:

                            st.error(
                                """
                                OpenAI API is not configured.

                                Add your API key under:

                                Streamlit Cloud →
                                Settings →
                                Secrets

                                OPENAI_API_KEY = "your_key"
                                """
                            )

                        else:

                            prompt = f"""

You are analysing the Indian NSE stock:

{selected_symbol}

Use ONLY the following calculated
technical data.

PRICE
-----
Close: ₹{selected_analysis['Close']:.2f}

MOVING AVERAGES
--------------
SMA20: ₹{selected_analysis['SMA20']:.2f}
SMA50: ₹{selected_analysis['SMA50']:.2f}
SMA200: ₹{selected_analysis['SMA200']:.2f}

MOMENTUM
--------
RSI: {selected_analysis['RSI']:.2f}

MACD: {selected_analysis['MACD']:.4f}

MACD Signal:
{selected_analysis['MACD Signal']:.4f}

VOLUME
------
Volume Ratio:
{selected_analysis['Volume Ratio']:.2f}x

DONCHIAN
--------
Upper:
₹{selected_analysis['Donchian Upper']:.2f}

Lower:
₹{selected_analysis['Donchian Lower']:.2f}

BREAKOUT CONDITIONS
-------------------

C1:
{selected_analysis['C1']}

C2:
{selected_analysis['C2']}

C3:
{selected_analysis['C3']}

C4:
{selected_analysis['C4']}

C5:
{selected_analysis['C5']}

Volume Confirmation:
{selected_analysis['Volume Confirm']}

RSI Confirmation:
{selected_analysis['RSI Confirm']}

MACD Confirmation:
{selected_analysis['MACD Confirm']}

Technical Score:
{selected_analysis['Score']}/10


Provide:

1. Overall Technical View

2. Trend Analysis

3. Momentum Analysis

4. Volume Analysis

5. Breakout Structure

6. Strengths

7. Risks

8. What to Monitor

9. Overall Technical Score


Do not invent any data.

Do not guarantee future returns.

Do not state that the stock will definitely
rise or fall.

Clearly distinguish technical evidence
from prediction.

This is educational technical analysis,
not personalized financial advice.

"""

                            with st.spinner(
                                "AI analysing..."
                            ):

                                try:

                                    response = (
                                        client.responses.create(

                                            model=
                                            "gpt-5.6-luna",

                                            instructions="""

You are an expert educational
technical-analysis assistant.

Use only supplied technical data.

Never invent prices,
indicator values or signals.

Never guarantee returns.

Explain the reasoning clearly.

""",

                                            input=prompt

                                        )
                                    )

                                    st.markdown(
                                        "### 🤖 AI Analysis"
                                    )

                                    st.write(
                                        response.output_text
                                    )

                                except Exception as e:

                                    error = str(e)

                                    if "429" in error:

                                        st.error(
                                            """
                                            OpenAI API returned
                                            a 429 error.

                                            Your API credit balance
                                            may be exhausted or
                                            the API rate limit may
                                            have been reached.
                                            """
                                        )

                                    elif "401" in error:

                                        st.error(
                                            """
                                            OpenAI API authentication
                                            failed.

                                            Check your API key in
                                            Streamlit Secrets.
                                            """
                                        )

                                    else:

                                        st.error(
                                            f"AI error: {error}"
                                        )

            # =================================================
            # SCORE DISTRIBUTION
            # =================================================

            st.subheader(
                "📊 Technical Score Distribution"
            )

            score_distribution = (
                results_df[
                    "Technical Score"
                ]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                score_distribution
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            st.subheader(
                "📥 Download Results"
            )

            csv = results_df.to_csv(
                index=False
            )

            st.download_button(

                "⬇️ Download Scanner CSV",

                data=csv,

                file_name=(
                    "NSE_Technical_Scanner.csv"
                ),

                mime="text/csv",

                width="stretch"

            )


# ============================================================
# AI ASSISTANT
# ============================================================

else:

    st.header(
        "🤖 AI Technical Assistant"
    )

    st.write(
        """
        Ask questions about RSI, MACD,
        moving averages, breakouts,
        candlestick patterns, support,
        resistance and technical analysis.
        """
    )

    if client is None:

        st.warning(
            """
            OpenAI API is not configured.

            Add this to Streamlit Cloud Secrets:

            OPENAI_API_KEY = "your_api_key"
            """
        )

    question = st.chat_input(
        "Ask a technical-analysis question..."
    )

    if question:

        st.chat_message(
            "user"
        ).write(
            question
        )

        with st.chat_message(
            "assistant"
        ):

            if client is None:

                st.error(
                    "OpenAI API key is not configured."
                )

            else:

                try:

                    response = (
                        client.responses.create(

                            model=
                            "gpt-5.6-luna",

                            instructions="""

You are an educational technical-analysis
assistant focused on Indian equity markets.

Explain technical analysis clearly.

Topics include:

RSI
MACD
Moving averages
Donchian channels
Volume
Breakouts
Support/resistance
Candlestick patterns
Trend analysis
Risk management
Elliott Wave

Do not invent live prices.

Do not guarantee returns.

If actual market data has not been
provided, clearly state that.

This is educational information,
not personalized financial advice.

""",

                            input=question

                        )
                    )

                    st.write(
                        response.output_text
                    )

                except Exception as e:

                    error = str(e)

                    if "429" in error:

                        st.error(
                            """
                            OpenAI API credit balance
                            exhausted or rate limit reached.
                            """
                        )

                    elif "401" in error:

                        st.error(
                            """
                            OpenAI API authentication failed.

                            Check your API key.
                            """
                        )

                    else:

                        st.error(
                            f"AI error: {error}"
                        )
