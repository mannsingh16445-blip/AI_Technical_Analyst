import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
import os

from io import StringIO
from dotenv import load_dotenv

# Optional OpenAI import
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
)

load_dotenv()


# ============================================================
# MOBILE RESPONSIVE CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .mobile-scanner {
        display: none;
    }

    @media only screen and (max-width: 768px) {

        .block-container {
            padding-top: 0.5rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }

        h1 {
            font-size: 1.6rem !important;
        }

        h2 {
            font-size: 1.3rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        [data-testid="stMetric"] {
            padding: 0.4rem 0.2rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 2.7rem;
        }

        input {
            font-size: 16px !important;
        }

        .desktop-scanner {
            display: none !important;
        }

        .mobile-scanner {
            display: block !important;
        }

        .stock-card {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .stock-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .stock-score {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .stock-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            font-size: 0.9rem;
        }

        [data-testid="stDataFrame"] {
            width: 100% !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    client = None


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "📈 AI Technical Analyst"
)

st.caption(
    "Technical Charts • Smart NSE Scanner • AI-Powered Analysis"
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
# LOAD FULL NSE EQUITY UNIVERSE
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_nse_equity_universe():

    url = (
        "https://nsearchives.nseindia.com/"
        "content/equities/EQUITY_L.csv"
    )

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36",

        "Accept":
            "text/csv,*/*",

        "Referer":
            "https://www.nseindia.com/"

    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text)
        )

        df.columns = [
            str(c)
            .strip()
            .upper()
            for c in df.columns
        ]

        if "SYMBOL" not in df.columns:

            return []

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

        return sorted(
            symbols
            .drop_duplicates()
            .tolist()
        )

    except Exception:

        return []


# ============================================================
# LOAD NIFTY 500
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_nifty500():

    urls = [

        (
            "https://nsearchives.nseindia.com/"
            "content/indices/ind_nifty500list.csv"
        ),

        (
            "https://www.niftyindices.com/"
            "IndexConstituent/ind_nifty500list.csv"
        )

    ]

    headers = {
        "User-Agent":
            "Mozilla/5.0"
    }

    for url in urls:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:

                continue

            df = pd.read_csv(
                StringIO(response.text)
            )

            df.columns = [
                str(c)
                .strip()
                .upper()
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

            if len(symbols) >= 400:

                return symbols

        except Exception:

            continue

    return []


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def download_batches(
    tickers,
    period="1y",
    batch_size=50
):

    all_data = {}

    ticker_list = list(
        dict.fromkeys(tickers)
    )

    for start in range(
        0,
        len(ticker_list),
        batch_size
    ):

        batch = ticker_list[
            start:start + batch_size
        ]

        yahoo_tickers = [
            symbol + ".NS"
            for symbol in batch
        ]

        try:

            data = yf.download(
                tickers=yahoo_tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker"
            )

            if data is None:
                continue

            if data.empty:
                continue


            # =================================================
            # SINGLE STOCK
            # =================================================

            if len(batch) == 1:

                symbol = batch[0]

                stock = data.copy()

                if isinstance(
                    stock.columns,
                    pd.MultiIndex
                ):

                    level0 = (
                        stock.columns
                        .get_level_values(0)
                        .tolist()
                    )

                    level1 = (
                        stock.columns
                        .get_level_values(1)
                        .tolist()
                    )

                    if "Open" in level0:

                        stock.columns = [
                            col[0]
                            for col in stock.columns
                        ]

                    elif "Open" in level1:

                        stock.columns = [
                            col[1]
                            for col in stock.columns
                        ]

                else:

                    stock.columns = [
                        str(col)
                        .strip()
                        .title()
                        for col in stock.columns
                    ]

                required = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                if all(
                    column in stock.columns
                    for column in required
                ):

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if not stock.empty:

                        all_data[symbol] = stock

                continue


            # =================================================
            # MULTIPLE STOCKS
            # =================================================

            if not isinstance(
                data.columns,
                pd.MultiIndex
            ):

                continue

            level0 = (
                data.columns
                .get_level_values(0)
                .unique()
                .tolist()
            )

            level1 = (
                data.columns
                .get_level_values(1)
                .unique()
                .tolist()
            )

            for symbol in batch:

                yahoo_symbol = (
                    symbol + ".NS"
                )

                try:

                    if (
                        yahoo_symbol
                        in level0
                    ):

                        stock = data[
                            yahoo_symbol
                        ].copy()

                    elif (
                        yahoo_symbol
                        in level1
                    ):

                        stock = data[
                            :,
                            yahoo_symbol
                        ].copy()

                    else:

                        continue

                    if isinstance(
                        stock.columns,
                        pd.MultiIndex
                    ):

                        stock.columns = [

                            col[0]
                            if isinstance(
                                col,
                                tuple
                            )
                            else col

                            for col
                            in stock.columns

                        ]

                    stock.columns = [
                        str(col)
                        .strip()
                        .title()
                        for col
                        in stock.columns
                    ]

                    required = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    if not all(
                        column in stock.columns
                        for column in required
                    ):

                        continue

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if not stock.empty:

                        all_data[symbol] = stock

                except Exception:

                    continue

        except Exception:

            continue

        time.sleep(0.2)

    return all_data


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(data):

    data = data.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data.dropna(
        subset=required
    )

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

    delta = data["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = (
        avg_gain /
        avg_loss
    )

    data["RSI14"] = (
        100 -
        (
            100 /
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
        ema12 -
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
    # DONCHIAN 3
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
# STAGE 1 FILTER
# ============================================================

def stage_one_filter(
    data,
    min_price,
    min_volume,
    min_turnover
):

    if len(data) < 210:

        return False

    latest = data.iloc[-1]

    required_values = [
        latest["Close"],
        latest["SMA200"],
        latest["AVG_VOLUME20"],
        latest["AVG_TURNOVER20"]
    ]

    if any(
        pd.isna(value)
        for value
        in required_values
    ):

        return False

    # Price

    if (
        latest["Close"]
        < min_price
    ):

        return False

    # Volume

    if (
        latest["AVG_VOLUME20"]
        < min_volume
    ):

        return False

    # Turnover

    turnover_crore = (
        latest["AVG_TURNOVER20"]
        / 10000000
    )

    if (
        turnover_crore
        < min_turnover
    ):

        return False

    # 200 SMA

    if (
        latest["Close"]
        <= latest["SMA200"]
    ):

        return False

    return True


# ============================================================
# STAGE 2 ANALYSIS
# ============================================================

def stage_two_analysis(data):

    if len(data) < 210:

        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    n_minus_2 = data.iloc[-3]

    # --------------------------------------------------------
    # C1
    # Recent Close > Previous Day High
    # --------------------------------------------------------

    condition_1 = (
        latest["Close"]
        >
        previous["High"]
    )

    # --------------------------------------------------------
    # C2
    # Previous Day High < N-2 High
    # --------------------------------------------------------

    condition_2 = (
        previous["High"]
        <
        n_minus_2["High"]
    )

    # --------------------------------------------------------
    # C3
    # Close < Donchian Upper
    # --------------------------------------------------------

    condition_3 = (
        latest["Close"]
        <
        latest["DONCHIAN_UPPER"]
    )

    # --------------------------------------------------------
    # C4
    # Low > Donchian Lower
    # --------------------------------------------------------

    condition_4 = (
        latest["Low"]
        >
        latest["DONCHIAN_LOWER"]
    )

    # --------------------------------------------------------
    # C5
    # Close > 200 SMA
    # --------------------------------------------------------

    condition_5 = (
        latest["Close"]
        >
        latest["SMA200"]
    )

    # --------------------------------------------------------
    # CONFIRMATIONS
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

    if condition_1:
        score += 2

    if condition_2:
        score += 1

    if condition_3:
        score += 1

    if condition_4:
        score += 1

    if condition_5:
        score += 2

    if volume_confirm:
        score += 1

    if rsi_confirm:
        score += 1

    if macd_confirm:
        score += 1

    return {

        "C1":
            condition_1,

        "C2":
            condition_2,

        "C3":
            condition_3,

        "C4":
            condition_4,

        "C5":
            condition_5,

        "Volume Confirm":
            volume_confirm,

        "RSI Confirm":
            rsi_confirm,

        "MACD Confirm":
            macd_confirm,

        "Score":
            score,

        "Close":
            latest["Close"],

        "SMA200":
            latest["SMA200"],

        "RSI":
            latest["RSI14"],

        "MACD":
            latest["MACD"],

        "MACD Signal":
            latest["MACD_SIGNAL"],

        "Volume Ratio":
            latest["VOLUME_RATIO"],

        "Turnover":
            latest["AVG_TURNOVER20"]
            / 10000000,

        "Donchian Upper":
            latest["DONCHIAN_UPPER"],

        "Donchian Lower":
            latest["DONCHIAN_LOWER"]

    }


# ============================================================
# SESSION STATE
# ============================================================

if "selected_stock" not in st.session_state:

    st.session_state.selected_stock = None


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
        "🚀 Smart Breakout Scanner",
        "🤖 AI Analyst"
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
            "6mo",
            "1y",
            "2y",
            "5y"
        ],
        index=1
    )

    load_chart = st.sidebar.button(
        "📈 Load Chart",
        type="primary"
    )

    if load_chart:

        with st.spinner(
            f"Loading {symbol}..."
        ):

            market = download_batches(
                [symbol],
                period,
                1
            )

        if symbol not in market:

            st.error(
                f"""
                Could not retrieve market data for
                **{symbol}.NS**.

                Please check the NSE symbol and
                try again.
                """
            )

            st.stop()

        data = market[
            symbol
        ].copy()

        data = calculate_indicators(
            data
        )

        if data.empty:

            st.error(
                "No usable market data found."
            )

            st.stop()

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

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
                y=data["DONCHIAN_UPPER"],
                mode="lines",
                name="Donchian Upper"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["DONCHIAN_LOWER"],
                mode="lines",
                name="Donchian Lower"
            )
        )

        fig.update_layout(
            title=(
                f"{symbol} — Technical Analysis"
            ),
            height=650,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        latest = data.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)

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

        # ----------------------------------------------------
        # TECHNICAL SCORE
        # ----------------------------------------------------

        analysis = stage_two_analysis(
            data
        )

        if analysis:

            score = analysis[
                "Score"
            ]

            if score >= 8:

                st.success(
                    f"🔥 Strong technical setup — "
                    f"{score}/10"
                )

            elif score >= 5:

                st.warning(
                    f"⚠️ Moderate technical setup — "
                    f"{score}/10"
                )

            else:

                st.info(
                    f"Technical setup — "
                    f"{score}/10"
                )


# ============================================================
# SMART BREAKOUT SCANNER
# ============================================================

elif module == "🚀 Smart Breakout Scanner":

    st.header(
        "🚀 Smart Two-Stage Breakout Scanner"
    )

    st.write(
        """
        **Stage 1:** Liquidity + price + 200 SMA

        **Stage 2:** Breakout structure + Donchian
        + RSI + MACD + volume
        """
    )

    # --------------------------------------------------------
    # LOAD UNIVERSES
    # --------------------------------------------------------

    with st.spinner(
        "Loading NSE stock universe..."
    ):

        nse_stocks = (
            load_nse_equity_universe()
        )

        nifty500 = (
            load_nifty500()
        )

    # --------------------------------------------------------
    # UNIVERSE SELECTOR
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

    elif universe == "Full NSE":

        stocks = nse_stocks

    else:

        stocks = []

    if (
        universe == "Nifty 500"
        and not stocks
    ):

        st.error(
            """
            Nifty 500 list could not be loaded.
            """
        )

        st.stop()

    if (
        universe == "Full NSE"
        and not stocks
    ):

        st.error(
            """
            Full NSE stock list could not be loaded.

            NSE may temporarily reject automated
            requests. Please try again later.
            """
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}**"
    )

    # --------------------------------------------------------
    # LIQUIDITY SETTINGS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stage 1 — Liquidity"
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

    # --------------------------------------------------------
    # TECHNICAL SETTINGS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stage 2 — Technical"
    )

    min_score = st.sidebar.slider(
        "Minimum Technical Score",
        0,
        10,
        5
    )

    require_volume = st.sidebar.checkbox(
        "Require Volume Confirmation",
        value=False
    )

    require_rsi = st.sidebar.checkbox(
        "Require RSI > 50",
        value=False
    )

    require_macd = st.sidebar.checkbox(
        "Require MACD > Signal",
        value=False
    )

    batch_size = st.sidebar.slider(
        "Download Batch Size",
        25,
        100,
        50,
        step=25
    )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    run_scanner = st.sidebar.button(
        "🚀 RUN SMART SCANNER",
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
### Core Conditions

**C1:** Recent Close > Previous Day High

**C2:** Previous Day High < High of N-2

**C3:** Recent Close < 3-Day Donchian Upper

**C4:** Recent Low > 3-Day Donchian Lower

**C5:** Recent Close > 200 SMA

### Confirmations

**Volume:** Current Volume ≥ 1.5 × 20-day average

**RSI:** RSI > 50

**MACD:** MACD > Signal

### Maximum Score: 10
            """
        )

    # ========================================================
    # SCANNER
    # ========================================================

    if run_scanner:

        if not stocks:

            st.error(
                "No stocks available."
            )

            st.stop()

        progress = st.progress(
            0,
            text="Downloading market data..."
        )

        market = download_batches(
            stocks,
            "1y",
            batch_size
        )

        progress.progress(
            30,
            text="Applying Stage 1 filters..."
        )

        stage1_stocks = []

        total = len(market)

        for i, (symbol, raw_data) in enumerate(
            market.items()
        ):

            try:

                data = calculate_indicators(
                    raw_data
                )

                if stage_one_filter(
                    data,
                    min_price,
                    min_volume,
                    min_turnover
                ):

                    stage1_stocks.append(
                        symbol
                    )

            except Exception:

                pass

            if total > 0:

                progress.progress(
                    30 +
                    int(
                        30 *
                        (i + 1) /
                        total
                    ),
                    text=(
                        f"Stage 1: "
                        f"{i + 1}/{total}"
                    )
                )

        # ----------------------------------------------------
        # STAGE 1 SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "🔎 Stage 1 Results"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Universe",
            len(stocks)
        )

        c2.metric(
            "Data Retrieved",
            len(market)
        )

        c3.metric(
            "Stage 1 Survivors",
            len(stage1_stocks)
        )

        if not stage1_stocks:

            progress.empty()

            st.warning(
                """
                No stocks survived Stage 1.

                Try lowering the liquidity filters.
                """
            )

            st.stop()

        # ----------------------------------------------------
        # STAGE 2
        # ----------------------------------------------------

        progress.progress(
            65,
            text="Running technical analysis..."
        )

        results = []

        total_stage2 = len(
            stage1_stocks
        )

        for i, symbol in enumerate(
            stage1_stocks
        ):

            try:

                data = calculate_indicators(
                    market[symbol]
                )

                analysis = stage_two_analysis(
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

                        "200 SMA":
                            round(
                                analysis["SMA200"],
                                2
                            ),

                        "RSI":
                            round(
                                analysis["RSI"],
                                1
                            ),

                        "Volume Ratio":
                            round(
                                analysis[
                                    "Volume Ratio"
                                ],
                                2
                            ),

                        "Avg Turnover ₹Cr":
                            round(
                                analysis[
                                    "Turnover"
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

                        "Volume":
                            "✓"
                            if analysis[
                                "Volume Confirm"
                            ]
                            else "✗",

                        "RSI Confirm":
                            "✓"
                            if analysis[
                                "RSI Confirm"
                            ]
                            else "✗",

                        "MACD Confirm":
                            "✓"
                            if analysis[
                                "MACD Confirm"
                            ]
                            else "✗",

                        "Technical Score":
                            analysis["Score"]

                    }

                )

            except Exception:

                pass

            progress.progress(
                65 +
                int(
                    35 *
                    (i + 1)
                    /
                    total_stage2
                ),
                text=(
                    f"Stage 2: "
                    f"{i + 1}/{total_stage2}"
                )
            )

        progress.empty()

        # ====================================================
        # RESULTS
        # ====================================================

        if not results:

            st.warning(
                """
                No stocks passed Stage 2.

                Try reducing the minimum technical score.
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
                f"🎯 {len(results_df)} "
                f"stocks passed Stage 2."
            )

            # =================================================
            # STOCK SELECTION
            # =================================================

            st.subheader(
                "🔍 Analyse a Scanned Stock"
            )

            available_stocks = (
                results_df[
                    "Stock"
                ]
                .tolist()
            )

            selected_stock = st.selectbox(
                "Select stock",
                available_stocks
            )

            st.session_state.selected_stock = (
                selected_stock
            )

            # =================================================
            # TOP STOCKS
            # =================================================

            st.subheader(
                "🏆 Top Early Breakout Candidates"
            )

            top = results_df[
                [
                    "Stock",
                    "Close",
                    "200 SMA",
                    "RSI",
                    "Volume Ratio",
                    "Technical Score"
                ]
            ].head(20)

            # Desktop

            st.markdown(
                '<div class="desktop-scanner">',
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

            # Mobile

            st.markdown(
                '<div class="mobile-scanner">',
                unsafe_allow_html=True
            )

            for _, row in top.iterrows():

                score = int(
                    row["Technical Score"]
                )

                if score >= 8:

                    badge = "🔥 STRONG"

                elif score >= 6:

                    badge = "🟢 POSITIVE"

                else:

                    badge = "🟡 WATCH"

                st.markdown(
                    f"""
                    <div class="stock-card">

                        <div class="stock-title">
                            📈 {row['Stock']}
                        </div>

                        <div class="stock-score">
                            {badge} —
                            Score {score}/10
                        </div>

                        <div class="stock-row">
                            <span>Close</span>
                            <strong>
                                ₹{row['Close']:.2f}
                            </strong>
                        </div>

                        <div class="stock-row">
                            <span>200 SMA</span>
                            <strong>
                                ₹{row['200 SMA']:.2f}
                            </strong>
                        </div>

                        <div class="stock-row">
                            <span>RSI</span>
                            <strong>
                                {row['RSI']:.1f}
                            </strong>
                        </div>

                        <div class="stock-row">
                            <span>Volume</span>
                            <strong>
                                {row['Volume Ratio']:.2f}x
                            </strong>
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
            # SELECTED STOCK DATA
            # =================================================

            selected_symbol = (
                st.session_state.selected_stock
            )

            if selected_symbol in market:

                selected_data = (
                    calculate_indicators(
                        market[selected_symbol]
                    )
                )

                selected_analysis = (
                    stage_two_analysis(
                        selected_data
                    )
                )

            else:

                selected_data = None

                selected_analysis = None

            # =================================================
            # SELECTED STOCK ANALYSIS
            # =================================================

            if (
                selected_data is not None
                and selected_analysis is not None
            ):

                st.divider()

                st.subheader(
                    f"📊 {selected_symbol} "
                    f"— Technical Analysis"
                )

                latest = (
                    selected_data.iloc[-1]
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Close",
                    f"₹{latest['Close']:.2f}"
                )

                m2.metric(
                    "RSI",
                    f"{latest['RSI14']:.1f}"
                )

                m3.metric(
                    "Volume",
                    f"{latest['VOLUME_RATIO']:.2f}x"
                )

                m4.metric(
                    "Technical Score",
                    f"{selected_analysis['Score']}/10"
                )

                # =============================================
                # SELECTED STOCK CHART
                # =============================================

                fig_selected = go.Figure()

                fig_selected.add_trace(
                    go.Candlestick(
                        x=selected_data.index,
                        open=selected_data["Open"],
                        high=selected_data["High"],
                        low=selected_data["Low"],
                        close=selected_data["Close"],
                        name=selected_symbol
                    )
                )

                fig_selected.add_trace(
                    go.Scatter(
                        x=selected_data.index,
                        y=selected_data["SMA20"],
                        mode="lines",
                        name="SMA 20"
                    )
                )

                fig_selected.add_trace(
                    go.Scatter(
                        x=selected_data.index,
                        y=selected_data["SMA50"],
                        mode="lines",
                        name="SMA 50"
                    )
                )

                fig_selected.add_trace(
                    go.Scatter(
                        x=selected_data.index,
                        y=selected_data["SMA200"],
                        mode="lines",
                        name="SMA 200"
                    )
                )

                fig_selected.add_trace(
                    go.Scatter(
                        x=selected_data.index,
                        y=selected_data[
                            "DONCHIAN_UPPER"
                        ],
                        mode="lines",
                        name="Donchian Upper"
                    )
                )

                fig_selected.add_trace(
                    go.Scatter(
                        x=selected_data.index,
                        y=selected_data[
                            "DONCHIAN_LOWER"
                        ],
                        mode="lines",
                        name="Donchian Lower"
                    )
                )

                fig_selected.update_layout(
                    title=(
                        f"{selected_symbol} "
                        "— Scanner Analysis"
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

                st.plotly_chart(
                    fig_selected,
                    width="stretch"
                )

                # =============================================
                # BREAKOUT CONDITIONS
                # =============================================

                st.subheader(
                    "🎯 Breakout Conditions"
                )

                condition_data = pd.DataFrame(
                    {

                        "Condition": [

                            "C1: Close > Previous High",

                            "C2: Previous High < N-2 High",

                            "C3: Close < Donchian Upper",

                            "C4: Low > Donchian Lower",

                            "C5: Close > 200 SMA",

                            "Volume Confirmation",

                            "RSI Confirmation",

                            "MACD Confirmation"

                        ],

                        "Status": [

                            "✓"
                            if selected_analysis["C1"]
                            else "✗",

                            "✓"
                            if selected_analysis["C2"]
                            else "✗",

                            "✓"
                            if selected_analysis["C3"]
                            else "✗",

                            "✓"
                            if selected_analysis["C4"]
                            else "✗",

                            "✓"
                            if selected_analysis["C5"]
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
                    condition_data,
                    width="stretch",
                    hide_index=True
                )

                # =============================================
                # AI DATA
                # =============================================

                ai_data = {

                    "Stock":
                        selected_symbol,

                    "Close":
                        round(
                            float(
                                latest["Close"]
                            ),
                            2
                        ),

                    "SMA20":
                        round(
                            float(
                                latest["SMA20"]
                            ),
                            2
                        ),

                    "SMA50":
                        round(
                            float(
                                latest["SMA50"]
                            ),
                            2
                        ),

                    "SMA200":
                        round(
                            float(
                                latest["SMA200"]
                            ),
                            2
                        ),

                    "RSI":
                        round(
                            float(
                                latest["RSI14"]
                            ),
                            2
                        ),

                    "MACD":
                        round(
                            float(
                                latest["MACD"]
                            ),
                            4
                        ),

                    "MACD Signal":
                        round(
                            float(
                                latest[
                                    "MACD_SIGNAL"
                                ]
                            ),
                            4
                        ),

                    "Volume Ratio":
                        round(
                            float(
                                latest[
                                    "VOLUME_RATIO"
                                ]
                            ),
                            2
                        ),

                    "Donchian Upper":
                        round(
                            float(
                                latest[
                                    "DONCHIAN_UPPER"
                                ]
                            ),
                            2
                        ),

                    "Donchian Lower":
                        round(
                            float(
                                latest[
                                    "DONCHIAN_LOWER"
                                ]
                            ),
                            2
                        ),

                    "C1":
                        selected_analysis["C1"],

                    "C2":
                        selected_analysis["C2"],

                    "C3":
                        selected_analysis["C3"],

                    "C4":
                        selected_analysis["C4"],

                    "C5":
                        selected_analysis["C5"],

                    "Volume Confirmation":
                        selected_analysis[
                            "Volume Confirm"
                        ],

                    "RSI Confirmation":
                        selected_analysis[
                            "RSI Confirm"
                        ],

                    "MACD Confirmation":
                        selected_analysis[
                            "MACD Confirm"
                        ],

                    "Technical Score":
                        selected_analysis[
                            "Score"
                        ]

                }

                # =============================================
                # AI ANALYSIS
                # =============================================

                st.subheader(
                    "🤖 AI Technical Analysis"
                )

                analyse_button = st.button(
                    "🤖 Analyse This Stock",
                    type="primary"
                )

                if analyse_button:

                    if client is None:

                        st.error(
                            """
                            OpenAI is not configured.

                            Check your Streamlit Secrets
                            and make sure API credits are
                            available.
                            """
                        )

                    else:

                        prompt = f"""
Analyse the following technical-analysis
data for {ai_data['Stock']}.

MARKET DATA
-----------

Stock:
{ai_data['Stock']}

Close:
₹{ai_data['Close']}

SMA20:
₹{ai_data['SMA20']}

SMA50:
₹{ai_data['SMA50']}

SMA200:
₹{ai_data['SMA200']}

RSI:
{ai_data['RSI']}

MACD:
{ai_data['MACD']}

MACD Signal:
{ai_data['MACD Signal']}

Volume Ratio:
{ai_data['Volume Ratio']}x

Donchian Upper:
₹{ai_data['Donchian Upper']}

Donchian Lower:
₹{ai_data['Donchian Lower']}


BREAKOUT CONDITIONS
-------------------

C1 Close > Previous High:
{ai_data['C1']}

C2 Previous High < N-2 High:
{ai_data['C2']}

C3 Close < Donchian Upper:
{ai_data['C3']}

C4 Low > Donchian Lower:
{ai_data['C4']}

C5 Close > 200 SMA:
{ai_data['C5']}

Volume Confirmation:
{ai_data['Volume Confirmation']}

RSI Confirmation:
{ai_data['RSI Confirmation']}

MACD Confirmation:
{ai_data['MACD Confirmation']}

Technical Score:
{ai_data['Technical Score']}/10


Provide a structured technical analysis
using ONLY the supplied data.

Use these sections:

1. Overall Technical View
2. Trend Analysis
3. Momentum Analysis
4. Volume Analysis
5. Breakout Structure
6. Strengths
7. Risks
8. What to Monitor
9. Overall Technical Score

Do not invent values.

Do not guarantee returns.

Do not present predictions as certainty.

This is educational technical analysis,
not personalized financial advice.
"""

                        with st.spinner(
                            f"Analysing "
                            f"{selected_symbol}..."
                        ):

                            try:

                                response = (
                                    client.responses.create(

                                        model=
                                        "gpt-5.6-luna",

                                        instructions="""
You are an expert educational
technical-analysis assistant.

Analyze ONLY the market data supplied
by the application.

Never invent prices or indicator values.

Never guarantee returns.

Clearly distinguish evidence from
prediction.

Explain the reasoning behind your
technical conclusion.
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

                                error_text = str(e)

                                if (
                                    "429"
                                    in error_text
                                ):

                                    st.error(
                                        """
                                        OpenAI API returned
                                        a 429 error.

                                        Check your API credit
                                        balance or rate limit.
                                        """
                                    )

                                elif (
                                    "401"
                                    in error_text
                                ):

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
                                        f"AI error: "
                                        f"{error_text}"
                                    )

            # =================================================
            # SCORE DISTRIBUTION
            # =================================================

            st.subheader(
                "📊 Score Distribution"
            )

            score_counts = (
                results_df[
                    "Technical Score"
                ]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                score_counts
            )

            # =================================================
            # FULL RESULTS
            # =================================================

            st.subheader(
                "📋 Detailed Results"
            )

            st.dataframe(
                results_df,
                width="stretch",
                hide_index=True
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            csv = results_df.to_csv(
                index=False
            )

            st.download_button(
                label=(
                    "⬇️ Download Scanner Results"
                ),
                data=csv,
                file_name=(
                    "NSE_Smart_Breakout_Scanner.csv"
                ),
                mime="text/csv"
            )


# ============================================================
# AI GENERAL ASSISTANT
# ============================================================

else:

    st.header(
        "🤖 AI Technical Analyst"
    )

    st.write(
        """
        Ask questions about technical analysis,
        indicators, chart patterns and trading setups.
        """
    )

    if client is None:

        st.warning(
            """
            OpenAI API is not configured.

            Go to Streamlit Cloud:

            Settings → Secrets

            and add:

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
                    "OpenAI API key is unavailable."
                )

            else:

                try:

                    response = (
                        client.responses.create(

                            model=
                            "gpt-5.6-luna",

                            instructions="""
You are an educational technical-analysis
assistant for Indian equity markets.

Explain:

- Candlestick patterns
- Support and resistance
- Moving averages
- RSI
- MACD
- Donchian channels
- Volume
- Breakouts
- Trend structure
- Elliott Wave concepts
- Risk management
- Position sizing

Do not invent live market prices.

Do not invent indicator values.

Do not guarantee returns.

If actual market data has not been
provided, clearly say that the value
cannot be verified.

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

                    error_text = str(e)

                    if (
                        "429"
                        in error_text
                    ):

                        st.error(
                            """
                            OpenAI API credit balance
                            exhausted or rate limit reached.

                            Please check your API billing
                            balance.
                            """
                        )

                    elif (
                        "401"
                        in error_text
                    ):

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
                            f"AI error: {error_text}"
                        )
