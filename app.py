import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
import os

from io import StringIO
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
)

load_dotenv()


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )
else:
    client = None


# ============================================================
# TITLE
# ============================================================

st.title("📈 AI Technical Analyst")

st.caption(
    "Technical charts • Smart NSE scanner • AI analysis"
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
# LOAD NSE EQUITY UNIVERSE
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
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/"
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
            str(c).strip().upper()
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

        symbols = sorted(
            symbols.drop_duplicates().tolist()
        )

        return symbols

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

        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"

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

            if len(symbols) >= 400:

                return sorted(
                    symbols.drop_duplicates().tolist()
                )

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
            ticker + ".NS"
            for ticker in batch
        ]

        try:

            data = yf.download(
                yahoo_tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=30
            )

            if data is None:
                continue

            if data.empty:
                continue

            for ticker in batch:

                yahoo_symbol = ticker + ".NS"

                try:

                    if len(batch) == 1:

                        stock = data.copy()

                    else:

                        if not isinstance(
                            data.columns,
                            pd.MultiIndex
                        ):

                            continue

                        level_zero = (
                            data.columns
                            .get_level_values(0)
                            .unique()
                        )

                        if yahoo_symbol not in level_zero:

                            continue

                        stock = data[
                            yahoo_symbol
                        ].copy()

                    # Fix MultiIndex

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
                            for col in stock.columns
                        ]

                    # Normalize names

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

                        all_data[ticker] = stock

                except Exception:

                    continue

        except Exception:

            continue

        time.sleep(0.20)

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
        ema12 - ema26
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
    # DONCHIAN CHANNEL
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

    values = [
        latest["Close"],
        latest["SMA200"],
        latest["AVG_VOLUME20"],
        latest["AVG_TURNOVER20"]
    ]

    if any(
        pd.isna(value)
        for value in values
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

    # Long-term trend

    if (
        latest["Close"]
        <= latest["SMA200"]
    ):

        return False

    return True


# ============================================================
# STAGE 2 TECHNICAL ANALYSIS
# ============================================================

def stage_two_analysis(data):

    if len(data) < 210:

        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    n_minus_2 = data.iloc[-3]

    # --------------------------------------------------------
    # YOUR FIVE CORE CONDITIONS
    # --------------------------------------------------------

    condition_1 = (
        latest["Close"]
        >
        previous["High"]
    )

    condition_2 = (
        previous["High"]
        <
        n_minus_2["High"]
    )

    condition_3 = (
        latest["Close"]
        <
        latest["DONCHIAN_UPPER"]
    )

    condition_4 = (
        latest["Low"]
        >
        latest["DONCHIAN_LOWER"]
    )

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

        "C1": condition_1,

        "C2": condition_2,

        "C3": condition_3,

        "C4": condition_4,

        "C5": condition_5,

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

    st.write(
        "Enter any NSE-listed stock symbol."
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

    if st.sidebar.button(
        "📈 Load Chart",
        type="primary"
    ):

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
                Could not retrieve data for
                **{symbol}.NS**

                Check that the NSE symbol is correct.
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
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ----------------------------------------------------
        # CURRENT VALUES
        # ----------------------------------------------------

        latest = data.iloc[-1]

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Close",
            f"₹{latest['Close']:.2f}"
        )

        col2.metric(
            "SMA 20",
            f"₹{latest['SMA20']:.2f}"
        )

        col3.metric(
            "SMA 50",
            f"₹{latest['SMA50']:.2f}"
        )

        col4.metric(
            "SMA 200",
            f"₹{latest['SMA200']:.2f}"
        )

        col5.metric(
            "RSI",
            f"{latest['RSI14']:.1f}"
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        analysis = stage_two_analysis(
            data
        )

        if analysis:

            st.subheader(
                "Technical Setup"
            )

            if analysis["Score"] >= 8:

                st.success(
                    f"🔥 Strong technical setup — "
                    f"Score {analysis['Score']}/10"
                )

            elif analysis["Score"] >= 5:

                st.warning(
                    f"⚠️ Moderate technical setup — "
                    f"Score {analysis['Score']}/10"
                )

            else:

                st.info(
                    f"Technical setup score: "
                    f"{analysis['Score']}/10"
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
        **Stage 1:** Liquidity + price + 200 SMA filter

        **Stage 2:** Early breakout + Donchian +
        RSI + MACD + volume analysis
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
    # UNIVERSE
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stock Universe"
    )

    universe = st.sidebar.selectbox(
        "Select Universe",
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

            Please select Nifty 50 or try again later.
            """
        )

        st.stop()

    if (
        universe == "Full NSE"
        and not stocks
    ):

        st.error(
            """
            Full NSE equity list could not be loaded.

            The NSE data server may temporarily reject
            automated requests.

            Please try again later.
            """
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}**"
    )

    # --------------------------------------------------------
    # STAGE 1
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
    # STAGE 2
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stage 2 — Technical"
    )

    min_score = st.sidebar.slider(
        "Minimum Technical Score",
        min_value=0,
        max_value=10,
        value=5
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
        min_value=25,
        max_value=100,
        value=50,
        step=25
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run_scanner = st.sidebar.button(
        "🚀 RUN SMART SCANNER",
        type="primary"
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    with st.expander(
        "📋 View Scanner Conditions"
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

### Score

C1 = 2 points

C2 = 1 point

C3 = 1 point

C4 = 1 point

C5 = 2 points

Volume = 1 point

RSI = 1 point

MACD = 1 point

**Maximum = 10 points**
            """
        )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if run_scanner:

        if not stocks:

            st.error(
                "No stocks available to scan."
            )

            st.stop()

        # ====================================================
        # STAGE 1 DOWNLOAD
        # ====================================================

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
            35,
            text=(
                "Applying Stage 1 filters..."
            )
        )

        stage1_stocks = []

        processed = 0

        total = len(market)

        # ====================================================
        # STAGE 1
        # ====================================================

        for symbol, raw_data in market.items():

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

            processed += 1

            if total > 0:

                progress.progress(
                    35 +
                    int(
                        25 *
                        processed /
                        total
                    ),
                    text=(
                        f"Stage 1: "
                        f"{processed}/{total}"
                    )
                )

        # ====================================================
        # STAGE 1 SUMMARY
        # ====================================================

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

                Try lowering:
                • Minimum price
                • Minimum average volume
                • Minimum turnover
                """
            )

            st.stop()

        # ====================================================
        # STAGE 2
        # ====================================================

        progress.progress(
            65,
            text="Running Stage 2 technical analysis..."
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

                results.append({

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
                })

            except Exception:

                pass

            progress.progress(
                65 +
                int(
                    35 *
                    (i + 1)
                    / total_stage2
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
                No stocks passed the Stage 2
                technical conditions.

                Try reducing the minimum score.
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

            # ------------------------------------------------
            # TOP STOCKS
            # ------------------------------------------------

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

            st.dataframe(
                top,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # SCORE DISTRIBUTION
            # ------------------------------------------------

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

            # ------------------------------------------------
            # FULL RESULTS
            # ------------------------------------------------

            st.subheader(
                "📋 Detailed Results"
            )

            st.dataframe(
                results_df,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            csv = results_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Scanner Results",
                data=csv,
                file_name=(
                    "NSE_Smart_Breakout_Scanner.csv"
                ),
                mime="text/csv"
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📈 Scanner Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Universe",
                len(stocks)
            )

            c2.metric(
                "Stage 1",
                len(stage1_stocks)
            )

            c3.metric(
                "Stage 2",
                len(results_df)
            )

            c4.metric(
                "Score ≥ 8",
                len(
                    results_df[
                        results_df[
                            "Technical Score"
                        ] >= 8
                    ]
                )
            )


# ============================================================
# AI ANALYST
# ============================================================

else:

    st.header(
        "🤖 AI Technical Analyst"
    )

    st.write(
        """
        Ask questions about technical analysis,
        chart patterns, indicators and trading setups.
        """
    )

    if client is None:

        st.warning(
            """
            OpenAI API key is not configured.

            Add your key to Streamlit Cloud:

            Settings → Secrets
            """
        )

    question = st.chat_input(
        "Ask your technical-analysis question..."
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
                    "OpenAI API key is missing."
                )

            else:

                try:

                    response = (
                        client.responses.create(
                            model="gpt-5.6-luna",

                            instructions="""
You are an educational technical-analysis
assistant for Indian equity markets.

Explain technical-analysis concepts clearly.

You may discuss:
- Candlestick patterns
- Support and resistance
- Moving averages
- RSI
- MACD
- Bollinger Bands
- Donchian channels
- Volume
- Breakouts
- Trend structure
- Elliott Wave concepts
- Risk management
- Position sizing

Do not invent live market prices,
technical indicator values or company data.

If the application has not supplied
actual market data, clearly state that
you cannot verify the current value.

Do not present predictions as certainty.

Use clear sections:
1. Interpretation
2. Technical reasoning
3. Risk
4. What to monitor

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

                    st.error(
                        f"AI error: {e}"
                    )
