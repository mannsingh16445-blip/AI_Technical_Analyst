import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
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
# NSE EQUITY UNIVERSE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
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

        from io import StringIO

        df = pd.read_csv(
            StringIO(response.text)
        )

        # Clean column names

        df.columns = [
            str(c).strip().upper()
            for c in df.columns
        ]

        # Find symbol column

        symbol_column = None

        for column in [
            "SYMBOL",
            "SYMBOL ",
            "SECURITY SYMBOL"
        ]:

            if column in df.columns:
                symbol_column = column
                break

        if symbol_column is None:

            return []

        symbols = (
            df[symbol_column]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        # Remove invalid values

        invalid = {
            "",
            "NAN",
            "NONE",
            "SYMBOL"
        }

        symbols = [
            symbol
            for symbol in symbols
            if symbol not in invalid
        ]

        return sorted(
            list(set(symbols))
        )

    except Exception as e:

        st.warning(
            "Could not automatically load the NSE "
            "equity universe. Using Nifty 50 as fallback."
        )

        return NIFTY50


# ============================================================
# NIFTY 500
#
# This is loaded from NSE's index constituents CSV if
# available. Otherwise the app uses Nifty 50 as fallback.
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty500():

    possible_urls = [

        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    for url in possible_urls:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                continue

            from io import StringIO

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
                .tolist()
            )

            symbols = [
                x for x in symbols
                if x not in [
                    "",
                    "NAN",
                    "NONE"
                ]
            ]

            if len(symbols) > 400:

                return sorted(
                    list(set(symbols))
                )

        except Exception:
            continue

    return NIFTY50


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def download_market_data(
    tickers,
    period="1y"
):

    yahoo_tickers = [
        ticker + ".NS"
        for ticker in tickers
    ]

    if not yahoo_tickers:
        return None

    for attempt in range(3):

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

            if data is not None and not data.empty:

                return data

        except Exception:

            pass

        time.sleep(2)

    return None


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calculate_indicators(data):

    data = data.copy()

    # --------------------------------------------------------
    # Moving averages
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

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

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

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["MACD_HIST"] = (
        data["MACD"] -
        data["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["VOLUME_AVG20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["VOLUME_RATIO"] = (
        data["Volume"] /
        data["VOLUME_AVG20"]
    )

    # Average traded value

    data["TURNOVER"] = (
        data["Close"] *
        data["Volume"]
    )

    data["TURNOVER_AVG20"] = (
        data["TURNOVER"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # Donchian Channel
    #
    # Previous 3 completed candles
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
# BREAKOUT CONDITIONS
# ============================================================

def check_breakout(data):

    if len(data) < 210:
        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    two_days_ago = data.iloc[-3]

    # --------------------------------------------------------
    # Five core conditions
    # --------------------------------------------------------

    c1 = (
        latest["Close"] >
        previous["High"]
    )

    c2 = (
        previous["High"] <
        two_days_ago["High"]
    )

    c3 = (
        latest["Close"] <
        latest["DONCHIAN_UPPER"]
    )

    c4 = (
        latest["Low"] >
        latest["DONCHIAN_LOWER"]
    )

    c5 = (
        latest["Close"] >
        latest["SMA200"]
    )

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    volume_confirm = (
        latest["VOLUME_RATIO"] >= 1.5
    )

    rsi_confirm = (
        latest["RSI14"] > 50
    )

    macd_confirm = (
        latest["MACD"] >
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

        "Core Setup":
            c1 and c2 and c3 and c4 and c5,

        "C1":
            c1,

        "C2":
            c2,

        "C3":
            c3,

        "C4":
            c4,

        "C5":
            c5,

        "Volume":
            volume_confirm,

        "RSI":
            rsi_confirm,

        "MACD":
            macd_confirm,

        "Score":
            score,

        "Close":
            latest["Close"],

        "SMA200":
            latest["SMA200"],

        "RSI Value":
            latest["RSI14"],

        "MACD Value":
            latest["MACD"],

        "Volume Ratio":
            latest["VOLUME_RATIO"],

        "Turnover Avg":
            latest["TURNOVER_AVG20"],

        "Donchian Upper":
            latest["DONCHIAN_UPPER"],

        "Donchian Lower":
            latest["DONCHIAN_LOWER"]
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📈 AI Technical Analyst"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Technical Chart",
        "Early Breakout Scanner"
    ]
)


# ============================================================
# TECHNICAL CHART
# ============================================================

if page == "Technical Chart":

    st.title(
        "📈 AI Technical Analyst"
    )

    st.caption(
        "Single-stock technical analysis"
    )

    symbol = st.sidebar.text_input(
        "NSE Stock Symbol",
        value="RELIANCE"
    ).strip().upper()

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

    market_data = download_market_data(
        [symbol],
        period
    )

    if market_data is None:

        st.error(
            "Unable to retrieve market data."
        )

        st.stop()

    ticker = symbol + ".NS"

    try:

        stock_data = market_data[
            ticker
        ].copy()

    except Exception:

        st.error(
            f"No data found for {symbol}."
        )

        st.stop()

    stock_data = stock_data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    stock_data = calculate_indicators(
        stock_data
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            name=symbol
        )
    )

    for column, name in [

        ("SMA20", "SMA 20"),

        ("SMA50", "SMA 50"),

        ("SMA200", "SMA 200"),

        (
            "DONCHIAN_UPPER",
            "Donchian Upper"
        ),

        (
            "DONCHIAN_LOWER",
            "Donchian Lower"
        )
    ]:

        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data[column],
                mode="lines",
                name=name
            )
        )

    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    latest = stock_data.iloc[-1]

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
        f"{latest['RSI14']:.2f}"
    )


# ============================================================
# EARLY BREAKOUT SCANNER
# ============================================================

else:

    st.title(
        "🚀 Early Breakout Scanner"
    )

    st.caption(
        "Full NSE / Nifty 500 / Nifty 50 technical scanner"
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

        nifty500 = load_nifty500()

    # --------------------------------------------------------
    # UNIVERSE SELECTOR
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stock Universe"
    )

    universe = st.sidebar.selectbox(
        "Select Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Full NSE",
            "Custom Number of Stocks"
        ]
    )

    if universe == "Nifty 50":

        selected_stocks = NIFTY50

    elif universe == "Nifty 500":

        selected_stocks = nifty500

    elif universe == "Full NSE":

        selected_stocks = nse_stocks

    else:

        max_stocks = len(
            nse_stocks
        )

        custom_number = st.sidebar.number_input(
            "Number of Stocks",
            min_value=50,
            max_value=max(
                50,
                max_stocks
            ),
            value=min(
                300,
                max_stocks
            ),
            step=50
        )

        selected_stocks = nse_stocks[
            :int(custom_number)
        ]

    # --------------------------------------------------------
    # LIQUIDITY FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Liquidity Filters"
    )

    min_price = st.sidebar.number_input(
        "Minimum Price ₹",
        min_value=0.0,
        value=20.0,
        step=5.0
    )

    min_avg_volume = st.sidebar.number_input(
        "Minimum Avg Volume",
        min_value=0,
        value=100000,
        step=50000
    )

    min_turnover_crore = st.sidebar.number_input(
        "Minimum Avg Turnover ₹ Crore",
        min_value=0.0,
        value=1.0,
        step=0.5
    )

    # --------------------------------------------------------
    # TECHNICAL FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Technical Filters"
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

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    st.info(
        f"Selected universe: "
        f"**{universe}** | "
        f"Stocks available: "
        f"**{len(selected_stocks)}**"
    )

    st.warning(
        "Full NSE scanning can take considerably longer "
        "than Nifty 50 scanning because market data must "
        "be retrieved for many securities."
    )

    # --------------------------------------------------------
    # SCAN BUTTON
    # --------------------------------------------------------

    scan_button = st.sidebar.button(
        "🔍 RUN SCANNER",
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
### Core setup

**1. Recent Close > Previous Day High**

**2. Previous Day High < High of N-2**

**3. Recent Close < Previous 3-Day Donchian Upper**

**4. Recent Low > Previous 3-Day Donchian Lower**

**5. Recent Close > 200 SMA**

### Confirmation

**Volume Ratio ≥ 1.5**

**RSI > 50**

**MACD > Signal**

### Score

Core conditions = **7 points**

Confirmations = **3 points**

Maximum = **10/10**
            """
        )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if scan_button:

        progress = st.progress(
            0,
            text="Starting scanner..."
        )

        # ----------------------------------------------------
        # Download data
        # ----------------------------------------------------

        progress.progress(
            10,
            text=(
                f"Downloading data for "
                f"{len(selected_stocks)} stocks..."
            )
        )

        market_data = download_market_data(
            selected_stocks,
            "2y"
        )

        if market_data is None:

            st.error(
                "Market data could not be downloaded."
            )

            st.stop()

        progress.progress(
            50,
            text="Calculating technical indicators..."
        )

        results = []

        total = len(
            selected_stocks
        )

        processed = 0

        # ----------------------------------------------------
        # PROCESS EACH STOCK
        # ----------------------------------------------------

        for symbol in selected_stocks:

            ticker = symbol + ".NS"

            try:

                stock_data = (
                    market_data[ticker]
                    .copy()
                )

                stock_data = stock_data.dropna(
                    subset=[
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]
                )

                if len(stock_data) < 210:

                    processed += 1

                    continue

                stock_data = (
                    calculate_indicators(
                        stock_data
                    )
                )

                latest = stock_data.iloc[-1]

                # --------------------------------------------
                # LIQUIDITY FILTER
                # --------------------------------------------

                if (
                    latest["Close"]
                    < min_price
                ):

                    processed += 1
                    continue

                if (
                    latest["VOLUME_AVG20"]
                    < min_avg_volume
                ):

                    processed += 1
                    continue

                avg_turnover_crore = (
                    latest["TURNOVER_AVG20"]
                    / 10000000
                )

                if (
                    avg_turnover_crore
                    < min_turnover_crore
                ):

                    processed += 1
                    continue

                # --------------------------------------------
                # BREAKOUT
                # --------------------------------------------

                signal = check_breakout(
                    stock_data
                )

                if signal is None:

                    processed += 1
                    continue

                # --------------------------------------------
                # SCORE FILTER
                # --------------------------------------------

                if (
                    signal["Score"]
                    < min_score
                ):

                    processed += 1
                    continue

                # --------------------------------------------
                # OPTIONAL CONFIRMATION FILTERS
                # --------------------------------------------

                if (
                    require_volume
                    and not signal["Volume"]
                ):

                    processed += 1
                    continue

                if (
                    require_rsi
                    and not signal["RSI"]
                ):

                    processed += 1
                    continue

                if (
                    require_macd
                    and not signal["MACD"]
                ):

                    processed += 1
                    continue

                # --------------------------------------------
                # ADD RESULT
                # --------------------------------------------

                results.append({

                    "Stock":
                        symbol,

                    "Close":
                        round(
                            signal["Close"],
                            2
                        ),

                    "200 SMA":
                        round(
                            signal["SMA200"],
                            2
                        ),

                    "RSI":
                        round(
                            signal["RSI Value"],
                            1
                        ),

                    "Volume Ratio":
                        round(
                            signal["Volume Ratio"],
                            2
                        ),

                    "Avg Turnover ₹Cr":
                        round(
                            avg_turnover_crore,
                            2
                        ),

                    "C1 Close > Prev High":
                        "✓"
                        if signal["C1"]
                        else "✗",

                    "C2 High Structure":
                        "✓"
                        if signal["C2"]
                        else "✗",

                    "C3 Below Donchian":
                        "✓"
                        if signal["C3"]
                        else "✗",

                    "C4 Above Donchian":
                        "✓"
                        if signal["C4"]
                        else "✗",

                    "C5 Above 200 SMA":
                        "✓"
                        if signal["C5"]
                        else "✗",

                    "Volume Confirm":
                        "✓"
                        if signal["Volume"]
                        else "✗",

                    "RSI Confirm":
                        "✓"
                        if signal["RSI"]
                        else "✗",

                    "MACD Confirm":
                        "✓"
                        if signal["MACD"]
                        else "✗",

                    "Technical Score":
                        signal["Score"]

                })

            except Exception:

                pass

            processed += 1

            progress.progress(
                50 +
                int(
                    50 *
                    processed /
                    total
                ),
                text=(
                    f"Scanning "
                    f"{symbol} "
                    f"({processed}/{total})"
                )
            )

        progress.empty()

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if not results:

            st.warning(
                "No stocks satisfied the selected "
                "conditions and liquidity filters."
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
                .reset_index(drop=True)
            )

            st.success(
                f"🎯 {len(results_df)} "
                f"stocks passed the scanner."
            )

            # ------------------------------------------------
            # TOP RESULTS
            # ------------------------------------------------

            st.subheader(
                "🏆 Top Opportunities"
            )

            top_columns = [
                "Stock",
                "Close",
                "200 SMA",
                "RSI",
                "Volume Ratio",
                "Technical Score"
            ]

            st.dataframe(
                results_df[
                    top_columns
                ].head(20),
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # DETAILED RESULTS
            # ------------------------------------------------

            st.subheader(
                "📊 Detailed Scanner Results"
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
                "⬇️ Download Scanner Results",
                data=csv,
                file_name=(
                    "NSE_early_breakout_scanner.csv"
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
                len(selected_stocks)
            )

            c2.metric(
                "Passed",
                len(results_df)
            )

            c3.metric(
                "Score ≥ 8",
                len(
                    results_df[
                        results_df[
                            "Technical Score"
                        ] >= 8
                    ]
                )
            )

            c4.metric(
                "Strong Volume",
                len(
                    results_df[
                        results_df[
                            "Volume Ratio"
                        ] >= 1.5
                    ]
                )
            )
