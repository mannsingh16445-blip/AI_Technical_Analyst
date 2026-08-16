import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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
# NIFTY 50 UNIVERSE
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
# SIDEBAR
# ============================================================

st.sidebar.title("📈 AI Technical Analyst")

page = st.sidebar.radio(
    "Select Module",
    [
        "Technical Chart",
        "Early Breakout Scanner"
    ]
)

# ============================================================
# DATA DOWNLOAD FUNCTION
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(tickers, period="2y"):

    yahoo_tickers = [
        ticker + ".NS"
        for ticker in tickers
    ]

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
# INDICATOR CALCULATION
# ============================================================

def calculate_indicators(data):

    data = data.copy()

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
    # RSI 14
    # --------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI14"] = 100 - (
        100 / (1 + rs)
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
    # VOLUME
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

    # --------------------------------------------------------
    # DONCHIAN CHANNEL
    #
    # Previous 3 completed candles only
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
# EARLY BREAKOUT CONDITIONS
# ============================================================

def check_breakout(data):

    if len(data) < 210:
        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    two_days_ago = data.iloc[-3]

    # --------------------------------------------------------
    # CONDITION 1
    # Recent Close > Previous Day High
    # --------------------------------------------------------

    condition_1 = (
        latest["Close"] >
        previous["High"]
    )

    # --------------------------------------------------------
    # CONDITION 2
    #
    # Previous Day High <
    # High of N-2 days
    #
    # N = latest day
    # N-1 = previous day
    # N-2 = two days ago
    # --------------------------------------------------------

    condition_2 = (
        previous["High"] <
        two_days_ago["High"]
    )

    # --------------------------------------------------------
    # CONDITION 3
    #
    # Recent Close <
    # Previous 3-day Donchian Upper
    # --------------------------------------------------------

    condition_3 = (
        latest["Close"] <
        latest["DONCHIAN_UPPER"]
    )

    # --------------------------------------------------------
    # CONDITION 4
    #
    # Recent Low >
    # Previous 3-day Donchian Lower
    # --------------------------------------------------------

    condition_4 = (
        latest["Low"] >
        latest["DONCHIAN_LOWER"]
    )

    # --------------------------------------------------------
    # CONDITION 5
    #
    # Recent Close > 200 SMA
    # --------------------------------------------------------

    condition_5 = (
        latest["Close"] >
        latest["SMA200"]
    )

    # --------------------------------------------------------
    # CORE SETUP
    # --------------------------------------------------------

    core_setup = (
        condition_1
        and condition_2
        and condition_3
        and condition_4
        and condition_5
    )

    # --------------------------------------------------------
    # CONFIRMATION CONDITIONS
    # --------------------------------------------------------

    volume_confirmation = (
        latest["VOLUME_RATIO"] >= 1.5
    )

    rsi_confirmation = (
        latest["RSI14"] > 50
    )

    macd_confirmation = (
        latest["MACD"] >
        latest["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # TECHNICAL SCORE
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

    if volume_confirmation:
        score += 1

    if rsi_confirmation:
        score += 1

    if macd_confirmation:
        score += 1

    return {
        "Core Setup": core_setup,
        "Condition 1": condition_1,
        "Condition 2": condition_2,
        "Condition 3": condition_3,
        "Condition 4": condition_4,
        "Condition 5": condition_5,
        "Volume Confirm": volume_confirmation,
        "RSI Confirm": rsi_confirmation,
        "MACD Confirm": macd_confirmation,
        "Score": score,
        "Close": latest["Close"],
        "SMA200": latest["SMA200"],
        "RSI": latest["RSI14"],
        "MACD": latest["MACD"],
        "Signal": latest["MACD_SIGNAL"],
        "Volume Ratio": latest["VOLUME_RATIO"],
        "Donchian Upper": latest["DONCHIAN_UPPER"],
        "Donchian Lower": latest["DONCHIAN_LOWER"]
    }


# ============================================================
# TECHNICAL CHART PAGE
# ============================================================

if page == "Technical Chart":

    st.title("📈 AI Technical Analyst")

    st.caption(
        "Single-stock technical analysis"
    )

    symbol = st.sidebar.text_input(
        "Enter NSE Stock Symbol",
        value="RELIANCE"
    ).strip().upper()

    period = st.sidebar.selectbox(
        "Chart Period",
        ["6mo", "1y", "2y", "5y"],
        index=1
    )

    ticker = symbol + ".NS"

    data = download_market_data(
        [symbol],
        period
    )

    if data is None:

        st.error(
            "Unable to retrieve market data."
        )

        st.stop()

    # Extract ticker data

    try:

        stock_data = data[ticker].copy()

    except Exception:

        st.error(
            f"No data found for {symbol}"
        )

        st.stop()

    stock_data = stock_data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    stock_data = calculate_indicators(
        stock_data
    )

    st.subheader(
        f"{symbol} Price Chart"
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
        ("DONCHIAN_UPPER", "Donchian Upper"),
        ("DONCHIAN_LOWER", "Donchian Lower")
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
        "Scans the selected NSE universe using your breakout strategy"
    )

    # --------------------------------------------------------
    # SCANNER SETTINGS
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Scanner Settings"
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

    scan_button = st.sidebar.button(
        "🔍 Run Scanner",
        type="primary"
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    with st.expander(
        "📋 Scanner Conditions",
        expanded=False
    ):

        st.write(
            """
            **Core Conditions**

            1. Recent Close > Previous Day High

            2. Previous Day High < High of N-2

            3. Recent Close < Previous 3-Day Donchian Upper

            4. Recent Low > Previous 3-Day Donchian Lower

            5. Recent Close > 200 SMA

            **Confirmation**

            • Volume Ratio >= 1.5

            • RSI > 50

            • MACD > Signal
            """
        )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if scan_button:

        progress = st.progress(
            0,
            text="Downloading market data..."
        )

        market_data = download_market_data(
            NIFTY50,
            "2y"
        )

        progress.progress(
            50,
            text="Calculating technical indicators..."
        )

        if market_data is None:

            st.error(
                "Unable to download market data. "
                "Please try again later."
            )

            st.stop()

        results = []

        total = len(NIFTY50)

        for index, symbol in enumerate(NIFTY50):

            ticker = symbol + ".NS"

            try:

                stock_data = market_data[ticker].copy()

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

                    continue

                stock_data = calculate_indicators(
                    stock_data
                )

                signal = check_breakout(
                    stock_data
                )

                if signal is None:

                    continue

                # ------------------------------------------------
                # Apply optional confirmation filters
                # ------------------------------------------------

                if signal["Score"] < min_score:

                    continue

                if (
                    require_volume
                    and not signal["Volume Confirm"]
                ):

                    continue

                if (
                    require_rsi
                    and not signal["RSI Confirm"]
                ):

                    continue

                if (
                    require_macd
                    and not signal["MACD Confirm"]
                ):

                    continue

                # ------------------------------------------------
                # Add result
                # ------------------------------------------------

                results.append({

                    "Stock": symbol,

                    "Close": round(
                        signal["Close"],
                        2
                    ),

                    "200 SMA": round(
                        signal["SMA200"],
                        2
                    ),

                    "RSI": round(
                        signal["RSI"],
                        1
                    ),

                    "MACD": round(
                        signal["MACD"],
                        2
                    ),

                    "Volume Ratio": round(
                        signal["Volume Ratio"],
                        2
                    ),

                    "Donchian Upper": round(
                        signal["Donchian Upper"],
                        2
                    ),

                    "Donchian Lower": round(
                        signal["Donchian Lower"],
                        2
                    ),

                    "C1 Close > Prev High":
                        "✓"
                        if signal["Condition 1"]
                        else "✗",

                    "C2 High Structure":
                        "✓"
                        if signal["Condition 2"]
                        else "✗",

                    "C3 Below Donchian Upper":
                        "✓"
                        if signal["Condition 3"]
                        else "✗",

                    "C4 Above Donchian Lower":
                        "✓"
                        if signal["Condition 4"]
                        else "✗",

                    "C5 Above 200 SMA":
                        "✓"
                        if signal["Condition 5"]
                        else "✗",

                    "Volume":
                        "✓"
                        if signal["Volume Confirm"]
                        else "✗",

                    "MACD":
                        "✓"
                        if signal["MACD Confirm"]
                        else "✗",

                    "Technical Score":
                        signal["Score"]

                })

            except Exception:
                continue

            progress.progress(
                50 +
                int(
                    50 *
                    (index + 1) /
                    total
                ),
                text=f"Scanning {symbol}..."
            )

        progress.empty()

        # --------------------------------------------------------
        # RESULTS
        # --------------------------------------------------------

        if not results:

            st.warning(
                "No stocks currently satisfy the selected "
                "conditions."
            )

        else:

            results_df = pd.DataFrame(
                results
            )

            results_df = results_df.sort_values(
                "Technical Score",
                ascending=False
            )

            st.success(
                f"🎯 {len(results_df)} stocks found"
            )

            # ----------------------------------------------------
            # TOP PICKS
            # ----------------------------------------------------

            st.subheader(
                "🏆 Top Scanner Results"
            )

            top = results_df.head(10)

            st.dataframe(
                top[
                    [
                        "Stock",
                        "Close",
                        "200 SMA",
                        "RSI",
                        "Volume Ratio",
                        "Technical Score"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # COMPLETE RESULTS
            # ----------------------------------------------------

            st.subheader(
                "📊 Detailed Results"
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # DOWNLOAD
            # ----------------------------------------------------

            csv = results_df.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Scanner Results",
                data=csv,
                file_name="early_breakout_scanner.csv",
                mime="text/csv"
            )

            # ----------------------------------------------------
            # SUMMARY
            # ----------------------------------------------------

            st.subheader(
                "📈 Scanner Summary"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Stocks Scanned",
                len(NIFTY50)
            )

            col2.metric(
                "Stocks Passed",
                len(results_df)
            )

            col3.metric(
                "Score ≥ 8",
                len(
                    results_df[
                        results_df["Technical Score"] >= 8
                    ]
                )
            )

            col4.metric(
                "Strong Volume",
                len(
                    results_df[
                        results_df["Volume Ratio"] >= 1.5
                    ]
                )
            )
