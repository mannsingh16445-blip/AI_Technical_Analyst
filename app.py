import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
from io import StringIO


# ============================================================
# PAGE CONFIG
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
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP",
    "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL",
    "BHARTIARTL", "CIPLA", "COALINDIA",
    "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI",
    "MAXHEALTH", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO"
]


# ============================================================
# LOAD NSE EQUITY UNIVERSE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_equity_universe():

    url = (
        "https://nsearchives.nseindia.com/"
        "content/equities/EQUITY_L.csv"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
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

            st.warning(
                "NSE returned an unexpected file format."
            )

            return []

        symbols = (
            df["SYMBOL"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        symbols = symbols[
            ~symbols.isin([
                "",
                "NAN",
                "NONE"
            ])
        ]

        symbols = sorted(
            symbols.drop_duplicates().tolist()
        )

        return symbols

    except Exception as e:

        st.warning(
            f"NSE stock list could not be downloaded: {e}"
        )

        return []


# ============================================================
# LOAD NIFTY 500
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
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
                ~symbols.isin([
                    "",
                    "NAN",
                    "NONE"
                ])
            ]

            if len(symbols) >= 400:

                return sorted(
                    symbols.drop_duplicates().tolist()
                )

        except Exception:
            continue

    return NIFTY50


# ============================================================
# BATCH DOWNLOAD
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_batches(
    tickers,
    period="1y",
    batch_size=50
):

    all_data = {}

    ticker_list = list(tickers)

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

            if data is None or data.empty:
                continue

            # ==================================================
            # PROCESS EACH STOCK
            # ==================================================

            for ticker in batch:

                yahoo_symbol = ticker + ".NS"

                try:

                    # ------------------------------------------
                    # SINGLE STOCK
                    # ------------------------------------------

                    if len(batch) == 1:

                        stock = data.copy()

                    # ------------------------------------------
                    # MULTIPLE STOCKS
                    # ------------------------------------------

                    else:

                        if yahoo_symbol not in data.columns.get_level_values(0):

                            continue

                        stock = data[
                            yahoo_symbol
                        ].copy()

                    # ------------------------------------------
                    # FIX MULTIINDEX COLUMNS
                    # ------------------------------------------

                    if isinstance(
                        stock.columns,
                        pd.MultiIndex
                    ):

                        stock.columns = [
                            col[0]
                            if isinstance(col, tuple)
                            else col
                            for col in stock.columns
                        ]

                    # ------------------------------------------
                    # NORMALIZE COLUMN NAMES
                    # ------------------------------------------

                    stock.columns = [
                        str(col).strip().title()
                        for col in stock.columns
                    ]

                    required = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    # ------------------------------------------
                    # CHECK REQUIRED COLUMNS
                    # ------------------------------------------

                    if not all(
                        col in stock.columns
                        for col in required
                    ):

                        continue

                    # ------------------------------------------
                    # CLEAN DATA
                    # ------------------------------------------

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if len(stock) > 0:

                        all_data[ticker] = stock

                except Exception:

                    continue

        except Exception:

            continue

        time.sleep(0.25)

    return all_data

# ============================================================
# INDICATORS
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

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
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
        data["MACD"] -
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
        data["Volume"] /
        data["AVG_VOLUME20"]
    )

    data["TURNOVER"] = (
        data["Close"] *
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
# STAGE 1
# LIQUIDITY + TREND FILTER
# ============================================================

def stage_one_filter(
    data,
    min_price,
    min_avg_volume,
    min_turnover_crore
):

    if len(data) < 210:
        return False

    latest = data.iloc[-1]

    if pd.isna(
        latest["SMA200"]
    ):
        return False

    if pd.isna(
        latest["AVG_VOLUME20"]
    ):
        return False

    if pd.isna(
        latest["AVG_TURNOVER20"]
    ):
        return False

    # Price filter

    if (
        latest["Close"]
        < min_price
    ):
        return False

    # Average volume

    if (
        latest["AVG_VOLUME20"]
        < min_avg_volume
    ):
        return False

    # Average turnover

    turnover_crore = (
        latest["AVG_TURNOVER20"]
        / 10000000
    )

    if (
        turnover_crore
        < min_turnover_crore
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
# STAGE 2
# BREAKOUT ENGINE
# ============================================================

def stage_two_analysis(data):

    if len(data) < 210:
        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    n_minus_2 = data.iloc[-3]

    # --------------------------------------------------------
    # CORE CONDITIONS
    # --------------------------------------------------------

    c1 = (
        latest["Close"]
        >
        previous["High"]
    )

    c2 = (
        previous["High"]
        <
        n_minus_2["High"]
    )

    c3 = (
        latest["Close"]
        <
        latest["DONCHIAN_UPPER"]
    )

    c4 = (
        latest["Low"]
        >
        latest["DONCHIAN_LOWER"]
    )

    c5 = (
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
            latest["Close"],

        "SMA200":
            latest["SMA200"],

        "RSI":
            latest["RSI14"],

        "MACD":
            latest["MACD"],

        "Signal":
            latest["MACD_SIGNAL"],

        "Volume Ratio":
            latest["VOLUME_RATIO"],

        "Avg Turnover":
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
    "📈 AI Technical Analyst"
)

page = st.sidebar.radio(
    "Module",
    [
        "Technical Chart",
        "Smart Breakout Scanner"
    ]
)


# ============================================================
# TECHNICAL CHART
# ============================================================

if page == "Technical Chart":

    st.title(
        "📈 Technical Analysis"
    )

    symbol = st.sidebar.text_input(
        "NSE Symbol",
        "RELIANCE"
    ).strip().upper()

    period = st.sidebar.selectbox(
        "Period",
        [
            "6mo",
            "1y",
            "2y",
            "5y"
        ],
        index=1
    )

    st.info(
        f"Loading {symbol}.NS..."
    )

    market = download_batches(
        [symbol],
        period,
        1
    )

    if not market:

        st.error(
            f"""
            Unable to retrieve **{symbol}.NS**

            Please check:
            1. The NSE symbol is correct.
            2. The stock is listed on NSE.
            3. Yahoo Finance is currently returning data.
            """
        )

        st.stop()

    if symbol not in market:

        st.error(
            f"No Yahoo Finance data available for {symbol}.NS"
        )

        st.stop()

    data = market[symbol].copy()

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing:

        st.error(
            f"Missing columns: {missing}"
        )

        st.stop()

    # ========================================================
    # INDICATORS
    # ========================================================

    data = calculate_indicators(
        data
    )

    if data.empty:

        st.error(
            "No usable historical data after processing."
        )

        st.stop()

    # ========================================================
    # CHART
    # ========================================================

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

    # ========================================================
    # SMA 20
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            mode="lines",
            name="SMA 20"
        )
    )

    # ========================================================
    # SMA 50
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            mode="lines",
            name="SMA 50"
        )
    )

    # ========================================================
    # SMA 200
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA200"],
            mode="lines",
            name="SMA 200"
        )
    )

    # ========================================================
    # DONCHIAN UPPER
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["DONCHIAN_UPPER"],
            mode="lines",
            name="Donchian Upper"
        )
    )

    # ========================================================
    # DONCHIAN LOWER
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["DONCHIAN_LOWER"],
            mode="lines",
            name="Donchian Lower"
        )
    )

    fig.update_layout(
        title=f"{symbol} — Technical Analysis",
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ========================================================
    # CURRENT DATA
    # ========================================================

    latest = data.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Close",
        f"₹{latest['Close']:.2f}"
    )

    col2.metric(
        "RSI",
        f"{latest['RSI14']:.1f}"
    )

    col3.metric(
        "200 SMA",
        f"₹{latest['SMA200']:.2f}"
    )

    col4.metric(
        "Volume Ratio",
        f"{latest['VOLUME_RATIO']:.2f}x"
    )


# ============================================================
# SMART SCANNER
# ============================================================

else:

    st.title(
        "🚀 Smart Two-Stage Breakout Scanner"
    )

    st.caption(
        "Stage 1: Liquidity + trend → "
        "Stage 2: breakout + momentum confirmation"
    )

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    nse_stocks = (
        load_nse_equity_universe()
    )

    nifty500 = (
        load_nifty500()
    )

    st.sidebar.subheader(
        "Universe"
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


    if universe == "Full NSE" and not stocks:

        st.error(
            """
            Full NSE stock list could not be loaded.

            Please try again later or select Nifty 500.
            """
        )

        st.stop()


    st.info(
        f"Universe selected: **{universe} — {len(stocks)} stocks**"
    )

    # --------------------------------------------------------
    # STAGE 1 FILTERS
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
        "Minimum Score",
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

    batch_size = st.sidebar.slider(
        "Download Batch Size",
        25,
        100,
        50,
        step=25
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.info(
        f"Universe selected: **{len(stocks)} stocks**"
    )

    st.write(
        """
        **Stage 1** removes stocks with insufficient
        price, liquidity or long-term trend.

        **Stage 2** applies your complete early-breakout
        strategy to the survivors.
        """
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run = st.sidebar.button(
        "🚀 RUN SMART SCANNER",
        type="primary"
    )

    if run:

        # ====================================================
        # STAGE 1 DOWNLOAD
        # ====================================================

        progress = st.progress(
            0,
            text="Stage 1: downloading market data..."
        )

        market = download_batches(
            stocks,
            "1y",
            batch_size
        )

        progress.progress(
            40,
            text="Stage 1: applying liquidity filters..."
        )

        stage1 = []

        total = len(market)

        for i, (
            symbol,
            raw_data
        ) in enumerate(
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

                    stage1.append(
                        symbol
                    )

            except Exception:

                pass

            if total > 0:

                progress.progress(
                    40 +
                    int(
                        20 *
                        (i + 1)
                        / total
                    ),
                    text=(
                        f"Stage 1: "
                        f"{symbol}"
                    )
                )

        # ====================================================
        # STAGE 1 RESULT
        # ====================================================

        progress.progress(
            65,
            text="Stage 1 completed..."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Universe",
            len(stocks)
        )

        col2.metric(
            "Data Retrieved",
            len(market)
        )

        col3.metric(
            "Stage 1 Survivors",
            len(stage1)
        )

        if not stage1:

            progress.empty()

            st.warning(
                "No stocks survived Stage 1. "
                "Try reducing the liquidity thresholds."
            )

            st.stop()

        # ====================================================
        # STAGE 2
        # ====================================================

        progress.progress(
            70,
            text="Stage 2: running breakout analysis..."
        )

        results = []

        for i, symbol in enumerate(
            stage1
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
                                "Avg Turnover"
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
                70 +
                int(
                    30 *
                    (i + 1)
                    / len(stage1)
                ),
                text=(
                    f"Stage 2: "
                    f"{symbol}"
                )
            )

        progress.empty()

        # ====================================================
        # RESULTS
        # ====================================================

        if not results:

            st.warning(
                "No stocks satisfied the Stage 2 "
                "technical conditions."
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
                f"stocks passed Stage 2."
            )

            # ------------------------------------------------
            # TOP PICKS
            # ------------------------------------------------

            st.subheader(
                "🏆 Top Early Breakout Candidates"
            )

            st.dataframe(
                results_df[
                    [
                        "Stock",
                        "Close",
                        "200 SMA",
                        "RSI",
                        "Volume Ratio",
                        "Technical Score"
                    ]
                ].head(20),
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # COMPLETE RESULTS
            # ------------------------------------------------

            st.subheader(
                "📊 Detailed Results"
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
                "⬇️ Download Results",
                csv,
                "smart_breakout_results.csv",
                "text/csv"
            )

            # ------------------------------------------------
            # SCORE DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📈 Score Distribution"
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
            # STAGE SUMMARY
            # ------------------------------------------------

            st.subheader(
                "🔎 Two-Stage Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Universe",
                len(stocks)
            )

            c2.metric(
                "Stage 1",
                len(stage1)
            )

            c3.metric(
                "Stage 2",
                len(results_df)
            )

            c4.metric(
                "High Score ≥ 8",
                len(
                    results_df[
                        results_df[
                            "Technical Score"
                        ] >= 8
                    ]
                )
            )
