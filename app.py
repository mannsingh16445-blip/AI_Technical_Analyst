import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Technical Analyst")
st.caption("Technical Analysis Dashboard")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE"
)

period = st.sidebar.selectbox(
    "Chart Period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

show_sma = st.sidebar.checkbox(
    "Show SMA 20 / 50 / 200",
    value=True
)

show_donchian = st.sidebar.checkbox(
    "Show Donchian Channel",
    value=True
)

show_rsi = st.sidebar.checkbox(
    "Show RSI",
    value=True
)

show_macd = st.sidebar.checkbox(
    "Show MACD",
    value=True
)

show_volume = st.sidebar.checkbox(
    "Show Volume",
    value=True
)

# ============================================================
# MARKET DATA
# ============================================================

ticker = symbol.upper().strip() + ".NS"

try:

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:

        st.error(
            f"No market data found for {symbol.upper()}."
        )

        st.stop()

    # Handle Yahoo Finance multi-level columns
    if hasattr(data.columns, "nlevels"):

        if data.columns.nlevels > 1:

            data.columns = data.columns.get_level_values(0)

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    # Simple Moving Averages
    data["SMA20"] = data["Close"].rolling(20).mean()

    data["SMA50"] = data["Close"].rolling(50).mean()

    data["SMA200"] = data["Close"].rolling(200).mean()

    # Donchian Channel - 3 periods
    data["Donchian_Upper"] = data["High"].rolling(3).max()

    data["Donchian_Lower"] = data["Low"].rolling(3).min()

    # Middle line
    data["Donchian_Middle"] = (
        data["Donchian_Upper"] +
        data["Donchian_Lower"]
    ) / 2

    # ========================================================
    # RSI
    # ========================================================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # ========================================================
    # MACD
    # ========================================================

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_Hist"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    # ========================================================
    # VOLUME
    # ========================================================

    data["Volume_Avg20"] = (
        data["Volume"].rolling(20).mean()
    )

    # ========================================================
    # CREATE CHART
    # ========================================================

    rows = 1

    if show_volume:
        rows += 1

    if show_rsi:
        rows += 1

    if show_macd:
        rows += 1

    row_heights = []

    for i in range(rows):

        if i == 0:
            row_heights.append(0.50)

        else:
            row_heights.append(
                0.50 / (rows - 1)
            )

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    # ========================================================
    # PRICE / CANDLESTICK
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=data.index,

            open=data["Open"],

            high=data["High"],

            low=data["Low"],

            close=data["Close"],

            name=symbol.upper()

        ),

        row=1,
        col=1

    )

    # ========================================================
    # SMA
    # ========================================================

    if show_sma:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA20"],
                name="SMA 20",
                mode="lines"
            ),

            row=1,
            col=1

        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA50"],
                name="SMA 50",
                mode="lines"
            ),

            row=1,
            col=1

        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA200"],
                name="SMA 200",
                mode="lines"
            ),

            row=1,
            col=1

        )

    # ========================================================
    # DONCHIAN CHANNEL
    # ========================================================

    if show_donchian:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Donchian_Upper"],
                name="Donchian Upper (3)",
                mode="lines"
            ),

            row=1,
            col=1

        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Donchian_Lower"],
                name="Donchian Lower (3)",
                mode="lines"
            ),

            row=1,
            col=1

        )

    current_row = 2

    # ========================================================
    # VOLUME
    # ========================================================

    if show_volume:

        fig.add_trace(

            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume"
            ),

            row=current_row,
            col=1

        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Volume_Avg20"],
                name="Avg Volume 20",
                mode="lines"
            ),

            row=current_row,
            col=1

        )

        current_row += 1

    # ========================================================
    # RSI
    # ========================================================

    if show_rsi:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["RSI"],
                name="RSI 14",
                mode="lines"
            ),

            row=current_row,
            col=1

        )

        fig.add_hline(
            y=70,
            row=current_row,
            col=1
        )

        fig.add_hline(
            y=30,
            row=current_row,
            col=1
        )

        current_row += 1

    # ========================================================
    # MACD
    # ========================================================

    if show_macd:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["MACD"],
                name="MACD",
                mode="lines"
            ),

            row=current_row,
            col=1

        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["MACD_Signal"],
                name="MACD Signal",
                mode="lines"
            ),

            row=current_row,
            col=1

        )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        title=f"{symbol.upper()} Technical Analysis",

        height=1000,

        xaxis_rangeslider_visible=False,

        hovermode="x unified",

        legend=dict(
            orientation="h"
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    latest = data.iloc[-1]

    st.subheader("📊 Current Technical Values")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Close",
        f"₹{float(latest['Close']):.2f}"
    )

    col2.metric(
        "SMA 20",
        f"₹{float(latest['SMA20']):.2f}"
    )

    col3.metric(
        "SMA 50",
        f"₹{float(latest['SMA50']):.2f}"
    )

    col4.metric(
        "SMA 200",
        f"₹{float(latest['SMA200']):.2f}"
    )

    col5.metric(
        "RSI",
        f"{float(latest['RSI']):.2f}"
    )

    st.subheader("🎯 Donchian Channel")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Upper (3)",
        f"₹{float(latest['Donchian_Upper']):.2f}"
    )

    col2.metric(
        "Middle",
        f"₹{float(latest['Donchian_Middle']):.2f}"
    )

    col3.metric(
        "Lower (3)",
        f"₹{float(latest['Donchian_Lower']):.2f}"
    )

    # ========================================================
    # TREND STATUS
    # ========================================================

    st.subheader("📈 Trend Status")

    close = float(latest["Close"])

    sma20 = float(latest["SMA20"])

    sma50 = float(latest["SMA50"])

    sma200 = float(latest["SMA200"])

    if close > sma200:

        st.success(
            "🟢 Price is ABOVE the 200 SMA"
        )

    else:

        st.warning(
            "🔴 Price is BELOW the 200 SMA"
        )

    if sma50 > sma200:

        st.success(
            "🟢 50 SMA is ABOVE 200 SMA — bullish trend structure"
        )

    else:

        st.warning(
            "🔴 50 SMA is BELOW 200 SMA — weak trend structure"
        )

except Exception as e:

    st.error(
        f"Error: {e}"
    )
