import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
).strip().upper()

period = st.sidebar.selectbox(
    "Chart Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

show_sma20 = st.sidebar.checkbox(
    "SMA 20",
    value=True
)

show_sma50 = st.sidebar.checkbox(
    "SMA 50",
    value=True
)

show_sma200 = st.sidebar.checkbox(
    "SMA 200",
    value=True
)

show_rsi = st.sidebar.checkbox(
    "RSI 14",
    value=True
)

ticker = symbol + ".NS"


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker, period):

    for attempt in range(3):

        try:

            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=20
            )

            if data is not None and not data.empty:
                return data

        except Exception:
            pass

        time.sleep(1)

    return None


try:

    with st.spinner(f"Loading {symbol} market data..."):

        data = get_stock_data(ticker, period)

    if data is None or data.empty:

        st.error(
            f"Unable to retrieve data for {symbol}. "
            "Please try again."
        )

        st.stop()

    # ========================================================
    # HANDLE YAHOO FINANCE COLUMNS
    # ========================================================

    if hasattr(data.columns, "nlevels"):

        if data.columns.nlevels > 1:

            data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        st.error(
            f"Missing market data: {', '.join(missing)}"
        )

        st.stop()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    data["SMA20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["SMA50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    data["SMA200"] = (
        data["Close"]
        .rolling(window=200)
        .mean()
    )

    # ========================================================
    # RSI 14
    # ========================================================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()

    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    data["RSI14"] = 100 - (
        100 / (1 + rs)
    )

    # ========================================================
    # CREATE SUBPLOTS
    # ========================================================

    if show_rsi:

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.70, 0.30]
        )

    else:

        fig = make_subplots(
            rows=1,
            cols=1
        )

    # ========================================================
    # PRICE CHART
    # ========================================================

    fig.add_trace(

        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol
        ),

        row=1,
        col=1
    )

    # ========================================================
    # SMA 20
    # ========================================================

    if show_sma20:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA20"],
                mode="lines",
                name="SMA 20"
            ),

            row=1,
            col=1
        )

    # ========================================================
    # SMA 50
    # ========================================================

    if show_sma50:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA50"],
                mode="lines",
                name="SMA 50"
            ),

            row=1,
            col=1
        )

    # ========================================================
    # SMA 200
    # ========================================================

    if show_sma200:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["SMA200"],
                mode="lines",
                name="SMA 200"
            ),

            row=1,
            col=1
        )

    # ========================================================
    # RSI PANEL
    # ========================================================

    if show_rsi:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["RSI14"],
                mode="lines",
                name="RSI 14"
            ),

            row=2,
            col=1
        )

        # RSI 70 line

        fig.add_hline(
            y=70,
            row=2,
            col=1
        )

        # RSI 50 line

        fig.add_hline(
            y=50,
            row=2,
            col=1
        )

        # RSI 30 line

        fig.add_hline(
            y=30,
            row=2,
            col=1
        )

        fig.update_yaxes(
            range=[0, 100],
            title_text="RSI",
            row=2,
            col=1
        )

    # ========================================================
    # CHART LAYOUT
    # ========================================================

    fig.update_layout(

        title=f"{symbol} Technical Analysis",

        height=850,

        xaxis_rangeslider_visible=False,

        hovermode="x unified",

        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    latest = data.iloc[-1]

    close = float(latest["Close"])

    sma20 = float(latest["SMA20"])

    sma50 = float(latest["SMA50"])

    sma200 = float(latest["SMA200"])

    rsi = float(latest["RSI14"])

    st.subheader("📊 Current Technical Values")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Current Price",
        f"₹{close:.2f}"
    )

    col2.metric(
        "SMA 20",
        f"₹{sma20:.2f}"
    )

    col3.metric(
        "SMA 50",
        f"₹{sma50:.2f}"
    )

    col4.metric(
        "SMA 200",
        f"₹{sma200:.2f}"
    )

    col5.metric(
        "RSI 14",
        f"{rsi:.2f}"
    )

    # ========================================================
    # RSI INTERPRETATION
    # ========================================================

    st.subheader("⚡ Momentum Status")

    if rsi >= 70:

        st.warning(
            f"🔴 RSI = {rsi:.2f} — Overbought zone"
        )

    elif rsi <= 30:

        st.success(
            f"🟢 RSI = {rsi:.2f} — Oversold zone"
        )

    elif rsi > 50:

        st.success(
            f"🟢 RSI = {rsi:.2f} — Positive momentum"
        )

    else:

        st.warning(
            f"🟡 RSI = {rsi:.2f} — Weak/neutral momentum"
        )

    # ========================================================
    # TREND STATUS
    # ========================================================

    st.subheader("📈 Moving Average Trend")

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
            "🟢 SMA 50 is ABOVE SMA 200"
        )

    else:

        st.warning(
            "🔴 SMA 50 is BELOW SMA 200"
        )

    if sma20 > sma50:

        st.success(
            "🟢 SMA 20 is ABOVE SMA 50"
        )

    else:

        st.warning(
            "🔴 SMA 20 is BELOW SMA 50"
        )

    # ========================================================
    # DATA STATUS
    # ========================================================

    st.caption(
        f"Data points: {len(data)} | "
        f"Last available date: {data.index[-1].date()}"
    )

    st.success(
        f"✅ {symbol} analysis loaded successfully"
    )


except Exception as e:

    st.error(
        f"Unexpected error: {e}"
    )
