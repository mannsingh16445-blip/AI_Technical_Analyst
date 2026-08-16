import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
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
    # CREATE CHART
    # ========================================================

    fig = go.Figure()

    # Candlestick

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

    # SMA 20

    if show_sma20:

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA20"],
                mode="lines",
                name="SMA 20"
            )
        )

    # SMA 50

    if show_sma50:

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA50"],
                mode="lines",
                name="SMA 50"
            )
        )

    # SMA 200

    if show_sma200:

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA200"],
                mode="lines",
                name="SMA 200"
            )
        )

    # ========================================================
    # CHART LAYOUT
    # ========================================================

    fig.update_layout(

        title=f"{symbol} Price + Moving Averages",

        xaxis_title="Date",

        yaxis_title="Price",

        height=700,

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

    st.subheader("📊 Current Technical Values")

    col1, col2, col3, col4 = st.columns(4)

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

    # ========================================================
    # TREND ANALYSIS
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
