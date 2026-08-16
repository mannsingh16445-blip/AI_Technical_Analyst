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

st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE"
).strip().upper()

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

ticker = symbol + ".NS"


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
            "Please try again in a few seconds."
        )

        st.stop()

    # Handle Yahoo Finance MultiIndex
    if hasattr(data.columns, "nlevels"):

        if data.columns.nlevels > 1:

            data.columns = data.columns.get_level_values(0)

    # Make sure required columns exist
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:

        st.error(
            f"Market data is missing: {', '.join(missing)}"
        )

        st.stop()

    # Remove rows without price data
    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if data.empty:

        st.error("No usable price data was returned.")

        st.stop()

    # Create chart
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

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # Latest values
    latest = data.iloc[-1]

    latest_close = float(latest["Close"])
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Close",
        f"₹{latest_close:.2f}"
    )

    col2.metric(
        "Day High",
        f"₹{latest_high:.2f}"
    )

    col3.metric(
        "Day Low",
        f"₹{latest_low:.2f}"
    )

    st.success(
        f"✅ {symbol} chart loaded successfully"
    )

except Exception as e:

    st.error(
        f"Unexpected error: {e}"
    )
