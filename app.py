import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

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
)

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

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
        st.error("No data found. Check the stock symbol.")
        st.stop()

    # Handle Yahoo Finance columns
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol.upper()
        )
    )

    fig.update_layout(
        title=f"{symbol.upper()} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    latest_close = float(data["Close"].iloc[-1])
    latest_high = float(data["High"].iloc[-1])
    latest_low = float(data["Low"].iloc[-1])

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

    st.success("✅ Market data and chart are working!")

except Exception as e:

    st.error(f"Error: {e}")
