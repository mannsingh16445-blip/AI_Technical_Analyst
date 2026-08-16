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

show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
show_sma50 = st.sidebar.checkbox("SMA 50", value=True)
show_sma200 = st.sidebar.checkbox("SMA 200", value=True)
show_rsi = st.sidebar.checkbox("RSI 14", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_volume = st.sidebar.checkbox("Volume", value=True)
show_donchian = st.sidebar.checkbox(
    "Donchian Channel (3)",
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
        "Close",
        "Volume"
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
            "Close",
            "Volume"
        ]
    )

    # ========================================================
    # SMA
    # ========================================================

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

    # ========================================================
    # RSI 14
    # ========================================================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI14"] = 100 - (
        100 / (1 + rs)
    )

    # ========================================================
    # MACD
    # ========================================================

    ema12 = (
        data["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["MACD_Histogram"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    # ========================================================
    # VOLUME
    # ========================================================

    data["Volume_Avg20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Volume_Avg20"]
    )

    # ========================================================
    # DONCHIAN CHANNEL - 3 PERIOD
    #
    # IMPORTANT:
    # shift(1) means today's channel uses only the
    # previous 3 completed candles.
    # ========================================================

    data["Donchian_Upper_3"] = (
        data["High"]
        .shift(1)
        .rolling(3)
        .max()
    )

    data["Donchian_Lower_3"] = (
        data["Low"]
        .shift(1)
        .rolling(3)
        .min()
    )

    data["Donchian_Middle_3"] = (
        data["Donchian_Upper_3"] +
        data["Donchian_Lower_3"]
    ) / 2

    # ========================================================
    # DONCHIAN POSITION
    # ========================================================

    data["Donchian_Range"] = (
        data["Donchian_Upper_3"] -
        data["Donchian_Lower_3"]
    )

    data["Distance_From_Donchian_Upper"] = (
        data["Donchian_Upper_3"] -
        data["Close"]
    )

    data["Distance_From_Donchian_Lower"] = (
        data["Close"] -
        data["Donchian_Lower_3"]
    )

    data["Donchian_Position"] = (
        (
            data["Close"] -
            data["Donchian_Lower_3"]
        )
        /
        data["Donchian_Range"]
    ) * 100

    # ========================================================
    # NUMBER OF CHART PANELS
    # ========================================================

    panel_count = 1

    if show_volume:
        panel_count += 1

    if show_rsi:
        panel_count += 1

    if show_macd:
        panel_count += 1

    row_heights = [0.50]

    if panel_count > 1:

        remaining_height = 0.50

        for _ in range(panel_count - 1):

            row_heights.append(
                remaining_height /
                (panel_count - 1)
            )

    fig = make_subplots(
        rows=panel_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
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
    # DONCHIAN CHANNEL
    # ========================================================

    if show_donchian:

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Donchian_Upper_3"],
                mode="lines",
                name="Donchian Upper (3)"
            ),

            row=1,
            col=1
        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Donchian_Lower_3"],
                mode="lines",
                name="Donchian Lower (3)"
            ),

            row=1,
            col=1
        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["Donchian_Middle_3"],
                mode="lines",
                name="Donchian Middle"
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
                mode="lines",
                name="Average Volume 20"
            ),

            row=current_row,
            col=1
        )

        fig.update_yaxes(
            title_text="Volume",
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
                y=data["RSI14"],
                mode="lines",
                name="RSI 14"
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
            y=50,
            row=current_row,
            col=1
        )

        fig.add_hline(
            y=30,
            row=current_row,
            col=1
        )

        fig.update_yaxes(
            range=[0, 100],
            title_text="RSI",
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
                mode="lines",
                name="MACD"
            ),

            row=current_row,
            col=1
        )

        fig.add_trace(

            go.Scatter(
                x=data.index,
                y=data["MACD_Signal"],
                mode="lines",
                name="MACD Signal"
            ),

            row=current_row,
            col=1
        )

        fig.add_trace(

            go.Bar(
                x=data.index,
                y=data["MACD_Histogram"],
                name="MACD Histogram"
            ),

            row=current_row,
            col=1
        )

        fig.add_hline(
            y=0,
            row=current_row,
            col=1
        )

        fig.update_yaxes(
            title_text="MACD",
            row=current_row,
            col=1
        )

    # ========================================================
    # CHART LAYOUT
    # ========================================================

    fig.update_layout(

        title=f"{symbol} Technical Analysis",

        height=1200,

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
    # LATEST VALUES
    # ========================================================

    latest = data.iloc[-1]

    close = float(latest["Close"])

    sma20 = float(latest["SMA20"])

    sma50 = float(latest["SMA50"])

    sma200 = float(latest["SMA200"])

    rsi = float(latest["RSI14"])

    macd = float(latest["MACD"])

    signal = float(latest["MACD_Signal"])

    histogram = float(
        latest["MACD_Histogram"]
    )

    volume = float(latest["Volume"])

    avg_volume = float(
        latest["Volume_Avg20"]
    )

    volume_ratio = float(
        latest["Volume_Ratio"]
    )

    donchian_upper = float(
        latest["Donchian_Upper_3"]
    )

    donchian_lower = float(
        latest["Donchian_Lower_3"]
    )

    donchian_middle = float(
        latest["Donchian_Middle_3"]
    )

    donchian_position = float(
        latest["Donchian_Position"]
    )

    # ========================================================
    # CURRENT TECHNICAL VALUES
    # ========================================================

    st.subheader(
        "📊 Current Technical Values"
    )

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
    # DONCHIAN VALUES
    # ========================================================

    st.subheader(
        "🎯 Donchian Channel — 3 Period"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Upper Channel",
        f"₹{donchian_upper:.2f}"
    )

    col2.metric(
        "Current Price",
        f"₹{close:.2f}"
    )

    col3.metric(
        "Middle Channel",
        f"₹{donchian_middle:.2f}"
    )

    col4.metric(
        "Lower Channel",
        f"₹{donchian_lower:.2f}"
    )

    st.write(
        f"**Position within channel:** "
        f"{donchian_position:.1f}%"
    )

    if donchian_position >= 80:

        st.info(
            "🔵 Price is close to the upper Donchian boundary."
        )

    elif donchian_position <= 20:

        st.warning(
            "🟡 Price is close to the lower Donchian boundary."
        )

    else:

        st.success(
            "🟢 Price is trading within the Donchian channel."
        )

    # ========================================================
    # VOLUME STATUS
    # ========================================================

    st.subheader(
        "📊 Volume Analysis"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Volume",
        f"{volume:,.0f}"
    )

    col2.metric(
        "20-Day Avg Volume",
        f"{avg_volume:,.0f}"
    )

    col3.metric(
        "Volume Ratio",
        f"{volume_ratio:.2f}×"
    )

    if volume_ratio >= 1.5:

        st.success(
            f"🟢 Strong volume: "
            f"{volume_ratio:.2f}× the 20-day average"
        )

    elif volume_ratio >= 1.0:

        st.info(
            f"🟡 Normal volume: "
            f"{volume_ratio:.2f}× the 20-day average"
        )

    else:

        st.warning(
            f"🔴 Weak volume: "
            f"{volume_ratio:.2f}× the 20-day average"
        )

    # ========================================================
    # MACD STATUS
    # ========================================================

    st.subheader("📉 MACD Status")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MACD",
        f"{macd:.2f}"
    )

    col2.metric(
        "Signal",
        f"{signal:.2f}"
    )

    col3.metric(
        "Histogram",
        f"{histogram:.2f}"
    )

    if macd > signal:

        st.success(
            "🟢 MACD is ABOVE Signal — bullish momentum"
        )

    else:

        st.warning(
            "🔴 MACD is BELOW Signal — bearish momentum"
        )

    # ========================================================
    # RSI STATUS
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
        f"Last available date: "
        f"{data.index[-1].date()}"
    )

    st.success(
        f"✅ {symbol} analysis loaded successfully"
    )


except Exception as e:

    st.error(
        f"Unexpected error: {e}"
    )
