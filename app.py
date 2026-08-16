import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Technical Analyst")
st.caption("Your personal AI-powered technical analysis assistant")

# Sidebar
st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input(
    "Enter NSE stock symbol",
    value="RELIANCE"
)

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

# Convert NSE symbol to Yahoo Finance format
ticker = symbol.upper().strip() + ".NS"

# Get market data
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
    else:

        # Remove multi-level columns if present
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        # Candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name=symbol.upper()
                )
            ]
        )

        fig.update_layout(
            title=f"{symbol.upper()} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Basic price information
        latest_close = float(data["Close"].iloc[-1])
        latest_high = float(data["High"].iloc[-1])
        latest_low = float(data["Low"].iloc[-1])

        col1, col2, col3 = st.columns(3)

        col1.metric("Latest Close", f"₹{latest_close:.2f}")
        col2.metric("Day High", f"₹{latest_high:.2f}")
        col3.metric("Day Low", f"₹{latest_low:.2f}")

except Exception as e:
    st.error(f"Error: {e}")


# AI Chat
st.divider()

st.subheader("🤖 Ask the AI Technical Analyst")

question = st.chat_input(
    "Ask something about technical analysis..."
)

if question:

    st.chat_message("user").write(question)

    with st.chat_message("assistant"):

        try:
            response = client.responses.create(
                model="gpt-5.6-mini",
                instructions="""
                You are an educational technical-analysis assistant.
                Explain technical-analysis concepts clearly.
                Do not invent market prices or indicator values.
                The application will later provide calculated indicators
                and market data separately.
                """,
                input=question
            )

            st.write(response.output_text)

        except Exception as e:
            st.error(f"AI error: {e}")