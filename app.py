import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
import os

from io import StringIO, BytesIO
import zipfile
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# MOBILE RESPONSIVE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .main .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Keep Plotly charts responsive */
    .js-plotly-plot,
    .plotly,
    .plot-container {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Prevent long stock names / messages from overflowing */
    .stMarkdown,
    .stText,
    p,
    h1, h2, h3, h4, h5, h6 {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* Dataframes should remain usable on small screens */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 42px;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        width: 100%;
    }

    /* ---------- MOBILE ---------- */

    @media only screen and (max-width: 768px) {

        /* Main page */
        .main .block-container {
            padding-top: 0.75rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 1rem !important;
        }

        /* Titles */
        h1 {
            font-size: 1.65rem !important;
            line-height: 1.2 !important;
        }

        h2 {
            font-size: 1.35rem !important;
            line-height: 1.25 !important;
        }

        h3 {
            font-size: 1.15rem !important;
            line-height: 1.3 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 85vw !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }

        /* Make Streamlit column layouts stack naturally */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: min(100%, 300px) !important;
            flex: 1 1 100% !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            padding: 0.45rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }

        /* Inputs */
        input,
        textarea,
        [data-baseweb="select"] {
            font-size: 16px !important;
        }

        /* Buttons become easy to tap */
        .stButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            min-height: 46px !important;
            font-size: 0.95rem !important;
        }

        /* Select boxes / number inputs */
        [data-testid="stSelectbox"],
        [data-testid="stNumberInput"],
        [data-testid="stTextInput"],
        [data-testid="stSlider"],
        [data-testid="stCheckbox"] {
            width: 100% !important;
        }

        /* Plotly chart */
        .js-plotly-plot,
        .plotly,
        .plot-container {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
        }

        /* Expander */
        [data-testid="stExpander"] {
            width: 100% !important;
        }

        /* Alerts */
        [data-testid="stAlert"] {
            font-size: 0.9rem !important;
        }

        /* Progress bar */
        [data-testid="stProgress"] {
            width: 100% !important;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }

        /* Reduce excessive vertical spacing */
        .element-container {
            margin-bottom: 0.25rem !important;
        }

    }

    /* ---------- VERY SMALL PHONES ---------- */

    @media only screen and (max-width: 480px) {

        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        h1 {
            font-size: 1.45rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        h3 {
            font-size: 1.05rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }

        [data-testid="stSidebar"] {
            min-width: 260px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


load_dotenv()


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is not None and OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    client = None


# ============================================================
# TITLE
# ============================================================

st.title("📈 AI Technical Analyst")

st.caption(
    "Technical charts • Smart NSE scanner • AI analysis"
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
# LOAD NSE EQUITY UNIVERSE
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
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

        df = pd.read_csv(
            StringIO(response.text)
        )

        df.columns = [
            str(c).strip().upper()
            for c in df.columns
        ]

        if "SYMBOL" not in df.columns:

            return []

        symbols = (
            df["SYMBOL"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        symbols = symbols[
            ~symbols.isin(
                [
                    "",
                    "NAN",
                    "NONE"
                ]
            )
        ]

        symbols = sorted(
            symbols.drop_duplicates().tolist()
        )

        return symbols

    except Exception:

        return []


# ============================================================
# LOAD NIFTY 500
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
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
                ~symbols.isin(
                    [
                        "",
                        "NAN",
                        "NONE"
                    ]
                )
            ]

            if len(symbols) >= 400:

                return sorted(
                    symbols.drop_duplicates().tolist()
                )

        except Exception:

            continue

    return []


# ============================================================
# LOAD NSE F&O STOCK UNIVERSE
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_fno_stocks():

    """
    Load NSE individual-stock F&O underlyings.

    Source priority:
    1. Current NFO symbol master
    2. NSE underlying-information page
    3. NSE API
    4. Built-in fallback list

    Index derivatives are excluded.
    """

    # --------------------------------------------------------
    # 1. CURRENT NFO SYMBOL MASTER
    # --------------------------------------------------------

    for url in [
        "https://api.shoonya.com/NFO_symbols.txt.zip",
        "https://shoonya.finvasia.com/NFO_symbols.txt.zip"
    ]:

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent":
                        "Mozilla/5.0"
                }
            )

            if response.status_code != 200:
                continue

            with zipfile.ZipFile(
                BytesIO(response.content)
            ) as z:

                names = z.namelist()

                if not names:
                    continue

                with z.open(names[0]) as f:
                    raw = f.read()

            df = pd.read_csv(
                BytesIO(raw),
                sep="|",
                dtype=str
            )

            df.columns = [
                str(c).strip().upper()
                for c in df.columns
            ]

            # The symbol master normally has a clean
            # underlying/SymbolName field.
            underlying_col = None

            for candidate in [
                "SYMBOLNAME",
                "UNDERLYING",
                "BASE_SYMBOL"
            ]:

                if candidate in df.columns:
                    underlying_col = candidate
                    break

            if underlying_col:

                if "INSTRUMENT" in df.columns:

                    instrument = (
                        df["INSTRUMENT"]
                        .astype(str)
                        .str.upper()
                    )

                    df = df[
                        instrument.isin([
                            "FUTSTK",
                            "OPTSTK"
                        ])
                    ]

                symbols = (
                    df[underlying_col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                symbols = symbols[
                    ~symbols.isin([
                        "",
                        "NAN",
                        "NONE",
                        "NULL",
                        "NIFTY",
                        "BANKNIFTY",
                        "FINNIFTY",
                        "MIDCPNIFTY",
                        "NIFTYNXT50"
                    ])
                ]

                symbols = sorted(
                    symbols.drop_duplicates().tolist()
                )

                if len(symbols) >= 100:
                    return symbols

        except Exception:
            continue

    # --------------------------------------------------------
    # 2. NSE UNDERLYING INFORMATION PAGE
    # --------------------------------------------------------

    nse_headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept":
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":
            "en-US,en;q=0.9",
        "Referer":
            "https://www.nseindia.com/"
    }

    for page_url in [
        "https://www.nseindia.com/products-services/"
        "equity-derivatives-list-underlyings-information",

        "https://www.nseindia.com/static/products-services/"
        "equity-derivatives-list-underlyings-information"
    ]:

        try:

            session = requests.Session()
            session.headers.update(nse_headers)

            session.get(
                "https://www.nseindia.com/",
                timeout=20
            )

            response = session.get(
                page_url,
                timeout=30
            )

            if response.status_code != 200:
                continue

            tables = pd.read_html(
                StringIO(response.text)
            )

            for table in tables:

                table.columns = [
                    str(c).strip().upper()
                    for c in table.columns
                ]

                if "SYMBOL" not in table.columns:
                    continue

                symbols = (
                    table["SYMBOL"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                symbols = symbols[
                    ~symbols.isin([
                        "",
                        "NAN",
                        "NONE",
                        "NULL",
                        "SYMBOL",
                        "NIFTY",
                        "BANKNIFTY",
                        "FINNIFTY",
                        "MIDCPNIFTY",
                        "NIFTYNXT50"
                    ])
                ]

                symbols = sorted(
                    symbols.drop_duplicates().tolist()
                )

                if len(symbols) >= 100:
                    return symbols

        except Exception:
            continue

    # --------------------------------------------------------
    # 3. NSE API
    # --------------------------------------------------------

    try:

        session = requests.Session()
        session.headers.update(nse_headers)

        session.get(
            "https://www.nseindia.com/",
            timeout=20
        )

        response = session.get(
            "https://www.nseindia.com/"
            "api/underlying-information",
            timeout=30
        )

        if response.status_code == 200:

            payload = response.json()

            data_block = payload.get(
                "data",
                {}
            )

            rows = data_block.get(
                "UnderlyingList",
                []
            )

            symbols = []

            for item in rows:

                if isinstance(item, dict):

                    symbol = (
                        item.get("symbol")
                        or item.get("SYMBOL")
                        or item.get("Symbol")
                    )

                else:

                    symbol = item

                if symbol:

                    symbol = (
                        str(symbol)
                        .strip()
                        .upper()
                    )

                    if symbol not in {
                        "",
                        "NAN",
                        "NONE",
                        "NULL",
                        "NIFTY",
                        "BANKNIFTY",
                        "FINNIFTY",
                        "MIDCPNIFTY",
                        "NIFTYNXT50"
                    }:

                        symbols.append(symbol)

            symbols = sorted(
                set(symbols)
            )

            if len(symbols) >= 100:
                return symbols

    except Exception:
        pass

    # --------------------------------------------------------
    # 4. BUILT-IN FALLBACK
    # --------------------------------------------------------
    #
    # Based on NSE's published individual-security list,
    # with six securities introduced for F&O from 01-Apr-2026.
    # This prevents the scanner from failing when NSE blocks
    # automated requests from Streamlit Cloud.
    # --------------------------------------------------------

    fallback = """
AARTIIND ABB ABBOTINDIA ACC ADANIENT ADANIPORTS ABCAPITAL ABFRL
ALKEM AMBUJACEM APOLLOHOSP APOLLOTYRE ASHOKLEY ASIANPAINT ASTRAL ATUL
AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BALKRISIND
BALRAMCHIN BANDHANBNK BANKBARODA BATAINDIA BERGEPAINT BEL BHARATFORG
BHEL BPCL BHARTIARTL BIOCON BSOFT BOSCHLTD BRITANNIA CANFINHOME CANBK
CHAMBLFERT CHOLAFIN CIPLA CUB COALINDIA COFORGE COLPAL CONCOR COROMANDEL
CROMPTON CUMMINSIND DABUR DALBHARAT DEEPAKNTR DELTACORP DIVISLAB DIXON
DLF LALPATHLAB DRREDDY EICHERMOT ESCORTS EXIDEIND GAIL GLENMARK GMRINFRA
GODREJCP GODREJPROP GRANULES GRASIM GUJGASLTD GNFC HAVELLS HCLTECH HDFCAMC
HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HAL HINDCOPPER HINDPETRO HINDUNILVR
HDFC ICICIBANK ICICIGI ICICIPRULI IDFCFIRSTB IDFC IBULHSGFIN INDIAMART IEX
IOC IRCTC IGL INDUSTOWER INDUSINDBK NAUKRI INFY INTELLECT INDIGO IPCALAB ITC
JINDALSTEL JKCEMENT JSWSTEEL JUBLFOOD KOTAKBANK L&TFH LTTS LTIM LT LAURUSLABS
LICHSGFIN LUPIN MGL M&MFIN M&M MANAPPURAM MARICO MARUTI MFSL METROPOLIS
MOTHERSON MPHASIS MRF MCX MUTHOOTFIN NATIONALUM NAVINFLUOR NESTLEIND NMDC NTPC
OBEROIRLTY ONGC OFSS PAGEIND PERSISTENT PETRONET PIIND PIDILITIND PEL POLYCAB
PFC POWERGRID PNB PVRINOX RAIN RBLBANK RECLTD RELIANCE SBICARD SBILIFE
SHREECEM SHRIRAMFIN SIEMENS SRF SBIN SAIL SUNPHARMA SUNTV SYNGENE TATACHEM
TATACOMM TCS TATACONSUM TATAMOTORS TATAPOWER TATASTEEL TECHM FEDERALBNK
INDIACEM INDHOTEL RAMCOCEM TITAN TORNTPHARM TRENT TVSMOTOR ULTRACEMCO UBL
MCDOWELL-N UPL VEDL IDEA VOLTAS WHIRLPOOL WIPRO ZEEL ZYDUSLIFE
ADANIPOWER COCHINSHIP HYUNDAI MOTILALOFS NAM-INDIA VMM
"""

    return sorted(
        set(
            x.strip().upper()
            for x in fallback.split()
            if x.strip()
        )
    )


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_batches(
    tickers,
    period="1y",
    batch_size=50
):

    all_data = {}

    ticker_list = list(dict.fromkeys(tickers))

    for start in range(
        0,
        len(ticker_list),
        batch_size
    ):

        batch = ticker_list[
            start:start + batch_size
        ]

        yahoo_tickers = [
            symbol + ".NS"
            for symbol in batch
        ]

        try:

            data = yf.download(
                tickers=yahoo_tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker"
            )

            if data is None or data.empty:
                continue

            # =================================================
            # SINGLE STOCK
            # =================================================

            if len(batch) == 1:

                symbol = batch[0]

                stock = data.copy()

                # Handle MultiIndex returned by yfinance

                if isinstance(
                    stock.columns,
                    pd.MultiIndex
                ):

                    # Case:
                    # ('Open', 'RELIANCE.NS')
                    #
                    # or:
                    # ('RELIANCE.NS', 'Open')

                    level0 = (
                        stock.columns
                        .get_level_values(0)
                        .tolist()
                    )

                    level1 = (
                        stock.columns
                        .get_level_values(1)
                        .tolist()
                    )

                    if "Open" in level0:

                        stock.columns = [
                            col[0]
                            for col in stock.columns
                        ]

                    elif "Open" in level1:

                        stock.columns = [
                            col[1]
                            for col in stock.columns
                        ]

                else:

                    stock.columns = [
                        str(col)
                        .strip()
                        .title()
                        for col in stock.columns
                    ]

                # ---------------------------------------------
                # Required columns
                # ---------------------------------------------

                required = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                if all(
                    column in stock.columns
                    for column in required
                ):

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if not stock.empty:

                        all_data[symbol] = stock

                continue

            # =================================================
            # MULTIPLE STOCKS
            # =================================================

            if not isinstance(
                data.columns,
                pd.MultiIndex
            ):

                continue

            level0 = (
                data.columns
                .get_level_values(0)
                .unique()
                .tolist()
            )

            level1 = (
                data.columns
                .get_level_values(1)
                .unique()
                .tolist()
            )

            for symbol in batch:

                yahoo_symbol = symbol + ".NS"

                try:

                    # -----------------------------------------
                    # Format 1:
                    #
                    # RELIANCE.NS
                    #   Open
                    #   High
                    # -----------------------------------------

                    if yahoo_symbol in level0:

                        stock = data[
                            yahoo_symbol
                        ].copy()

                    # -----------------------------------------
                    # Format 2:
                    #
                    # Open
                    # High
                    # ...
                    # RELIANCE.NS
                    # -----------------------------------------

                    elif yahoo_symbol in level1:

                        stock = data[
                            :,
                            yahoo_symbol
                        ].copy()

                    else:

                        continue

                    # -----------------------------------------
                    # Flatten columns
                    # -----------------------------------------

                    if isinstance(
                        stock.columns,
                        pd.MultiIndex
                    ):

                        stock.columns = [
                            col[0]
                            if isinstance(
                                col,
                                tuple
                            )
                            else col
                            for col in stock.columns
                        ]

                    stock.columns = [
                        str(col)
                        .strip()
                        .title()
                        for col in stock.columns
                    ]

                    required = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    if not all(
                        column in stock.columns
                        for column in required
                    ):

                        continue

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if not stock.empty:

                        all_data[symbol] = stock

                except Exception:

                    continue

        except Exception:

            continue

        time.sleep(0.2)

    return all_data

# ============================================================
# TECHNICAL INDICATORS
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

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = (
        avg_gain /
        avg_loss
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
        data["MACD"]
        -
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
        data["Volume"]
        /
        data["AVG_VOLUME20"]
    )

    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    data["TURNOVER"] = (
        data["Close"]
        *
        data["Volume"]
    )

    data["AVG_TURNOVER20"] = (
        data["TURNOVER"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # DONCHIAN CHANNEL
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
# STAGE 1 FILTER
# ============================================================

def stage_one_filter(
    data,
    min_price,
    min_volume,
    min_turnover
):

    if len(data) < 210:

        return False

    latest = data.iloc[-1]

    values = [
        latest["Close"],
        latest["SMA200"],
        latest["AVG_VOLUME20"],
        latest["AVG_TURNOVER20"]
    ]

    if any(
        pd.isna(value)
        for value in values
    ):

        return False

    # Price

    if (
        latest["Close"]
        < min_price
    ):

        return False

    # Volume

    if (
        latest["AVG_VOLUME20"]
        < min_volume
    ):

        return False

    # Turnover

    turnover_crore = (
        latest["AVG_TURNOVER20"]
        / 10000000
    )

    if (
        turnover_crore
        < min_turnover
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
# STAGE 2 TECHNICAL ANALYSIS
# ============================================================

def stage_two_analysis(data):

    if len(data) < 210:

        return None

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    n_minus_2 = data.iloc[-3]

    # --------------------------------------------------------
    # YOUR FIVE CORE CONDITIONS
    # --------------------------------------------------------

    condition_1 = (
        latest["Close"]
        >
        previous["High"]
    )

    condition_2 = (
        previous["High"]
        <
        n_minus_2["High"]
    )

    condition_3 = (
        latest["Close"]
        <
        latest["DONCHIAN_UPPER"]
    )

    condition_4 = (
        latest["Low"]
        >
        latest["DONCHIAN_LOWER"]
    )

    condition_5 = (
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

    if volume_confirm:
        score += 1

    if rsi_confirm:
        score += 1

    if macd_confirm:
        score += 1

    return {

        "C1": condition_1,

        "C2": condition_2,

        "C3": condition_3,

        "C4": condition_4,

        "C5": condition_5,

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

        "MACD Signal":
            latest["MACD_SIGNAL"],

        "Volume Ratio":
            latest["VOLUME_RATIO"],

        "Turnover":
            latest["AVG_TURNOVER20"]
            / 10000000,

        "Donchian Upper":
            latest["DONCHIAN_UPPER"],

        "Donchian Lower":
            latest["DONCHIAN_LOWER"]
    }



# ============================================================
# RSI / WMA MULTI-TIMEFRAME SCANNER
# ============================================================

def calculate_rsi_wilder(series, period=14):
    """Calculate RSI using Wilder-style exponential smoothing."""
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_wma(series, period=21):
    """Weighted moving average of a price series."""
    weights = np.arange(1, period + 1, dtype=float)

    return series.rolling(
        period
    ).apply(
        lambda values: np.dot(values, weights) / weights.sum(),
        raw=True
    )


def prepare_rsi_wma_data(data):
    """
    Prepare the exact indicators represented by the
    supplied scanner conditions:

    RSI(9)
    WMA(Close, 21)
    """
    data = data.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        return pd.DataFrame()

    data = data.dropna(
        subset=required
    )

    if data.empty:
        return data

    data["RSI9"] = calculate_rsi_wilder(
        data["Close"],
        9
    )

    data["WMA21_CLOSE"] = calculate_wma(
        data["Close"],
        21
    )

    return data.dropna(
        subset=[
            "RSI9",
            "WMA21_CLOSE"
        ]
    )


def crossed_above_rsi_wma(data):
    """
    Exact interpretation of:
    RSI(9) Crossed above WMA(Close,21)

    The comparison is intentionally kept exactly as
    written in the supplied scanner screenshot.
    """
    if len(data) < 2:
        return False

    previous = data.iloc[-2]
    latest = data.iloc[-1]

    values = [
        previous["RSI9"],
        previous["WMA21_CLOSE"],
        latest["RSI9"],
        latest["WMA21_CLOSE"]
    ]

    if any(pd.isna(value) for value in values):
        return False

    return (
        previous["RSI9"] <= previous["WMA21_CLOSE"]
        and
        latest["RSI9"] > latest["WMA21_CLOSE"]
    )


def rsi_wma_scan_result(
    data,
    rsi9_threshold
):
    """
    Scan one timeframe using:

    1. RSI(9) crossed above WMA(Close,21)
    2. RSI(9) > threshold
    """
    prepared = prepare_rsi_wma_data(data)

    if prepared.empty or len(prepared) < 30:
        return None

    latest = prepared.iloc[-1]

    cross_condition = crossed_above_rsi_wma(
        prepared
    )

    rsi_condition = (
        latest["RSI9"] > rsi9_threshold
    )

    return {
        "Cross": cross_condition,
        "RSI9 > Threshold": rsi_condition,
        "RSI9": float(latest["RSI9"]),
        "WMA21 Close": float(latest["WMA21_CLOSE"]),
        "Close": float(latest["Close"]),
        "Volume": float(latest["Volume"]),
        "Date": prepared.index[-1]
    }


def resample_to_weekly(data):
    """
    Convert daily OHLCV data into completed weekly bars.
    Friday is used as the weekly anchor.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(
            data.index
        )

    weekly = data.resample(
        "W-FRI"
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }
    )

    return weekly.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def download_rsi_wma_batches(
    tickers,
    timeframe,
    batch_size=50
):
    """
    Download data for the RSI/WMA scanner.

    Daily:
        1 year / 1 day

    Weekly:
        3 years daily data, then resampled to
        completed weekly bars

    Hourly:
        60 days / 60 minute bars
    """
    all_data = {}

    ticker_list = list(
        dict.fromkeys(tickers)
    )

    if timeframe == "Daily":
        period = "1y"
        interval = "1d"

    elif timeframe == "Weekly":
        period = "3y"
        interval = "1d"

    else:
        period = "60d"
        interval = "60m"

    for start in range(
        0,
        len(ticker_list),
        batch_size
    ):

        batch = ticker_list[
            start:start + batch_size
        ]

        yahoo_tickers = [
            symbol + ".NS"
            for symbol in batch
        ]

        try:

            data = yf.download(
                tickers=yahoo_tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker"
            )

            if data is None or data.empty:
                continue

            # ------------------------------------------------
            # Single stock
            # ------------------------------------------------

            if len(batch) == 1:

                symbol = batch[0]

                stock = data.copy()

                if isinstance(
                    stock.columns,
                    pd.MultiIndex
                ):

                    level0 = (
                        stock.columns
                        .get_level_values(0)
                        .tolist()
                    )

                    level1 = (
                        stock.columns
                        .get_level_values(1)
                        .tolist()
                    )

                    if "Open" in level0:

                        stock.columns = [
                            col[0]
                            for col in stock.columns
                        ]

                    elif "Open" in level1:

                        stock.columns = [
                            col[1]
                            for col in stock.columns
                        ]

                stock.columns = [
                    str(col)
                    .strip()
                    .title()
                    for col in stock.columns
                ]

                required = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                if all(
                    column in stock.columns
                    for column in required
                ):

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if timeframe == "Weekly":

                        stock = resample_to_weekly(
                            stock
                        )

                    if not stock.empty:

                        all_data[symbol] = stock

                continue

            # ------------------------------------------------
            # Multiple stocks
            # ------------------------------------------------

            if not isinstance(
                data.columns,
                pd.MultiIndex
            ):
                continue

            level0 = (
                data.columns
                .get_level_values(0)
                .unique()
                .tolist()
            )

            level1 = (
                data.columns
                .get_level_values(1)
                .unique()
                .tolist()
            )

            for symbol in batch:

                yahoo_symbol = (
                    symbol + ".NS"
                )

                try:

                    if yahoo_symbol in level0:

                        stock = data[
                            yahoo_symbol
                        ].copy()

                    elif yahoo_symbol in level1:

                        stock = data[
                            :,
                            yahoo_symbol
                        ].copy()

                    else:

                        continue

                    if isinstance(
                        stock.columns,
                        pd.MultiIndex
                    ):

                        stock.columns = [
                            col[0]
                            if isinstance(
                                col,
                                tuple
                            )
                            else col
                            for col in stock.columns
                        ]

                    stock.columns = [
                        str(col)
                        .strip()
                        .title()
                        for col in stock.columns
                    ]

                    required = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    if not all(
                        column in stock.columns
                        for column in required
                    ):
                        continue

                    stock = stock[
                        required
                    ].copy()

                    stock = stock.dropna(
                        subset=required
                    )

                    if timeframe == "Weekly":

                        stock = resample_to_weekly(
                            stock
                        )

                    if not stock.empty:

                        all_data[symbol] = stock

                except Exception:

                    continue

        except Exception:

            continue

        time.sleep(0.15)

    return all_data


def run_rsi_wma_scanner(
    stocks,
    mode,
    batch_size=50,
    hourly_confirmation=False
):
    """
    Run the scanner.

    Daily:
        Daily RSI9 cross above Daily WMA(Close,21)
        Daily RSI9 > 55

    Weekly:
        Weekly RSI9 cross above Weekly WMA(Close,21)
        Weekly RSI9 > 50

    Hourly:
        Hourly RSI9 cross above Hourly WMA(Close,21)
        Hourly RSI9 > 55

    Multi-Timeframe:
        Weekly conditions AND Daily conditions.
        Optional hourly confirmation can be enabled.
    """

    if mode == "Daily":

        market = download_rsi_wma_batches(
            stocks,
            "Daily",
            batch_size
        )

        results = []

        for symbol, data in market.items():

            analysis = rsi_wma_scan_result(
                data,
                55
            )

            if analysis is None:
                continue

            if (
                analysis["Cross"]
                and
                analysis["RSI9 > Threshold"]
            ):

                results.append(
                    {
                        "Stock": symbol,
                        "Close": round(
                            analysis["Close"],
                            2
                        ),
                        "RSI 9": round(
                            analysis["RSI9"],
                            2
                        ),
                        "WMA 21 Close": round(
                            analysis["WMA21 Close"],
                            2
                        ),
                        "RSI 9 Threshold": 55,
                        "RSI9 Cross": "✓",
                        "Signal": "Daily Bullish RSI/WMA"
                    }
                )

        return results, {
            "Data Retrieved": len(market)
        }


    if mode == "Weekly":

        market = download_rsi_wma_batches(
            stocks,
            "Weekly",
            batch_size
        )

        results = []

        for symbol, data in market.items():

            analysis = rsi_wma_scan_result(
                data,
                50
            )

            if analysis is None:
                continue

            if (
                analysis["Cross"]
                and
                analysis["RSI9 > Threshold"]
            ):

                results.append(
                    {
                        "Stock": symbol,
                        "Close": round(
                            analysis["Close"],
                            2
                        ),
                        "RSI 9": round(
                            analysis["RSI9"],
                            2
                        ),
                        "WMA 21 Close": round(
                            analysis["WMA21 Close"],
                            2
                        ),
                        "RSI 9 Threshold": 50,
                        "RSI9 Cross": "✓",
                        "Signal": "Weekly Bullish RSI/WMA"
                    }
                )

        return results, {
            "Data Retrieved": len(market)
        }


    if mode == "Hourly":

        market = download_rsi_wma_batches(
            stocks,
            "Hourly",
            batch_size
        )

        results = []

        for symbol, data in market.items():

            analysis = rsi_wma_scan_result(
                data,
                55
            )

            if analysis is None:
                continue

            if (
                analysis["Cross"]
                and
                analysis["RSI9 > Threshold"]
            ):

                results.append(
                    {
                        "Stock": symbol,
                        "Close": round(
                            analysis["Close"],
                            2
                        ),
                        "RSI 9": round(
                            analysis["RSI9"],
                            2
                        ),
                        "WMA 21 Close": round(
                            analysis["WMA21 Close"],
                            2
                        ),
                        "RSI 9 Threshold": 55,
                        "RSI9 Cross": "✓",
                        "Signal": "Hourly Bullish RSI/WMA"
                    }
                )

        return results, {
            "Data Retrieved": len(market)
        }


    # ========================================================
    # MULTI-TIMEFRAME
    # ========================================================

    daily_market = download_rsi_wma_batches(
        stocks,
        "Daily",
        batch_size
    )

    weekly_market = download_rsi_wma_batches(
        stocks,
        "Weekly",
        batch_size
    )

    hourly_market = {}

    if hourly_confirmation:

        hourly_market = download_rsi_wma_batches(
            stocks,
            "Hourly",
            batch_size
        )

    results = []

    for symbol in stocks:

        daily_analysis = None
        weekly_analysis = None
        hourly_analysis = None

        if symbol in daily_market:

            daily_analysis = (
                rsi_wma_scan_result(
                    daily_market[symbol],
                    55
                )
            )

        if symbol in weekly_market:

            weekly_analysis = (
                rsi_wma_scan_result(
                    weekly_market[symbol],
                    50
                )
            )

        if hourly_confirmation and symbol in hourly_market:

            hourly_analysis = (
                rsi_wma_scan_result(
                    hourly_market[symbol],
                    55
                )
            )

        if (
            daily_analysis is None
            or
            weekly_analysis is None
        ):
            continue

        daily_pass = (
            daily_analysis["Cross"]
            and
            daily_analysis[
                "RSI14 > Threshold"
            ]
        )

        weekly_pass = (
            weekly_analysis["Cross"]
            and
            weekly_analysis[
                "RSI14 > Threshold"
            ]
        )

        hourly_pass = True

        if hourly_confirmation:

            hourly_pass = (
                hourly_analysis is not None
                and
                hourly_analysis["Cross"]
                and
                hourly_analysis[
                    "RSI14 > Threshold"
                ]
            )

        if (
            daily_pass
            and
            weekly_pass
            and
            hourly_pass
        ):

            results.append(
                {
                    "Stock": symbol,

                    "Daily Close": round(
                        daily_analysis["Close"],
                        2
                    ),

                    "Daily RSI9": round(
                        daily_analysis["RSI9"],
                        2
                    ),

                    "Daily WMA21": round(
                        daily_analysis["WMA21 Close"],
                        2
                    ),

                    "Daily RSI9 Threshold": 55,

                    "Weekly Close": round(
                        weekly_analysis["Close"],
                        2
                    ),

                    "Weekly RSI9": round(
                        weekly_analysis["RSI9"],
                        2
                    ),

                    "Weekly WMA21": round(
                        weekly_analysis["WMA21 Close"],
                        2
                    ),

                    "Weekly RSI9 Threshold": 50,

                    "Daily RSI9 Cross": "✓",

                    "Weekly RSI9 Cross": "✓",

                    "Hourly RSI9 Cross":
                        "✓"
                        if hourly_confirmation
                        else "—",

                    "Signal":
                        "Multi-Timeframe Bullish"
                }
            )

    return results, {
        "Daily Data Retrieved":
            len(daily_market),

        "Weekly Data Retrieved":
            len(weekly_market),

        "Hourly Data Retrieved":
            len(hourly_market)
            if hourly_confirmation
            else 0
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Control Panel"
)

module = st.sidebar.radio(
    "Select Module",
    [
        "📊 Technical Chart",
        "🚀 Smart Breakout Scanner",
        "📡 RSI/WMA Timeframe Scanner",
        "🏆 Top 10 Momentum Stocks",
        "🤖 AI Analyst"
    ]
)


# ============================================================
# TECHNICAL CHART
# ============================================================

if module == "📊 Technical Chart":

    st.header(
        "📊 Technical Chart"
    )

    st.write(
        "Enter any NSE-listed stock symbol."
    )

    symbol = st.sidebar.text_input(
        "NSE Symbol",
        value="RELIANCE"
    )

    symbol = (
        symbol
        .strip()
        .upper()
    )

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

    if st.sidebar.button(
        "📈 Load Chart",
        type="primary"
    ):

        with st.spinner(
            f"Loading {symbol}..."
        ):

            market = download_batches(
                [symbol],
                period,
                1
            )

        if symbol not in market:

            st.error(
                f"""
                Could not retrieve data for
                **{symbol}.NS**

                Check that the NSE symbol is correct.
                """
            )

            st.stop()

        data = market[
            symbol
        ].copy()

        data = calculate_indicators(
            data
        )

        if data.empty:

            st.error(
                "No usable market data found."
            )

            st.stop()

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

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

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA20"],
                mode="lines",
                name="SMA 20"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA50"],
                mode="lines",
                name="SMA 50"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA200"],
                mode="lines",
                name="SMA 200"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["DONCHIAN_UPPER"],
                mode="lines",
                name="Donchian Upper"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["DONCHIAN_LOWER"],
                mode="lines",
                name="Donchian Lower"
            )
        )

        fig.update_layout(
            title=(
                f"{symbol} — Technical Analysis"
            ),
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ----------------------------------------------------
        # CURRENT VALUES
        # ----------------------------------------------------

        latest = data.iloc[-1]

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
            f"{latest['RSI14']:.1f}"
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        analysis = stage_two_analysis(
            data
        )

        if analysis:

            st.subheader(
                "Technical Setup"
            )

            if analysis["Score"] >= 8:

                st.success(
                    f"🔥 Strong technical setup — "
                    f"Score {analysis['Score']}/10"
                )

            elif analysis["Score"] >= 5:

                st.warning(
                    f"⚠️ Moderate technical setup — "
                    f"Score {analysis['Score']}/10"
                )

            else:

                st.info(
                    f"Technical setup score: "
                    f"{analysis['Score']}/10"
                )


# ============================================================
# SMART BREAKOUT SCANNER
# ============================================================

elif module == "🚀 Smart Breakout Scanner":

    st.header(
        "🚀 Smart Two-Stage Breakout Scanner"
    )

    st.write(
        """
        **Stage 1:** Liquidity + price + 200 SMA filter

        **Stage 2:** Early breakout + Donchian +
        RSI + MACD + volume analysis
        """
    )

    # --------------------------------------------------------
    # LOAD UNIVERSES
    # --------------------------------------------------------

    with st.spinner(
        "Loading NSE stock universes..."
    ):

        nse_stocks = (
            load_nse_equity_universe()
        )

        nifty500 = (
            load_nifty500()
        )

        fno_stocks = (
            load_fno_stocks()
        )

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Stock Universe"
    )

    universe = st.sidebar.selectbox(
        "Select Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "NSE F&O Stocks",
            "Full NSE"
        ]
    )

    if universe == "Nifty 50":

        stocks = NIFTY50

    elif universe == "Nifty 500":

        stocks = nifty500

    elif universe == "NSE F&O Stocks":

        stocks = fno_stocks

    elif universe == "Full NSE":

        stocks = nse_stocks

    else:

        stocks = []

    if (
        universe == "Nifty 500"
        and not stocks
    ):

        st.error(
            """
            Nifty 500 list could not be loaded.

            Please select Nifty 50 or try again later.
            """
        )

        st.stop()

    if (
        universe == "NSE F&O Stocks"
        and not stocks
    ):

        st.error(
            """
            NSE F&O stock list could not be loaded.

            The NSE derivatives-underlying list may be
            temporarily unavailable. Please try again later.
            """
        )

        st.stop()


    if (
        universe == "Full NSE"
        and not stocks
    ):

        st.error(
            """
            Full NSE equity list could not be loaded.

            The NSE data server may temporarily reject
            automated requests.

            Please try again later.
            """
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}**"
    )

    if universe == "NSE F&O Stocks":

        st.caption(
            "NSE F&O Stocks = individual stocks with "
            "current equity-derivatives underlyings on NSE. "
            "Index derivatives are excluded."
        )

    # --------------------------------------------------------
    # STAGE 1
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

    batch_size = st.sidebar.slider(
        "Download Batch Size",
        min_value=25,
        max_value=100,
        value=50,
        step=25
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run_scanner = st.sidebar.button(
        "🚀 RUN SMART SCANNER",
        type="primary"
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    with st.expander(
        "📋 View Scanner Conditions"
    ):

        st.markdown(
            """
### Core Conditions

**C1:** Recent Close > Previous Day High

**C2:** Previous Day High < High of N-2

**C3:** Recent Close < 3-Day Donchian Upper

**C4:** Recent Low > 3-Day Donchian Lower

**C5:** Recent Close > 200 SMA

### Confirmations

**Volume:** Current Volume ≥ 1.5 × 20-day average

**RSI:** RSI > 50

**MACD:** MACD > Signal

### Score

C1 = 2 points

C2 = 1 point

C3 = 1 point

C4 = 1 point

C5 = 2 points

Volume = 1 point

RSI = 1 point

MACD = 1 point

**Maximum = 10 points**
            """
        )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if run_scanner:

        if not stocks:

            st.error(
                "No stocks available to scan."
            )

            st.stop()

        # ====================================================
        # STAGE 1 DOWNLOAD
        # ====================================================

        progress = st.progress(
            0,
            text="Downloading market data..."
        )

        market = download_batches(
            stocks,
            "1y",
            batch_size
        )

        progress.progress(
            35,
            text=(
                "Applying Stage 1 filters..."
            )
        )

        stage1_stocks = []

        processed = 0

        total = len(market)

        # ====================================================
        # STAGE 1
        # ====================================================

        for symbol, raw_data in market.items():

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

                    stage1_stocks.append(
                        symbol
                    )

            except Exception:

                pass

            processed += 1

            if total > 0:

                progress.progress(
                    35 +
                    int(
                        25 *
                        processed /
                        total
                    ),
                    text=(
                        f"Stage 1: "
                        f"{processed}/{total}"
                    )
                )

        # ====================================================
        # STAGE 1 SUMMARY
        # ====================================================

        st.subheader(
            "🔎 Stage 1 Results"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Universe",
            len(stocks)
        )

        c2.metric(
            "Data Retrieved",
            len(market)
        )

        c3.metric(
            "Stage 1 Survivors",
            len(stage1_stocks)
        )

        if not stage1_stocks:

            progress.empty()

            st.warning(
                """
                No stocks survived Stage 1.

                Try lowering:
                • Minimum price
                • Minimum average volume
                • Minimum turnover
                """
            )

            st.stop()

        # ====================================================
        # STAGE 2
        # ====================================================

        progress.progress(
            65,
            text="Running Stage 2 technical analysis..."
        )

        results = []

        total_stage2 = len(
            stage1_stocks
        )

        for i, symbol in enumerate(
            stage1_stocks
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
                                "Turnover"
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
                65 +
                int(
                    35 *
                    (i + 1)
                    / total_stage2
                ),
                text=(
                    f"Stage 2: "
                    f"{i + 1}/{total_stage2}"
                )
            )

        progress.empty()

        # ====================================================
        # RESULTS
        # ====================================================

        if not results:

            st.warning(
                """
                No stocks passed the Stage 2
                technical conditions.

                Try reducing the minimum score.
                """
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
                .reset_index(
                    drop=True
                )
            )

            st.success(
                f"🎯 {len(results_df)} "
                f"stocks passed Stage 2."
            )

            # ------------------------------------------------
            # TOP STOCKS
            # ------------------------------------------------

            st.subheader(
                "🏆 Top Early Breakout Candidates"
            )

            top = results_df[
                [
                    "Stock",
                    "Close",
                    "200 SMA",
                    "RSI",
                    "Volume Ratio",
                    "Technical Score"
                ]
            ].head(20)

            st.dataframe(
                top,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # SCORE DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📊 Score Distribution"
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
            # FULL RESULTS
            # ------------------------------------------------

            st.subheader(
                "📋 Detailed Results"
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
                label="⬇️ Download Scanner Results",
                data=csv,
                file_name=(
                    "NSE_Smart_Breakout_Scanner.csv"
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
                len(stocks)
            )

            c2.metric(
                "Stage 1",
                len(stage1_stocks)
            )

            c3.metric(
                "Stage 2",
                len(results_df)
            )

            c4.metric(
                "Score ≥ 8",
                len(
                    results_df[
                        results_df[
                            "Technical Score"
                        ] >= 8
                    ]
                )
            )


# ============================================================
# AI ANALYST
# ============================================================


# ============================================================
# RSI / WMA TIMEFRAME SCANNER
# ============================================================

elif module == "📡 RSI/WMA Timeframe Scanner":

    st.header(
        "📡 RSI / WMA Multi-Timeframe Scanner"
    )

    st.write(
        """
        Scanner based on the conditions shown in your
        Chartink screenshot. You can scan **Daily, Weekly,
        Hourly**, or use **Multi-Timeframe confirmation**.
        """
    )

    st.info(
        """
        **Important:** The screenshot specifies
        **RSI(9) crossed above WMA(Close, 21)**.
        The program therefore compares RSI(9) directly with
        WMA of Close(21), exactly as written in the supplied
        conditions.
        """
    )

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    with st.spinner(
        "Loading NSE stock universes..."
    ):

        nse_stocks = (
            load_nse_equity_universe()
        )

        nifty500 = (
            load_nifty500()
        )

        fno_stocks = (
            load_fno_stocks()
        )


    st.sidebar.subheader(
        "RSI/WMA Scanner"
    )

    scan_mode = st.sidebar.selectbox(
        "Scan Timeframe",
        [
            "Daily",
            "Weekly",
            "Hourly",
            "Multi-Timeframe"
        ]
    )

    universe = st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "NSE F&O Stocks",
            "Full NSE"
        ]
    )


    if universe == "Nifty 50":

        stocks = list(
            NIFTY50
        )

    elif universe == "Nifty 500":

        stocks = list(
            nifty500
        )[:500]

    elif universe == "NSE F&O Stocks":

        stocks = list(
            fno_stocks
        )

    elif universe == "Full NSE":

        stocks = list(
            nse_stocks
        )

    else:

        stocks = []


    if not stocks:

        st.error(
            f"""
            No stocks are available for the selected
            universe: **{universe}**.

            Please check the NSE/Nifty list connection.
            """
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}** | "
        f"Timeframe: **{scan_mode}**"
    )

    if universe == "NSE F&O Stocks":

        st.caption(
            "NSE F&O Stocks = individual stocks with "
            "current equity-derivative contracts. "
            "Index derivatives are excluded."
        )

    # --------------------------------------------------------
    # TIMEFRAME CONDITIONS
    # --------------------------------------------------------

    if scan_mode == "Daily":

        st.markdown(
            """
            ### Daily Conditions

            **D1:** Daily RSI(9) crossed above
            Daily WMA(Daily Close, 21)

            **D2:** Daily RSI(9) > **55**
            """
        )

        st.success(
            "Daily scan = D1 + D2"
        )

    elif scan_mode == "Weekly":

        st.markdown(
            """
            ### Weekly Conditions

            **W1:** Weekly RSI(9) crossed above
            Weekly WMA(Weekly Close, 21)

            **W2:** Weekly RSI(9) > **50**
            """
        )

        st.success(
            "Weekly scan = W1 + W2"
        )

    elif scan_mode == "Hourly":

        st.markdown(
            """
            ### Hourly Conditions

            **H1:** Hourly RSI(9) crossed above
            Hourly WMA(Hourly Close, 21)

            **H2:** Hourly RSI(9) > **55**
            """
        )

        st.success(
            "Hourly scan = H1 + H2"
        )

        st.warning(
            """
            Yahoo Finance intraday data is limited compared
            with daily data. The hourly scanner therefore
            uses the recent 60-day 60-minute dataset.
            """
        )

    else:

        st.markdown(
            """
            ### Multi-Timeframe Conditions

            **Weekly**

            **W1:** Weekly RSI(9) crossed above
            Weekly WMA(Weekly Close, 21)

            **W2:** Weekly RSI(9) > **50**

            **Daily**

            **D1:** Daily RSI(9) crossed above
            Daily WMA(Daily Close, 21)

            **D2:** Daily RSI(9) > **55**
            """
        )

        hourly_confirmation = st.sidebar.checkbox(
            "Require Hourly Confirmation",
            value=False
        )

        if hourly_confirmation:

            st.markdown(
                """
                **Hourly confirmation**

                **H1:** Hourly RSI(9) crossed above
                Hourly WMA(Hourly Close, 21)

                **H2:** Hourly RSI(9) > **55**
                """
            )

        else:

            st.caption(
                "Hourly confirmation is optional."
            )

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    batch_size = st.sidebar.slider(
        "Download Batch Size",
        min_value=25,
        max_value=100,
        value=50,
        step=25
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    run_rsi_scanner = st.sidebar.button(
        "📡 RUN RSI/WMA SCANNER",
        type="primary"
    )

    # --------------------------------------------------------
    # CONDITIONS EXPANDER
    # --------------------------------------------------------

    with st.expander(
        "📋 View Exact Scanner Conditions"
    ):

        st.markdown(
            """
            ### RSI(9) / WMA conditions

            **1. Weekly RSI(9) crossed above Weekly
            WMA(Weekly Close, 21)**

            **2. Daily RSI(9) crossed above Daily
            WMA(Daily Close, 21)**

            **3. Daily RSI(9) > 55**

            **4. Weekly RSI(9) > 50**

            ### Timeframe adaptation

            **Daily Scan**

            • RSI(9) crossed above WMA(Close,21)  
            • RSI(9) > 55

            **Weekly Scan**

            • RSI(9) crossed above WMA(Close,21)  
            • RSI(9) > 50

            **Hourly Scan**

            • RSI(9) crossed above WMA(Close,21)  
            • RSI(9) > 55

            **Multi-Timeframe Scan**

            • Weekly conditions  
            **AND**  
            • Daily conditions  
            • Optional Hourly confirmation
            """
        )

    # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if run_rsi_scanner:

        progress = st.progress(
            0,
            text="Starting RSI/WMA scanner..."
        )

        try:

            if scan_mode == "Multi-Timeframe":

                progress.progress(
                    10,
                    text="Preparing multi-timeframe scan..."
                )

            else:

                progress.progress(
                    10,
                    text=f"Downloading {scan_mode} data..."
                )

            results, stats = (
                run_rsi_wma_scanner(
                    stocks,
                    scan_mode,
                    batch_size,
                    (
                        hourly_confirmation
                        if scan_mode ==
                        "Multi-Timeframe"
                        else False
                    )
                )
            )

            progress.progress(
                100,
                text="Scan completed."
            )

            time.sleep(0.2)

            progress.empty()

        except Exception as e:

            progress.empty()

            st.error(
                f"Scanner error: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # DOWNLOAD SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📊 Scan Summary"
        )

        if scan_mode == "Multi-Timeframe":

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Universe",
                len(stocks)
            )

            c2.metric(
                "Daily Data",
                stats.get(
                    "Daily Data Retrieved",
                    0
                )
            )

            c3.metric(
                "Weekly Data",
                stats.get(
                    "Weekly Data Retrieved",
                    0
                )
            )

            c4.metric(
                "Stocks Passing",
                len(results)
            )

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Universe",
                len(stocks)
            )

            c2.metric(
                "Data Retrieved",
                stats.get(
                    "Data Retrieved",
                    0
                )
            )

            c3.metric(
                "Stocks Passing",
                len(results)
            )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if not results:

            st.warning(
                f"""
                No stocks passed all the selected
                **{scan_mode}** conditions.

                This is not necessarily an error.
                Cross-over conditions are intentionally selective.
                """
            )

        else:

            results_df = pd.DataFrame(
                results
            )

            st.success(
                f"🎯 {len(results_df)} stocks passed "
                f"the {scan_mode} RSI/WMA scan."
            )

            # ------------------------------------------------
            # TOP RESULTS
            # ------------------------------------------------

            st.subheader(
                "🏆 Stocks Passing the Scanner"
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
                label="⬇️ Download RSI/WMA Results",
                data=csv,
                file_name=(
                    f"RSI_WMA_{scan_mode.replace(' ', '_')}"
                    "_Scanner.csv"
                ),
                mime="text/csv"
            )

            # ------------------------------------------------
            # MOBILE STOCK CARDS
            # ------------------------------------------------

            st.subheader(
                "📱 Mobile-Friendly Results"
            )

            for _, row in results_df.head(20).iterrows():

                stock_name = row["Stock"]

                with st.expander(
                    f"📈 {stock_name}"
                ):

                    if scan_mode == "Multi-Timeframe":

                        c1, c2 = st.columns(2)

                        c1.metric(
                            "Daily RSI9 Threshold",
                            f"{row['Daily RSI9 Threshold']:.2f}"
                        )

                        c2.metric(
                            "Weekly RSI9 Threshold",
                            f"{row['Weekly RSI9 Threshold']:.2f}"
                        )

                        st.write(
                            f"Daily RSI9: "
                            f"**{row['Daily RSI9']:.2f}**"
                        )

                        st.write(
                            f"Daily WMA21: "
                            f"**{row['Daily WMA21']:.2f}**"
                        )

                        st.write(
                            f"Weekly RSI9: "
                            f"**{row['Weekly RSI9']:.2f}**"
                        )

                        st.write(
                            f"Weekly WMA21: "
                            f"**{row['Weekly WMA21']:.2f}**"
                        )

                        st.success(
                            "✓ Daily + Weekly conditions passed"
                        )

                    else:

                        c1, c2 = st.columns(2)

                        c1.metric(
                            "RSI 9",
                            f"{row['RSI 9']:.2f}"
                        )

                        c2.metric(
                            "RSI 9 Threshold",
                            f"{row['RSI 9 Threshold']}"
                        )

                        st.write(
                            f"Close: "
                            f"**₹{row['Close']:.2f}**"
                        )

                        st.write(
                            f"WMA 21 Close: "
                            f"**₹{row['WMA 21 Close']:.2f}**"
                        )

                        st.success(
                            "✓ RSI9 crossed above WMA21"
                        )

            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            st.subheader(
                "ℹ️ Scanner Interpretation"
            )

            if scan_mode == "Daily":

                st.write(
                    """
                    These stocks have a fresh Daily RSI(9)
                    cross above WMA(Close,21) and Daily
                    RSI(9) above 55 on the latest available
                    daily bar.
                    """
                )

            elif scan_mode == "Weekly":

                st.write(
                    """
                    These stocks have a fresh Weekly RSI(9)
                    cross above WMA(Weekly Close,21) and
                    Weekly RSI(9) above 50 on the latest
                    available completed weekly bar.
                    """
                )

            elif scan_mode == "Hourly":

                st.write(
                    """
                    These stocks have a fresh Hourly RSI(9)
                    cross above WMA(Hourly Close,21) and
                    Hourly RSI(9) above 55 on the latest
                    available hourly bar.
                    """
                )

            else:

                st.write(
                    """
                    These stocks satisfy the Weekly and Daily
                    conditions simultaneously. If Hourly
                    confirmation was enabled, the latest hourly
                    RSI/WMA conditions must also pass.
                    """
                )

elif module == "🏆 Top 10 Momentum Stocks":

    st.header(
        "🏆 Top 10 Momentum Stocks"
    )

    st.write(
        """
        A combined ranking model using your existing
        **Smart Breakout conditions + RSI(9)/WMA(21)
        Daily + Weekly + Hourly confirmation**.

        The scanner ranks stocks by technical strength;
        it does not guarantee future returns.
        """
    )

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------

    with st.spinner(
        "Loading stock universes..."
    ):

        nse_stocks = (
            load_nse_equity_universe()
        )

        nifty500 = (
            load_nifty500()
        )

        fno_stocks = (
            load_fno_stocks()
        )

    st.sidebar.subheader(
        "🏆 Top 10 Scanner"
    )

    universe = st.sidebar.selectbox(
        "Stock Universe",
        [
            "NSE F&O Stocks",
            "Nifty 50",
            "Nifty 500",
            "Full NSE"
        ],
        index=0
    )

    if universe == "NSE F&O Stocks":

        stocks = list(
            fno_stocks
        )

    elif universe == "Nifty 50":

        stocks = list(
            NIFTY50
        )

    elif universe == "Nifty 500":

        stocks = list(
            nifty500
        )[:500]

    else:

        stocks = list(
            nse_stocks
        )

    if not stocks:

        st.error(
            f"No stocks are available for **{universe}**."
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}**"
    )

    if universe == "NSE F&O Stocks":

        st.caption(
            "Recommended universe: individual NSE F&O "
            "stocks. Index derivatives are excluded."
        )

    batch_size = st.sidebar.slider(
        "Download Batch Size",
        min_value=25,
        max_value=100,
        value=50,
        step=25
    )

    only_full_mtf = st.sidebar.checkbox(
        "Only show stocks passing Daily + Weekly",
        value=False
    )

    include_hourly = st.sidebar.checkbox(
        "Include Hourly RSI(9) in ranking",
        value=True
    )

    run_top10 = st.sidebar.button(
        "🏆 FIND TOP 10 STOCKS",
        type="primary"
    )

    with st.expander(
        "📐 View Ranking Method"
    ):

        st.markdown(
            """
            ### Combined Score — 100 points

            **Smart Breakout — 40 points**

            • Your existing C1–C5 conditions  
            • Volume confirmation  
            • RSI confirmation  
            • MACD confirmation  

            **Daily RSI/WMA — 25 points**

            • RSI(9) crossed above WMA(Close,21): 15  
            • RSI(9) > 55: 10  

            **Weekly RSI/WMA — 20 points**

            • RSI(9) crossed above WMA(Close,21): 10  
            • RSI(9) > 50: 10  

            **Hourly RSI/WMA — 15 points**

            • RSI(9) crossed above WMA(Close,21): 8  
            • RSI(9) > 55: 7  

            The ranking is a **technical-strength ranking**, not
            an investment recommendation.
            """
        )

    if run_top10:

        progress = st.progress(
            0,
            text="Starting Top-10 scan..."
        )

        try:

            # ------------------------------------------------
            # DAILY DATA — BREAKOUT + DAILY RSI/WMA
            # ------------------------------------------------

            progress.progress(
                10,
                text="Downloading daily market data..."
            )

            daily_market = download_batches(
                stocks,
                "1y",
                batch_size
            )

            daily_rsi_market = (
                download_rsi_wma_batches(
                    stocks,
                    "Daily",
                    batch_size
                )
            )

            # ------------------------------------------------
            # WEEKLY
            # ------------------------------------------------

            progress.progress(
                40,
                text="Calculating weekly RSI(9) signals..."
            )

            weekly_market = (
                download_rsi_wma_batches(
                    stocks,
                    "Weekly",
                    batch_size
                )
            )

            # ------------------------------------------------
            # HOURLY
            # ------------------------------------------------

            hourly_market = {}

            if include_hourly:

                progress.progress(
                    60,
                    text="Calculating hourly RSI(9) signals..."
                )

                hourly_market = (
                    download_rsi_wma_batches(
                        stocks,
                        "Hourly",
                        batch_size
                    )
                )

            # ------------------------------------------------
            # BUILD RANKING
            # ------------------------------------------------

            progress.progress(
                75,
                text="Building combined technical scores..."
            )

            rows = []

            for symbol in stocks:

                # --------------------------------------------
                # SMART BREAKOUT
                # --------------------------------------------

                breakout_score = 0
                breakout = None
                daily_data = daily_market.get(
                    symbol
                )

                if (
                    daily_data is not None
                    and not daily_data.empty
                ):

                    try:

                        daily_indicators = (
                            calculate_indicators(
                                daily_data
                            )
                        )

                        breakout = (
                            stage_two_analysis(
                                daily_indicators
                            )
                        )

                        if breakout:

                            breakout_score = (
                                breakout["Score"]
                                / 10
                                * 40
                            )

                    except Exception:

                        breakout = None

                # --------------------------------------------
                # DAILY RSI/WMA
                # --------------------------------------------

                daily_signal = None
                daily_score = 0

                if symbol in daily_rsi_market:

                    try:

                        daily_signal = (
                            rsi_wma_scan_result(
                                daily_rsi_market[
                                    symbol
                                ],
                                55
                            )
                        )

                        if daily_signal:

                            if daily_signal["Cross"]:
                                daily_score += 15

                            if daily_signal[
                                "RSI9 > Threshold"
                            ]:
                                daily_score += 10

                    except Exception:

                        daily_signal = None

                # --------------------------------------------
                # WEEKLY RSI/WMA
                # --------------------------------------------

                weekly_signal = None
                weekly_score = 0

                if symbol in weekly_market:

                    try:

                        weekly_signal = (
                            rsi_wma_scan_result(
                                weekly_market[
                                    symbol
                                ],
                                50
                            )
                        )

                        if weekly_signal:

                            if weekly_signal["Cross"]:
                                weekly_score += 10

                            if weekly_signal[
                                "RSI9 > Threshold"
                            ]:
                                weekly_score += 10

                    except Exception:

                        weekly_signal = None

                # --------------------------------------------
                # HOURLY RSI/WMA
                # --------------------------------------------

                hourly_signal = None
                hourly_score = 0

                if include_hourly and symbol in hourly_market:

                    try:

                        hourly_signal = (
                            rsi_wma_scan_result(
                                hourly_market[
                                    symbol
                                ],
                                55
                            )
                        )

                        if hourly_signal:

                            if hourly_signal["Cross"]:
                                hourly_score += 8

                            if hourly_signal[
                                "RSI9 > Threshold"
                            ]:
                                hourly_score += 7

                    except Exception:

                        hourly_signal = None

                # --------------------------------------------
                # TOTAL SCORE
                # --------------------------------------------

                total_score = (
                    breakout_score
                    + daily_score
                    + weekly_score
                    + hourly_score
                )

                daily_pass = (
                    daily_signal is not None
                    and daily_signal["Cross"]
                    and daily_signal[
                        "RSI9 > Threshold"
                    ]
                )

                weekly_pass = (
                    weekly_signal is not None
                    and weekly_signal["Cross"]
                    and weekly_signal[
                        "RSI9 > Threshold"
                    ]
                )

                hourly_pass = (
                    hourly_signal is not None
                    and hourly_signal["Cross"]
                    and hourly_signal[
                        "RSI9 > Threshold"
                    ]
                )

                full_mtf = (
                    daily_pass
                    and weekly_pass
                    and (
                        not include_hourly
                        or hourly_pass
                    )
                )

                # --------------------------------------------
                # FILTER
                # --------------------------------------------

                if only_full_mtf and not full_mtf:
                    continue

                close = np.nan
                breakout_rsi = np.nan
                volume_ratio = np.nan
                breakout_score_display = round(
                    breakout_score,
                    1
                )

                if breakout:

                    close = breakout["Close"]
                    breakout_rsi = breakout["RSI"]
                    volume_ratio = breakout[
                        "Volume Ratio"
                    ]

                daily_rsi = (
                    daily_signal["RSI9"]
                    if daily_signal
                    else np.nan
                )

                weekly_rsi = (
                    weekly_signal["RSI9"]
                    if weekly_signal
                    else np.nan
                )

                hourly_rsi = (
                    hourly_signal["RSI9"]
                    if hourly_signal
                    else np.nan
                )

                rows.append({

                    "Rank": 0,

                    "Stock": symbol,

                    "Total Score":
                        round(
                            total_score,
                            1
                        ),

                    "Breakout Score":
                        breakout_score_display,

                    "Daily Score":
                        daily_score,

                    "Weekly Score":
                        weekly_score,

                    "Hourly Score":
                        hourly_score,

                    "Daily RSI9":
                        round(
                            daily_rsi,
                            2
                        )
                        if not pd.isna(
                            daily_rsi
                        )
                        else np.nan,

                    "Weekly RSI9":
                        round(
                            weekly_rsi,
                            2
                        )
                        if not pd.isna(
                            weekly_rsi
                        )
                        else np.nan,

                    "Hourly RSI9":
                        round(
                            hourly_rsi,
                            2
                        )
                        if not pd.isna(
                            hourly_rsi
                        )
                        else np.nan,

                    "Breakout RSI14":
                        round(
                            breakout_rsi,
                            2
                        )
                        if not pd.isna(
                            breakout_rsi
                        )
                        else np.nan,

                    "Volume Ratio":
                        round(
                            volume_ratio,
                            2
                        )
                        if not pd.isna(
                            volume_ratio
                        )
                        else np.nan,

                    "Close":
                        round(
                            close,
                            2
                        )
                        if not pd.isna(
                            close
                        )
                        else np.nan,

                    "Daily Pass":
                        "✓"
                        if daily_pass
                        else "—",

                    "Weekly Pass":
                        "✓"
                        if weekly_pass
                        else "—",

                    "Hourly Pass":
                        "✓"
                        if hourly_pass
                        else "—",

                    "Full MTF":
                        "🔥"
                        if full_mtf
                        else "—"
                })

            results_df = pd.DataFrame(
                rows
            )

            if not results_df.empty:

                results_df = (
                    results_df
                    .sort_values(
                        [
                            "Total Score",
                            "Weekly Score",
                            "Daily Score",
                            "Breakout Score"
                        ],
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                results_df["Rank"] = (
                    results_df.index + 1
                )

            progress.progress(
                100,
                text="Top-10 scan completed."
            )

            time.sleep(0.2)
            progress.empty()

        except Exception as e:

            progress.empty()

            st.error(
                f"Top-10 scanner error: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if results_df.empty:

            st.warning(
                """
                No stocks satisfied the selected filters.

                Try disabling "Only show stocks passing
                Daily + Weekly" or select a broader universe.
                """
            )

        else:

            top10 = results_df.head(10).copy()

            st.success(
                f"🏆 Top {len(top10)} stocks ranked from "
                f"{len(stocks)} stocks."
            )

            # -----------------------------------------------
            # TOP 3
            # -----------------------------------------------

            st.subheader(
                "🥇 Top 3"
            )

            top_cols = st.columns(
                min(3, len(top10))
            )

            for index, (_, row) in enumerate(
                top10.head(3).iterrows()
            ):

                with top_cols[index]:

                    score = row[
                        "Total Score"
                    ]

                    if score >= 75:

                        st.success(
                            f"🔥 #{int(row['Rank'])} "
                            f"{row['Stock']}"
                        )

                    elif score >= 55:

                        st.warning(
                            f"⚡ #{int(row['Rank'])} "
                            f"{row['Stock']}"
                        )

                    else:

                        st.info(
                            f"📈 #{int(row['Rank'])} "
                            f"{row['Stock']}"
                        )

                    st.metric(
                        "Technical Score",
                        f"{score:.1f}/100"
                    )

                    st.write(
                        f"Daily RSI9: "
                        f"**{row['Daily RSI9']}**"
                    )

                    st.write(
                        f"Weekly RSI9: "
                        f"**{row['Weekly RSI9']}**"
                    )

                    if include_hourly:

                        st.write(
                            f"Hourly RSI9: "
                            f"**{row['Hourly RSI9']}**"
                        )

            # -----------------------------------------------
            # TABLE
            # -----------------------------------------------

            st.subheader(
                "📊 Top 10 Ranking"
            )

            display_columns = [
                "Rank",
                "Stock",
                "Total Score",
                "Breakout Score",
                "Daily Score",
                "Weekly Score",
                "Hourly Score",
                "Daily RSI9",
                "Weekly RSI9",
                "Hourly RSI9",
                "Volume Ratio",
                "Close",
                "Daily Pass",
                "Weekly Pass",
                "Hourly Pass",
                "Full MTF"
            ]

            st.dataframe(
                top10[
                    display_columns
                ],
                width="stretch",
                hide_index=True
            )

            # -----------------------------------------------
            # MOBILE CARDS
            # -----------------------------------------------

            st.subheader(
                "📱 Mobile-Friendly Top 10"
            )

            for _, row in top10.iterrows():

                score = row[
                    "Total Score"
                ]

                title = (
                    f"#{int(row['Rank'])} "
                    f"{row['Stock']} — "
                    f"{score:.1f}/100"
                )

                with st.expander(
                    title
                ):

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Total Score",
                        f"{score:.1f}/100"
                    )

                    c2.metric(
                        "Breakout",
                        f"{row['Breakout Score']:.1f}/40"
                    )

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Daily RSI9",
                        (
                            f"{row['Daily RSI9']:.2f}"
                            if not pd.isna(
                                row["Daily RSI9"]
                            )
                            else "N/A"
                        )
                    )

                    c2.metric(
                        "Weekly RSI9",
                        (
                            f"{row['Weekly RSI9']:.2f}"
                            if not pd.isna(
                                row["Weekly RSI9"]
                            )
                            else "N/A"
                        )
                    )

                    if include_hourly:

                        st.write(
                            f"Hourly RSI9: "
                            f"**{row['Hourly RSI9']}**"
                        )

                    st.write(
                        f"Daily: **{row['Daily Pass']}** | "
                        f"Weekly: **{row['Weekly Pass']}** | "
                        f"Hourly: **{row['Hourly Pass']}**"
                    )

                    if row["Full MTF"] == "🔥":

                        st.success(
                            "🔥 Full multi-timeframe confirmation"
                        )

                    else:

                        st.info(
                            "Partial multi-timeframe confirmation"
                        )

                    if not pd.isna(
                        row["Close"]
                    ):

                        st.write(
                            f"Latest Close: "
                            f"**₹{row['Close']:.2f}**"
                        )

                    if not pd.isna(
                        row["Volume Ratio"]
                    ):

                        st.write(
                            f"Volume Ratio: "
                            f"**{row['Volume Ratio']:.2f}×**"
                        )

            # -----------------------------------------------
            # CSV DOWNLOAD
            # -----------------------------------------------

            csv = top10.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Top 10 Results",
                data=csv,
                file_name="Top_10_Momentum_Stocks.csv",
                mime="text/csv"
            )

            # -----------------------------------------------
            # OPTIONAL AI INTERPRETATION
            # -----------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Interpretation"
            )

            st.caption(
                "This is optional. It requires a working "
                "OpenAI API balance."
            )

            if client is None:

                st.info(
                    "Add your OpenAI API key in Streamlit "
                    "Secrets to enable AI interpretation."
                )

            else:

                if st.button(
                    "🤖 Explain Top 10",
                    type="secondary"
                ):

                    compact = top10[
                        [
                            "Rank",
                            "Stock",
                            "Total Score",
                            "Breakout Score",
                            "Daily Score",
                            "Weekly Score",
                            "Hourly Score",
                            "Daily RSI9",
                            "Weekly RSI9",
                            "Hourly RSI9",
                            "Volume Ratio",
                            "Close"
                        ]
                    ].to_dict(
                        orient="records"
                    )

                    prompt = f"""
                    Analyze these NSE technical scanner results
                    as an educational technical analyst.

                    Explain:
                    1. Which 3 stocks have the strongest setups.
                    2. Why their combined scores are high.
                    3. Which stocks have the strongest
                       multi-timeframe confirmation.
                    4. Which stocks have weak or missing
                       confirmation.
                    5. What should be checked on the chart
                       before considering a trade.

                    Do not invent prices or indicators.
                    Do not guarantee returns.
                    Use only the supplied data.

                    Scanner results:
                    {compact}
                    """

                    try:

                        with st.spinner(
                            "Generating AI interpretation..."
                        ):

                            response = client.responses.create(
                                model="gpt-5.6-mini",
                                instructions=(
                                    "You are an educational "
                                    "technical-analysis assistant. "
                                    "Never guarantee returns and "
                                    "never invent missing data."
                                ),
                                input=prompt
                            )

                        st.markdown(
                            response.output_text
                        )

                    except Exception as e:

                        st.error(
                            f"AI interpretation error: {e}"
                        )


else:

    st.header(
        "🤖 AI Technical Analyst"
    )

    st.write(
        """
        Ask questions about technical analysis,
        chart patterns, indicators and trading setups.
        """
    )

    if client is None:

        st.warning(
            """
            OpenAI API key is not configured.

            Add your key to Streamlit Cloud:

            Settings → Secrets
            """
        )

    question = st.chat_input(
        "Ask your technical-analysis question..."
    )

    if question:

        st.chat_message(
            "user"
        ).write(
            question
        )

        with st.chat_message(
            "assistant"
        ):

            if client is None:

                st.error(
                    "OpenAI API key is missing."
                )

            else:

                try:

                    response = (
                        client.responses.create(
                            model="gpt-5.6-luna",

                            instructions="""
You are an educational technical-analysis
assistant for Indian equity markets.

Explain technical-analysis concepts clearly.

You may discuss:
- Candlestick patterns
- Support and resistance
- Moving averages
- RSI
- MACD
- Bollinger Bands
- Donchian channels
- Volume
- Breakouts
- Trend structure
- Elliott Wave concepts
- Risk management
- Position sizing

Do not invent live market prices,
technical indicator values or company data.

If the application has not supplied
actual market data, clearly state that
you cannot verify the current value.

Do not present predictions as certainty.

Use clear sections:
1. Interpretation
2. Technical reasoning
3. Risk
4. What to monitor

This is educational information,
not personalized financial advice.
""",

                            input=question
                        )
                    )

                    st.write(
                        response.output_text
                    )

                except Exception as e:

                    st.error(
                        f"AI error: {e}"
                    )

