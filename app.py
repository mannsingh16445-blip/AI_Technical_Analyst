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
# WEEKLY TREND / MOMENTUM SCREEN
# ============================================================

def calculate_weekly_trend_screen(
    data,
    as_of=None,
    completed_only=False
):
    """
    User-specified weekly scanner:
    1) Close > SMA20
    2) Close > SMA50
    3) Close >= 90% of 20-week maximum Close
    4) ATR14 < ATR14 four weeks ago
    5) Current week Close > Open
    6) Previous week Close > Open
    7) Two weeks ago Close > Open
    8) Volume > Volume SMA20
    9) Close > 100
    10) Volume > 1,000,000
    """

    if data is None or data.empty:
        return None

    df=data.copy()

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    req=["Open","High","Low","Close","Volume"]

    if any(c not in df.columns for c in req):
        return None

    df=df.dropna(subset=req)

    if len(df)<260:
        return None

    df.index=pd.to_datetime(df.index)

    # For historical backtesting, never use data after the
    # signal date. This prevents look-ahead bias.
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        df = df.loc[df.index <= as_of]

    if df.empty:
        return None

    weekly=df.resample("W-FRI").agg({
        "Open":"first",
        "High":"max",
        "Low":"min",
        "Close":"last",
        "Volume":"sum"
    }).dropna(subset=req)

    # A historical weekly signal must use a completed weekly candle.
    # We conservatively treat Friday's close as the completion point.
    if completed_only and as_of is not None:
        weekday = as_of.weekday()  # Monday=0 ... Friday=4

        if weekday < 4:
            current_week_end = (
                as_of
                + pd.Timedelta(
                    days=(4 - weekday)
                )
            ).normalize()

            weekly = weekly.loc[
                weekly.index < current_week_end
            ]

    if len(weekly)<60:
        return None

    weekly["SMA20"]=weekly["Close"].rolling(20).mean()
    weekly["SMA50"]=weekly["Close"].rolling(50).mean()

    prev_close=weekly["Close"].shift(1)

    tr=pd.concat([
        weekly["High"]-weekly["Low"],
        (weekly["High"]-prev_close).abs(),
        (weekly["Low"]-prev_close).abs()
    ],axis=1).max(axis=1)

    weekly["ATR14"]=tr.rolling(14).mean()
    weekly["MAX20_CLOSE"]=weekly["Close"].rolling(20).max()
    weekly["VOL_SMA20"]=weekly["Volume"].rolling(20).mean()

    cur=weekly.iloc[-1]

    conditions={
        "Close > SMA20": cur["Close"] > cur["SMA20"],
        "Close > SMA50": cur["Close"] > cur["SMA50"],
        "Close >= 90% of 20W Max": cur["Close"] >= cur["MAX20_CLOSE"]*0.90,
        "ATR14 < ATR14 4W Ago": cur["ATR14"] < weekly["ATR14"].shift(4).iloc[-1],
        "Current Week Green": cur["Close"] > cur["Open"],
        "1 Week Ago Green": weekly["Close"].shift(1).iloc[-1] > weekly["Open"].shift(1).iloc[-1],
        "2 Weeks Ago Green": weekly["Close"].shift(2).iloc[-1] > weekly["Open"].shift(2).iloc[-1],
        "Volume > Volume SMA20": cur["Volume"] > cur["VOL_SMA20"],
        "Close > 100": cur["Close"] > 100,
        "Volume > 1M": cur["Volume"] > 1000000
    }

    # All required values are available once 60 weekly bars exist.
    passed=all(bool(v) for v in conditions.values())

    return {
        "Pass":passed,
        "Conditions":conditions,
        "Close":float(cur["Close"]),
        "SMA20":float(cur["SMA20"]),
        "SMA50":float(cur["SMA50"]),
        "ATR14":float(cur["ATR14"]),
        "ATR14_4W":float(weekly["ATR14"].shift(4).iloc[-1]),
        "MAX20_CLOSE":float(cur["MAX20_CLOSE"]),
        "Volume":float(cur["Volume"]),
        "VOL_SMA20":float(cur["VOL_SMA20"]),
        "WeeklyDate":weekly.index[-1]
    }


# ============================================================
# DAILY TREND / 50-150-200 SMA SCREEN
# ============================================================

def calculate_daily_trend_screen(data):
    """
    Daily conditions supplied by the user:
    Close>SMA150; Close>SMA200; SMA150>SMA200;
    SMA200 rising; SMA50>SMA150; SMA50>SMA200;
    Close>=1.25*252D Low; Close>=0.75*252D High;
    Close>SMA50; Volume>100000.
    """

    if data is None or data.empty:
        return None

    df=data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    required=["Open","High","Low","Close","Volume"]

    if any(c not in df.columns for c in required):
        return None

    df=df.dropna(subset=required)

    if len(df)<252:
        return None

    close=pd.to_numeric(df["Close"],errors="coerce")
    volume=pd.to_numeric(df["Volume"],errors="coerce")

    sma50=close.rolling(50).mean()
    sma150=close.rolling(150).mean()
    sma200=close.rolling(200).mean()

    min252=close.rolling(252).min()
    max252=close.rolling(252).max()

    c=float(close.iloc[-1])
    v=float(volume.iloc[-1])
    s50=float(sma50.iloc[-1])
    s150=float(sma150.iloc[-1])
    s200=float(sma200.iloc[-1])
    prev_s200=float(sma200.iloc[-2])
    lo=float(min252.iloc[-1])
    hi=float(max252.iloc[-1])

    conditions={
        "Close > SMA150": c>s150,
        "Close > SMA200": c>s200,
        "SMA150 > SMA200": s150>s200,
        "SMA200 rising": s200>prev_s200,
        "SMA50 > SMA150": s50>s150,
        "SMA50 > SMA200": s50>s200,
        "Close >= 1.25 x 252D Low": c>=lo*1.25,
        "Close >= 0.75 x 252D High": c>=hi*0.75,
        "Close > SMA50": c>s50,
        "Volume > 100000": v>100000
    }

    return {
        "Pass": all(conditions.values()),
        "Conditions": conditions,
        "Close": c,
        "Volume": v,
        "SMA50": s50,
        "SMA150": s150,
        "SMA200": s200,
        "Previous SMA200": prev_s200,
        "252D Low": lo,
        "252D High": hi
    }


# ============================================================
# STRATEGY-SPECIFIC HISTORICAL SIGNAL HELPERS
# ============================================================

def rsi_wma_signal_asof(
    data,
    as_of,
    threshold,
    timeframe="Daily"
):
    """
    Evaluate ONLY the RSI(9)/WMA(Close,21) rules for one
    timeframe using information available on or before as_of.

    Daily:
        RSI(9) crossed above WMA(21)
        RSI(9) > threshold

    Weekly:
        The daily data is resampled to W-FRI and only a
        completed weekly bar is eligible.
    """

    if data is None or data.empty:
        return None

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    as_of = pd.Timestamp(as_of)
    df = df.loc[df.index <= as_of]

    if timeframe == "Weekly":
        df = resample_to_weekly(df)

        # W-FRI labels the weekly bar Friday. Therefore a bar
        # labelled after as_of is not yet completed/available.
        df = df.loc[df.index <= as_of]

    prepared = prepare_rsi_wma_data(df)

    if prepared.empty or len(prepared) < 30:
        return None

    latest = prepared.iloc[-1]

    cross = crossed_above_rsi_wma(prepared)

    threshold_pass = (
        float(latest["RSI9"])
        > float(threshold)
    )

    return {
        "Pass": bool(
            cross and threshold_pass
        ),
        "Cross": bool(cross),
        "Threshold": bool(threshold_pass),
        "RSI9": float(latest["RSI9"]),
        "WMA21": float(latest["WMA21_CLOSE"]),
        "Date": prepared.index[-1]
    }


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def download_hourly_backtest_data(
    tickers
):
    """
    Yahoo Finance currently provides a much shorter history for
    60-minute data than daily data. This helper therefore downloads
    the maximum practical recent 60-day 60-minute history.

    It is used ONLY when an hourly condition is selected in the
    historical backtester.
    """

    results = {}

    for symbol in list(
        dict.fromkeys(tickers)
    ):

        ticker = (
            symbol
            if symbol.endswith(".NS")
            else symbol + ".NS"
        )

        try:

            df = yf.download(
                tickers=ticker,
                period="60d",
                interval="60m",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column"
            )

            if df is None or df.empty:
                continue

            if isinstance(
                df.columns,
                pd.MultiIndex
            ):
                # Single ticker downloads can still arrive with
                # a MultiIndex. Take the first ticker level.
                try:
                    df = df.xs(
                        ticker,
                        axis=1,
                        level=1
                    )
                except Exception:
                    df.columns = (
                        df.columns
                        .get_level_values(0)
                    )

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            if any(
                c not in df.columns
                for c in required
            ):
                continue

            df = df.dropna(
                subset=required
            )

            if not df.empty:
                results[symbol] = df

        except Exception:
            continue

    return results


def hourly_rsi_wma_signal_asof(
    hourly_data,
    as_of,
    threshold
):
    """
    Evaluate the exact Hourly RSI/WMA conditions using only
    hourly bars available on or before the signal date/time.

    For the daily backtest engine, the latest available hourly
    bar on the signal date is used. The actual trade is entered
    only on subsequent daily sessions.
    """

    if (
        hourly_data is None
        or hourly_data.empty
    ):
        return None

    df = hourly_data.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        df.index = pd.to_datetime(
            df.index
        )

    # Work by calendar date so the hourly confirmation belongs
    # to the same signal day.
    date = pd.Timestamp(
        as_of
    ).date()

    df = df.loc[
        df.index.date <= date
    ]

    if df.empty:
        return None

    # Use only bars on the signal day for the final hourly
    # confirmation, while the indicator history itself contains
    # all preceding hourly bars.
    prepared = prepare_rsi_wma_data(
        df
    )

    if prepared.empty or len(prepared) < 30:
        return None

    latest = prepared.iloc[-1]

    cross = crossed_above_rsi_wma(
        prepared
    )

    threshold_pass = (
        float(latest["RSI9"])
        > float(threshold)
    )

    return {
        "Pass": bool(
            cross and threshold_pass
        ),
        "Cross": bool(cross),
        "Threshold": bool(
            threshold_pass
        ),
        "RSI9": float(
            latest["RSI9"]
        ),
        "WMA21": float(
            latest["WMA21_CLOSE"]
        ),
        "Date": prepared.index[-1]
    }


def top20_score_asof(
    data,
    as_of,
    hourly_data=None,
    include_hourly=True
):
    """
    Historical version of the app's Top-20 scoring model.

    It uses ONLY the Top-20 model's own components:

      Smart Breakout          40 points
      Daily RSI/WMA           25 points
      Weekly RSI/WMA          20 points
      Hourly RSI/WMA          15 points

    It does NOT import an unrelated scanner condition.

    The score is returned for research/backtesting. Ranking the
    whole universe into exactly 20 names is a portfolio-selection
    problem; the backtest uses a configurable minimum score so
    that each historical signal can be tested independently.
    """

    if data is None or data.empty:
        return None

    df = data.copy()

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        df.index = pd.to_datetime(
            df.index
        )

    df = df.loc[
        df.index <= pd.Timestamp(as_of)
    ]

    if len(df) < 210:
        return None

    breakout = stage_two_analysis(
        calculate_indicators(df)
    )

    breakout_score = (
        40.0
        if breakout is None
        else min(
            float(
                breakout["Score"]
            ) / 10.0 * 40.0,
            40.0
        )
    )

    # Preserve the scanner's point-by-point logic rather than
    # replacing it with a generic RSI score.
    breakout_score = 0.0

    if breakout is not None:

        breakout_score = (
            float(
                breakout["Score"]
            )
            / 10.0
            * 40.0
        )

    daily = rsi_wma_signal_asof(
        df,
        as_of,
        55,
        "Daily"
    )

    daily_score = 0.0

    if daily is not None:

        if daily["Cross"]:
            daily_score += 15

        if daily["Threshold"]:
            daily_score += 10

    weekly = rsi_wma_signal_asof(
        df,
        as_of,
        50,
        "Weekly"
    )

    weekly_score = 0.0

    if weekly is not None:

        if weekly["Cross"]:
            weekly_score += 10

        if weekly["Threshold"]:
            weekly_score += 10

    hourly = None
    hourly_score = 0.0

    if (
        include_hourly
        and hourly_data is not None
    ):

        hourly = (
            hourly_rsi_wma_signal_asof(
                hourly_data,
                as_of,
                55
            )
        )

        if hourly is not None:

            if hourly["Cross"]:
                hourly_score += 8

            if hourly["Threshold"]:
                hourly_score += 7

    total = (
        breakout_score
        + daily_score
        + weekly_score
        + hourly_score
    )

    daily_pass = (
        daily is not None
        and daily["Pass"]
    )

    weekly_pass = (
        weekly is not None
        and weekly["Pass"]
    )

    hourly_pass = (
        hourly is not None
        and hourly["Pass"]
    )

    full_mtf = (
        daily_pass
        and weekly_pass
        and (
            not include_hourly
            or hourly_pass
        )
    )

    return {
        "Total Score": float(total),
        "Breakout Score": float(breakout_score),
        "Daily Score": float(daily_score),
        "Weekly Score": float(weekly_score),
        "Hourly Score": float(hourly_score),
        "Daily Pass": bool(daily_pass),
        "Weekly Pass": bool(weekly_pass),
        "Hourly Pass": bool(hourly_pass),
        "Full MTF": bool(full_mtf),
        "Breakout": breakout,
        "Daily": daily,
        "Weekly": weekly,
        "Hourly": hourly
    }



# ============================================================
# AUTOMATIC TRADE PLAN
# ============================================================

def calculate_trade_plan(data):
    """
    Create an educational technical trade plan from daily OHLCV.

    Method:
      Entry       = 0.25% above the latest close
      Stop Loss   = tighter of recent swing support and 1.5 ATR
                    is avoided; stop is placed below both levels
      Target 1    = Entry + 2R
      Target 2    = Entry + 3R
      Risk/Reward = Target / Entry-to-stop risk

    The levels are calculated from price/volatility data only.
    They are not guaranteed execution levels or recommendations.
    """

    if data is None or data.empty:
        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    if any(
        column not in data.columns
        for column in required
    ):
        return None

    df = data.copy()

    df = df.dropna(
        subset=required
    )

    if len(df) < 20:
        return None

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce"
    )

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    atr14 = (
        true_range
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    latest_close = float(
        close.iloc[-1]
    )

    latest_high = float(
        high.iloc[-1]
    )

    # Recent swing support excludes today's candle.
    swing_low_10 = float(
        low.shift(1)
        .rolling(10)
        .min()
        .iloc[-1]
    )

    swing_low_20 = float(
        low.shift(1)
        .rolling(20)
        .min()
        .iloc[-1]
    )

    support_candidates = [
        swing_low_10,
        swing_low_20
    ]

    # Use Donchian lower if it exists.
    if "DONCHIAN_LOWER" in df.columns:

        donchian_support = (
            pd.to_numeric(
                df["DONCHIAN_LOWER"],
                errors="coerce"
            )
            .iloc[-1]
        )

        if not pd.isna(
            donchian_support
        ):

            support_candidates.append(
                float(donchian_support)
            )

    support = min(
        support_candidates
    )

    if pd.isna(atr14) or atr14 <= 0:
        return None

    # Entry slightly above current close to avoid treating
    # the current close as a guaranteed fill.
    entry = max(
        latest_close * 1.0025,
        latest_high * 1.001
    )

    # Stop must be below recent support and below a
    # volatility-based 1.5 ATR level.
    atr_stop = (
        entry
        - 1.5 * float(atr14)
    )

    support_stop = (
        support * 0.995
    )

    stop_loss = min(
        atr_stop,
        support_stop
    )

    risk = (
        entry
        - stop_loss
    )

    if risk <= 0:
        return None

    target1 = (
        entry
        + 2.0 * risk
    )

    target2 = (
        entry
        + 3.0 * risk
    )

    risk_pct = (
        risk
        / entry
        * 100
    )

    target1_pct = (
        (target1 - entry)
        / entry
        * 100
    )

    target2_pct = (
        (target2 - entry)
        / entry
        * 100
    )

    rr1 = (
        target1 - entry
    ) / risk

    rr2 = (
        target2 - entry
    ) / risk

    return {

        "Entry":
            entry,

        "Stop Loss":
            stop_loss,

        "Target 1":
            target1,

        "Target 2":
            target2,

        "Risk":
            risk,

        "Risk %":
            risk_pct,

        "Target 1 %":
            target1_pct,

        "Target 2 %":
            target2_pct,

        "R:R T1":
            rr1,

        "R:R T2":
            rr2,

        "ATR14":
            float(atr14),

        "Support":
            support
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
        "📅 Weekly Trend Scanner",
        "📈 Daily Trend 50/150/200 Scanner",
        "🏆 Top 20 Momentum Stocks",
        "📊 Backtest & Performance",
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

elif module == "📅 Weekly Trend Scanner":

    st.header("📅 Weekly Trend Scanner")

    st.write(
        "Scanner based on the 10 weekly conditions supplied by you."
    )

    with st.spinner("Loading NSE stock universes..."):

        fno_stocks=load_fno_stocks()
        nifty500=load_nifty500()
        nse_stocks=load_nse_equity_universe()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        ["NSE F&O Stocks","Nifty 50","Nifty 500","Full NSE"],
        index=0,
        key="weekly_trend_universe"
    )

    if universe=="NSE F&O Stocks":
        stocks=list(fno_stocks)
    elif universe=="Nifty 50":
        stocks=list(NIFTY50)
    elif universe=="Nifty 500":
        stocks=list(nifty500)[:500]
    else:
        stocks=list(nse_stocks)

    max_stocks=st.sidebar.slider(
        "Maximum Stocks",
        10,
        min(500,max(10,len(stocks))),
        min(100,max(10,len(stocks))),
        10,
        key="weekly_trend_max"
    )

    period=st.sidebar.selectbox(
        "Data Period",
        ["2y","3y","5y","10y"],
        index=1,
        key="weekly_trend_period"
    )

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{min(max_stocks,len(stocks))}**"
    )

    with st.expander("📐 Conditions used"):

        st.markdown(
            """
            1. Weekly Close > Weekly SMA(20)  
            2. Weekly Close > Weekly SMA(50)  
            3. Weekly Close ≥ 90% of 20-week maximum Close  
            4. Weekly ATR(14) < ATR(14) four weeks ago  
            5. Current Weekly Close > Weekly Open  
            6. 1 week ago Close > 1 week ago Open  
            7. 2 weeks ago Close > 2 weeks ago Open  
            8. Weekly Volume > Weekly SMA(Volume,20)  
            9. Weekly Close > 100  
            10. Weekly Volume > 1,000,000
            """
        )

    if st.sidebar.button(
        "🔎 RUN WEEKLY SCAN",
        type="primary",
        key="weekly_trend_run"
    ):

        selected=stocks[:max_stocks]

        progress=st.progress(
            0,
            text="Scanning weekly conditions..."
        )

        rows=[]

        market=download_batches(
            selected,
            period,
            50
        )

        for n,symbol in enumerate(selected):

            result=calculate_weekly_trend_screen(
                market.get(symbol)
            )

            if result is not None:

                passed=sum(
                    bool(v)
                    for v in result["Conditions"].values()
                )

                rows.append({
                    "Stock":symbol,
                    "Status":"🟢 PASS" if result["Pass"] else "—",
                    "Passed":f"{passed}/10",
                    "Weekly Close":round(result["Close"],2),
                    "SMA20":round(result["SMA20"],2),
                    "SMA50":round(result["SMA50"],2),
                    "20W Max":round(result["MAX20_CLOSE"],2),
                    "ATR14":round(result["ATR14"],2),
                    "ATR14 4W Ago":round(result["ATR14_4W"],2),
                    "Volume":int(result["Volume"]),
                    "Volume SMA20":int(result["VOL_SMA20"])
                })

            progress.progress(
                int(100*(n+1)/max(1,len(selected))),
                text=f"Scanning {symbol}..."
            )

        progress.empty()

        result_df=pd.DataFrame(rows)

        if result_df.empty:

            st.warning("No usable weekly data was returned.")

        else:

            passed_df=result_df[
                result_df["Status"]=="🟢 PASS"
            ].copy()

            st.success(
                f"**{len(passed_df)}** stocks passed all 10 conditions "
                f"out of **{len(result_df)}** tested."
            )

            if not passed_df.empty:

                st.subheader(
                    "🟢 Stocks Passing All 10 Conditions"
                )

                st.dataframe(
                    passed_df,
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Weekly Results",
                    passed_df.to_csv(index=False),
                    "Weekly_Trend_10_Conditions.csv",
                    "text/csv"
                )

            else:

                st.info(
                    "No stocks currently pass all 10 conditions."
                )

            with st.expander("📋 Show all tested stocks"):

                st.dataframe(
                    result_df,
                    width="stretch",
                    hide_index=True
                )

elif module == "📈 Daily Trend 50/150/200 Scanner":

    st.header("📈 Daily Trend 50/150/200 Scanner")

    st.write(
        "Exact implementation of the 10 daily conditions supplied by you."
    )

    with st.spinner("Loading NSE stock universes..."):

        fno_stocks=load_fno_stocks()
        nifty500=load_nifty500()
        nse_stocks=load_nse_equity_universe()

    st.sidebar.subheader("📈 Daily Trend Scanner")

    universe=st.sidebar.selectbox(
        "Stock Universe",
        ["NSE F&O Stocks","Nifty 50","Nifty 500","Full NSE"],
        index=0,
        key="daily_trend_universe"
    )

    if universe=="NSE F&O Stocks":
        stocks=list(fno_stocks)
    elif universe=="Nifty 50":
        stocks=list(NIFTY50)
    elif universe=="Nifty 500":
        stocks=list(nifty500)[:500]
    else:
        stocks=list(nse_stocks)

    max_stocks=st.sidebar.slider(
        "Maximum Stocks to Scan",
        min_value=10,
        max_value=min(500,max(10,len(stocks))),
        value=min(100,max(10,len(stocks))),
        step=10,
        key="daily_trend_max_stocks"
    )

    period=st.sidebar.selectbox(
        "Historical Data",
        ["2y","3y","5y","10y"],
        index=1,
        key="daily_trend_period"
    )

    run_scan=st.sidebar.button(
        "🔎 RUN DAILY TREND SCAN",
        type="primary",
        key="daily_trend_run"
    )

    with st.expander("📐 Exact Scanner Conditions"):

        st.markdown(
            """
            1. Daily Close > Daily SMA(150)
            2. Daily Close > Daily SMA(200)
            3. Daily SMA(150) > Daily SMA(200)
            4. Daily SMA(200) > 1-day-ago Daily SMA(200)
            5. Daily SMA(50) > Daily SMA(150)
            6. Daily SMA(50) > Daily SMA(200)
            7. Daily Close ≥ Daily Min(252, Close) × 1.25
            8. Daily Close ≥ Daily Max(252, Close) × 0.75
            9. Daily Close > Daily SMA(50)
            10. Daily Volume > 100,000
            """
        )

    if run_scan:

        selected=stocks[:max_stocks]

        progress=st.progress(0,text="Downloading daily trend data...")

        try:

            market=download_batches(selected,period,50)
            rows=[]

            for n,symbol in enumerate(selected):

                data=market.get(symbol)

                if data is None or data.empty:
                    continue

                result=calculate_daily_trend_screen(data)

                if result is None:
                    continue

                passed_count=sum(
                    bool(v)
                    for v in result["Conditions"].values()
                )

                rows.append({
                    "Stock":symbol,
                    "Status":
                        "🟢 PASS"
                        if result["Pass"]
                        else "—",
                    "Conditions Passed":
                        f"{passed_count}/10",
                    "Close":round(result["Close"],2),
                    "SMA50":round(result["SMA50"],2),
                    "SMA150":round(result["SMA150"],2),
                    "SMA200":round(result["SMA200"],2),
                    "Prev SMA200":round(result["Previous SMA200"],2),
                    "252D Low":round(result["252D Low"],2),
                    "252D High":round(result["252D High"],2),
                    "Volume":int(result["Volume"])
                })

                progress.progress(
                    int(100*(n+1)/max(1,len(selected))),
                    text=f"Scanning {symbol}..."
                )

            progress.empty()

            result_df=pd.DataFrame(rows)

            if result_df.empty:

                st.warning("No usable daily data was returned.")

            else:

                passed_df=result_df[
                    result_df["Status"]=="🟢 PASS"
                ].copy()

                st.success(
                    f"Daily trend scan completed: "
                    f"**{len(passed_df)} stocks passed** "
                    f"out of **{len(result_df)} tested**."
                )

                if not passed_df.empty:

                    st.subheader(
                        "🟢 Stocks Passing All 10 Conditions"
                    )

                    st.dataframe(
                        passed_df,
                        width="stretch",
                        hide_index=True
                    )

                    st.download_button(
                        "⬇️ Download Daily Trend Results",
                        passed_df.to_csv(index=False),
                        "Daily_Trend_50_150_200_Scanner.csv",
                        "text/csv"
                    )

                else:

                    st.info(
                        "No stocks currently pass all 10 conditions."
                    )

                with st.expander("📋 Show all tested stocks"):

                    st.dataframe(
                        result_df,
                        width="stretch",
                        hide_index=True
                    )

        except Exception as e:

            progress.empty()
            st.error(
                f"Daily trend scanner error: {e}"
            )

elif module == "🏆 Top 20 Momentum Stocks":

    st.header(
        "🏆 Top 20 Momentum Stocks"
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
        "🏆 Top 20 Scanner"
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

    run_top20 = st.sidebar.button(
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

            ### 🎯 Automatic Trade Plan

            • Entry = approximately 0.25% above the latest close  
            • Stop Loss = below recent support and volatility-adjusted
              using 1.5× ATR(14)  
            • Target 1 = 2R  
            • Target 2 = 3R  
            • Risk/Reward is calculated from Entry → Stop Loss  

            These are algorithmic reference levels, not guaranteed
            execution prices or personalized investment advice.

            ### 🏷️ Trade Setup Quality

            **🟢 A+ Setup**
            - Full Daily + Weekly + Hourly confirmation
            - Score ≥ 80
            - R:R to Target 1 ≥ 1:2

            **🟢 A Setup**
            - Daily + Weekly confirmation
            - Score ≥ 65

            **🟡 B Setup**
            - Daily or Weekly confirmation
            - Score ≥ 50

            **🟡 C Setup**
            - Partial technical confirmation
            - Score ≥ 35

            **🔴 Avoid**
            - Weak technical confirmation
            """
        )

    if run_top20:

        progress = st.progress(
            0,
            text="Starting Top-20 scan..."
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
                # AUTOMATIC TRADE PLAN
                # --------------------------------------------

                trade_plan = (
                    calculate_trade_plan(
                        daily_data
                    )
                )

                # --------------------------------------------
                # TOTAL SCORE
                # --------------------------------------------

                total_score = (
                    breakout_score
                    + daily_score
                    + weekly_score
                    + hourly_score
                )

                # --------------------------------------------
                # MULTI-TIMEFRAME PASS CONDITIONS
                # --------------------------------------------

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
                # TRADE SETUP QUALITY
                # --------------------------------------------

                # A+ = strong multi-timeframe alignment
                # A  = strong Daily + Weekly confirmation
                # B  = useful momentum but incomplete confirmation
                # C  = partial/weak setup
                # Avoid = weak technical confirmation

                if (
                    full_mtf
                    and total_score >= 80
                    and (
                        not pd.isna(
                            trade_plan["R:R T1"]
                        )
                        if trade_plan
                        else False
                    )
                    and (
                        trade_plan["R:R T1"] >= 2
                        if trade_plan
                        else False
                    )
                ):

                    setup_grade = "A+"
                    setup_label = "🟢 A+ Setup"

                elif (
                    daily_pass
                    and weekly_pass
                    and total_score >= 65
                ):

                    setup_grade = "A"
                    setup_label = "🟢 A Setup"

                elif (
                    (
                        daily_pass
                        or weekly_pass
                    )
                    and total_score >= 50
                ):

                    setup_grade = "B"
                    setup_label = "🟡 B Setup"

                elif total_score >= 35:

                    setup_grade = "C"
                    setup_label = "🟡 C Setup"

                else:

                    setup_grade = "Avoid"
                    setup_label = "🔴 Avoid"

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

                    "Setup Grade":
                        setup_grade,

                    "Setup":
                        setup_label,

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

                    "Entry":
                        round(
                            trade_plan["Entry"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "Stop Loss":
                        round(
                            trade_plan["Stop Loss"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "Target 1":
                        round(
                            trade_plan["Target 1"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "Target 2":
                        round(
                            trade_plan["Target 2"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "Risk %":
                        round(
                            trade_plan["Risk %"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "R:R T1":
                        round(
                            trade_plan["R:R T1"],
                            2
                        )
                        if trade_plan
                        else np.nan,

                    "R:R T2":
                        round(
                            trade_plan["R:R T2"],
                            2
                        )
                        if trade_plan
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

                grade_order = {
                    "A+": 5,
                    "A": 4,
                    "B": 3,
                    "C": 2,
                    "Avoid": 1
                }

                results_df["_Grade Rank"] = (
                    results_df["Setup Grade"]
                    .map(grade_order)
                    .fillna(0)
                )

                results_df = (
                    results_df
                    .sort_values(
                        [
                            "_Grade Rank",
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

                results_df = results_df.drop(
                    columns=["_Grade Rank"]
                )

                results_df["Rank"] = (
                    results_df.index + 1
                )

            progress.progress(
                100,
                text="Top-20 scan completed."
            )

            time.sleep(0.2)
            progress.empty()

        except Exception as e:

            progress.empty()

            st.error(
                f"Top-20 scanner error: {e}"
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

            top20 = results_df.head(20).copy()

            st.success(
                f"🏆 Top {len(top20)} stocks ranked from "
                f"{len(stocks)} stocks."
            )

            # -----------------------------------------------
            # TOP 3
            # -----------------------------------------------

            st.subheader(
                "🥇 Top 3"
            )

            top_cols = st.columns(
                min(3, len(top20))
            )

            for index, (_, row) in enumerate(
                top20.head(3).iterrows()
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

                    if row["Setup Grade"] == "A+":

                        st.success(
                            "🟢 A+ Setup — Strong MTF confirmation"
                        )

                    elif row["Setup Grade"] == "A":

                        st.success(
                            "🟢 A Setup — Strong Daily + Weekly confirmation"
                        )

                    elif row["Setup Grade"] == "B":

                        st.warning(
                            "🟡 B Setup — Good momentum, incomplete confirmation"
                        )

                    elif row["Setup Grade"] == "C":

                        st.warning(
                            "🟡 C Setup — Partial confirmation"
                        )

                    else:

                        st.error(
                            "🔴 Avoid — Weak technical confirmation"
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
                "📊 Top 20 Ranking"
            )

            display_columns = [
                "Rank",
                "Stock",
                "Setup",
                "Setup Grade",
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
                "Entry",
                "Stop Loss",
                "Target 1",
                "Target 2",
                "Risk %",
                "R:R T1",
                "R:R T2",
                "Daily Pass",
                "Weekly Pass",
                "Hourly Pass",
                "Full MTF"
            ]

            st.dataframe(
                top20[
                    display_columns
                ],
                width="stretch",
                hide_index=True
            )

            # -----------------------------------------------
            # MOBILE CARDS
            # -----------------------------------------------

            st.subheader(
                "📱 Mobile-Friendly Top 20"
            )

            for _, row in top20.iterrows():

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
                        "Setup",
                        row["Setup"]
                    )

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Grade",
                        row["Setup Grade"]
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
                        row["Entry"]
                    ):

                        st.markdown(
                            "### 🎯 Trade Plan"
                        )

                        p1, p2 = st.columns(2)

                        p1.metric(
                            "Entry",
                            f"₹{row['Entry']:.2f}"
                        )

                        p2.metric(
                            "Stop Loss",
                            f"₹{row['Stop Loss']:.2f}"
                        )

                        p1, p2 = st.columns(2)

                        p1.metric(
                            "Target 1",
                            f"₹{row['Target 1']:.2f}"
                        )

                        p2.metric(
                            "Target 2",
                            f"₹{row['Target 2']:.2f}"
                        )

                        st.write(
                            f"Risk: **{row['Risk %']:.2f}%** | "
                            f"R:R to T1: **1:{row['R:R T1']:.1f}** | "
                            f"R:R to T2: **1:{row['R:R T2']:.1f}**"
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

            csv = top20.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Top 20 Results",
                data=csv,
                file_name="Top_20_Momentum_Stocks.csv",
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
                    "🤖 Explain Top 20",
                    type="secondary"
                ):

                    compact = top20[
                        [
                            "Rank",
                            "Stock",
                            "Setup",
                            "Setup Grade",
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
                            "Entry",
                            "Stop Loss",
                            "Target 1",
                            "Target 2",
                            "Risk %",
                            "R:R T1",
                            "R:R T2"
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
                    6. Explain why the A+ and A setups rank above
                       the B/C setups.

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


elif module == "📊 Backtest & Performance":

    st.header(
        "📊 Backtest & Performance"
    )

    st.write(
        """
        Test the scanner rules on historical daily data without
        using future information. The engine enters only after a
        signal is confirmed at the close and evaluates the next
        trading sessions for Entry, Stop Loss and Targets.
        """
    )

    st.sidebar.subheader(
        "📊 Backtest Settings"
    )

    with st.spinner(
        "Loading stock universes..."
    ):

        fno_stocks = load_fno_stocks()
        nifty500 = load_nifty500()
        nse_stocks = load_nse_equity_universe()

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
        stocks = list(fno_stocks)
    elif universe == "Nifty 50":
        stocks = list(NIFTY50)
    elif universe == "Nifty 500":
        stocks = list(nifty500)[:500]
    else:
        stocks = list(nse_stocks)

    strategy = st.sidebar.selectbox(
        "Strategy to Backtest",
        [
            "Smart Breakout",
            "Daily RSI(9)/WMA(21)",
            "Weekly RSI(9)/WMA(21)",
            "Hourly RSI(9)/WMA(21)",
            "Multi-Timeframe RSI/WMA",
            "Combined Breakout + Daily RSI(9)",
            "Weekly Trend — 10 Conditions",
            "Daily Trend — 10 Conditions",
            "Top 20 Momentum Model"
        ]
    )

    period = st.sidebar.selectbox(
        "Historical Period",
        [
            "1y",
            "2y",
            "3y",
            "5y",
            "10y"
        ],
        index=3
    )

    max_stocks = st.sidebar.slider(
        "Maximum Stocks to Test",
        min_value=10,
        max_value=min(250, max(10, len(stocks))),
        value=min(50, max(10, len(stocks))),
        step=10
    )

    if strategy in [
        "Smart Breakout",
        "Combined Breakout + Daily RSI(9)"
    ]:

        min_breakout_score = st.sidebar.slider(
            "Minimum Breakout Score",
            min_value=5,
            max_value=10,
            value=7,
            help="Used only for Smart Breakout and Combined strategies."
        )

    else:

        min_breakout_score = None

    if strategy == "Top 20 Momentum Model":

        top20_min_score = st.sidebar.slider(
            "Minimum Top-20 Model Score",
            min_value=35,
            max_value=100,
            value=65,
            step=5,
            help=(
                "The Top-20 scanner is a ranking model. "
                "For historical trade testing, this threshold "
                "defines a qualifying Top-20-model setup."
            )
        )

        top20_include_hourly = st.sidebar.checkbox(
            "Include Hourly component",
            value=True,
            help=(
                "Hourly history is limited by Yahoo Finance; "
                "when enabled, this backtest is effectively "
                "limited to the recent hourly-history window."
            )
        )

    else:

        top20_min_score = None
        top20_include_hourly = False

    if strategy == "Multi-Timeframe RSI/WMA":

        mtf_hourly_confirmation = st.sidebar.checkbox(
            "Require Hourly Confirmation",
            value=False
        )

    else:

        mtf_hourly_confirmation = False

    holding_days = st.sidebar.slider(
        "Maximum Holding Days",
        min_value=5,
        max_value=60,
        value=20,
        step=5
    )

    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)",
        min_value=10000.0,
        value=100000.0,
        step=10000.0
    )

    position_pct = st.sidebar.slider(
        "Capital per Trade (%)",
        min_value=5,
        max_value=100,
        value=10,
        step=5
    )


    run_backtest = st.sidebar.button(
        "🚀 RUN BACKTEST",
        type="primary"
    )

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks selected: **{min(max_stocks, len(stocks))}** | "
        f"Strategy: **{strategy}**"
    )

    strategy_condition_note = {
        "Smart Breakout":
            "Only C1–C5 + Smart Breakout confirmations are tested.",
        "Daily RSI(9)/WMA(21)":
            "Only Daily RSI(9) cross + RSI(9) > 55 are tested.",
        "Weekly RSI(9)/WMA(21)":
            "Only Weekly RSI(9) cross + RSI(9) > 50 are tested.",
        "Hourly RSI(9)/WMA(21)":
            "Only Hourly RSI(9) cross + RSI(9) > 55 are tested.",
        "Multi-Timeframe RSI/WMA":
            "Only Weekly + Daily RSI/WMA conditions are tested, plus optional Hourly confirmation.",
        "Combined Breakout + Daily RSI(9)":
            "Only Smart Breakout AND Daily RSI(9)/WMA(21) are required.",
        "Weekly Trend — 10 Conditions":
            "Only the 10 Weekly Trend conditions are tested.",
        "Daily Trend — 10 Conditions":
            "Only the 10 Daily Trend conditions are tested.",
        "Top 20 Momentum Model":
            "Only the Top-20 model's 100-point scoring components are tested."
    }

    st.caption(
        "🔒 Active backtest rule set: "
        + strategy_condition_note.get(
            strategy,
            "Selected strategy conditions only."
        )
    )

    if strategy in [
        "Hourly RSI(9)/WMA(21)",
        "Multi-Timeframe RSI/WMA"
    ] and (
        strategy == "Hourly RSI(9)/WMA(21)"
        or mtf_hourly_confirmation
    ):

        st.warning(
            "Hourly historical testing is limited by Yahoo Finance "
            "to the available recent 60-day 60-minute dataset."
        )

    if strategy == "Top 20 Momentum Model" and top20_include_hourly:

        st.warning(
            "The Top-20 model's hourly component uses the available "
            "recent 60-day hourly history. Disable the hourly component "
            "for a longer daily/weekly historical test."
        )

    with st.expander(
        "📐 Backtest Rules"
    ):

        st.markdown(
            """
            ### 🔒 Strategy-isolated backtesting

            **Smart Breakout**
            - ONLY the existing Smart Breakout C1–C5 rules and its
              built-in confirmations/score are used.
            - Daily/Weekly/Hourly RSI scanners are NOT added.

            **Daily RSI(9)/WMA(21)**
            - ONLY Daily RSI(9) crossed above WMA(Close,21).
            - ONLY Daily RSI(9) > 55.
            - No Smart Breakout or trend-filter conditions.

            **Weekly RSI(9)/WMA(21)**
            - ONLY Weekly RSI(9) crossed above WMA(Close,21).
            - ONLY Weekly RSI(9) > 50.
            - Only completed weekly bars are eligible.

            **Hourly RSI(9)/WMA(21)**
            - ONLY Hourly RSI(9) crossed above WMA(Close,21).
            - ONLY Hourly RSI(9) > 55.
            - Yahoo Finance hourly history is limited, so this test uses
              the available recent 60-day hourly dataset.

            **Multi-Timeframe RSI/WMA**
            - Weekly RSI/WMA conditions AND Daily RSI/WMA conditions.
            - Hourly confirmation is optional and is used only if enabled.

            **Combined Breakout + Daily RSI(9)**
            - Smart Breakout AND Daily RSI(9)/WMA(21).
            - This combination is intentional and is the ONLY reason those
              two condition families appear together.

            **Weekly Trend — 10 Conditions**
            - ONLY the 10 supplied Weekly Trend conditions.
            - Completed weekly bars only.
            - No Smart Breakout, RSI/WMA, or Daily Trend conditions.

            **Daily Trend — 10 Conditions**
            - ONLY the 10 supplied Daily 50/150/200 + 252-day range +
              volume conditions.
            - No Smart Breakout, RSI/WMA, or Weekly Trend conditions.

            **Top 20 Momentum Model**
            - ONLY the Top-20 scanner's own 100-point components:
              Smart Breakout 40 + Daily RSI/WMA 25 +
              Weekly RSI/WMA 20 + optional Hourly RSI/WMA 15.
            - A configurable minimum score is used for historical trade
              qualification because a "Top 20" ranking is a universe-level
              portfolio-selection problem rather than a single-stock signal.

            ### Common trade management
            **Entry:** approximately 0.25% above the signal-day close.
            A fill occurs only when a subsequent daily High reaches Entry.

            **Stop Loss:** below recent support and volatility-adjusted
            using ATR(14), consistent with the app trade-plan logic.

            **Target 1:** 2R.
            **Target 2:** 3R.

            If both stop and target are reached within the same daily
            candle, the conservative assumption is that the stop is hit
            first.

            These common items are trade-management rules, not additional
            scanner-selection conditions.
            """
        )

    if run_backtest:

        selected_stocks = stocks[:max_stocks]
        progress = st.progress(
            0,
            text="Downloading historical data..."
        )

        try:

            historical = download_batches(
                selected_stocks,
                period,
                50
            )

            # Hourly history is downloaded ONLY when the selected
            # scanner actually requires it. This keeps other strategy
            # backtests fast and prevents unrelated data from entering
            # their calculations.
            needs_hourly = (
                strategy == "Hourly RSI(9)/WMA(21)"
                or (
                    strategy == "Multi-Timeframe RSI/WMA"
                    and mtf_hourly_confirmation
                )
                or (
                    strategy == "Top 20 Momentum Model"
                    and top20_include_hourly
                )
            )

            if needs_hourly:

                progress.progress(
                    20,
                    text=(
                        "Downloading recent 60-day "
                        "hourly history..."
                    )
                )

                hourly_history = (
                    download_hourly_backtest_data(
                        selected_stocks
                    )
                )

            else:

                hourly_history = {}

            progress.progress(
                25,
                text="Scanning historical signals..."
            )

            trades = []

            for stock_index, symbol in enumerate(selected_stocks):

                data = historical.get(symbol)

                if data is None or data.empty:
                    continue

                data = data.copy()

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                required = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]

                if any(c not in data.columns for c in required):
                    continue

                data = data.dropna(subset=required)

                if len(data) < 220:
                    continue

                indicators = calculate_indicators(data)
                rsi_data = prepare_rsi_wma_data(data)

                # Prevent overlapping signals for the same stock.
                next_allowed_index = 0

                for i in range(210, len(indicators) - holding_days - 1):

                    if i < next_allowed_index:
                        continue

                    hist = indicators.iloc[:i + 1]
                    signal_date = hist.index[-1]

                    # ====================================================
                    # STRATEGY-ISOLATED HISTORICAL SIGNAL LOGIC
                    # ====================================================

                    breakout = None
                    breakout_pass = False

                    rsi_pass = False
                    rsi_value = np.nan

                    weekly_result = None
                    weekly_pass = False

                    daily_trend_result = None
                    daily_trend_pass = False

                    weekly_rsi = None
                    hourly_rsi = None
                    mtf_daily = None
                    mtf_weekly = None
                    mtf_hourly = None
                    top20_model = None

                    # --------------------------------------------
                    # 1. SMART BREAKOUT ONLY
                    # --------------------------------------------
                    if strategy == "Smart Breakout":

                        breakout = stage_two_analysis(
                            hist
                        )

                        breakout_pass = (
                            breakout is not None
                            and breakout["Score"]
                            >= min_breakout_score
                        )

                        signal = breakout_pass

                    # --------------------------------------------
                    # 2. DAILY RSI/WMA ONLY
                    # --------------------------------------------
                    elif strategy == "Daily RSI(9)/WMA(21)":

                        daily_rsi_result = (
                            rsi_wma_signal_asof(
                                data,
                                signal_date,
                                55,
                                "Daily"
                            )
                        )

                        if daily_rsi_result:

                            rsi_pass = (
                                daily_rsi_result["Pass"]
                            )

                            rsi_value = (
                                daily_rsi_result["RSI9"]
                            )

                        signal = rsi_pass

                    # --------------------------------------------
                    # 3. WEEKLY RSI/WMA ONLY
                    # --------------------------------------------
                    elif strategy == "Weekly RSI(9)/WMA(21)":

                        weekly_rsi = (
                            rsi_wma_signal_asof(
                                data,
                                signal_date,
                                50,
                                "Weekly"
                            )
                        )

                        signal = (
                            weekly_rsi is not None
                            and weekly_rsi["Pass"]
                            and weekly_rsi["Date"]
                            == pd.Timestamp(signal_date)
                        )

                    # --------------------------------------------
                    # 4. HOURLY RSI/WMA ONLY
                    # --------------------------------------------
                    elif strategy == "Hourly RSI(9)/WMA(21)":

                        hourly_rsi = (
                            hourly_rsi_wma_signal_asof(
                                hourly_history.get(
                                    symbol
                                ),
                                signal_date,
                                55
                            )
                            if symbol in hourly_history
                            else None
                        )

                        signal = (
                            hourly_rsi is not None
                            and hourly_rsi["Pass"]
                        )

                    # --------------------------------------------
                    # 5. MULTI-TIMEFRAME RSI/WMA ONLY
                    # --------------------------------------------
                    elif strategy == "Multi-Timeframe RSI/WMA":

                        mtf_daily = (
                            rsi_wma_signal_asof(
                                data,
                                signal_date,
                                55,
                                "Daily"
                            )
                        )

                        mtf_weekly = (
                            rsi_wma_signal_asof(
                                data,
                                signal_date,
                                50,
                                "Weekly"
                            )
                        )

                        mtf_hourly = None

                        if mtf_hourly_confirmation:

                            mtf_hourly = (
                                hourly_rsi_wma_signal_asof(
                                    hourly_history.get(
                                        symbol
                                    ),
                                    signal_date,
                                    55
                                )
                                if symbol in hourly_history
                                else None
                            )

                        daily_ok = (
                            mtf_daily is not None
                            and mtf_daily["Pass"]
                        )

                        weekly_ok = (
                            mtf_weekly is not None
                            and mtf_weekly["Pass"]
                            and mtf_weekly["Date"]
                            == pd.Timestamp(signal_date)
                        )

                        hourly_ok = (
                            not mtf_hourly_confirmation
                            or (
                                mtf_hourly is not None
                                and mtf_hourly["Pass"]
                            )
                        )

                        signal = (
                            daily_ok
                            and weekly_ok
                            and hourly_ok
                        )

                    # --------------------------------------------
                    # 6. INTENTIONAL COMBINED STRATEGY
                    # --------------------------------------------
                    elif strategy == "Combined Breakout + Daily RSI(9)":

                        breakout = stage_two_analysis(
                            hist
                        )

                        breakout_pass = (
                            breakout is not None
                            and breakout["Score"]
                            >= min_breakout_score
                        )

                        daily_rsi_result = (
                            rsi_wma_signal_asof(
                                data,
                                signal_date,
                                55,
                                "Daily"
                            )
                        )

                        if daily_rsi_result:

                            rsi_pass = (
                                daily_rsi_result["Pass"]
                            )

                            rsi_value = (
                                daily_rsi_result["RSI9"]
                            )

                        signal = (
                            breakout_pass
                            and rsi_pass
                        )

                    # --------------------------------------------
                    # 7. WEEKLY TREND — ONLY ITS 10 CONDITIONS
                    # --------------------------------------------
                    elif strategy == "Weekly Trend — 10 Conditions":

                        weekly_result = (
                            calculate_weekly_trend_screen(
                                data.iloc[:i + 1],
                                as_of=signal_date,
                                completed_only=True
                            )
                        )

                        weekly_pass = (
                            weekly_result is not None
                            and weekly_result["Pass"]
                            and weekly_result["WeeklyDate"]
                            == pd.Timestamp(signal_date)
                        )

                        signal = weekly_pass

                    # --------------------------------------------
                    # 8. DAILY TREND — ONLY ITS 10 CONDITIONS
                    # --------------------------------------------
                    elif strategy == "Daily Trend — 10 Conditions":

                        daily_trend_result = (
                            calculate_daily_trend_screen(
                                data.iloc[:i + 1]
                            )
                        )

                        daily_trend_pass = (
                            daily_trend_result is not None
                            and daily_trend_result["Pass"]
                        )

                        signal = daily_trend_pass

                    # --------------------------------------------
                    # 9. TOP-20 MODEL — ONLY ITS OWN SCORE
                    # --------------------------------------------
                    elif strategy == "Top 20 Momentum Model":

                        top20_model = (
                            top20_score_asof(
                                data,
                                signal_date,
                                hourly_data=(
                                    hourly_history.get(
                                        symbol
                                    )
                                    if top20_include_hourly
                                    else None
                                ),
                                include_hourly=(
                                    top20_include_hourly
                                )
                            )
                        )

                        signal = (
                            top20_model is not None
                            and top20_model[
                                "Total Score"
                            ] >= top20_min_score
                        )

                    else:

                        signal = False

                    if not signal:
                        continue


                    signal_close = float(
                        indicators.iloc[i]["Close"]
                    )

                    # Build trade plan using information available
                    # only through the signal bar.
                    plan = calculate_trade_plan(
                        indicators.iloc[:i + 1]
                    )

                    if plan is None:
                        continue

                    entry = float(plan["Entry"])
                    stop = float(plan["Stop Loss"])
                    target1 = float(plan["Target 1"])
                    target2 = float(plan["Target 2"])

                    if not (
                        stop < entry < target1 < target2
                    ):
                        continue

                    future = indicators.iloc[
                        i + 1:
                        min(
                            i + 1 + holding_days,
                            len(indicators)
                        )
                    ]

                    entry_pos = None
                    entry_date = None

                    # Find first future day that trades through entry.
                    for j, (_, candle) in enumerate(future.iterrows()):

                        day_high = float(candle["High"])

                        if day_high >= entry:
                            entry_pos = j
                            entry_date = candle.name
                            break

                    if entry_pos is None:
                        continue

                    post_entry = future.iloc[
                        entry_pos:
                    ]

                    exit_date = post_entry.index[-1]
                    exit_price = float(
                        post_entry.iloc[-1]["Close"]
                    )
                    outcome = "Time Exit"
                    r_multiple = (
                        exit_price - entry
                    ) / (entry - stop)

                    for _, candle in post_entry.iterrows():

                        day_low = float(candle["Low"])
                        day_high = float(candle["High"])

                        # Conservative same-day assumption:
                        # if stop and target both occur, count stop first.
                        if day_low <= stop:

                            exit_price = stop
                            exit_date = candle.name
                            outcome = "Stop Loss"
                            r_multiple = -1.0
                            break

                        if day_high >= target2:

                            exit_price = target2
                            exit_date = candle.name
                            outcome = "Target 2"
                            r_multiple = 3.0
                            break

                        if day_high >= target1:

                            exit_price = target1
                            exit_date = candle.name
                            outcome = "Target 1"
                            r_multiple = 2.0
                            break

                    # Strategy-specific label only.
                    if strategy == "Smart Breakout":

                        score = (
                            breakout["Score"]
                            if breakout is not None
                            else 0
                        )

                        grade = (
                            "A+"
                            if score >= 9
                            else "A"
                            if score >= 8
                            else "B"
                        )

                    elif strategy == "Combined Breakout + Daily RSI(9)":

                        score = (
                            breakout["Score"]
                            if breakout is not None
                            else 0
                        )

                        grade = (
                            "A+"
                            if score >= 9
                            else "A"
                            if score >= 8
                            else "B"
                        )

                    elif strategy == "Daily RSI(9)/WMA(21)":

                        grade = "Daily RSI Qualified"

                    elif strategy == "Weekly RSI(9)/WMA(21)":

                        grade = "Weekly RSI Qualified"

                    elif strategy == "Hourly RSI(9)/WMA(21)":

                        grade = "Hourly RSI Qualified"

                    elif strategy == "Multi-Timeframe RSI/WMA":

                        grade = "MTF RSI Qualified"

                    elif strategy == "Weekly Trend — 10 Conditions":

                        grade = "Weekly Trend Qualified"

                    elif strategy == "Daily Trend — 10 Conditions":

                        grade = "Daily Trend Qualified"

                    elif strategy == "Top 20 Momentum Model":

                        score = (
                            top20_model["Total Score"]
                            if top20_model is not None
                            else 0
                        )

                        grade = (
                            "A+"
                            if score >= 80
                            else "A"
                            if score >= 65
                            else "B"
                        )

                    else:

                        grade = "Qualified"

                    trades.append({

                        "Stock": symbol,
                        "Signal Date": signal_date,
                        "Entry Date": entry_date,
                        "Exit Date": exit_date,
                        "Setup Grade": grade,
                        "Entry": round(entry, 2),
                        "Stop Loss": round(stop, 2),
                        "Target 1": round(target1, 2),
                        "Target 2": round(target2, 2),
                        "Exit Price": round(exit_price, 2),
                        "Outcome": outcome,
                        "R Multiple": round(r_multiple, 2),
                        "RSI9":
                            (
                                round(
                                    rsi_value,
                                    2
                                )
                                if (
                                    strategy
                                    in [
                                        "Daily RSI(9)/WMA(21)",
                                        "Combined Breakout + Daily RSI(9)"
                                    ]
                                    and not pd.isna(
                                        rsi_value
                                    )
                                )
                                else np.nan
                            ),

                        "Breakout Score":
                            (
                                breakout["Score"]
                                if (
                                    strategy
                                    in [
                                        "Smart Breakout",
                                        "Combined Breakout + Daily RSI(9)"
                                    ]
                                    and breakout is not None
                                )
                                else np.nan
                            ),

                        "Weekly RSI9":
                            (
                                round(
                                    weekly_rsi["RSI9"],
                                    2
                                )
                                if (
                                    strategy
                                    == "Weekly RSI(9)/WMA(21)"
                                    and weekly_rsi is not None
                                )
                                else np.nan
                            ),

                        "Hourly RSI9":
                            (
                                round(
                                    hourly_rsi["RSI9"],
                                    2
                                )
                                if (
                                    strategy
                                    == "Hourly RSI(9)/WMA(21)"
                                    and hourly_rsi is not None
                                )
                                else np.nan
                            ),

                        "Top20 Score":
                            (
                                round(
                                    top20_model[
                                        "Total Score"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "Top 20 Momentum Model"
                                    and top20_model is not None
                                )
                                else np.nan
                            ),

                        "Weekly Trend Pass":
                            (
                                "✓"
                                if (
                                    strategy
                                    == "Weekly Trend — 10 Conditions"
                                    and weekly_pass
                                )
                                else "—"
                            ),

                        "Daily Trend Pass":
                            (
                                "✓"
                                if (
                                    strategy
                                    == "Daily Trend — 10 Conditions"
                                    and daily_trend_pass
                                )
                                else "—"
                            )
                    })

                    # Avoid repeatedly entering the same stock while a
                    # previous signal's holding window is still active.
                    next_allowed_index = i + entry_pos + holding_days + 1

                progress.progress(
                    25 + int(
                        60 * (stock_index + 1)
                        / max(1, len(selected_stocks))
                    ),
                    text=f"Testing {symbol}..."
                )

            progress.progress(
                100,
                text="Backtest completed."
            )

            time.sleep(0.2)
            progress.empty()

            if not trades:

                st.warning(
                    "No historical trades were generated. "
                    "Try a broader universe, longer period, or lower "
                    "minimum breakout score."
                )

            else:

                trades_df = pd.DataFrame(trades)

                # ------------------------------------------------
                # PERFORMANCE METRICS
                # ------------------------------------------------

                total_trades = len(trades_df)
                wins = int(
                    (trades_df["R Multiple"] > 0).sum()
                )
                losses = int(
                    (trades_df["R Multiple"] <= 0).sum()
                )

                win_rate = (
                    wins / total_trades * 100
                )

                avg_r = float(
                    trades_df["R Multiple"].mean()
                )

                median_r = float(
                    trades_df["R Multiple"].median()
                )

                gross_profit_r = float(
                    trades_df.loc[
                        trades_df["R Multiple"] > 0,
                        "R Multiple"
                    ].sum()
                )

                gross_loss_r = abs(float(
                    trades_df.loc[
                        trades_df["R Multiple"] < 0,
                        "R Multiple"
                    ].sum()
                ))

                profit_factor = (
                    gross_profit_r / gross_loss_r
                    if gross_loss_r > 0
                    else np.inf
                )

                expectancy = avg_r

                # ------------------------------------------------
                # EQUITY CURVE
                # ------------------------------------------------

                ordered = trades_df.sort_values(
                    "Exit Date"
                ).reset_index(drop=True)

                risk_per_trade = (
                    initial_capital
                    * position_pct
                    / 100
                )

                ordered["PnL ₹"] = (
                    ordered["R Multiple"]
                    * risk_per_trade
                )

                ordered["Equity"] = (
                    initial_capital
                    + ordered["PnL ₹"].cumsum()
                )

                ordered["Peak Equity"] = (
                    ordered["Equity"].cummax()
                )

                ordered["Drawdown"] = (
                    ordered["Equity"]
                    - ordered["Peak Equity"]
                )

                max_drawdown = float(
                    ordered["Drawdown"].min()
                )

                max_drawdown_pct = float(
                    (
                        ordered["Drawdown"]
                        / ordered["Peak Equity"]
                        * 100
                    ).min()
                )

                final_equity = float(
                    ordered["Equity"].iloc[-1]
                )

                total_return = (
                    (final_equity / initial_capital - 1)
                    * 100
                )

                trading_days = (
                    pd.to_datetime(ordered["Exit Date"]).max()
                    - pd.to_datetime(ordered["Exit Date"]).min()
                ).days

                years = max(
                    trading_days / 365.25,
                    1 / 365.25
                )

                cagr = (
                    (final_equity / initial_capital)
                    ** (1 / years)
                    - 1
                ) * 100

                # ------------------------------------------------
                # DASHBOARD
                # ------------------------------------------------

                st.success(
                    f"Backtest completed: {total_trades} trades"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Win Rate",
                    f"{win_rate:.1f}%"
                )

                c2.metric(
                    "Average R",
                    f"{avg_r:.2f}R"
                )

                c3.metric(
                    "Profit Factor",
                    (
                        "∞"
                        if np.isinf(profit_factor)
                        else f"{profit_factor:.2f}"
                    )
                )

                c4.metric(
                    "Total Return",
                    f"{total_return:.1f}%"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Expectancy",
                    f"{expectancy:.2f}R/trade"
                )

                c2.metric(
                    "Median R",
                    f"{median_r:.2f}R"
                )

                c3.metric(
                    "CAGR*",
                    f"{cagr:.1f}%"
                )

                c4.metric(
                    "Risk/Trade",
                    f"₹{risk_per_trade:,.0f}"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Total Trades",
                    f"{total_trades}"
                )

                c2.metric(
                    "Wins / Losses",
                    f"{wins} / {losses}"
                )

                c3.metric(
                    "Max Drawdown",
                    f"₹{max_drawdown:,.0f}"
                )

                c4.metric(
                    "Max DD %",
                    f"{max_drawdown_pct:.1f}%"
                )

                st.subheader(
                    "📈 Equity Curve"
                )

                equity_chart = go.Figure()

                equity_chart.add_trace(
                    go.Scatter(
                        x=ordered["Exit Date"],
                        y=ordered["Equity"],
                        mode="lines",
                        name="Equity"
                    )
                )

                equity_chart.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    annotation_text="Initial Capital"
                )

                equity_chart.update_layout(
                    height=450,
                    xaxis_title="Exit Date",
                    yaxis_title="Portfolio Value (₹)",
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(
                    equity_chart,
                    width="stretch"
                )

                # ------------------------------------------------
                # OUTCOME DISTRIBUTION
                # ------------------------------------------------

                st.subheader(
                    "🎯 Trade Outcomes"
                )

                outcome_counts = (
                    ordered["Outcome"]
                    .value_counts()
                )

                outcome_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=outcome_counts.index,
                            y=outcome_counts.values,
                            text=outcome_counts.values,
                            textposition="auto"
                        )
                    ]
                )

                outcome_fig.update_layout(
                    height=350,
                    xaxis_title="Outcome",
                    yaxis_title="Number of Trades"
                )

                st.plotly_chart(
                    outcome_fig,
                    width="stretch"
                )

                # ------------------------------------------------
                # GRADE PERFORMANCE
                # ------------------------------------------------

                st.subheader(
                    "🏷️ Setup Grade Performance"
                )

                grade_summary = (
                    ordered
                    .groupby("Setup Grade")
                    .agg(
                        Trades=("R Multiple", "count"),
                        Win_Rate=(
                            "R Multiple",
                            lambda x: (x > 0).mean() * 100
                        ),
                        Avg_R=(
                            "R Multiple",
                            "mean"
                        ),
                        Total_R=(
                            "R Multiple",
                            "sum"
                        )
                    )
                    .reset_index()
                )

                grade_summary["Win_Rate"] = (
                    grade_summary["Win_Rate"]
                    .round(1)
                )

                grade_summary["Avg_R"] = (
                    grade_summary["Avg_R"]
                    .round(2)
                )

                grade_summary["Total_R"] = (
                    grade_summary["Total_R"]
                    .round(2)
                )

                st.dataframe(
                    grade_summary,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # MONTHLY PERFORMANCE
                # ------------------------------------------------

                st.subheader(
                    "📅 Monthly Performance"
                )

                monthly = ordered.copy()

                monthly["Month"] = pd.to_datetime(
                    monthly["Exit Date"]
                ).dt.to_period("M").astype(str)

                monthly_summary = (
                    monthly
                    .groupby("Month")
                    .agg(
                        Trades=("R Multiple", "count"),
                        Total_R=("R Multiple", "sum"),
                        Avg_R=("R Multiple", "mean"),
                        Win_Rate=(
                            "R Multiple",
                            lambda x: (x > 0).mean() * 100
                        )
                    )
                    .reset_index()
                )

                monthly_summary["Total_R"] = (
                    monthly_summary["Total_R"].round(2)
                )
                monthly_summary["Avg_R"] = (
                    monthly_summary["Avg_R"].round(2)
                )
                monthly_summary["Win_Rate"] = (
                    monthly_summary["Win_Rate"].round(1)
                )

                st.dataframe(
                    monthly_summary,
                    width="stretch",
                    hide_index=True
                )

                # ------------------------------------------------
                # TRADE LOG
                # ------------------------------------------------

                st.subheader(
                    "📋 Historical Trade Log"
                )

                st.dataframe(
                    ordered[
                        [
                            "Stock",
                            "Signal Date",
                            "Entry Date",
                            "Exit Date",
                            "Setup Grade",
                            "Entry",
                            "Stop Loss",
                            "Target 1",
                            "Target 2",
                            "Exit Price",
                            "Outcome",
                            "R Multiple",
                            "RSI9",
                            "Breakout Score",
                            "PnL ₹",
                            "Equity"
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )

                csv = ordered.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download Backtest Trade Log",
                    data=csv,
                    file_name="scanner_backtest_trade_log.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Backtest results are historical simulations and "
                    "do not guarantee future performance. They also "
                    "do not model brokerage, taxes, slippage, gaps, "
                    "corporate actions or liquidity constraints. The "
                    "equity curve is a fixed-risk R-multiple simulation, "
                    "not a statement of actual portfolio returns. "
                    "*CAGR is illustrative under the same fixed-risk "
                    "assumption."
                )

        except Exception as e:

            progress.empty()

            st.error(
                f"Backtest error: {e}"
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

