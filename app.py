import os

# Keep the Streamlit Cloud process thread-safe.
# yfinance is also forced to use serial downloads below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time

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

    url=(
        "https://nsearchives.nseindia.com/"
        "content/equities/EQUITY_L.csv"
    )

    headers={
        "User-Agent":(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept":"text/csv,*/*",
        "Referer":"https://www.nseindia.com/"
    }

    # CSV with session/cookies.
    try:

        session=requests.Session()
        session.headers.update(headers)

        try:
            session.get(
                "https://www.nseindia.com/",
                timeout=15
            )
        except Exception:
            pass

        response=session.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        df=pd.read_csv(
            StringIO(response.text)
        )

        df.columns=[
            str(c).strip().upper()
            for c in df.columns
        ]

        if "SYMBOL" in df.columns:

            symbols=(
                df["SYMBOL"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            symbols=symbols[
                ~symbols.isin([
                    "",
                    "NAN",
                    "NONE",
                    "NULL"
                ])
            ]

            symbols=sorted(
                symbols.drop_duplicates().tolist()
            )

            if len(symbols)>=500:
                return symbols

    except Exception:
        pass

    # NSE equity-master API fallback.
    # The endpoint returns the currently active equity universe.
    for api_url in [
        "https://www.nseindia.com/api/equity-master",
        "https://www.nseindia.com/api/equity-stockIndices"
        "?index=NIFTY%20500"
    ]:

        try:

            session=requests.Session()
            session.headers.update({
                "User-Agent":(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0 Safari/537.36"
                ),
                "Accept":"application/json,text/plain,*/*",
                "Accept-Language":"en-US,en;q=0.9",
                "Referer":"https://www.nseindia.com/"
            })

            session.get(
                "https://www.nseindia.com/",
                timeout=20
            )

            response=session.get(
                api_url,
                timeout=30
            )

            if response.status_code!=200:
                continue

            payload=response.json()

            data=payload.get("data",[])

            symbols=[]

            for item in data:

                if not isinstance(item,dict):
                    continue

                symbol=(
                    item.get("symbol")
                    or item.get("SYMBOL")
                )

                if symbol is None:
                    continue

                symbol=str(symbol).strip().upper()

                if symbol in {
                    "",
                    "NAN",
                    "NONE",
                    "NULL",
                    "NIFTY",
                    "NIFTY 50",
                    "NIFTY 500"
                }:
                    continue

                symbols.append(symbol)

            symbols=sorted(set(symbols))

            if len(symbols)>=500:
                return symbols

        except Exception:
            continue

    return []





# ============================================================
# NSE INDEX LOADING HELPERS
# ============================================================

def _load_nse_index_from_api(index_name, minimum_count=1):
    """Load an NSE index constituent list through the NSE API."""
    try:
        session=requests.Session()
        session.headers.update({
            "User-Agent":(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            ),
            "Accept":"application/json,text/plain,*/*",
            "Accept-Language":"en-US,en;q=0.9",
            "Referer":"https://www.nseindia.com/"
        })

        try:
            session.get(
                "https://www.nseindia.com/",
                timeout=15
            )
        except Exception:
            pass

        url=(
            "https://www.nseindia.com/api/equity-stockIndices"
            "?index="+requests.utils.quote(index_name)
        )

        response=session.get(
            url,
            timeout=30
        )

        if response.status_code!=200:
            return []

        payload=response.json()
        data=payload.get("data",[])

        symbols=[]

        for item in data:
            if not isinstance(item,dict):
                continue

            symbol=item.get("symbol") or item.get("SYMBOL")

            if symbol is None:
                continue

            symbol=str(symbol).strip().upper()

            if symbol in {
                "","NAN","NONE","NULL",
                "NIFTY","BANKNIFTY","FINNIFTY",
                "MIDCPNIFTY","NIFTYNXT50"
            }:
                continue

            symbols.append(symbol)

        symbols=sorted(set(symbols))

        return symbols if len(symbols)>=minimum_count else []

    except Exception:
        return []


def _load_nse_index_with_multiple_sources(
    index_name,
    csv_urls,
    minimum_count=1
):
    """
    Try archive CSV endpoints first and then the NSE API.
    Returns [] only when all sources fail.
    """

    headers={
        "User-Agent":(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept":"text/csv,text/plain,application/json,*/*",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://www.nseindia.com/"
    }

    for url in csv_urls:
        try:
            session=requests.Session()
            session.headers.update(headers)

            try:
                session.get(
                    "https://www.nseindia.com/",
                    timeout=15
                )
            except Exception:
                pass

            response=session.get(
                url,
                timeout=30
            )

            if response.status_code!=200:
                continue

            df=pd.read_csv(
                StringIO(response.text)
            )

            df.columns=[
                str(c).strip().upper()
                for c in df.columns
            ]

            if "SYMBOL" not in df.columns:
                continue

            symbols=(
                df["SYMBOL"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            symbols=symbols[
                ~symbols.isin([
                    "","NAN","NONE","NULL"
                ])
            ]

            symbols=sorted(
                symbols.drop_duplicates().tolist()
            )

            if len(symbols)>=minimum_count:
                return symbols

        except Exception:
            continue

    return _load_nse_index_from_api(
        index_name,
        minimum_count
    )


# ============================================================
# LOAD NIFTY 500
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_nifty500():
    """
    Load Nifty 500 with multiple fallbacks.

    Priority:
      1. Official NSE/Nifty Indices CSV
      2. GitHub-hosted symbol snapshot
      3. NSE API

    The GitHub snapshot is used only as a connectivity fallback
    because Streamlit Cloud can intermittently receive a blocked
    response from NSE's CSV endpoint.
    """

    official_urls=[
        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://archives.nseindia.com/"
        "content/indices/ind_nifty500list.csv",

        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"
    ]

    # Stable fallback snapshot containing Nifty 500 symbols.
    # It is intentionally capped to 500 symbols after parsing.
    fallback_urls=[
        "https://raw.githubusercontent.com/"
        "ganeshbiyer/Nse_Historical_Data/main/"
        "nifty500_symbols.csv"
    ]

    symbols=_load_nse_index_with_multiple_sources(
        "NIFTY 500",
        official_urls,
        400
    )

    if symbols:
        return sorted(set(symbols))[:500]

    # --------------------------------------------------------
    # GitHub fallback
    # --------------------------------------------------------
    headers={
        "User-Agent":(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept":"text/csv,text/plain,*/*"
    }

    for url in fallback_urls:
        try:

            response=requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code!=200:
                continue

            df=pd.read_csv(
                StringIO(response.text)
            )

            df.columns=[
                str(c).strip().upper()
                for c in df.columns
            ]

            if "SYMBOL" not in df.columns:
                continue

            symbols=(
                df["SYMBOL"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            symbols=symbols[
                ~symbols.isin([
                    "","NAN","NONE","NULL"
                ])
            ]

            symbols=sorted(
                symbols.drop_duplicates().tolist()
            )

            # This fallback currently contains 501 rows in the
            # source file; the app must expose exactly 500.
            if len(symbols)>=500:
                return symbols[:500]

        except Exception:
            continue

    # --------------------------------------------------------
    # Final NSE API fallback
    # --------------------------------------------------------
    symbols=_load_nse_index_from_api(
        "NIFTY 500",
        400
    )

    if symbols:
        return sorted(set(symbols))[:500]

    return []


# ============================================================
# ADDITIONAL NIFTY INDEX UNIVERSES
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_nifty_index_constituents(
    index_key,
    minimum_count=1
):
    """
    Robust index constituent loader.

    Source order:
      1. NSE official archive
      2. Nifty Indices CSV
      3. PKScreener GitHub cache
      4. NSE API

    The GitHub cache is a dedicated fallback for Streamlit Cloud
    because NSE can reject automated requests intermittently.
    """

    url_map={
        "NIFTY_MIDCAP_100":[
            "https://nsearchives.nseindia.com/"
            "content/indices/ind_niftymidcap100list.csv",

            "https://archives.nseindia.com/"
            "content/indices/ind_niftymidcap100list.csv",

            "https://www.niftyindices.com/"
            "IndexConstituent/ind_niftymidcap100list.csv"
        ],

        "NIFTY_SMALLCAP_250":[
            "https://nsearchives.nseindia.com/"
            "content/indices/ind_niftysmallcap250list.csv",

            "https://archives.nseindia.com/"
            "content/indices/ind_niftysmallcap250list.csv",

            "https://www.niftyindices.com/"
            "IndexConstituent/ind_niftysmallcap250list.csv"
        ]
    }

    repo_map={
        "NIFTY_MIDCAP_100":
            "https://raw.githubusercontent.com/"
            "pkjmesra/PKScreener/actions-data-download/"
            "results/Indices/ind_niftymidcap100list.csv",

        "NIFTY_SMALLCAP_250":
            "https://raw.githubusercontent.com/"
            "pkjmesra/PKScreener/actions-data-download/"
            "results/Indices/ind_niftysmallcap250list.csv"
    }

    headers={
        "User-Agent":(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept":"text/csv,text/plain,application/json,*/*",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":"https://www.nseindia.com/"
    }

    def parse_symbols(response_text):

        # First try normal CSV parsing with a SYMBOL column.
        try:

            df=pd.read_csv(
                StringIO(response_text)
            )

            df.columns=[
                str(c).strip().upper()
                for c in df.columns
            ]

            if "SYMBOL" in df.columns:

                symbols=(
                    df["SYMBOL"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                symbols=symbols[
                    ~symbols.isin([
                        "","NAN","NONE","NULL"
                    ])
                ]

                return sorted(
                    symbols.drop_duplicates().tolist()
                )

        except Exception:
            pass

        # Fallback for NSE CSV layouts where SYMBOL is the
        # third column, matching the PKNSETools implementation.
        try:

            reader=csv.reader(
                response_text.strip().splitlines()
            )

            next(reader)

            symbols=[]

            for row in reader:

                if len(row)<3:
                    continue

                symbol=str(row[2]).strip().upper()

                if symbol in {
                    "","NAN","NONE","NULL"
                }:
                    continue

                symbols.append(symbol)

            return sorted(set(symbols))

        except Exception:
            return []

    # --------------------------------------------------------
    # 1 + 2. Official sources
    # --------------------------------------------------------
    for url in url_map.get(index_key,[]):

        try:

            response=requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code!=200:
                continue

            symbols=parse_symbols(response.text)

            if len(symbols)>=minimum_count:
                return symbols

        except Exception:
            continue

    # --------------------------------------------------------
    # 3. PKScreener GitHub cache
    #
    # This is the important new fallback.
    # PKNSETools documents the same REPO_INDEX_MAP approach
    # for reliable constituent retrieval.
    # --------------------------------------------------------
    repo_url=repo_map.get(index_key)

    if repo_url:

        try:

            response=requests.get(
                repo_url,
                headers={
                    "User-Agent":"Mozilla/5.0",
                    "Accept":"text/csv,text/plain,*/*"
                },
                timeout=30
            )

            if response.status_code==200:

                symbols=parse_symbols(
                    response.text
                )

                if len(symbols)>=minimum_count:
                    return symbols

        except Exception:
            pass

    # --------------------------------------------------------
    # 4. NSE API
    # --------------------------------------------------------
    api_name=index_key.replace("_"," ")

    symbols=_load_nse_index_from_api(
        api_name,
        minimum_count
    )

    if symbols:
        return symbols

    return []


def load_nifty_midcap100():
    return load_nifty_index_constituents(
        "NIFTY_MIDCAP_100",
        80
    )


def load_nifty_smallcap250():
    return load_nifty_index_constituents(
        "NIFTY_SMALLCAP_250",
        200
    )



def resolve_stock_universe(
    universe,
    nse_stocks,
    nifty500,
    fno_stocks,
    nifty_midcap100,
    nifty_smallcap250
):
    """
    Central universe resolver used by scanners and backtester.
    """

    if universe == "Nifty 50":
        return list(NIFTY50)

    if universe == "Nifty 500":
        return list(nifty500)[:500]

    if universe == "Nifty Midcap 100":
        return list(nifty_midcap100)[:100]

    if universe == "Nifty Smallcap 250":
        return list(nifty_smallcap250)[:250]

    if universe == "NSE F&O Stocks":
        return list(fno_stocks)

    if universe == "Full NSE":
        return list(nse_stocks)

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
                threads=False,
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
# 120-DAY HIGH BREAKOUT / LIQUIDITY SCANNER
# ============================================================

def calculate_120day_breakout_screen(data):
    """
    Exact implementation of the four conditions supplied
    in the user's scanner screenshot:

    1. Daily Close >
       1 day ago Max(120, Daily High)

    2. 1 day ago Close <
       2 days ago Max(120, Daily High)

    3. NSE Value in lakhs > 50

    4. Daily Close >
       1 day ago Close * 1.03

    NSE Value in lakhs is calculated as:
        Close * Volume / 100000

    This converts traded value from rupees to lakh rupees.
    """

    if data is None or data.empty:
        return None

    df = data.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
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
        return None

    df = df.dropna(
        subset=required
    ).copy()

    if len(df) < 123:
        return None

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    # "1 day ago Max(120, Daily High)"
    # = previous 120 completed daily highs.
    max120_1d_ago = (
        high
        .rolling(120)
        .max()
        .shift(1)
    )

    # "2 days ago Max(120, Daily High)"
    # = 120-day high available two sessions before
    # the current signal day.
    max120_2d_ago = (
        high
        .rolling(120)
        .max()
        .shift(2)
    )

    current_close = float(
        close.iloc[-1]
    )

    previous_close = float(
        close.iloc[-2]
    )

    previous_120_high = float(
        max120_1d_ago.iloc[-1]
    )

    two_days_ago_120_high = float(
        max120_2d_ago.iloc[-1]
    )

    current_volume = float(
        volume.iloc[-1]
    )

    # NSE traded value in lakh rupees.
    nse_value_lakhs = (
        current_close
        * current_volume
        / 100000.0
    )

    conditions = {

        "Close > 1D Ago Max(120, High)":
            current_close
            > previous_120_high,

        "1D Ago Close < 2D Ago Max(120, High)":
            previous_close
            < two_days_ago_120_high,

        "NSE Value in lakhs > 50":
            nse_value_lakhs
            > 50.0,

        "Close > 1D Ago Close × 1.03":
            current_close
            > previous_close * 1.03
    }

    return {

        "Pass":
            all(
                conditions.values()
            ),

        "Conditions":
            conditions,

        "Close":
            current_close,

        "Previous Close":
            previous_close,

        "120D High 1D Ago":
            previous_120_high,

        "120D High 2D Ago":
            two_days_ago_120_high,

        "Volume":
            current_volume,

        "NSE Value Lakhs":
            nse_value_lakhs
    }


def calculate_120day_breakout_asof(
    data,
    as_of
):
    """
    Historical, look-ahead-safe version of the exact
    four-condition 120-day breakout scanner.
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

    as_of = pd.Timestamp(
        as_of
    )

    df = df.loc[
        df.index <= as_of
    ]

    return calculate_120day_breakout_screen(
        df
    )



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


# ============================================================
# HOURLY DONCHIAN + 200 SMA + RSI(9) BREAKOUT SCANNER
# ============================================================

def calculate_hourly_donchian_breakout(
    hourly_data,
    as_of=None
):
    """
    Exact implementation of the supplied Chartink hourly rules.

    H1: [0] 1-hour Close > [-1] 1-hour High
    H2: [-1] 1-hour High < [-2] 1-hour High
    H3: [0] 1-hour Close > [0] 1-hour SMA(Close, 200)
    H4: [-1] 1-hour Low > [0] 1-hour Donchian Lower Band(5)
    H5: [-1] 1-hour High < [0] 1-hour Donchian Upper Band(5)
    H6: [0] 1-hour RSI(9) >= 55

    Donchian bands are calculated literally from the current
    5-bar window:
        Upper = rolling 5-bar High maximum
        Lower = rolling 5-bar Low minimum

    For historical use, only bars <= as_of are considered.
    """

    if hourly_data is None or hourly_data.empty:
        return None

    df = hourly_data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)

        # Handle timezone mismatch safely.
        try:
            if getattr(df.index, "tz", None) is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(df.index.tz)
            elif getattr(df.index, "tz", None) is None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
        except Exception:
            cutoff = pd.Timestamp(as_of).tz_localize(None)

        df = df.loc[df.index <= cutoff]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if any(c not in df.columns for c in required):
        return None

    df = df.dropna(subset=required).copy()

    if len(df) < 205:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    sma200 = close.rolling(200).mean()

    donchian_upper = high.rolling(5).max()
    donchian_lower = low.rolling(5).min()

    # RSI(9) using the app's existing Wilder RSI implementation.
    rsi9 = calculate_rsi_wilder(close, 9)

    work = pd.DataFrame(
        {
            "Close": close,
            "High": high,
            "Low": low,
            "SMA200": sma200,
            "DonchianUpper5": donchian_upper,
            "DonchianLower5": donchian_lower,
            "RSI9": rsi9,
        },
        index=df.index
    ).dropna()

    if len(work) < 3:
        return None

    latest = work.iloc[-1]
    prev = work.iloc[-2]
    prev2 = work.iloc[-3]

    conditions = {
        "H1 Close > Previous Hour High":
            float(latest["Close"]) > float(prev["High"]),

        "H2 Previous Hour High < 2-Hours-Ago High":
            float(prev["High"]) < float(prev2["High"]),

        "H3 Close > Hourly SMA(200)":
            float(latest["Close"]) > float(latest["SMA200"]),

        "H4 Previous Hour Low > Current Donchian Lower(5)":
            float(prev["Low"]) > float(latest["DonchianLower5"]),

        "H5 Previous Hour High < Current Donchian Upper(5)":
            float(prev["High"]) < float(latest["DonchianUpper5"]),

        "H6 Hourly RSI(9) >= 55":
            float(latest["RSI9"]) >= 55.0,
    }

    return {
        "Pass": all(conditions.values()),
        "Conditions": conditions,
        "Date": work.index[-1],
        "Close": float(latest["Close"]),
        "Previous High": float(prev["High"]),
        "Two Hours Ago High": float(prev2["High"]),
        "Previous Low": float(prev["Low"]),
        "SMA200": float(latest["SMA200"]),
        "Donchian Upper5": float(latest["DonchianUpper5"]),
        "Donchian Lower5": float(latest["DonchianLower5"]),
        "RSI9": float(latest["RSI9"]),
    }


def hourly_donchian_breakout_asof(
    hourly_data,
    as_of
):
    """
    Look-ahead-safe historical version.

    For a daily backtest signal date, inspect all hourly bars
    available on that calendar date and use the latest hourly
    bar that satisfies all six conditions.

    The trade itself is still entered only on subsequent daily
    sessions by the common backtest engine.
    """

    if hourly_data is None or hourly_data.empty:
        return None

    df = hourly_data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    signal_date = pd.Timestamp(as_of).date()

    # All hourly history through the signal date.
    try:
        date_mask = df.index.date <= signal_date
        available = df.loc[date_mask]
    except Exception:
        available = df.copy()

    if available.empty:
        return None

    # Evaluate each hourly bar in chronological order by passing
    # a prefix of the data to the exact condition engine.
    # This avoids using any future hourly bar.
    passed = None

    # We need at least 205 bars for SMA(200).
    for i in range(204, len(available)):
        prefix = available.iloc[:i + 1]
        result = calculate_hourly_donchian_breakout(prefix)

        if result is None:
            continue

        # Only accept signals occurring on the requested date.
        result_date = pd.Timestamp(result["Date"]).date()

        if (
            result_date == signal_date
            and result["Pass"]
        ):
            passed = result

    return passed



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
                threads=False,
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
# SMART BREAKOUT DRAWDOWN OPTIMIZER
# ============================================================

def _optimizer_atr14(data):
    high=pd.to_numeric(data["High"],errors="coerce")
    low=pd.to_numeric(data["Low"],errors="coerce")
    close=pd.to_numeric(data["Close"],errors="coerce")
    prev_close=close.shift(1)

    tr=pd.concat(
        [
            high-low,
            (high-prev_close).abs(),
            (low-prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(14).mean()


def _optimizer_rsi9(close):
    return calculate_rsi_wilder(
        pd.to_numeric(close,errors="coerce"),
        9
    )


def _optimizer_trade(
    indicators,
    signal_index,
    holding_days,
    entry_buffer
):
    """
    Same entry/SL/2R/3R framework as the app, except that
    entry_buffer is configurable for the optimizer.
    """

    if signal_index >= len(indicators)-2:
        return None

    hist=indicators.iloc[:signal_index+1]

    plan=calculate_trade_plan(hist)

    if plan is None:
        return None

    close=float(hist.iloc[-1]["Close"])
    entry=close*(1.0+entry_buffer/100.0)

    # Preserve the app's existing stop calculation.
    stop=float(plan["Stop Loss"])

    if not stop < entry:
        return None

    risk=entry-stop

    if risk<=0:
        return None

    target1=entry+2*risk
    target2=entry+3*risk

    future=indicators.iloc[
        signal_index+1:
        min(
            signal_index+1+holding_days,
            len(indicators)
        )
    ]

    if future.empty:
        return None

    entry_date=None
    entry_pos=None

    for j,(_,candle) in enumerate(
        future.iterrows()
    ):

        if float(candle["High"])>=entry:
            entry_date=candle.name
            entry_pos=j
            break

    if entry_pos is None:
        return None

    post=future.iloc[entry_pos:]

    exit_date=post.index[-1]
    exit_price=float(
        post.iloc[-1]["Close"]
    )
    outcome="Time Exit"
    r_multiple=(
        exit_price-entry
    )/risk

    for _,candle in post.iterrows():

        day_low=float(candle["Low"])
        day_high=float(candle["High"])

        # Conservative same-candle rule:
        # stop is considered first.
        if day_low<=stop:

            exit_price=stop
            exit_date=candle.name
            outcome="Stop Loss"
            r_multiple=-1.0
            break

        if day_high>=target2:

            exit_price=target2
            exit_date=candle.name
            outcome="Target 2"
            r_multiple=3.0
            break

        if day_high>=target1:

            exit_price=target1
            exit_date=candle.name
            outcome="Target 1"
            r_multiple=2.0
            break

    return {
        "Signal Date":hist.index[-1],
        "Entry Date":entry_date,
        "Exit Date":exit_date,
        "Entry":entry,
        "Stop Loss":stop,
        "Target 1":target1,
        "Target 2":target2,
        "Outcome":outcome,
        "R":r_multiple
    }


def run_smart_breakout_optimizer(
    historical,
    score_values,
    rsi_min_values,
    rsi_max_values,
    volume_ratio_values,
    atr_pct_values,
    entry_buffer_values,
    require_sma200_rising,
    holding_days
):
    """
    Runs the Smart Breakout strategy with optional drawdown-
    reduction filters.

    The original Smart Breakout conditions are always preserved.
    The optimizer only adds the user-selected filters.

    Ranking is based on:
      - Calmar proxy
      - Maximum drawdown
      - Profit Factor
      - Number of trades

    Equity model:
      1R = 1% of current equity.
    """

    prepared={}

    for symbol,data in historical.items():

        if data is None or data.empty:
            continue

        df=data.copy()

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns=df.columns.get_level_values(0)

        required=[
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

        df=df.dropna(
            subset=required
        ).copy()

        if len(df)<220:
            continue

        ind=calculate_indicators(df)

        ind["OPT_RSI9"]=_optimizer_rsi9(
            ind["Close"]
        )

        ind["OPT_ATR14"]=_optimizer_atr14(
            ind
        )

        ind["OPT_ATR_PCT"]=(
            ind["OPT_ATR14"]
            / ind["Close"]
            *100
        )

        ind["OPT_SMA200_SLOPE20"]=(
            ind["SMA200"]
            >
            ind["SMA200"].shift(20)
        )

        prepared[symbol]=ind

    results=[]

    for score_min in score_values:
        for rsi_min in rsi_min_values:
            for rsi_max in rsi_max_values:
                for vol_min in volume_ratio_values:
                    for atr_max in atr_pct_values:
                        for entry_buffer in entry_buffer_values:

                            trade_list=[]

                            for symbol,ind in prepared.items():

                                next_allowed=0

                                for i in range(
                                    210,
                                    len(ind)-holding_days-1
                                ):

                                    if i<next_allowed:
                                        continue

                                    hist=ind.iloc[:i+1]

                                    breakout=stage_two_analysis(
                                        hist
                                    )

                                    if (
                                        breakout is None
                                        or float(
                                            breakout["Score"]
                                        )<float(score_min)
                                    ):
                                        continue

                                    latest=hist.iloc[-1]

                                    rsi=float(
                                        latest["OPT_RSI9"]
                                    )

                                    atr_pct=float(
                                        latest["OPT_ATR_PCT"]
                                    )

                                    vol_ratio=float(
                                        latest["VOLUME_RATIO"]
                                    )

                                    if pd.isna(rsi):
                                        continue

                                    if pd.isna(atr_pct):
                                        continue

                                    if pd.isna(vol_ratio):
                                        continue

                                    # Optional RSI window.
                                    if not (
                                        rsi>=float(rsi_min)
                                        and rsi<=float(rsi_max)
                                    ):
                                        continue

                                    # Volume confirmation.
                                    if vol_ratio<float(
                                        vol_min
                                    ):
                                        continue

                                    # Volatility filter.
                                    if atr_pct>float(
                                        atr_max
                                    ):
                                        continue

                                    # Rising SMA200 filter.
                                    if require_sma200_rising:

                                        if not bool(
                                            latest[
                                                "OPT_SMA200_SLOPE20"
                                            ]
                                        ):
                                            continue

                                    trade=_optimizer_trade(
                                        ind,
                                        i,
                                        holding_days,
                                        entry_buffer
                                    )

                                    if trade is None:
                                        continue

                                    trade["Stock"]=symbol
                                    trade["Score"]=float(
                                        breakout["Score"]
                                    )
                                    trade_list.append(
                                        trade
                                    )

                                    # Same non-overlap behaviour
                                    # as the main backtester.
                                    try:
                                        exit_pos=ind.index.get_loc(
                                            trade["Exit Date"]
                                        )
                                        next_allowed=(
                                            exit_pos+1
                                        )
                                    except Exception:
                                        next_allowed=i+1

                            n=len(trade_list)

                            if n==0:
                                continue

                            r_values=np.array(
                                [
                                    float(t["R"])
                                    for t in trade_list
                                ],
                                dtype=float
                            )

                            wins=r_values[
                                r_values>0
                            ]

                            losses=r_values[
                                r_values<0
                            ]

                            win_rate=(
                                len(wins)/n*100
                            )

                            gross_profit=(
                                wins.sum()
                                if len(wins)
                                else 0.0
                            )

                            gross_loss=abs(
                                losses.sum()
                            )

                            profit_factor=(
                                gross_profit/gross_loss
                                if gross_loss>0
                                else np.inf
                            )

                            # 1R = 1% of current equity.
                            equity=100000.0
                            peak=equity
                            max_dd=0.0

                            equity_curve=[]

                            for r_mult in r_values:

                                equity*=(
                                    1.0
                                    +0.01*r_mult
                                )

                                peak=max(
                                    peak,
                                    equity
                                )

                                dd=(
                                    peak-equity
                                )/peak*100

                                max_dd=max(
                                    max_dd,
                                    dd
                                )

                                equity_curve.append(
                                    equity
                                )

                            total_return=(
                                equity/100000.0-1
                            )*100

                            start=min(
                                t["Entry Date"]
                                for t in trade_list
                            )

                            end=max(
                                t["Exit Date"]
                                for t in trade_list
                            )

                            days=max(
                                1,
                                (
                                    pd.Timestamp(end)
                                    -
                                    pd.Timestamp(start)
                                ).days
                            )

                            years=days/365.25

                            cagr=(
                                (
                                    equity/100000.0
                                )**(
                                    1/max(
                                        years,
                                        0.01
                                    )
                                )-1
                            )*100

                            calmar=(
                                cagr/max_dd
                                if max_dd>0
                                else np.nan
                            )

                            results.append(
                                {
                                    "Score Min":
                                        score_min,
                                    "RSI Min":
                                        rsi_min,
                                    "RSI Max":
                                        rsi_max,
                                    "Volume Ratio Min":
                                        vol_min,
                                    "ATR % Max":
                                        atr_max,
                                    "Entry Buffer %":
                                        entry_buffer,
                                    "SMA200 Rising":
                                        (
                                            "Yes"
                                            if require_sma200_rising
                                            else "No"
                                        ),
                                    "Trades":
                                        n,
                                    "Win Rate %":
                                        round(
                                            win_rate,
                                            2
                                        ),
                                    "Profit Factor":
                                        round(
                                            profit_factor,
                                            2
                                        )
                                        if np.isfinite(
                                            profit_factor
                                        )
                                        else np.inf,
                                    "Net R":
                                        round(
                                            r_values.sum(),
                                            2
                                        ),
                                    "Total Return %":
                                        round(
                                            total_return,
                                            2
                                        ),
                                    "CAGR %":
                                        round(
                                            cagr,
                                            2
                                        ),
                                    "Max Drawdown %":
                                        round(
                                            max_dd,
                                            2
                                        ),
                                    "Calmar":
                                        round(
                                            calmar,
                                            2
                                        )
                                        if not pd.isna(
                                            calmar
                                        )
                                        else np.nan
                                }
                            )

    return pd.DataFrame(results)



# ============================================================
# CHART PATTERN RECOGNITION ENGINE
# ============================================================

def _pattern_clean_ohlcv(data):
    if data is None or data.empty:
        return None

    df=data.copy()

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    required=["Open","High","Low","Close","Volume"]

    if any(c not in df.columns for c in required):
        return None

    df=df.dropna(subset=required).copy()

    if len(df)<40:
        return None

    for c in required:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    return df.dropna(subset=required)


def _find_pattern_pivots(df, window=3):
    high=df["High"].values
    low=df["Low"].values

    highs=[]
    lows=[]

    for i in range(window,len(df)-window):

        h=high[i]
        l=low[i]

        if h>=max(high[i-window:i+window+1]):
            highs.append((i,float(h)))

        if l<=min(low[i-window:i+window+1]):
            lows.append((i,float(l)))

    return highs,lows


def _nearest_pivots(pivots, start, end, kind="high"):
    return [
        p for p in pivots
        if start<=p[0]<=end
    ]


def _pattern_trade_plan(df, pattern, level, direction):
    close=float(df["Close"].iloc[-1])
    atr_series=_optimizer_atr14(df)
    atr=float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else close*0.02

    if direction=="Bullish":
        entry=max(close,level)*1.0025
        stop=min(
            level-1.2*atr,
            float(df["Low"].tail(10).min())
        )
        if stop>=entry:
            stop=entry-1.5*atr

        risk=entry-stop

        return {
            "Entry":entry,
            "Stop Loss":stop,
            "Target 1":entry+2*risk,
            "Target 2":entry+3*risk,
            "Risk":risk
        }

    entry=min(close,level)*0.9975
    stop=max(
        level+1.2*atr,
        float(df["High"].tail(10).max())
    )

    if stop<=entry:
        stop=entry+1.5*atr

    risk=stop-entry

    return {
        "Entry":entry,
        "Stop Loss":stop,
        "Target 1":entry-2*risk,
        "Target 2":entry-3*risk,
        "Risk":risk
    }


def _pattern_result(
    name,
    stage,
    direction,
    confidence,
    breakout_level,
    start_index,
    end_index,
    df,
    details=None
):
    current=float(df["Close"].iloc[-1])
    level=float(breakout_level)

    if direction=="Bullish":
        breakout=current>level
        distance=(current-level)/level*100
    else:
        breakout=current<level
        distance=(level-current)/level*100

    # Avoid calling a pattern "confirmed" when price is only
    # marginally through the level.
    confirmed=bool(
        breakout and distance>=0.25
    )

    if confirmed:
        stage="CONFIRMED BREAKOUT"

    elif breakout:
        stage="BREAKOUT"

    elif abs(distance)<=2.0:
        stage="NEAR BREAKOUT"

    else:
        stage="FORMING"

    volume_avg=float(
        df["Volume"].rolling(20).mean().iloc[-1]
    )

    volume_ratio=(
        float(df["Volume"].iloc[-1])/volume_avg
        if volume_avg>0
        else 1.0
    )

    if volume_ratio>=1.5:
        confidence+=8
    elif volume_ratio<0.7:
        confidence-=5

    confidence=int(max(0,min(99,confidence)))

    trade=_pattern_trade_plan(
        df,
        name,
        level,
        direction
    )

    return {
        "Pattern":name,
        "Stage":stage,
        "Direction":direction,
        "Confidence":confidence,
        "Breakout Level":level,
        "Current Price":current,
        "Distance %":distance,
        "Start Bar":start_index,
        "End Bar":end_index,
        "Entry":trade["Entry"],
        "Stop Loss":trade["Stop Loss"],
        "Target 1":trade["Target 1"],
        "Target 2":trade["Target 2"],
        "Risk":trade["Risk"],
        "Volume Ratio":volume_ratio,
        "Details":details or ""
    }


def _detect_head_shoulders(df, inverse=False):
    highs,lows=_find_pattern_pivots(df,3)

    pivots=sorted(
        [(i,p,"H") for i,p in highs]+
        [(i,p,"L") for i,p in lows],
        key=lambda x:x[0]
    )

    if len(pivots)<5:
        return None

    candidates=[]

    for j in range(len(pivots)-4):

        p=pivots[j:j+5]

        if [x[2] for x in p] != (
            ["L","H","L","H","L"]
            if inverse
            else ["H","L","H","L","H"]
        ):
            continue

        a,b,c,d,e=[x[1] for x in p]

        if not inverse:
            # H&S: middle high is the head.
            if not (c>a and c>e):
                continue

            shoulder_similarity=abs(a-e)/c
            if shoulder_similarity>0.12:
                continue

            neck_similarity=abs(b-d)/c
            if neck_similarity>0.10:
                continue

            neckline=(b+d)/2

            # Head should be meaningfully above shoulders.
            if c<=max(a,e)*1.03:
                continue

            confidence=65
            confidence+=int(
                max(
                    0,
                    12*(1-shoulder_similarity/0.12)
                )
            )

            return _pattern_result(
                "Head & Shoulders",
                "FORMING",
                "Bearish",
                confidence,
                neckline,
                p[0][0],
                p[-1][0],
                df,
                "Head higher than both shoulders; neckline formed by two troughs."
            )

        else:
            # Inverse H&S: middle low is the head.
            if not (c<a and c<e):
                continue

            shoulder_similarity=abs(a-e)/abs(c)
            if shoulder_similarity>0.12:
                continue

            neck_similarity=abs(b-d)/abs(c)
            if neck_similarity>0.10:
                continue

            neckline=(b+d)/2

            if c>=min(a,e)*0.97:
                continue

            confidence=65
            confidence+=int(
                max(
                    0,
                    12*(1-shoulder_similarity/0.12)
                )
            )

            return _pattern_result(
                "Inverse Head & Shoulders",
                "FORMING",
                "Bullish",
                confidence,
                neckline,
                p[0][0],
                p[-1][0],
                df,
                "Head lower than both shoulders; neckline formed by two peaks."
            )

    return None


def _detect_double_top_bottom(df, bottom=False):
    highs,lows=_find_pattern_pivots(df,3)

    pivots=sorted(
        [(i,p,"H") for i,p in highs]+
        [(i,p,"L") for i,p in lows],
        key=lambda x:x[0]
    )

    wanted=["L","H","L"] if bottom else ["H","L","H"]

    for j in range(len(pivots)-2):

        p=pivots[j:j+3]

        if [x[2] for x in p]!=wanted:
            continue

        a,b,c=[x[1] for x in p]

        if bottom:
            if abs(a-c)/max(a,c)>0.04:
                continue

            if b>=min(a,c)*1.03:
                level=b
                return _pattern_result(
                    "Double Bottom",
                    "FORMING",
                    "Bullish",
                    70,
                    level,
                    p[0][0],
                    p[-1][0],
                    df,
                    "Two comparable lows separated by a rebound."
                )
        else:
            if abs(a-c)/max(a,c)>0.04:
                continue

            if b<=max(a,c)*0.97:
                level=b
                return _pattern_result(
                    "Double Top",
                    "FORMING",
                    "Bearish",
                    70,
                    level,
                    p[0][0],
                    p[-1][0],
                    df,
                    "Two comparable highs separated by a decline."
                )

    return None


def _detect_cup_handle(df):
    # Use the most recent 30–160 bars and test a rounded cup
    # followed by a relatively shallow handle.
    n=len(df)
    close=df["Close"].values

    for length in range(
        min(160,n-1),
        39,
        -5
    ):

        start=n-length-1
        segment=df.iloc[start:n]

        left=float(segment["High"].iloc[:max(5,length//5)].max())
        right=float(segment["High"].iloc[-max(8,length//5):].max())

        cup_bottom=float(
            segment["Low"].iloc[length//4:3*length//4].min()
        )

        if left<=0 or right<=0:
            continue

        rim_similarity=abs(left-right)/max(left,right)

        if rim_similarity>0.10:
            continue

        if cup_bottom>=min(left,right)*0.92:
            continue

        # Require the bottom to be away from both rims.
        bottom_pos=int(
            segment["Low"].values.argmin()
        )

        if not (
            length*0.20
            <=bottom_pos
            <=length*0.80
        ):
            continue

        # Handle is the most recent 10–25% of the pattern.
        handle_len=max(
            5,
            min(
                25,
                length//5
            )
        )

        handle=segment.iloc[-handle_len:]

        handle_low=float(handle["Low"].min())

        if handle_low<right*0.90:
            continue

        neckline=min(left,right)

        current=float(df["Close"].iloc[-1])

        confidence=68

        confidence+=int(
            max(
                0,
                10*(1-rim_similarity/0.10)
            )
        )

        if current>neckline:
            confidence+=10

        return _pattern_result(
            "Cup & Handle",
            "FORMING",
            "Bullish",
            confidence,
            neckline,
            start,
            n-1,
            df,
            "Rounded cup with comparable rims followed by a shallow handle."
        )

    return None


def _detect_triangles(df):
    highs,lows=_find_pattern_pivots(df,3)

    recent_highs=[
        p for p in highs
        if p[0]>=max(0,len(df)-70)
    ]

    recent_lows=[
        p for p in lows
        if p[0]>=max(0,len(df)-70)
    ]

    if len(recent_highs)<3 or len(recent_lows)<3:
        return None

    hs=recent_highs[-4:]
    ls=recent_lows[-4:]

    xh=np.array([p[0] for p in hs],dtype=float)
    yh=np.array([p[1] for p in hs],dtype=float)

    xl=np.array([p[0] for p in ls],dtype=float)
    yl=np.array([p[1] for p in ls],dtype=float)

    hslope=np.polyfit(xh,yh,1)[0]
    lslope=np.polyfit(xl,yl,1)[0]

    avg_price=float(df["Close"].iloc[-1])

    # Normalize slopes by price per bar.
    hs_norm=hslope/avg_price
    ls_norm=lslope/avg_price

    if abs(hs_norm)<0.0015 and ls_norm>0.0003:
        return _pattern_result(
            "Ascending Triangle",
            "FORMING",
            "Bullish",
            72,
            max(yh),
            hs[0][0],
            len(df)-1,
            df,
            "Flat/resistant highs with rising lows."
        )

    if hs_norm<-0.0003 and abs(ls_norm)<0.0015:
        return _pattern_result(
            "Descending Triangle",
            "FORMING",
            "Bearish",
            72,
            min(yl),
            hs[0][0],
            len(df)-1,
            df,
            "Falling highs with relatively flat support."
        )

    if hs_norm<-0.0002 and ls_norm>0.0002:
        level=(max(yh)+min(yl))/2

        return _pattern_result(
            "Symmetrical Triangle",
            "FORMING",
            "Neutral",
            68,
            level,
            min(
                hs[0][0],
                ls[0][0]
            ),
            len(df)-1,
            df,
            "Converging falling highs and rising lows."
        )

    return None


def detect_chart_patterns(data):
    """
    Detect the supported classical chart patterns.

    Returns a list of the strongest recent pattern candidates.
    This is a quantitative heuristic detector, not an image/
    computer-vision classifier.
    """

    df=_pattern_clean_ohlcv(data)

    if df is None:
        return []

    candidates=[]

    detectors=[
        lambda x:_detect_head_shoulders(x,False),
        lambda x:_detect_head_shoulders(x,True),
        lambda x:_detect_double_top_bottom(x,False),
        lambda x:_detect_double_top_bottom(x,True),
        _detect_cup_handle,
        _detect_triangles
    ]

    for detector in detectors:

        try:
            result=detector(df)

            if result is not None:
                candidates.append(result)

        except Exception:
            continue

    # Remove duplicate pattern names and keep highest confidence.
    unique={}

    for item in candidates:

        name=item["Pattern"]

        if (
            name not in unique
            or item["Confidence"]>unique[name]["Confidence"]
        ):
            unique[name]=item

    return sorted(
        unique.values(),
        key=lambda x:(
            x["Confidence"],
            1 if x["Stage"]=="CONFIRMED BREAKOUT" else 0
        ),
        reverse=True
    )


# ============================================================
# CCI + EMA9/21/200 + RSI9/WMA21 STRATEGY
# ============================================================

def calculate_cci(data, period=20):
    high=pd.to_numeric(data["High"],errors="coerce")
    low=pd.to_numeric(data["Low"],errors="coerce")
    close=pd.to_numeric(data["Close"],errors="coerce")

    typical=(high+low+close)/3.0
    sma=typical.rolling(period).mean()

    mean_dev=typical.rolling(period).apply(
        lambda x: np.mean(np.abs(x-np.mean(x))),
        raw=True
    )

    return (typical-sma)/(0.015*mean_dev.replace(0,np.nan))


def calculate_wma(series, period=21):
    weights=np.arange(1,period+1)
    denominator=weights.sum()

    return pd.to_numeric(
        series,
        errors="coerce"
    ).rolling(period).apply(
        lambda x: np.dot(x,weights)/denominator,
        raw=True
    )


def prepare_cci_ema_rsi_strategy(data):
    df=data.copy()

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    required=["Open","High","Low","Close","Volume"]

    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df=df.dropna(subset=required).copy()

    for c in required:
        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df["EMA9"]=df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"]=df["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["EMA200"]=df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["CCI20"]=calculate_cci(
        df,
        20
    )

    df["RSI9"]=calculate_rsi_wilder(
        df["Close"],
        9
    )

    df["RSI9_WMA21"]=calculate_wma(
        df["RSI9"],
        21
    )

    return df.dropna(
        subset=[
            "EMA9",
            "EMA21",
            "EMA200",
            "CCI20",
            "RSI9",
            "RSI9_WMA21"
        ]
    )


def cci_ema_rsi_entry_condition(
    row,
    ema200_near_pct=2.0
):
    """
    NEW ENTRY RULES — ALL 5 CONDITIONS MUST PASS

    1. EMA9 and EMA21 must both be ABOVE EMA200.
       Neither EMA is allowed below EMA200.
       Both must also be within the configured upper
       distance from EMA200. Default = 2%.

    2. Closing price > EMA200.

    3. Daily CCI(20) > 100.

    4. RSI(9) > 60 and RSI(9) < 70.

    5. RSI(9) > WMA(21) of RSI(9).
    """

    ema200=float(row["EMA200"])

    if ema200<=0:
        return False

    ema9=float(row["EMA9"])
    ema21=float(row["EMA21"])
    close=float(row["Close"])

    ema9_distance=(
        (ema9-ema200)/ema200*100.0
    )

    ema21_distance=(
        (ema21-ema200)/ema200*100.0
    )

    ema_position_ok=(
        ema9>ema200
        and
        ema21>ema200
        and
        0.0<=ema9_distance<=ema200_near_pct
        and
        0.0<=ema21_distance<=ema200_near_pct
    )

    return bool(
        ema_position_ok
        and close>ema200
        and float(row["CCI20"])>100.0
        and float(row["RSI9"])>60.0
        and float(row["RSI9"])<70.0
        and float(row["RSI9"])>float(row["RSI9_WMA21"])
    )


def cci_ema_rsi_exit_condition(row):
    """
    NEW EXIT RULES — AT LEAST 2 OF 3 CONDITIONS MUST PASS

    X1. EMA9 crosses EMA200 from above in a downward direction.
        Previous EMA9 >= previous EMA200 AND
        current EMA9 < current EMA200.

    X2. WMA(21) of RSI(9) > RSI(9).

    X3. CCI(20) < 100.

    EXIT when the number of satisfied conditions >= 2.
    """

    x1=(
        float(row["EMA9"])<float(row["EMA200"])
        and
        float(row["EMA9_PREV"])>=float(row["EMA200_PREV"])
    )

    x2=(
        float(row["RSI9_WMA21"])
        >
        float(row["RSI9"])
    )

    x3=(
        float(row["CCI20"])<100.0
    )

    exit_count=int(x1)+int(x2)+int(x3)

    return bool(exit_count>=2)




def add_cci_ema_rsi_legacy_conditions(
    df,
    ema200_near_pct=3.0,
    rsi_wma_50_tolerance=2.0
):
    """
    Previous-style baseline used only for comparison.

    Entry:
      - EMA9 and EMA21 are near EMA200 (absolute distance)
      - Close > EMA200
      - CCI20 > 100
      - RSI9 between 60 and 70
      - RSI9 > WMA21

    Exit:
      - EMA9 and EMA21 both slope down
      OR
      - WMA21 > RSI9 and both RSI9/WMA21 are around 50.

    This function is isolated from the new strategy.
    """

    out=df.copy()

    out["EMA9_PREV"]=out["EMA9"].shift(1)
    out["EMA21_PREV"]=out["EMA21"].shift(1)
    out["EMA200_PREV"]=out["EMA200"].shift(1)

    ema9_distance=(
        (out["EMA9"]-out["EMA200"])
        /out["EMA200"].abs()*100.0
    )

    ema21_distance=(
        (out["EMA21"]-out["EMA200"])
        /out["EMA200"].abs()*100.0
    )

    out["LEGACY_ENTRY_SIGNAL"]=(
        (ema9_distance.abs()<=ema200_near_pct)
        &
        (ema21_distance.abs()<=ema200_near_pct)
        &
        (out["Close"]>out["EMA200"])
        &
        (out["CCI20"]>100.0)
        &
        (out["RSI9"]>60.0)
        &
        (out["RSI9"]<70.0)
        &
        (out["RSI9"]>out["RSI9_WMA21"])
    )

    ema9_down=(
        out["EMA9"]<out["EMA9_PREV"]
    )

    ema21_down=(
        out["EMA21"]<out["EMA21_PREV"]
    )

    around_50=(
        (out["RSI9"]-50.0).abs()
        <=rsi_wma_50_tolerance
    ) & (
        (out["RSI9_WMA21"]-50.0).abs()
        <=rsi_wma_50_tolerance
    )

    out["LEGACY_EXIT_SIGNAL"]=(
        (ema9_down & ema21_down)
        |
        (
            (out["RSI9_WMA21"]>out["RSI9"])
            & around_50
        )
    )

    return out


def add_cci_ema_rsi_conditions(
    df,
    ema200_near_pct=2.0,
    rsi_wma_50_tolerance=None
):
    out=df.copy()

    out["EMA9_PREV"]=out["EMA9"].shift(1)
    out["EMA21_PREV"]=out["EMA21"].shift(1)
    out["EMA200_PREV"]=out["EMA200"].shift(1)

    out["ENTRY_SIGNAL"]=out.apply(
        lambda r:cci_ema_rsi_entry_condition(
            r,
            ema200_near_pct
        ),
        axis=1
    )

    # Individual exit conditions are exposed for transparency.
    out["EXIT_X1_EMA9_CROSS_200"]=(
        (out["EMA9"]<out["EMA200"])
        &
        (out["EMA9_PREV"]>=out["EMA200_PREV"])
    )

    out["EXIT_X2_WMA_ABOVE_RSI"]=(
        out["RSI9_WMA21"]>out["RSI9"]
    )

    out["EXIT_X3_CCI_BELOW_100"]=(
        out["CCI20"]<100.0
    )

    out["EXIT_CONDITION_COUNT"]=(
        out["EXIT_X1_EMA9_CROSS_200"].astype(int)
        +
        out["EXIT_X2_WMA_ABOVE_RSI"].astype(int)
        +
        out["EXIT_X3_CCI_BELOW_100"].astype(int)
    )

    out["EXIT_SIGNAL"]=(
        out["EXIT_CONDITION_COUNT"]>=2
    )

    out["EMA9_DISTANCE_EMA200_%"]=(
        (out["EMA9"]-out["EMA200"])
        /out["EMA200"].abs()*100
    )

    out["EMA21_DISTANCE_EMA200_%"]=(
        (out["EMA21"]-out["EMA200"])
        /out["EMA200"].abs()*100
    )

    return out



def backtest_cci_ema_rsi_strategy(
    data,
    ema200_near_pct=2.0,
    rsi_wma_50_tolerance=None,
    max_holding_days=120,
    max_loss_pct=20.0,
    trailing_enabled=False,
    trail_activation_pct=10.0,
    trailing_stop_pct=10.0,
    profit_booking_enabled=False,
    target1_pct=15.0,
    target1_booking_pct=25.0,
    target2_pct=25.0,
    target2_booking_pct=25.0,
    improved_exit_enabled=False,
    improved_exit_activation_pct=5.0,
    ema200_breakdown_exit_enabled=False,
    strategy_mode="new"
):
    """
    Strategy-specific backtest.

    Entry is unchanged.

    Risk/profit management:
      - Hard maximum-loss stop from entry.
      - Optional partial profit booking:
          Target 1 = +15%, book 25%.
          Target 2 = +25%, book another 25%.
        The remaining 50% is allowed to run.
      - Optional trailing stop on the remaining position.
      - Optional improved/profit-protection exit:
          after the trade reaches the configured profit threshold,
          exit on EMA9 < EMA21 OR RSI9 < WMA21.
      - Original technical exit remains active.
      - Stop is checked before intraday profit targets when both
        appear in the same daily candle (conservative assumption).
      - Trailing stop uses the peak known before the current candle
        to avoid look-ahead bias.
    """

    df=prepare_cci_ema_rsi_strategy(data)

    if df.empty:
        return {"Trades":[],"Data":df}

    if strategy_mode=="legacy":
        df=add_cci_ema_rsi_legacy_conditions(
            df,
            ema200_near_pct,
            (
                rsi_wma_50_tolerance
                if rsi_wma_50_tolerance is not None
                else 2.0
            )
        )

        entry_signal_column="LEGACY_ENTRY_SIGNAL"
        exit_signal_column="LEGACY_EXIT_SIGNAL"

    else:
        df=add_cci_ema_rsi_conditions(
            df,
            ema200_near_pct,
            rsi_wma_50_tolerance
        )

        entry_signal_column="ENTRY_SIGNAL"
        exit_signal_column="EXIT_SIGNAL"

    df["EMA9_PREV"]=df["EMA9"].shift(1)
    df["EMA21_PREV"]=df["EMA21"].shift(1)
    df["EMA200_PREV"]=df["EMA200"].shift(1)

    trades=[]
    in_position=False
    entry_pos=None
    entry_price=None
    entry_date=None
    peak_price=None
    trailing_active=False

    remaining_qty=1.0
    realized_pnl=0.0
    target1_booked=False
    target2_booked=False
    profit_booked_pct=0.0
    profit_booking_events=[]

    i=1

    def reset_position():
        return (
            False,None,None,None,None,False,
            1.0,0.0,False,False,0.0,[]
        )

    while i<len(df)-1:

        row=df.iloc[i]

        if not in_position:

            if bool(row[entry_signal_column]):

                entry_pos=i+1

                if entry_pos>=len(df):
                    break

                entry_price=float(
                    df.iloc[entry_pos]["Open"]
                )
                entry_date=df.index[entry_pos]

                in_position=True
                peak_price=entry_price
                trailing_active=False

                remaining_qty=1.0
                realized_pnl=0.0
                target1_booked=False
                target2_booked=False
                profit_booked_pct=0.0
                profit_booking_events=[]

                i=entry_pos+1
                continue

        else:

            day_open=float(row["Open"])
            day_low=float(row["Low"])
            day_high=float(row["High"])

            # ------------------------------------------------
            # 1. HARD MAX-LOSS STOP
            # ------------------------------------------------
            hard_stop=(
                entry_price*
                (1.0-max_loss_pct/100.0)
            )

            # ------------------------------------------------
            # 2. TRAILING STOP
            # Peak is the peak known BEFORE today's candle.
            # ------------------------------------------------
            if (
                trailing_enabled
                and not trailing_active
                and peak_price>=(
                    entry_price*
                    (1.0+trail_activation_pct/100.0)
                )
            ):
                trailing_active=True

            trailing_stop=None

            if trailing_enabled and trailing_active:
                trailing_stop=(
                    peak_price*
                    (1.0-trailing_stop_pct/100.0)
                )

            active_stop=hard_stop

            if trailing_stop is not None:
                active_stop=max(
                    hard_stop,
                    trailing_stop
                )

            # ------------------------------------------------
            # 3. STOP FIRST — CONSERVATIVE SAME-CANDLE RULE
            # ------------------------------------------------
            stop_reason=None

            if day_open<=active_stop:
                stop_price=day_open
                stop_reason=(
                    "Maximum Loss Stop"
                    if active_stop==hard_stop
                    else "Trailing Stop"
                )

            elif day_low<=active_stop:
                stop_price=active_stop
                stop_reason=(
                    "Maximum Loss Stop"
                    if active_stop==hard_stop
                    else "Trailing Stop"
                )

            else:
                stop_price=None

            if stop_price is not None:

                # Remaining position exits at stop.
                realized_pnl += (
                    (stop_price-entry_price)
                    /entry_price
                    *remaining_qty
                )

                exit_date=df.index[i]

                trades.append(
                    {
                        "Entry Date":entry_date,
                        "Entry":entry_price,
                        "Exit Date":exit_date,
                        "Exit":stop_price,
                        "P&L %":realized_pnl*100,
                        "Holding Days":(
                            pd.Timestamp(exit_date)-
                            pd.Timestamp(entry_date)
                        ).days,
                        "Exit Reason":stop_reason,
                        "Peak Price":peak_price,
                        "Trailing Active":trailing_active,
                        "Profit Booked %":profit_booked_pct,
                        "Profit Booking Events":
                            "; ".join(profit_booking_events)
                    }
                )

                (
                    in_position,entry_pos,entry_price,entry_date,
                    peak_price,trailing_active,remaining_qty,
                    realized_pnl,target1_booked,target2_booked,
                    profit_booked_pct,profit_booking_events
                )=reset_position()

                i+=1
                continue

            # ------------------------------------------------
            # 4. PARTIAL PROFIT BOOKING
            #
            # Targets are evaluated using today's high.
            # If the stop was not hit, targets can be booked.
            # ------------------------------------------------
            if profit_booking_enabled:

                target1_price=(
                    entry_price*
                    (1.0+target1_pct/100.0)
                )

                target2_price=(
                    entry_price*
                    (1.0+target2_pct/100.0)
                )

                if (
                    not target1_booked
                    and remaining_qty>0
                    and day_high>=target1_price
                ):

                    qty=min(
                        target1_booking_pct/100.0,
                        remaining_qty
                    )

                    realized_pnl += (
                        (target1_price-entry_price)
                        /entry_price
                        *qty
                    )

                    remaining_qty-=qty
                    target1_booked=True
                    profit_booked_pct+=qty*100

                    profit_booking_events.append(
                        f"T1 +{target1_pct:.0f}%: "
                        f"booked {qty*100:.0f}%"
                    )

                if (
                    not target2_booked
                    and remaining_qty>0
                    and day_high>=target2_price
                ):

                    qty=min(
                        target2_booking_pct/100.0,
                        remaining_qty
                    )

                    realized_pnl += (
                        (target2_price-entry_price)
                        /entry_price
                        *qty
                    )

                    remaining_qty-=qty
                    target2_booked=True
                    profit_booked_pct+=qty*100

                    profit_booking_events.append(
                        f"T2 +{target2_pct:.0f}%: "
                        f"booked {qty*100:.0f}%"
                    )

            # ------------------------------------------------
            # 5. NEW EMA200 BREAKDOWN EXIT
            #
            # User-specified condition:
            #   1. WMA21 > RSI9
            #   2. Either EMA9 OR EMA21 has crossed below EMA200
            #   3. Close is below EMA200
            #   4. EMA9 and EMA21 are both sloping down
            #
            # "Crossed below" is interpreted strictly as:
            # yesterday's EMA >= EMA200 and today's EMA < EMA200.
            # ------------------------------------------------
            wma_rsi_exit = (
                float(row["RSI9_WMA21"])
                > float(row["RSI9"])
            )

            ema9_crossed_below_200 = (
                float(row["EMA9"]) < float(row["EMA200"])
                and
                float(row["EMA9_PREV"]) >= float(row["EMA200_PREV"])
            )

            ema21_crossed_below_200 = (
                float(row["EMA21"]) < float(row["EMA200"])
                and
                float(row["EMA21_PREV"]) >= float(row["EMA200_PREV"])
            )

            price_below_200 = (
                float(row["Close"]) < float(row["EMA200"])
            )

            both_short_emas_down = (
                float(row["EMA9"]) < float(row["EMA9_PREV"])
                and
                float(row["EMA21"]) < float(row["EMA21_PREV"])
            )

            ema200_breakdown_exit = (
                ema200_breakdown_exit_enabled
                and
                wma_rsi_exit
                and
                (
                    ema9_crossed_below_200
                    or
                    ema21_crossed_below_200
                )
                and
                price_below_200
                and
                both_short_emas_down
            )

            # ------------------------------------------------
            # 6. IMPROVED EXIT / PROFIT PROTECTION
            #
            # Only applies after the trade has first reached
            # the configured profit threshold. This prevents
            # the improved exit from prematurely closing normal
            # losing trades; the hard stop handles those.
            # ------------------------------------------------
            improved_exit=False

            if improved_exit_enabled:

                profit_threshold=(
                    entry_price*
                    (1.0+improved_exit_activation_pct/100.0)
                )

                reached_profit=(
                    peak_price>=profit_threshold
                    or
                    day_high>=profit_threshold
                )

                early_deterioration=(
                    float(row["EMA9"])
                    <
                    float(row["EMA21"])
                    or
                    float(row["RSI9"])
                    <
                    float(row["RSI9_WMA21"])
                )

                improved_exit=(
                    reached_profit
                    and early_deterioration
                )

            # ------------------------------------------------
            # 6. ORIGINAL TECHNICAL EXIT
            # ------------------------------------------------
            original_exit=bool(row[exit_signal_column])

            if (
                original_exit
                or ema200_breakdown_exit
                or improved_exit
            ):

                exit_pos=i+1

                if exit_pos>=len(df):
                    exit_pos=len(df)-1

                exit_price=float(
                    df.iloc[exit_pos]["Open"]
                )
                exit_date=df.index[exit_pos]

                # Remaining position exits at next open.
                realized_pnl += (
                    (exit_price-entry_price)
                    /entry_price
                    *remaining_qty
                )

                if ema200_breakdown_exit:
                    exit_reason=(
                        "EMA200 Breakdown Exit"
                    )
                elif improved_exit:
                    exit_reason=(
                        "Improved Profit-Protection Exit"
                    )
                else:
                    exit_reason=(
                        "CCI+EMA+RSI 2-of-3 Exit"
                    )

                trades.append(
                    {
                        "Entry Date":entry_date,
                        "Entry":entry_price,
                        "Exit Date":exit_date,
                        "Exit":exit_price,
                        "P&L %":realized_pnl*100,
                        "Holding Days":(
                            pd.Timestamp(exit_date)-
                            pd.Timestamp(entry_date)
                        ).days,
                        "Exit Reason":exit_reason,
                        "Peak Price":peak_price,
                        "Trailing Active":trailing_active,
                        "Profit Booked %":profit_booked_pct,
                        "Profit Booking Events":
                            "; ".join(profit_booking_events)
                    }
                )

                (
                    in_position,entry_pos,entry_price,entry_date,
                    peak_price,trailing_active,remaining_qty,
                    realized_pnl,target1_booked,target2_booked,
                    profit_booked_pct,profit_booking_events
                )=reset_position()

                i=exit_pos+1
                continue

            # ------------------------------------------------
            # 7. UPDATE PEAK AFTER ALL PRIOR-DAY-BASED STOPS
            # ------------------------------------------------
            peak_price=max(
                peak_price,
                day_high
            )

            if (
                trailing_enabled
                and peak_price>=(
                    entry_price*
                    (1.0+trail_activation_pct/100.0)
                )
            ):
                trailing_active=True

            # ------------------------------------------------
            # 8. MAXIMUM HOLDING PERIOD
            # ------------------------------------------------
            if i-entry_pos>=max_holding_days:

                exit_price=float(
                    df.iloc[i]["Close"]
                )
                exit_date=df.index[i]

                realized_pnl += (
                    (exit_price-entry_price)
                    /entry_price
                    *remaining_qty
                )

                trades.append(
                    {
                        "Entry Date":entry_date,
                        "Entry":entry_price,
                        "Exit Date":exit_date,
                        "Exit":exit_price,
                        "P&L %":realized_pnl*100,
                        "Holding Days":(
                            pd.Timestamp(exit_date)-
                            pd.Timestamp(entry_date)
                        ).days,
                        "Exit Reason":
                            "Maximum holding period",
                        "Peak Price":peak_price,
                        "Trailing Active":trailing_active,
                        "Profit Booked %":profit_booked_pct,
                        "Profit Booking Events":
                            "; ".join(profit_booking_events)
                    }
                )

                (
                    in_position,entry_pos,entry_price,entry_date,
                    peak_price,trailing_active,remaining_qty,
                    realized_pnl,target1_booked,target2_booked,
                    profit_booked_pct,profit_booking_events
                )=reset_position()

        i+=1

    # Close any open position at final close.
    if in_position and entry_date is not None:

        exit_date=df.index[-1]
        exit_price=float(df["Close"].iloc[-1])

        realized_pnl += (
            (exit_price-entry_price)
            /entry_price
            *remaining_qty
        )

        trades.append(
            {
                "Entry Date":entry_date,
                "Entry":entry_price,
                "Exit Date":exit_date,
                "Exit":exit_price,
                "P&L %":realized_pnl*100,
                "Holding Days":(
                    pd.Timestamp(exit_date)-
                    pd.Timestamp(entry_date)
                ).days,
                "Exit Reason":"End of data",
                "Peak Price":peak_price,
                "Trailing Active":trailing_active,
                "Profit Booked %":profit_booked_pct,
                "Profit Booking Events":
                    "; ".join(profit_booking_events)
            }
        )

    return {
        "Trades":trades,
        "Data":df
    }




def compare_cci_ema_rsi_strategies(
    market,
    new_ema_range_pct=2.0,
    legacy_ema_range_pct=3.0,
    legacy_rsi_50_tolerance=2.0,
    max_holding_days=120,
    max_loss_pct=20.0
):
    """
    Run the previous-style and new strategy over the exact same
    market data and risk settings, with trailing/profit booking/
    optional additional exits disabled.

    This isolates the effect of the new entry/exit rules.
    """

    rows=[]
    new_trades=[]
    legacy_trades=[]

    for symbol,data in market.items():

        if data is None or data.empty:
            continue

        new_bt=backtest_cci_ema_rsi_strategy(
            data,
            new_ema_range_pct,
            None,
            max_holding_days,
            max_loss_pct,
            False,
            10.0,
            10.0,
            False,
            15.0,
            25.0,
            25.0,
            25.0,
            False,
            5.0,
            False,
            "new"
        )

        old_bt=backtest_cci_ema_rsi_strategy(
            data,
            legacy_ema_range_pct,
            legacy_rsi_50_tolerance,
            max_holding_days,
            max_loss_pct,
            False,
            10.0,
            10.0,
            False,
            15.0,
            25.0,
            25.0,
            25.0,
            False,
            5.0,
            False,
            "legacy"
        )

        nt=new_bt["Trades"]
        ot=old_bt["Trades"]

        new_trades.extend(
            [
                dict(t,Stock=symbol)
                for t in nt
            ]
        )

        legacy_trades.extend(
            [
                dict(t,Stock=symbol)
                for t in ot
            ]
        )

        ns=summarize_cci_ema_rsi_backtest(nt)
        os=summarize_cci_ema_rsi_backtest(ot)

        rows.append({
            "Stock":symbol,
            "New Trades":ns["Trades"],
            "Previous Trades":os["Trades"],
            "New Return %":ns["Net Return %"],
            "Previous Return %":os["Net Return %"],
            "New Max DD %":ns["Max Drawdown %"],
            "Previous Max DD %":os["Max Drawdown %"],
            "New Win Rate %":ns["Win Rate %"],
            "Previous Win Rate %":os["Win Rate %"],
            "New Profit Factor":ns["Profit Factor"],
            "Previous Profit Factor":os["Profit Factor"]
        })

    new_summary=summarize_cci_ema_rsi_backtest(new_trades)
    old_summary=summarize_cci_ema_rsi_backtest(legacy_trades)

    return {
        "New Summary":new_summary,
        "Previous Summary":old_summary,
        "Stock Comparison":pd.DataFrame(rows),
        "New Trades":new_trades,
        "Previous Trades":legacy_trades
    }


def summarize_cci_ema_rsi_backtest(trades):
    if not trades:
        return {
            "Trades":0,
            "Win Rate %":0,
            "Profit Factor":0,
            "Net Return %":0,
            "Max Drawdown %":0,
            "Average Trade %":0
        }

    t=pd.DataFrame(trades)

    pnl=t["P&L %"].astype(float)

    wins=pnl[pnl>0]
    losses=pnl[pnl<0]

    pf=(
        wins.sum()/abs(losses.sum())
        if len(losses)
        else np.inf
    )

    equity=(1+pnl/100).cumprod()
    peak=equity.cummax()
    dd=(equity/peak-1)*100

    return {
        "Trades":len(t),
        "Win Rate %":round(
            (pnl>0).mean()*100,
            2
        ),
        "Profit Factor":(
            round(pf,2)
            if np.isfinite(pf)
            else np.inf
        ),
        "Net Return %":round(
            (equity.iloc[-1]-1)*100,
            2
        ),
        "Max Drawdown %":round(
            abs(dd.min()),
            2
        ),
        "Average Trade %":round(
            pnl.mean(),
            2
        )
    }



# ============================================================
# BUY / SELL SIGNAL ENGINE
# ============================================================

def generate_scanner_signal(
    daily_data,
    scanner_name,
    hourly_data=None
):
    """
    Convert the selected scanner's current behaviour into:
    STRONG BUY / BUY / WATCH-HOLD / SELL / STRONG SELL.

    This is a ranking/signal layer only. It does not alter the
    underlying scanner or its strategy-specific backtest rules.
    """

    if daily_data is None or daily_data.empty:
        return None

    df=daily_data.copy()

    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)

    required=["Open","High","Low","Close","Volume"]

    if any(c not in df.columns for c in required):
        return None

    df=df.dropna(subset=required).copy()

    if len(df)<30:
        return None

    close=pd.to_numeric(df["Close"],errors="coerce")
    high=pd.to_numeric(df["High"],errors="coerce")
    low=pd.to_numeric(df["Low"],errors="coerce")
    volume=pd.to_numeric(df["Volume"],errors="coerce")

    rsi9=calculate_rsi_wilder(close,9)
    wma21=(
        rsi9.rolling(21)
        .apply(
            lambda x: np.dot(
                x,
                np.arange(1,22)
            )/231.0,
            raw=True
        )
    )

    sma50=close.rolling(50).mean()
    sma200=close.rolling(200).mean()

    bullish=0
    bearish=0
    reasons=[]
    warnings=[]

    c=float(close.iloc[-1])
    pc=float(close.iloc[-2])
    r=float(rsi9.iloc[-1]) if not pd.isna(rsi9.iloc[-1]) else np.nan
    w=float(wma21.iloc[-1]) if not pd.isna(wma21.iloc[-1]) else np.nan

    # Common price behaviour.
    if c>pc:
        bullish+=5
        reasons.append("Price is rising")
    else:
        bearish+=5
        warnings.append("Price is falling")

    # RSI behaviour.
    if not pd.isna(r):
        if r>=60:
            bullish+=8
            reasons.append(f"RSI(9) strong ({r:.1f})")
        elif r>=55:
            bullish+=5
            reasons.append(f"RSI(9) bullish ({r:.1f})")
        elif r<45:
            bearish+=8
            warnings.append(f"RSI(9) weak ({r:.1f})")
        elif r<50:
            bearish+=5
            warnings.append(f"RSI(9) below 50 ({r:.1f})")

    if not pd.isna(r) and not pd.isna(w):
        if r>w:
            bullish+=7
            reasons.append("RSI(9) above WMA(21)")
        else:
            bearish+=7
            warnings.append("RSI(9) below WMA(21)")

    # Trend behaviour.
    if len(df)>=200 and not pd.isna(sma200.iloc[-1]):
        if c>float(sma200.iloc[-1]):
            bullish+=8
            reasons.append("Price above SMA(200)")
        else:
            bearish+=8
            warnings.append("Price below SMA(200)")

    if len(df)>=50 and not pd.isna(sma50.iloc[-1]):
        if c>float(sma50.iloc[-1]):
            bullish+=5
        else:
            bearish+=5

    # Volume behaviour.
    av=float(volume.rolling(20).mean().iloc[-1])
    if av>0 and not pd.isna(av):
        vr=float(volume.iloc[-1])/av
        if vr>=1.5:
            bullish+=7
            reasons.append(f"Volume expansion ({vr:.1f}x)")
        elif vr<0.7:
            bearish+=3
            warnings.append("Volume below 20-day average")

    name=str(scanner_name)

    # Scanner-specific behaviour.
    if name=="Smart Breakout":
        # stage_two_analysis() expects the application's calculated
        # indicator columns (DONCHIAN_UPPER/LOWER, VOLUME_RATIO,
        # RSI14, SMA200, MACD, etc.), while the signal engine
        # intentionally receives raw OHLCV data.
        smart_data=calculate_indicators(
            df
        )

        x=stage_two_analysis(
            smart_data
        )
        if x:
            s=float(x.get("Score",0))
            bullish+=min(25,int(s*2.5))
            if s>=8:
                reasons.append(f"Smart Breakout score {s:.0f}/10")
            elif s<5:
                bearish+=10
                warnings.append(f"Smart Breakout score {s:.0f}/10")

    elif name=="120-Day High Breakout":
        x=calculate_120day_breakout_screen(df)
        if x:
            passed=sum(bool(v) for v in x["Conditions"].values())
            bullish+=int(passed*7.5)
            if x["Pass"]:
                reasons.append("120-day breakout confirmed")
            else:
                warnings.append(f"120-day breakout {passed}/4")

            if c>float(x["120D High 1D Ago"]):
                bullish+=10
                reasons.append("Price holding above breakout level")
            else:
                bearish+=10
                warnings.append("Price below breakout level")

    elif name=="Hourly Donchian Breakout":
        if hourly_data is not None and not hourly_data.empty:
            x=calculate_hourly_donchian_breakout(hourly_data)
            if x:
                passed=sum(bool(v) for v in x["Conditions"].values())
                bullish+=int(passed*5.8)
                if x["Pass"]:
                    reasons.append("Hourly Donchian 6/6 confirmed")
                else:
                    warnings.append(f"Hourly Donchian {passed}/6")
                if x["RSI9"]>=55:
                    bullish+=5
                else:
                    bearish+=5
        else:
            warnings.append("Hourly data unavailable")

    elif name=="Daily RSI(9)/WMA(21)":
        if not pd.isna(r) and not pd.isna(w):
            if r>w:
                bullish+=10
                reasons.append("Daily RSI/WMA bullish")
            else:
                bearish+=10
                warnings.append("Daily RSI/WMA bearish")
            if r>=55:
                bullish+=10
            else:
                bearish+=5

    elif name=="Weekly Trend":
        x=calculate_weekly_trend_screen(df)
        if x:
            passed=sum(bool(v) for v in x["Conditions"].values())
            bullish+=int(passed/len(x["Conditions"])*30)
            if x["Pass"]:
                reasons.append("Weekly trend fully confirmed")
            else:
                warnings.append(f"Weekly trend {passed}/10")

    elif name=="Daily Trend":
        x=calculate_daily_trend_screen(df)
        if x:
            passed=sum(bool(v) for v in x["Conditions"].values())
            bullish+=int(passed/len(x["Conditions"])*30)
            if x["Pass"]:
                reasons.append("Daily trend fully confirmed")
            else:
                warnings.append(f"Daily trend {passed}/10")

    elif name=="Multi-Timeframe":
        if r>w:
            bullish+=10
            reasons.append("Daily RSI/WMA bullish")
        else:
            bearish+=10
            warnings.append("Daily RSI/WMA bearish")

        weekly=rsi_wma_signal_asof(
            df,
            df.index[-1],
            50,
            "Weekly"
        )
        if weekly and weekly["Pass"]:
            bullish+=15
            reasons.append("Weekly RSI/WMA confirmed")
        else:
            bearish+=5

    elif name=="Top 20 Momentum":
        if c>float(sma50.iloc[-1]):
            bullish+=10
            reasons.append("Momentum above SMA50")
        else:
            bearish+=10
            warnings.append("Momentum below SMA50")

    bullish=max(0,min(100,bullish))
    bearish=max(0,min(100,bearish))
    net=bullish-bearish

    if net>=45 and bullish>=60:
        signal="🟢 STRONG BUY"
    elif net>=20 and bullish>=45:
        signal="🟢 BUY"
    elif net<=-45 and bearish>=50:
        signal="🔴 STRONG SELL"
    elif net<=-20 and bearish>=35:
        signal="🔴 SELL"
    else:
        signal="🟡 WATCH / HOLD"

    strength=max(0,min(100,int(50+net)))

    trade=None
    if signal in ["🟢 BUY","🟢 STRONG BUY"]:
        try:
            trade=calculate_trade_plan(df)
        except Exception:
            trade=None

    return {
        "Signal":signal,
        "Signal Strength":strength,
        "Bullish Score":bullish,
        "Bearish Score":bearish,
        "Net Score":net,
        "RSI9":round(r,2) if not pd.isna(r) else np.nan,
        "WMA21":round(w,2) if not pd.isna(w) else np.nan,
        "Reasons":reasons,
        "Warnings":warnings,
        "Trade Plan":trade
    }



# ============================================================
# MINERVINI SEPA + VCP TECHNICAL SCANNER
# ============================================================

def _minervini_prepare(df):
    """Prepare daily OHLCV data for the mechanical SEPA/VCP scan.

    Fundamentals are deliberately NOT fabricated here. This module uses
    the book-supported technical/price-volume portion of SEPA. Fundamental
    fields are shown as unavailable until a reliable quarterly fundamentals
    source is connected.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=x.columns.get_level_values(0)

    required=["Open","High","Low","Close","Volume"]
    if any(c not in x.columns for c in required):
        return pd.DataFrame()

    x=x.dropna(subset=required).copy()
    for c in required:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=required)

    x["SMA50"]=x["Close"].rolling(50).mean()
    x["SMA150"]=x["Close"].rolling(150).mean()
    x["SMA200"]=x["Close"].rolling(200).mean()
    x["VOL_SMA50"]=x["Volume"].rolling(50).mean()
    x["HIGH_252"]=x["High"].rolling(252).max()
    x["LOW_252"]=x["Low"].rolling(252).min()

    x["RET_63"]=x["Close"].pct_change(63)
    x["RET_126"]=x["Close"].pct_change(126)
    x["RET_252"]=x["Close"].pct_change(252)

    # Relative-strength line versus Nifty 50 is filled by the scanner when
    # a benchmark series is available.
    x["RS_LINE"]=np.nan
    x["RS_LINE_20D_SLOPE"]=np.nan
    x["RS_LINE_50D_SLOPE"]=np.nan

    return x


def _minervini_vcp_metrics(df, window_days=60):
    """Mechanical approximation of successive VCP contractions.

    The books describe progressively smaller contractions and declining
    volume. The implementation uses four equal recent segments so the
    result is deterministic and backtestable rather than subjective chart
    drawing.
    """
    result={
        "VCP Valid":False,
        "VCP Score":0,
        "Base Days":0,
        "T1 %":np.nan,
        "T2 %":np.nan,
        "T3 %":np.nan,
        "T4 %":np.nan,
        "Volume T1":np.nan,
        "Volume T2":np.nan,
        "Volume T3":np.nan,
        "Volume T4":np.nan,
        "Final Tightness %":np.nan,
        "Pivot":np.nan,
        "Pivot Distance %":np.nan,
        "VCP Reason":""
    }

    if df is None or len(df)<max(220,window_days):
        result["VCP Reason"]="Insufficient history"
        return result

    x=df.tail(window_days).copy()
    n=len(x)
    seg=max(5,n//4)
    chunks=[x.iloc[i*seg:(i+1)*seg] for i in range(3)]
    chunks.append(x.iloc[3*seg:])
    if any(len(c)<5 for c in chunks):
        result["VCP Reason"]="Insufficient contraction segments"
        return result

    depths=[]
    vols=[]
    for c in chunks:
        hi=float(c["High"].max())
        lo=float(c["Low"].min())
        depth=(hi-lo)/hi*100 if hi else np.nan
        depths.append(depth)
        vols.append(float(c["Volume"].mean()))

    t1,t2,t3,t4=depths
    v1,v2,v3,v4=vols
    final_tight=t4

    contraction_ok=bool(
        np.isfinite(t1) and np.isfinite(t2) and np.isfinite(t3) and np.isfinite(t4)
        and t1>t2>t3
        and t4<=t3
    )
    strong_contraction=bool(
        contraction_ok and t2<=0.80*t1 and t3<=0.80*t2
    )
    volume_contracting=bool(v1>v2>v3 and v4<=v3)
    final_dryup=bool(v4<=0.70*v1)
    tight=bool(final_tight<=8.0)
    very_tight=bool(final_tight<=5.0)

    pivot=float(chunks[-1]["High"].max())
    close=float(x["Close"].iloc[-1])
    pivot_distance=(close/pivot-1)*100 if pivot else np.nan

    score=0
    score += 8 if contraction_ok else 0
    score += 4 if strong_contraction else 0
    score += 5 if volume_contracting else 0
    score += 3 if final_dryup else 0
    score += 3 if very_tight else (2 if tight else 0)
    score += 2 if -3<=pivot_distance<=3 else 0

    valid=bool(contraction_ok and volume_contracting and tight)
    reason=[]
    if contraction_ok: reason.append("T1>T2>T3>T4")
    if strong_contraction: reason.append("strong contraction")
    if volume_contracting: reason.append("volume contracting")
    if final_dryup: reason.append("final volume dry-up")
    if very_tight: reason.append("final range <=5%")

    result.update({
        "VCP Valid":valid,
        "VCP Score":min(25,score),
        "Base Days":n,
        "T1 %":t1,
        "T2 %":t2,
        "T3 %":t3,
        "T4 %":t4,
        "Volume T1":v1,
        "Volume T2":v2,
        "Volume T3":v3,
        "Volume T4":v4,
        "Final Tightness %":final_tight,
        "Pivot":pivot,
        "Pivot Distance %":pivot_distance,
        "VCP Reason":", ".join(reason) if reason else "VCP not confirmed"
    })
    return result


def _minervini_score_row(df, rs_rank, benchmark=None, window_days=60,
                         min_volume_lakhs=5.0, breakout_volume_mult=1.20,
                         chase_pct=3.0):
    x=_minervini_prepare(df)
    if x.empty or len(x)<252:
        return None

    # Benchmark-relative RS line.
    if benchmark is not None and not benchmark.empty:
        b=benchmark.copy()
        if isinstance(b.columns,pd.MultiIndex):
            b.columns=b.columns.get_level_values(0)
        if "Close" in b.columns:
            bclose=pd.to_numeric(b["Close"],errors="coerce").dropna()
            aligned=bclose.reindex(x.index).ffill()
            x["RS_LINE"]=x["Close"]/aligned
            x["RS_LINE_20D_SLOPE"]=x["RS_LINE"]/x["RS_LINE"].shift(20)-1
            x["RS_LINE_50D_SLOPE"]=x["RS_LINE"]/x["RS_LINE"].shift(50)-1

    r=x.iloc[-1]
    prev=x.iloc[-2]

    # ----- Trend Template: hard gate -----
    t1=bool(r["Close"]>r["SMA150"])
    t2=bool(r["Close"]>r["SMA200"])
    t3=bool(r["SMA150"]>r["SMA200"])
    t4=bool(r["SMA200"]>x["SMA200"].iloc[-21])
    t5=bool(r["SMA50"]>r["SMA150"] and r["SMA50"]>r["SMA200"])
    t6=bool(r["Close"]>r["SMA50"])
    t7=bool(r["Close"]>=1.30*r["LOW_252"])
    t8=bool(r["Close"]>=0.75*r["HIGH_252"])
    trend_pass=all([t1,t2,t3,t4,t5,t6,t7,t8])

    # ----- RS / leadership -----
    rs_rank=float(rs_rank) if pd.notna(rs_rank) else 0.0
    rs_score=(
        10 if rs_rank>=95 else
        9 if rs_rank>=90 else
        7 if rs_rank>=85 else
        5 if rs_rank>=80 else
        3 if rs_rank>=75 else
        1 if rs_rank>=70 else 0
    )
    rs20=float(r.get("RS_LINE_20D_SLOPE",np.nan))
    rs50=float(r.get("RS_LINE_50D_SLOPE",np.nan))
    rsline_score=(2 if pd.notna(rs20) and rs20>0 else 0)+(2 if pd.notna(rs50) and rs50>0 else 0)
    if pd.notna(r.get("RS_LINE",np.nan)) and pd.notna(x["RS_LINE"].tail(252).max()):
        if r["RS_LINE"]>=x["RS_LINE"].tail(252).max()*0.98:
            rsline_score+=1

    dist_high=(1-r["Close"]/r["HIGH_252"])*100 if r["HIGH_252"] else np.nan
    price_score=(5 if dist_high<=5 else 4 if dist_high<=10 else 3 if dist_high<=15 else 2 if dist_high<=20 else 1 if dist_high<=25 else 0)

    # ----- VCP -----
    vcp=_minervini_vcp_metrics(x,window_days)
    vcp_score=int(vcp["VCP Score"])

    # ----- Pivot / breakout -----
    pivot=vcp["Pivot"]
    close=float(r["Close"])
    vol=float(r["Volume"])
    vol50=float(r["VOL_SMA50"])
    vol_ratio=vol/vol50 if vol50 and np.isfinite(vol50) else np.nan
    breakout=bool(pd.notna(pivot) and close>pivot)
    in_entry_zone=bool(pd.notna(pivot) and pivot>0 and close<=pivot*(1+chase_pct/100))
    breakout_volume_score=(5 if pd.notna(vol_ratio) and vol_ratio>=2 else 4 if pd.notna(vol_ratio) and vol_ratio>=1.5 else 3 if pd.notna(vol_ratio) and vol_ratio>=1.2 else 1 if pd.notna(vol_ratio) and vol_ratio>=1 else 0)
    pivot_quality=3 if pd.notna(pivot) and abs(close/pivot-1)<=0.03 else 2 if pd.notna(pivot) and abs(close/pivot-1)<=0.05 else 0
    distance_score=5 if pd.notna(pivot) and 0<=close/pivot-1<=0.03 else 4 if pd.notna(pivot) and -0.03<=close/pivot-1<0 else 1 if pd.notna(pivot) else 0
    pivot_score=min(15,pivot_quality+breakout_volume_score+distance_score)

    # ----- Market/liquidity -----
    liquidity_lakhs=(vol*close)/100000 if pd.notna(vol) else 0
    liquidity_score=5 if liquidity_lakhs>=50 else 4 if liquidity_lakhs>=25 else 3 if liquidity_lakhs>=10 else 2 if liquidity_lakhs>=min_volume_lakhs else 0
    market_score=5 if t4 and t2 else 3 if t2 else 0

    # Technical 100-point implementation. Fundamental fields are kept out
    # rather than inventing data; see UI note in the scanner.
    total=int(min(100,25*int(trend_pass)+20+vcp_score+pivot_score+liquidity_score+market_score))
    # Leadership contributes up to 20, but if trend is not passed the stock
    # is rejected regardless of score.
    leadership=min(20,rs_score+rsline_score+price_score)
    total=int(min(100,25*int(trend_pass)+leadership+vcp_score+pivot_score+liquidity_score+market_score))

    breakout_confirmed=bool(
        trend_pass and vcp["VCP Valid"] and breakout and
        pd.notna(vol_ratio) and vol_ratio>=breakout_volume_mult and
        in_entry_zone
    )
    watch=bool(
        trend_pass and vcp["VCP Valid"] and
        pd.notna(pivot) and close<=pivot*(1+chase_pct/100)
    )
    if breakout_confirmed and total>=80:
        status="🚀 BUY"
    elif breakout and not in_entry_zone:
        status="⚠️ BREAKOUT — DON'T CHASE"
    elif watch and total>=70:
        status="🟡 VCP WATCH"
    elif trend_pass and total>=70:
        status="🟢 SEPA QUALIFIED"
    elif trend_pass:
        status="🟠 TREND PASS — WEAK SETUP"
    else:
        status="🔴 REJECT"

    return {
        "Score":total,
        "Trend Template":"PASS" if trend_pass else "FAIL",
        "RS Rank":round(rs_rank,1),
        "RS Line":"Rising" if pd.notna(rs20) and rs20>0 else "Weak/Flat",
        "6M Return %":round(float(r["RET_126"])*100,2) if pd.notna(r["RET_126"]) else np.nan,
        "12M Return %":round(float(r["RET_252"])*100,2) if pd.notna(r["RET_252"]) else np.nan,
        "52W High Distance %":round(dist_high,2) if pd.notna(dist_high) else np.nan,
        "VCP":"PASS" if vcp["VCP Valid"] else "FAIL",
        "VCP Score":vcp_score,
        "Contractions":f'{vcp["T1 %"]:.1f}% → {vcp["T2 %"]:.1f}% → {vcp["T3 %"]:.1f}% → {vcp["T4 %"]:.1f}%' if all(pd.notna(vcp[k]) for k in ["T1 %","T2 %","T3 %","T4 %"]) else "-",
        "Final Tightness %":round(vcp["Final Tightness %"],2) if pd.notna(vcp["Final Tightness %"]) else np.nan,
        "Pivot":round(pivot,2) if pd.notna(pivot) else np.nan,
        "Pivot Distance %":round((close/pivot-1)*100,2) if pd.notna(pivot) and pivot else np.nan,
        "Volume / SMA50":round(vol_ratio,2) if pd.notna(vol_ratio) else np.nan,
        "Liquidity ₹L":round(liquidity_lakhs,1),
        "Status":status,
        "Close":round(close,2),
        "Trend Score":25 if trend_pass else 0,
        "Leadership Score":leadership,
        "VCP Component":vcp_score,
        "Pivot Component":pivot_score,
        "Market/Liquidity":market_score+liquidity_score,
        "VCP Detail":vcp["VCP Reason"],
        "Breakout Confirmed":breakout_confirmed,
        "Chase Warning":bool(breakout and not in_entry_zone),
    }


def run_minervini_scanner(market, benchmark=None, window_days=60,
                          min_volume_lakhs=5.0, breakout_volume_mult=1.20,
                          chase_pct=3.0):
    """Run the technical SEPA/VCP scanner and compute universe RS ranks."""
    prepared={}
    momentum=[]
    for symbol,raw in market.items():
        x=_minervini_prepare(raw)
        if x.empty or len(x)<252:
            continue
        momentum.append({
            "Symbol":symbol,
            "M3":x["RET_63"].iloc[-1],
            "M6":x["RET_126"].iloc[-1],
            "M12":x["RET_252"].iloc[-1]
        })
        prepared[symbol]=x

    if not momentum:
        return pd.DataFrame()

    m=pd.DataFrame(momentum).set_index("Symbol")
    p3=m["M3"].rank(pct=True)*100
    p6=m["M6"].rank(pct=True)*100
    p12=m["M12"].rank(pct=True)*100
    m["RS_Rank"]=0.40*p3+0.35*p6+0.25*p12

    rows=[]
    for symbol,x in prepared.items():
        try:
            out=_minervini_score_row(
                x,
                m.loc[symbol,"RS_Rank"],
                benchmark=benchmark,
                window_days=window_days,
                min_volume_lakhs=min_volume_lakhs,
                breakout_volume_mult=breakout_volume_mult,
                chase_pct=chase_pct
            )
            if out:
                out["Stock"]=symbol
                rows.append(out)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result=pd.DataFrame(rows)
    order={"🚀 BUY":0,"🟡 VCP WATCH":1,"🟢 SEPA QUALIFIED":2,"⚠️ BREAKOUT — DON'T CHASE":3,"🟠 TREND PASS — WEAK SETUP":4,"🔴 REJECT":5}
    result["_order"]=result["Status"].map(order).fillna(9)
    result=result.sort_values(["_order","Score","RS Rank"],ascending=[True,False,False]).drop(columns=["_order"])
    return result.reset_index(drop=True)



def _minervini_historical_snapshot(
    prepared,
    benchmark,
    asof,
    window_days,
    min_volume_lakhs,
    breakout_volume_mult,
    chase_pct
):
    """
    Evaluate the EXACT same Minervini technical scanner rules
    using data available only through `asof`.

    This prevents look-ahead: the scanner never sees future bars.
    """
    rows=[]

    # Universe-relative RS must also be historical.
    momentum={}
    for symbol,x in prepared.items():
        if asof not in x.index:
            continue
        hist=x.loc[:asof]
        if len(hist)<252:
            continue
        r=hist.iloc[-1]
        if all(pd.notna(r.get(k,np.nan)) for k in
               ["RET_63","RET_126","RET_252"]):
            momentum[symbol]={
                "M3":float(r["RET_63"]),
                "M6":float(r["RET_126"]),
                "M12":float(r["RET_252"])
            }

    if not momentum:
        return pd.DataFrame()

    m=pd.DataFrame(momentum).T
    rs_rank=(
        0.40*m["M3"].rank(pct=True)*100
        +0.35*m["M6"].rank(pct=True)*100
        +0.25*m["M12"].rank(pct=True)*100
    )

    for symbol in m.index:
        hist=prepared[symbol].loc[:asof]
        try:
            bench_hist=(
                benchmark.loc[:asof]
                if benchmark is not None and not benchmark.empty
                else benchmark
            )
            out=_minervini_score_row(
                hist,
                rs_rank.loc[symbol],
                benchmark=bench_hist,
                window_days=window_days,
                min_volume_lakhs=min_volume_lakhs,
                breakout_volume_mult=breakout_volume_mult,
                chase_pct=chase_pct
            )
            if out:
                out["Stock"]=symbol
                rows.append(out)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result=pd.DataFrame(rows)
    return result


def backtest_minervini_sepa_vcp(
    market,
    benchmark=None,
    window_days=60,
    min_volume_lakhs=5.0,
    breakout_volume_mult=1.20,
    chase_pct=3.0,
    stop_loss_pct=8.0,
    max_holding_days=60,
    starting_capital=1000000.0,
    max_positions=10,
    position_risk_pct=1.0
):
    """
    Historical Minervini SEPA/VCP backtest.

    ENTRY:
      The same scanner BUY rules are evaluated on each completed
      daily bar. When a BUY signal appears, the position is entered
      at the NEXT day's open. This avoids look-ahead bias.

    EXIT:
      1) 8% (configurable) hard stop, reflecting the book's
         risk-control philosophy.
      2) Breakout failure: after entry, a daily close below the
         breakout pivot.
      3) If the trade is profitable, a close below the 50-day SMA
         is treated as a trend-failure exit.
      4) Maximum holding period.

    If stop and another exit are both possible on the same bar,
    the stop is assumed first (conservative).

    Position sizing:
      risk-based sizing using position_risk_pct of current equity,
      capped by equal-capital allocation across max_positions.
    """

    prepared={}
    for symbol,raw in market.items():
        try:
            x=_minervini_prepare(raw)
            if len(x)>=252:
                prepared[symbol]=x
        except Exception:
            continue

    if not prepared:
        return {
            "trades":[],
            "equity":pd.DataFrame(),
            "summary":{}
        }

    # Build a common daily calendar from the prepared data.
    all_dates=sorted(
        set().union(
            *[set(x.index) for x in prepared.values()]
        )
    )

    if len(all_dates)<253:
        return {
            "trades":[],
            "equity":pd.DataFrame(),
            "summary":{}
        }

    equity=float(starting_capital)
    cash=equity
    open_positions={}
    trades=[]
    equity_rows=[]

    # Cache scanner snapshots by date.
    snapshot_cache={}

    def get_snapshot(dt):
        key=pd.Timestamp(dt)
        if key not in snapshot_cache:
            snapshot_cache[key]=_minervini_historical_snapshot(
                prepared,
                benchmark,
                key,
                window_days,
                min_volume_lakhs,
                breakout_volume_mult,
                chase_pct
            )
        return snapshot_cache[key]

    # Only dates for which all relevant lookback data can exist.
    dates=all_dates

    for i,dt in enumerate(dates[:-1]):

        dt=pd.Timestamp(dt)
        next_dt=pd.Timestamp(dates[i+1])

        # ----------------------------------------------------
        # EXIT OPEN POSITIONS USING TODAY'S COMPLETED BAR
        # ----------------------------------------------------
        for symbol in list(open_positions.keys()):

            pos=open_positions[symbol]
            x=prepared[symbol]

            if dt not in x.index:
                continue

            r=x.loc[dt]
            close=float(r["Close"])
            low=float(r["Low"])
            high=float(r["High"])

            entry=float(pos["entry"])
            shares=int(pos["shares"])
            stop=float(pos["stop"])
            pivot=float(pos["pivot"])
            held=int(
                (dt-pos["entry_date"]).days
            )

            exit_price=None
            reason=None

            # Conservative stop-first assumption.
            if low<=stop:
                exit_price=stop
                reason="8% Risk Stop"
            elif close<pivot:
                exit_price=close
                reason="Failed Breakout"
            elif (
                close>entry
                and pd.notna(r.get("SMA50",np.nan))
                and close<float(r["SMA50"])
            ):
                exit_price=close
                reason="50 SMA Trend Failure"
            elif held>=max_holding_days:
                exit_price=close
                reason="Maximum Holding Period"

            if exit_price is not None:

                proceeds=shares*exit_price
                pnl=shares*(exit_price-entry)
                pnl_pct=(exit_price/entry-1)*100

                cash+=proceeds

                trades.append({
                    "Stock":symbol,
                    "Entry Date":pos["entry_date"],
                    "Exit Date":dt,
                    "Entry":round(entry,2),
                    "Exit":round(exit_price,2),
                    "Shares":shares,
                    "Pivot":round(pivot,2),
                    "Stop":round(stop,2),
                    "PnL":round(pnl,2),
                    "Return %":round(pnl_pct,2),
                    "Holding Days":held,
                    "Score":pos["score"],
                    "RS Rank":pos["rs_rank"],
                    "VCP Score":pos["vcp_score"],
                    "Exit Reason":reason
                })

                del open_positions[symbol]

        # ----------------------------------------------------
        # MARK-TO-MARKET EQUITY
        # ----------------------------------------------------
        mtm=cash

        for symbol,pos in open_positions.items():
            x=prepared[symbol]
            if dt in x.index:
                mtm+=pos["shares"]*float(x.loc[dt,"Close"])

        equity=mtm
        equity_rows.append({
            "Date":dt,
            "Equity":equity,
            "Open Positions":len(open_positions)
        })

        # ----------------------------------------------------
        # HISTORICAL SCANNER SIGNAL
        # ----------------------------------------------------
        snapshot=get_snapshot(dt)

        if snapshot.empty:
            continue

        buys=snapshot[
            snapshot["Status"]=="🚀 BUY"
        ].copy()

        if buys.empty:
            continue

        # Do not buy stocks already held.
        buys=buys[
            ~buys["Stock"].isin(open_positions.keys())
        ]

        if buys.empty:
            continue

        # Strongest signals first.
        buys=buys.sort_values(
            ["Score","RS Rank","VCP Score"],
            ascending=False
        )

        available_slots=max(
            0,
            int(max_positions)-len(open_positions)
        )

        if available_slots<=0:
            continue

        for _,sig in buys.head(available_slots).iterrows():

            symbol=sig["Stock"]

            if symbol not in prepared:
                continue

            x=prepared[symbol]

            if next_dt not in x.index:
                continue

            next_bar=x.loc[next_dt]
            entry=float(next_bar["Open"])

            if not np.isfinite(entry) or entry<=0:
                continue

            pivot=float(sig["Pivot"])

            # Signal was generated at today's close, so the next
            # day's open can gap above the configured chase zone.
            # We still use the scanner's exact signal; we do not
            # invent a new signal after the gap.
            stop=entry*(1-stop_loss_pct/100)

            # Risk-based position size.
            risk_amount=equity*(position_risk_pct/100)
            risk_per_share=entry-stop
            risk_shares=(
                int(risk_amount/risk_per_share)
                if risk_per_share>0 else 0
            )

            capital_limit=equity/max(
                1,
                int(max_positions)
            )
            capital_shares=int(
                capital_limit/entry
            )

            shares=max(
                0,
                min(risk_shares,capital_shares)
            )

            if shares<=0:
                continue

            cost=shares*entry

            if cost>cash:
                shares=int(cash/entry)

            if shares<=0:
                continue

            cash-=shares*entry

            open_positions[symbol]={
                "entry_date":next_dt,
                "entry":entry,
                "shares":shares,
                "stop":stop,
                "pivot":pivot,
                "score":float(sig["Score"]),
                "rs_rank":float(sig["RS Rank"]),
                "vcp_score":float(sig["VCP Score"])
            }

    # Close remaining positions on final available bar.
    final_dt=pd.Timestamp(dates[-1])

    for symbol,pos in list(open_positions.items()):

        x=prepared[symbol]

        if final_dt not in x.index:
            continue

        exit_price=float(x.loc[final_dt,"Close"])
        shares=int(pos["shares"])

        cash+=shares*exit_price

        pnl=shares*(exit_price-pos["entry"])
        pnl_pct=(exit_price/pos["entry"]-1)*100

        held=int(
            (final_dt-pos["entry_date"]).days
        )

        trades.append({
            "Stock":symbol,
            "Entry Date":pos["entry_date"],
            "Exit Date":final_dt,
            "Entry":round(pos["entry"],2),
            "Exit":round(exit_price,2),
            "Shares":shares,
            "Pivot":round(pos["pivot"],2),
            "Stop":round(pos["stop"],2),
            "PnL":round(pnl,2),
            "Return %":round(pnl_pct,2),
            "Holding Days":held,
            "Score":pos["score"],
            "RS Rank":pos["rs_rank"],
            "VCP Score":pos["vcp_score"],
            "Exit Reason":"Backtest End"
        })

        del open_positions[symbol]

    # Rebuild equity curve after final exits.
    equity_df=pd.DataFrame(equity_rows)

    if equity_df.empty:
        equity_df=pd.DataFrame(
            [{"Date":final_dt,"Equity":cash,"Open Positions":0}]
        )
    else:
        equity_df.loc[
            equity_df.index[-1],
            "Equity"
        ]=cash
        equity_df.loc[
            equity_df.index[-1],
            "Open Positions"
        ]=0

    trades_df=pd.DataFrame(trades)

    if trades_df.empty:
        summary={
            "Starting Capital":starting_capital,
            "Ending Capital":cash,
            "Net Profit":cash-starting_capital,
            "Return %":(cash/starting_capital-1)*100,
            "Trades":0,
            "Win Rate %":0.0,
            "Profit Factor":0.0,
            "Max Drawdown %":0.0
        }
    else:
        wins=trades_df[trades_df["PnL"]>0]["PnL"]
        losses=trades_df[trades_df["PnL"]<0]["PnL"]

        peak=equity_df["Equity"].cummax()
        dd=(equity_df["Equity"]/peak-1)*100

        summary={
            "Starting Capital":starting_capital,
            "Ending Capital":cash,
            "Net Profit":cash-starting_capital,
            "Return %":(cash/starting_capital-1)*100,
            "Trades":len(trades_df),
            "Win Rate %":(
                (trades_df["PnL"]>0).mean()*100
            ),
            "Profit Factor":(
                wins.sum()/abs(losses.sum())
                if losses.sum()!=0 else np.inf
            ),
            "Max Drawdown %":float(dd.min()),
            "Average Trade %":float(
                trades_df["Return %"].mean()
            ),
            "Average Holding Days":float(
                trades_df["Holding Days"].mean()
            )
        }

    return {
        "trades":trades,
        "trades_df":trades_df,
        "equity":equity_df,
        "summary":summary
    }


@st.cache_data(ttl=900, show_spinner=False)
def download_nifty50_benchmark(period="2y"):
    """Serial benchmark download; avoids thread-heavy calls on Streamlit Cloud."""
    try:
        d=yf.download(
            tickers="^NSEI",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if isinstance(d,pd.DataFrame) and not d.empty:
            if isinstance(d.columns,pd.MultiIndex):
                d.columns=d.columns.get_level_values(0)
            return d
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Control Panel"
)

module = st.sidebar.radio(
    "Select Module",
    [
        "🎯 CCI + EMA + RSI Strategy",
        "📚 Kratter Momentum Scanner",
        "🔥 Momentum Catalyst Scanner",
        "📊 Options Next-Day Analyzer",
            "🏆 Minervini SEPA + VCP Scanner",
        "🚀 Smart Breakout Scanner",
        "🎯 Buy / Sell Signal Engine",
        "📐 Chart Pattern Scanner",
        "🧪 Smart Breakout Drawdown Optimizer",
        "📊 Backtest & Performance",
        "🏆 Top 20 Stocks",
        "🤖 AI Analyst"
    ]
)

st.sidebar.success(
    "Focused mode: selected modules only."
)


# ============================================================
# TECHNICAL CHART
# ============================================================



# Local indicator helpers for the independent Options Analyzer.
# These deliberately do not depend on the Minervini module.

def _options_rsi(series, period=14):
    s=pd.to_numeric(series,errors="coerce")
    delta=s.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()
    avg_loss=loss.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    rsi=100-(100/(1+rs))
    rsi= rsi.where(avg_loss.ne(0), 100.0)
    rsi= rsi.where(~((avg_gain==0)&(avg_loss==0)), 50.0)
    return rsi


def _options_atr(df, period=14):
    high=pd.to_numeric(df["High"],errors="coerce")
    low=pd.to_numeric(df["Low"],errors="coerce")
    close=pd.to_numeric(df["Close"],errors="coerce")
    prev_close=close.shift(1)

    tr=pd.concat([
        high-low,
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ],axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()


def _options_cci(df, period=20):
    high=pd.to_numeric(df["High"],errors="coerce")
    low=pd.to_numeric(df["Low"],errors="coerce")
    close=pd.to_numeric(df["Close"],errors="coerce")

    tp=(high+low+close)/3
    sma=tp.rolling(period,min_periods=period).mean()
    mad=tp.rolling(period,min_periods=period).apply(
        lambda x: np.mean(np.abs(x-np.mean(x))),
        raw=True
    )

    return (tp-sma)/(0.015*mad.replace(0,np.nan))


# ============================================================
# OPTIONS NEXT-DAY ANALYZER
# ============================================================

def _read_options_eod_csv(uploaded_file):
    """Read the user's end-of-day derivatives CSV.
    The supplied format contains four metadata lines before the header.
    """
    if uploaded_file is None:
        return pd.DataFrame(), {}

    raw=uploaded_file.getvalue()
    meta={}
    lines=raw.decode("utf-8-sig",errors="replace").splitlines()

    for line in lines[:4]:
        if ":" in line:
            k,v=line.split(":",1)
            meta[k.strip()]=v.strip().rstrip(",")

    # Detect the actual header instead of hard-coding skiprows.
    header_idx=None
    for i,line in enumerate(lines[:15]):
        if "Stock Name" in line and "Cumulative Future OI" in line:
            header_idx=i
            break

    if header_idx is None:
        raise ValueError(
            "Could not find the derivatives CSV header. "
            "Expected fields such as Stock Name, Symbol, "
            "Cumulative Future OI and Cumulative Call OI."
        )

    data=pd.read_csv(
        StringIO("\n".join(lines[header_idx:])),
        engine="python"
    )

    data.columns=[
        str(c).strip()
        for c in data.columns
    ]

    numeric_cols=[
        "Close","Chg %","Lot Size","Cumulative Future OI",
        "OI Chg %","Volume (Times)","Delivery (Times)",
        "Cumulative Call OI","Cumulative Put OI",
        "Put Call Ratio (PCR)","PCR Change 1D"
    ]

    for c in numeric_cols:
        if c in data.columns:
            data[c]=pd.to_numeric(
                data[c].astype(str).str.replace(",","",regex=False)
                .str.replace("%","",regex=False),
                errors="coerce"
            )

    if "Symbol" in data.columns:
        data["Symbol"]=(
            data["Symbol"].astype(str).str.upper().str.strip()
        )

    return data,meta


def _options_position_classification(price_change, oi_change):
    """Classic futures price/OI interpretation."""
    if pd.isna(price_change) or pd.isna(oi_change):
        return "Unknown"

    if price_change>0 and oi_change>0:
        return "Long Buildup"
    if price_change<0 and oi_change>0:
        return "Short Buildup"
    if price_change>0 and oi_change<0:
        return "Short Covering"
    if price_change<0 and oi_change<0:
        return "Long Unwinding"

    return "Neutral"


def _options_fibonacci_levels(df, lookback=60):
    if df is None or df.empty or "High" not in df or "Low" not in df:
        return {}

    x=df.tail(int(lookback)).copy()
    if len(x)<10:
        return {}

    hi=float(x["High"].max())
    lo=float(x["Low"].min())

    hi_idx=x["High"].idxmax()
    lo_idx=x["Low"].idxmin()

    # Latest dominant swing: determine whether low preceded high.
    bullish=lo_idx<hi_idx

    rng=hi-lo
    if rng<=0:
        return {}

    if bullish:
        levels={
            "Fib 23.6":hi-rng*0.236,
            "Fib 38.2":hi-rng*0.382,
            "Fib 50.0":hi-rng*0.500,
            "Fib 61.8":hi-rng*0.618,
            "Fib 78.6":hi-rng*0.786,
            "Swing High":hi,
            "Swing Low":lo
        }
    else:
        # For a recent downward swing, retracement levels are measured
        # upward from the low.
        levels={
            "Fib 23.6":lo+rng*0.236,
            "Fib 38.2":lo+rng*0.382,
            "Fib 50.0":lo+rng*0.500,
            "Fib 61.8":lo+rng*0.618,
            "Fib 78.6":lo+rng*0.786,
            "Swing High":hi,
            "Swing Low":lo
        }

    levels["Swing Direction"]="Bullish Swing" if bullish else "Bearish Swing"
    return levels


def _options_technical_snapshot(df, fib_lookback=60):
    if df is None or df.empty:
        return {}

    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=x.columns.get_level_values(0)

    req=["Open","High","Low","Close","Volume"]
    if any(c not in x.columns for c in req):
        return {}

    x=x.dropna(subset=req).copy()
    if len(x)<220:
        return {}

    for c in req:
        x[c]=pd.to_numeric(x[c],errors="coerce")

    x["EMA20"]=x["Close"].ewm(span=20,adjust=False).mean()
    x["EMA50"]=x["Close"].ewm(span=50,adjust=False).mean()
    x["EMA200"]=x["Close"].ewm(span=200,adjust=False).mean()
    x["RSI14"]=_options_rsi(x["Close"],14)
    x["CCI20"]=_options_cci(x,20)
    x["ATR14"]=_options_atr(x,14)
    x["VOL_SMA20"]=x["Volume"].rolling(20).mean()

    r=x.iloc[-1]
    prev=x.iloc[-2]

    fib=_options_fibonacci_levels(x,fib_lookback)
    close=float(r["Close"])
    ema20=float(r["EMA20"])
    ema50=float(r["EMA50"])
    ema200=float(r["EMA200"])
    rsi=float(r["RSI14"])
    cci=float(r["CCI20"])
    atr=float(r["ATR14"])

    trend_points=0
    if close>ema200:
        trend_points+=8
    if ema50>ema200:
        trend_points+=5
    if ema20>ema50:
        trend_points+=4
    if close>ema20:
        trend_points+=3
    if ema200>float(x["EMA200"].iloc[-21]):
        trend_points+=5

    trend="Bullish" if trend_points>=18 else (
        "Bearish" if trend_points<=8 else "Neutral"
    )

    # Find closest Fibonacci levels above and below current price.
    fib_numeric={
        k:v for k,v in fib.items()
        if isinstance(v,(int,float,np.floating))
        and k.startswith("Fib")
    }

    supports=[
        (k,v) for k,v in fib_numeric.items()
        if v<=close
    ]
    resistances=[
        (k,v) for k,v in fib_numeric.items()
        if v>=close
    ]

    support=max(supports,key=lambda z:z[1]) if supports else None
    resistance=min(resistances,key=lambda z:z[1]) if resistances else None

    support_score=0
    resistance_score=0

    if support:
        dist=(close-support[1])/close*100
        if dist<=2: support_score=10
        elif dist<=4: support_score=7
        elif dist<=7: support_score=4

    if resistance:
        dist=(resistance[1]-close)/close*100
        if dist<=2: resistance_score=10
        elif dist<=4: resistance_score=7
        elif dist<=7: resistance_score=4

    return {
        "Close":close,
        "EMA20":ema20,
        "EMA50":ema50,
        "EMA200":ema200,
        "RSI14":rsi,
        "CCI20":cci,
        "ATR14":atr,
        "Trend":trend,
        "Trend Points":trend_points,
        "Fib":fib,
        "Fib Support":support[0] if support else "",
        "Fib Support Price":support[1] if support else np.nan,
        "Fib Resistance":resistance[0] if resistance else "",
        "Fib Resistance Price":resistance[1] if resistance else np.nan,
        "Fib Support Score":support_score,
        "Fib Resistance Score":resistance_score,
        "Volume Ratio":(
            float(r["Volume"])/float(r["VOL_SMA20"])
            if float(r["VOL_SMA20"])>0 else np.nan
        ),
        "Bullish Momentum":(
            close>ema20 and ema20>ema50 and rsi>=55
        ),
        "Bearish Momentum":(
            close<ema20 and ema20<ema50 and rsi<=45
        )
    }


def _options_score_row(row, tech):
    """Generate deterministic next-day Call/Put/Selling scores.
    No option premium/IV is assumed when it is absent from the CSV.
    """

    chg=float(row.get("Chg %",np.nan))
    oi_chg=float(row.get("OI Chg %",np.nan))
    pcr=float(row.get("Put Call Ratio (PCR)",np.nan))
    pcr_chg=float(row.get("PCR Change 1D",np.nan))
    call_oi=float(row.get("Cumulative Call OI",np.nan))
    put_oi=float(row.get("Cumulative Put OI",np.nan))
    vol_mult=float(row.get("Volume (Times)",np.nan))
    del_mult=float(row.get("Delivery (Times)",np.nan))

    pos=_options_position_classification(chg,oi_chg)
    oi_trend=str(row.get("OI Trend","")).strip()

    call=0.0
    put=0.0
    sell=0.0

    # ---------------- Technical layer: 35 points ----------------
    if tech:
        trend=tech.get("Trend")
        if trend=="Bullish":
            call+=15
        elif trend=="Bearish":
            put+=15

        if tech.get("Bullish Momentum"):
            call+=10
        if tech.get("Bearish Momentum"):
            put+=10

        call+=float(tech.get("Fib Support Score",0))
        put+=float(tech.get("Fib Resistance Score",0))

    # ---------------- Futures positioning: 25 points ----------------
    if pos=="Long Buildup":
        call+=20
    elif pos=="Short Covering":
        call+=14
    elif pos=="Short Buildup":
        put+=20
    elif pos=="Long Unwinding":
        put+=14

    if "AggressiveNewLong" in oi_trend:
        call+=5
    elif "AggressiveNewShort" in oi_trend:
        put+=5
    elif "NewLong" in oi_trend:
        call+=3
    elif "NewShort" in oi_trend:
        put+=3

    # ---------------- OI / PCR structure: 25 points ----------------
    if not pd.isna(pcr):
        if pcr>=1.10:
            call+=8
        elif pcr>=0.90:
            call+=4

        if pcr<=0.70:
            put+=8
        elif pcr<=0.90:
            put+=4

    if not pd.isna(pcr_chg):
        if pcr_chg>0:
            call+=min(5,abs(pcr_chg)*100)
        elif pcr_chg<0:
            put+=min(5,abs(pcr_chg)*100)

    if call_oi>0 and put_oi>0:
        # Relative put/call positioning gives a modest confirmation,
        # not the entire signal.
        total=call_oi+put_oi
        put_share=put_oi/total
        if put_share>=0.60:
            call+=5
        elif put_share<=0.40:
            put+=5

    # ---------------- Participation: 15 points ----------------
    if not pd.isna(vol_mult):
        if vol_mult>=1.5:
            if chg>0: call+=5
            elif chg<0: put+=5
        elif vol_mult>=1.0:
            if chg>0: call+=3
            elif chg<0: put+=3

    if not pd.isna(del_mult):
        if del_mult>=1.5:
            if chg>0: call+=3
            elif chg<0: put+=3

    # Selling score: range-bound + balanced positioning + OI.
    if tech and tech.get("Trend")=="Neutral":
        sell+=20

    if not pd.isna(pcr) and 0.85<=pcr<=1.15:
        sell+=15

    if not pd.isna(oi_chg) and abs(oi_chg)>=5:
        sell+=10

    if not pd.isna(vol_mult) and vol_mult<1.0:
        sell+=10

    if tech:
        # Near Fib boundaries/range without momentum favours defined
        # resistance/support selling setups.
        if tech.get("Fib Support Score",0)>=7:
            sell+=5
        if tech.get("Fib Resistance Score",0)>=7:
            sell+=5

    call=min(100,round(call,1))
    put=min(100,round(put,1))
    sell=min(100,round(sell,1))

    if call>=80 and call>put and call>sell:
        signal="🟢 Strong Call Candidate"
    elif put>=80 and put>call and put>sell:
        signal="🔴 Strong Put Candidate"
    elif sell>=75 and sell>call and sell>put:
        signal="🟡 Option Selling Candidate"
    elif max(call,put)>=65:
        signal="🔵 Option Buying Candidate"
    else:
        signal="⚪ Avoid / Wait"

    return {
        "Position":pos,
        "Call Score":call,
        "Put Score":put,
        "Selling Score":sell,
        "Signal":signal
    }


def run_options_next_day_analysis(option_df, market_data,
                                  fib_lookback=60):
    """Combine the uploaded EOD derivatives file with latest OHLCV."""
    rows=[]

    if option_df is None or option_df.empty:
        return pd.DataFrame()

    for _,row in option_df.iterrows():
        symbol=str(row.get("Symbol","")).strip().upper()
        if not symbol:
            continue

        price=float(row["Close"]) if not pd.isna(row.get("Close",np.nan)) else np.nan
        tech=_options_technical_snapshot(
            market_data.get(symbol),
            fib_lookback
        )

        scores=_options_score_row(row,tech)

        out={
            "Stock":row.get("Stock Name",symbol),
            "Symbol":symbol,
            "EOD Date":row.get("Date",""),
            "Close":price,
            "Daily Chg %":row.get("Chg %",np.nan),
            "Future OI Chg %":row.get("OI Chg %",np.nan),
            "PCR":row.get("Put Call Ratio (PCR)",np.nan),
            "PCR Chg 1D":row.get("PCR Change 1D",np.nan),
            "OI Trend":row.get("OI Trend",""),
            "Call OI":row.get("Cumulative Call OI",np.nan),
            "Put OI":row.get("Cumulative Put OI",np.nan),
            "Volume x":row.get("Volume (Times)",np.nan),
            "Delivery x":row.get("Delivery (Times)",np.nan),
            "Position":scores["Position"],
            "Call Score":scores["Call Score"],
            "Put Score":scores["Put Score"],
            "Selling Score":scores["Selling Score"],
            "Signal":scores["Signal"],
        }

        if tech:
            out.update({
                "Trend":tech["Trend"],
                "RSI14":round(tech["RSI14"],2),
                "CCI20":round(tech["CCI20"],2),
                "EMA20":round(tech["EMA20"],2),
                "EMA50":round(tech["EMA50"],2),
                "EMA200":round(tech["EMA200"],2),
                "Fib Support":tech["Fib Support"],
                "Fib Support Price":tech["Fib Support Price"],
                "Fib Resistance":tech["Fib Resistance"],
                "Fib Resistance Price":tech["Fib Resistance Price"],
                "ATR14":round(tech["ATR14"],2),
                "Volume Ratio":round(tech["Volume Ratio"],2),
            })

        rows.append(out)

    result=pd.DataFrame(rows)

    if not result.empty:
        # Prioritize actionable signals, then score.
        priority={
            "🟢 Strong Call Candidate":4,
            "🔴 Strong Put Candidate":4,
            "🟡 Option Selling Candidate":3,
            "🔵 Option Buying Candidate":2,
            "⚪ Avoid / Wait":1
        }
        result["_priority"]=result["Signal"].map(priority).fillna(0)
        result=result.sort_values(
            ["_priority","Call Score","Put Score","Selling Score"],
            ascending=[False,False,False,False]
        ).drop(columns=["_priority"])

    return result




# ============================================================
# OPTIONS PHASE 1 HELPERS
# ============================================================

def _options_phase1_plan(row):
    close=float(row.get("Close", np.nan))
    atr=float(row.get("ATR14", np.nan))
    support=float(row.get("Fib Support Price", np.nan))
    resistance=float(row.get("Fib Resistance Price", np.nan))
    call=float(row.get("Call Score",0) or 0)
    put=float(row.get("Put Score",0) or 0)
    sell=float(row.get("Selling Score",0) or 0)
    trend=str(row.get("Trend",""))
    position=str(row.get("Position",""))
    pcr=float(row.get("PCR",np.nan))
    pcrchg=float(row.get("PCR Chg 1D",np.nan))
    volume=float(row.get("Volume x",np.nan))
    delivery=float(row.get("Delivery x",np.nan))
    oi_trend=str(row.get("OI Trend",""))

    reasons=[]

    if call>=put and call>=sell and call>=65:
        direction="CALL"
        preferred="ATM / nearest 1-step ITM CE"
        trigger=close+(0.20*atr if np.isfinite(atr) and atr>0 else close*0.005)
        invalidation=support if np.isfinite(support) and support>0 else close-(atr if np.isfinite(atr) and atr>0 else close*0.02)
        target1=resistance if np.isfinite(resistance) and resistance>trigger else close+(atr if np.isfinite(atr) and atr>0 else close*0.02)
        target2=target1+(atr if np.isfinite(atr) and atr>0 else close*0.02)

        if trend=="Bullish": reasons.append(("Bullish trend structure",15))
        if position in ("Long Buildup","Short Covering"): reasons.append((position,15))
        if "AggressiveNewLong" in oi_trend: reasons.append(("Aggressive new long",5))
        if np.isfinite(pcr) and pcr>=1.10: reasons.append(("PCR bullish (>1.10)",8))
        elif np.isfinite(pcr) and pcr>=0.90: reasons.append(("PCR supportive",4))
        if np.isfinite(pcrchg) and pcrchg>0: reasons.append(("PCR improving",min(5,round(abs(pcrchg)*100,1))))
        if np.isfinite(support) and close>support: reasons.append(("Price above Fib support",7))
        if np.isfinite(volume) and volume>=1.5: reasons.append(("Strong volume participation",5))
        if np.isfinite(delivery) and delivery>=1.5: reasons.append(("Strong delivery participation",3))

    elif put>call and put>=sell and put>=65:
        direction="PUT"
        preferred="ATM / nearest 1-step ITM PE"
        trigger=close-(0.20*atr if np.isfinite(atr) and atr>0 else close*0.005)
        invalidation=resistance if np.isfinite(resistance) and resistance>0 else close+(atr if np.isfinite(atr) and atr>0 else close*0.02)
        target1=support if np.isfinite(support) and support<trigger else close-(atr if np.isfinite(atr) and atr>0 else close*0.02)
        target2=target1-(atr if np.isfinite(atr) and atr>0 else close*0.02)

        if trend=="Bearish": reasons.append(("Bearish trend structure",15))
        if position in ("Short Buildup","Long Unwinding"): reasons.append((position,15))
        if "AggressiveNewShort" in oi_trend: reasons.append(("Aggressive new short",5))
        if np.isfinite(pcr) and pcr<=0.70: reasons.append(("PCR bearish (<0.70)",8))
        elif np.isfinite(pcr) and pcr<=0.90: reasons.append(("PCR bearish",4))
        if np.isfinite(pcrchg) and pcrchg<0: reasons.append(("PCR declining",min(5,round(abs(pcrchg)*100,1))))
        if np.isfinite(resistance) and close<resistance: reasons.append(("Price below Fib resistance",7))
        if np.isfinite(volume) and volume>=1.5: reasons.append(("Strong volume participation",5))
        if np.isfinite(delivery) and delivery>=1.5: reasons.append(("Strong delivery participation",3))

    else:
        direction="WAIT"
        preferred="No contract until directional confirmation"
        trigger=invalidation=target1=target2=np.nan
        reasons.append(("Directional scores not sufficiently separated",10))

    seen=set()
    clean=[]
    for reason,points in reasons:
        if reason not in seen:
            clean.append((reason,float(points)))
            seen.add(reason)

    return {
        "Direction":direction,
        "Preferred Contract":preferred,
        "Underlying Trigger":trigger,
        "Underlying Invalidation":invalidation,
        "Underlying Target 1":target1,
        "Underlying Target 2":target2,
        "Why Signal":" | ".join(r for r,_ in clean),
        "Explanation Points":round(sum(p for _,p in clean),1)
    }


def _options_phase1_contract_note(direction):
    if direction=="CALL":
        return "Prefer ATM or nearest 1-step ITM CE. Avoid far OTM contracts because premium/IV are unavailable."
    if direction=="PUT":
        return "Prefer ATM or nearest 1-step ITM PE. Avoid far OTM contracts because premium/IV are unavailable."
    return "Wait for directional confirmation before selecting an option contract."


def _options_phase1_rr(direction, entry, sl, t1, t2):
    vals=[entry,sl,t1,t2]
    if not all(np.isfinite(v) for v in vals):
        return np.nan,np.nan
    if direction=="CALL":
        risk=entry-sl
        return ((t1-entry)/risk if risk>0 else np.nan,
                (t2-entry)/risk if risk>0 else np.nan)
    if direction=="PUT":
        risk=sl-entry
        return ((entry-t1)/risk if risk>0 else np.nan,
                (entry-t2)/risk if risk>0 else np.nan)
    return np.nan,np.nan



# ============================================================
# OPTIONS PHASE 2: BOLLINGER BANDS + ADX + ATR
# ============================================================

def _options_bollinger_bands(df, period=20, std_mult=2.0):
    close=pd.to_numeric(df["Close"],errors="coerce")
    mid=close.rolling(period,min_periods=period).mean()
    std=close.rolling(period,min_periods=period).std(ddof=0)
    upper=mid+(std_mult*std)
    lower=mid-(std_mult*std)
    width=(upper-lower)/mid.replace(0,np.nan)
    percent_b=(close-lower)/(upper-lower).replace(0,np.nan)
    return mid,upper,lower,width,percent_b


def _options_adx(df, period=14):
    high=pd.to_numeric(df["High"],errors="coerce")
    low=pd.to_numeric(df["Low"],errors="coerce")
    close=pd.to_numeric(df["Close"],errors="coerce")

    up_move=high.diff()
    down_move=-low.diff()

    plus_dm=up_move.where(
        (up_move>down_move)&(up_move>0),0.0
    )
    minus_dm=down_move.where(
        (down_move>up_move)&(down_move>0),0.0
    )

    prev_close=close.shift(1)
    tr=pd.concat([
        high-low,
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ],axis=1).max(axis=1)

    atr=tr.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di=100*plus_dm.ewm(
        alpha=1/period,adjust=False,min_periods=period
    ).mean()/atr.replace(0,np.nan)

    minus_di=100*minus_dm.ewm(
        alpha=1/period,adjust=False,min_periods=period
    ).mean()/atr.replace(0,np.nan)

    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)

    adx=dx.ewm(
        alpha=1/period,adjust=False,min_periods=period
    ).mean()

    return adx,plus_di,minus_di


def _options_phase2_snapshot(df):
    """Calculate BB(20,2), ADX(14), +DI/-DI and ATR(14)."""
    if df is None or df.empty:
        return {}

    x=_options_normalize_columns(df)

    required=["High","Low","Close"]
    if any(c not in x.columns for c in required):
        return {}

    x=x.dropna(subset=required).copy()
    if len(x)<60:
        return {}

    for c in required:
        x[c]=pd.to_numeric(x[c],errors="coerce")

    mid,upper,lower,width,pctb=_options_bollinger_bands(x,20,2.0)
    adx,plus_di,minus_di=_options_adx(x,14)

    # ATR is calculated with the same Wilder-style smoothing used
    # in the existing independent Options Analyzer.
    atr=_options_atr(x,14)

    r=x.iloc[-1]
    prev=x.iloc[-2]

    vals={
        "BB Middle":float(mid.iloc[-1]),
        "BB Upper":float(upper.iloc[-1]),
        "BB Lower":float(lower.iloc[-1]),
        "BB Width":float(width.iloc[-1]),
        "BB %B":float(pctb.iloc[-1]),
        "BB Width 20D Percentile":float(
            width.tail(120).rank(pct=True).iloc[-1]*100
        ),
        "ADX14":float(adx.iloc[-1]),
        "+DI14":float(plus_di.iloc[-1]),
        "-DI14":float(minus_di.iloc[-1]),
        "ATR14":float(atr.iloc[-1]),
        "ATR %":float(
            atr.iloc[-1]/r["Close"]*100
        ) if float(r["Close"]) else np.nan,
        "BB Expansion":bool(
            np.isfinite(width.iloc[-1])
            and np.isfinite(width.iloc[-2])
            and width.iloc[-1]>width.iloc[-2]
        ),
        "BB Squeeze":bool(
            np.isfinite(width.iloc[-1])
            and width.iloc[-1]<=
            width.tail(120).quantile(0.20)
        ),
        "ADX Rising":bool(
            np.isfinite(adx.iloc[-1])
            and np.isfinite(adx.iloc[-2])
            and adx.iloc[-1]>adx.iloc[-2]
        ),
        "DI Bullish":bool(plus_di.iloc[-1]>minus_di.iloc[-1]),
        "DI Bearish":bool(minus_di.iloc[-1]>plus_di.iloc[-1])
    }

    close=float(r["Close"])

    if close>vals["BB Upper"]:
        bb_state="Upper Band Breakout"
    elif close<vals["BB Lower"]:
        bb_state="Lower Band Breakdown"
    elif close>vals["BB Middle"]:
        bb_state="Above BB Midline"
    else:
        bb_state="Below BB Midline"

    if vals["BB Squeeze"] and vals["BB Expansion"]:
        bb_state+=" + Squeeze Release"

    if vals["ADX14"]>=25 and vals["ADX Rising"]:
        adx_state="Strongening Trend"
    elif vals["ADX14"]>=25:
        adx_state="Strong Trend"
    elif vals["ADX14"]>=20 and vals["ADX Rising"]:
        adx_state="Developing Trend"
    else:
        adx_state="Weak / Range"

    vals["BB State"]=bb_state
    vals["ADX State"]=adx_state

    # Phase-2 confluence points are kept separate from the original score.
    bullish=0
    bearish=0
    selling=0

    if close>vals["BB Middle"]:
        bullish+=4
    if close<vals["BB Middle"]:
        bearish+=4
    if close>=vals["BB Upper"]:
        bullish+=5
    if close<=vals["BB Lower"]:
        bearish+=5

    if vals["BB Expansion"]:
        if close>vals["BB Middle"]:
            bullish+=3
        elif close<vals["BB Middle"]:
            bearish+=3

    if vals["BB Squeeze"]:
        selling+=5
        if vals["BB Expansion"]:
            if close>vals["BB Middle"]:
                bullish+=3
            elif close<vals["BB Middle"]:
                bearish+=3

    if vals["ADX14"]>=25:
        if vals["DI Bullish"]:
            bullish+=5
        elif vals["DI Bearish"]:
            bearish+=5
    elif vals["ADX14"]>=20 and vals["ADX Rising"]:
        if vals["DI Bullish"]:
            bullish+=3
        elif vals["DI Bearish"]:
            bearish+=3
    else:
        selling+=5

    if vals["ADX14"]<20:
        selling+=5

    return {
        **vals,
        "BB Bullish Points":min(15,bullish),
        "BB Bearish Points":min(15,bearish),
        "BB/ADX Selling Points":min(15,selling)
    }



# ============================================================
# OPTIONS PHASE 3: CONFLUENCE + CONFLICT + MARKET REGIME
# ============================================================

def _options_phase3_confluence(row):
    call=put=sell=0.0
    bullish=[]; bearish=[]; neutral=[]; conflicts=[]

    position=str(row.get("Position",""))
    oi_trend=str(row.get("OI Trend",""))
    chg=float(row.get("Daily Chg %",np.nan))
    trend=str(row.get("Trend",""))
    close=float(row.get("Close",np.nan))
    ema20=float(row.get("EMA20",np.nan))
    ema50=float(row.get("EMA50",np.nan))
    ema200=float(row.get("EMA200",np.nan))

    if position=="Long Buildup": call+=20; bullish.append("Long buildup")
    elif position=="Short Covering": call+=14; bullish.append("Short covering")
    elif position=="Short Buildup": put+=20; bearish.append("Short buildup")
    elif position=="Long Unwinding": put+=14; bearish.append("Long unwinding")

    if "AggressiveNewLong" in oi_trend:
        call+=5; bullish.append("Aggressive new long")
    elif "AggressiveNewShort" in oi_trend:
        put+=5; bearish.append("Aggressive new short")

    ema_bull=all(np.isfinite(x) for x in [close,ema20,ema50,ema200]) and close>ema20>ema50>ema200
    ema_bear=all(np.isfinite(x) for x in [close,ema20,ema50,ema200]) and close<ema20<ema50<ema200

    if ema_bull: call+=15; bullish.append("EMA20 > EMA50 > EMA200")
    elif ema_bear: put+=15; bearish.append("EMA20 < EMA50 < EMA200")
    elif trend=="Bullish": call+=9; bullish.append("Bullish trend")
    elif trend=="Bearish": put+=9; bearish.append("Bearish trend")

    bb_state=str(row.get("BB State",""))
    if "Upper Band Breakout" in bb_state:
        call+=10; bullish.append("Upper Bollinger breakout")
    elif "Lower Band Breakdown" in bb_state:
        put+=10; bearish.append("Lower Bollinger breakdown")
    elif "Above BB Midline" in bb_state:
        call+=5; bullish.append("Above Bollinger midline")
    elif "Below BB Midline" in bb_state:
        put+=5; bearish.append("Below Bollinger midline")
    if "Squeeze Release" in bb_state:
        if "Upper" in bb_state: call+=3
        elif "Lower" in bb_state: put+=3

    adx=float(row.get("ADX14",np.nan)); plus=float(row.get("+DI14",np.nan)); minus=float(row.get("-DI14",np.nan))
    adx_rising=bool(row.get("ADX Rising",False))
    if np.isfinite(adx) and adx>=25:
        if plus>minus: call+=10; bullish.append(f"ADX {adx:.1f} +DI dominant")
        elif minus>plus: put+=10; bearish.append(f"ADX {adx:.1f} -DI dominant")
    elif np.isfinite(adx) and adx>=20 and adx_rising:
        if plus>minus: call+=6; bullish.append("Developing ADX +DI dominant")
        elif minus>plus: put+=6; bearish.append("Developing ADX -DI dominant")
    elif np.isfinite(adx) and adx<20:
        sell+=5; neutral.append("ADX below 20")

    support=float(row.get("Fib Support Price",np.nan)); resistance=float(row.get("Fib Resistance Price",np.nan))
    if np.isfinite(close) and np.isfinite(support):
        d=(close-support)/close*100
        if 0<=d<=2: call+=10; bullish.append("Within 2% of Fib support")
        elif 0<=d<=4: call+=5
    if np.isfinite(close) and np.isfinite(resistance):
        d=(resistance-close)/close*100
        if 0<=d<=2: put+=10; bearish.append("Within 2% of Fib resistance")
        elif 0<=d<=4: put+=5

    pcr=float(row.get("PCR",np.nan)); pcrchg=float(row.get("PCR Chg 1D",np.nan))
    call_oi=float(row.get("Call OI",np.nan)); put_oi=float(row.get("Put OI",np.nan))
    if np.isfinite(pcr):
        if pcr>=1.10: call+=5; bullish.append(f"PCR bullish ({pcr:.2f})")
        elif pcr<=0.70: put+=5; bearish.append(f"PCR bearish ({pcr:.2f})")
        elif .85<=pcr<=1.15: sell+=2; neutral.append("PCR near balance")
    if np.isfinite(pcrchg):
        if pcrchg>0: call+=3; bullish.append("PCR improving")
        elif pcrchg<0: put+=3; bearish.append("PCR declining")
    if np.isfinite(call_oi) and np.isfinite(put_oi) and call_oi>0 and put_oi>0:
        share=put_oi/(call_oi+put_oi)
        if share>=.60: call+=2; bullish.append("Put OI dominance")
        elif share<=.40: put+=2; bearish.append("Call OI dominance")

    vol=float(row.get("Volume x",np.nan)); delivery=float(row.get("Delivery x",np.nan))
    if np.isfinite(vol) and vol>=1.5:
        if chg>0: call+=5; bullish.append("High volume with positive price")
        elif chg<0: put+=5; bearish.append("High volume with negative price")
    if np.isfinite(delivery) and delivery>=1.5:
        if chg>0: call+=3
        elif chg<0: put+=3

    squeeze=bool(row.get("BB Squeeze",False))
    if squeeze: sell+=5; neutral.append("Bollinger squeeze")
    if np.isfinite(adx) and adx<20: sell+=5

    if position in ("Long Buildup","Short Covering") and ("Lower Band Breakdown" in bb_state or (np.isfinite(adx) and adx>=25 and minus>plus)):
        conflicts.append("Bullish derivatives conflict with bearish momentum")
    if position in ("Short Buildup","Long Unwinding") and ("Upper Band Breakout" in bb_state or (np.isfinite(adx) and adx>=25 and plus>minus)):
        conflicts.append("Bearish derivatives conflict with bullish momentum")
    if np.isfinite(pcr) and pcr>=1.10 and (bb_state.startswith("Lower") or ema_bear):
        conflicts.append("Bullish PCR conflicts with bearish technical structure")
    if np.isfinite(pcr) and pcr<=.70 and (bb_state.startswith("Upper") or ema_bull):
        conflicts.append("Bearish PCR conflicts with bullish technical structure")

    penalty=min(15,5*len(conflicts))
    call=max(0,call-penalty); put=max(0,put-penalty)

    if ema_bull and np.isfinite(adx) and adx>=25 and plus>minus and ("Upper" in bb_state or "Squeeze Release" in bb_state):
        regime="🚀 Trending Bull"
    elif ema_bear and np.isfinite(adx) and adx>=25 and minus>plus and ("Lower" in bb_state or "Squeeze Release" in bb_state):
        regime="🔻 Trending Bear"
    elif squeeze and np.isfinite(adx) and adx<20:
        regime="🟡 Range / Compression"
    elif np.isfinite(adx) and adx<20:
        regime="🟡 Weak / Range"
    elif trend=="Bullish":
        regime="🟢 Bullish Developing"
    elif trend=="Bearish":
        regime="🔴 Bearish Developing"
    else:
        regime="⚪ Mixed / Neutral"

    return {
        "Phase 3 Call Score":round(min(100,call),1),
        "Phase 3 Put Score":round(min(100,put),1),
        "Phase 3 Selling Score":round(min(100,sell),1),
        "Market Regime":regime,
        "Conflict Status":"No major conflict" if not conflicts else f"⚠️ {len(conflicts)} conflict(s)",
        "Conflict Details":" | ".join(conflicts) if conflicts else "None",
        "Bullish Factors":" | ".join(bullish),
        "Bearish Factors":" | ".join(bearish),
        "Neutral Factors":" | ".join(neutral),
        "Conflict Count":len(conflicts)
    }


def _options_phase3_final_signal(row):
    c=float(row.get("Phase 3 Call Score",0) or 0)
    p=float(row.get("Phase 3 Put Score",0) or 0)
    s=float(row.get("Phase 3 Selling Score",0) or 0)
    conflicts=int(row.get("Conflict Count",0) or 0)
    regime=str(row.get("Market Regime",""))
    if conflicts>=2: return "⚠️ Conflicted / Wait"
    if c>=80 and c>p+8 and regime in ("🚀 Trending Bull","🟢 Bullish Developing"):
        return "🟢 Strong Call Candidate"
    if p>=80 and p>c+8 and regime in ("🔻 Trending Bear","🔴 Bearish Developing"):
        return "🔴 Strong Put Candidate"
    if s>=75 and regime in ("🟡 Range / Compression","🟡 Weak / Range"):
        return "🟡 Option Selling Candidate"
    if max(c,p)>=65: return "🔵 Option Buying Candidate"
    return "⚪ Avoid / Wait"



# ============================================================
# OPTIONS PHASE 4: BACKTESTING & VALIDATION
# ============================================================

def _options_phase4_signal_plan(signal, close, atr, support, resistance):
    """Convert a historical signal into underlying trigger/SL/targets."""
    if not np.isfinite(close):
        return None

    atr_use=atr if np.isfinite(atr) and atr>0 else close*0.02

    if signal=="🟢 Strong Call Candidate":
        trigger=close+0.20*atr_use
        sl=support if np.isfinite(support) and support<close else close-atr_use
        if sl>=trigger:
            sl=close-atr_use
        t1=resistance if np.isfinite(resistance) and resistance>trigger else trigger+atr_use
        t2=t1+atr_use
        return "CALL",trigger,sl,t1,t2

    if signal=="🔴 Strong Put Candidate":
        trigger=close-0.20*atr_use
        sl=resistance if np.isfinite(resistance) and resistance>close else close+atr_use
        if sl<=trigger:
            sl=close+atr_use
        t1=support if np.isfinite(support) and support<trigger else trigger-atr_use
        t2=t1-atr_use
        return "PUT",trigger,sl,t1,t2

    return None


def _options_phase4_evaluate_path(signal, trigger, sl, t1, t2,
                                  next_open, next_high, next_low,
                                  next_close):
    """Evaluate the next trading session using OHLC only.

    Conservative ambiguity rule:
    if both SL and T1 are touched in the same candle, SL is assumed
    to occur first. This avoids overstating historical performance.
    """
    if not all(np.isfinite(v) for v in [
        trigger,sl,t1,t2,next_open,next_high,next_low,next_close
    ]):
        return {
            "Outcome":"Insufficient Data",
            "Return %":np.nan,
            "Target 1 Hit":False,
            "Target 2 Hit":False,
            "Stop Hit":False,
            "Triggered":False
        }

    if signal=="CALL":
        triggered=next_high>=trigger
        if not triggered:
            return {
                "Outcome":"Not Triggered",
                "Return %":(next_close-next_open)/next_open*100,
                "Target 1 Hit":False,
                "Target 2 Hit":False,
                "Stop Hit":False,
                "Triggered":False
            }

        stop_hit=next_low<=sl
        t1_hit=next_high>=t1
        t2_hit=next_high>=t2

        if stop_hit:
            exit_price=sl
            outcome="Stop Hit"
        elif t2_hit:
            exit_price=t2
            outcome="Target 2 Hit"
        elif t1_hit:
            exit_price=t1
            outcome="Target 1 Hit"
        else:
            exit_price=next_close
            outcome="Close Exit"

        return {
            "Outcome":outcome,
            "Return %":(exit_price-trigger)/trigger*100,
            "Target 1 Hit":bool(t1_hit),
            "Target 2 Hit":bool(t2_hit),
            "Stop Hit":bool(stop_hit),
            "Triggered":True
        }

    if signal=="PUT":
        triggered=next_low<=trigger
        if not triggered:
            return {
                "Outcome":"Not Triggered",
                "Return %":(next_open-next_close)/next_open*100,
                "Target 1 Hit":False,
                "Target 2 Hit":False,
                "Stop Hit":False,
                "Triggered":False
            }

        stop_hit=next_high>=sl
        t1_hit=next_low<=t1
        t2_hit=next_low<=t2

        if stop_hit:
            exit_price=sl
            outcome="Stop Hit"
        elif t2_hit:
            exit_price=t2
            outcome="Target 2 Hit"
        elif t1_hit:
            exit_price=t1
            outcome="Target 1 Hit"
        else:
            exit_price=next_close
            outcome="Close Exit"

        return {
            "Outcome":outcome,
            "Return %":(trigger-exit_price)/trigger*100,
            "Target 1 Hit":bool(t1_hit),
            "Target 2 Hit":bool(t2_hit),
            "Stop Hit":bool(stop_hit),
            "Triggered":True
        }

    return {
        "Outcome":"Unsupported",
        "Return %":np.nan,
        "Target 1 Hit":False,
        "Target 2 Hit":False,
        "Stop Hit":False,
        "Triggered":False
    }


def _options_phase4_build_backtest(history, min_score=80):
    """Backtest Phase-3 signals using a dictionary of symbol -> OHLC DataFrame.

    The historical signal itself must already be computed from data available
    on that signal date. The following trading session is used for validation.
    """
    records=[]

    for symbol, df in history.items():
        if df is None or df.empty:
            continue

        x=df.copy()
        x=_options_normalize_columns(x)

        needed=["Open","High","Low","Close"]
        if any(c not in x.columns for c in needed):
            continue

        x=x.dropna(subset=needed).copy()
        if len(x)<80:
            continue

        for c in needed:
            x[c]=pd.to_numeric(x[c],errors="coerce")

        # Compute all Phase-3 technical indicators on each historical date.
        # Existing functions are intentionally reused to keep the validation
        # rules identical to the live analyzer.
        close=x["Close"]
        x["EMA20"]=close.ewm(span=20,adjust=False,min_periods=20).mean()
        x["EMA50"]=close.ewm(span=50,adjust=False,min_periods=50).mean()
        x["EMA200"]=close.ewm(span=200,adjust=False,min_periods=200).mean()

        if len(x)<205:
            continue

        # Lightweight daily volume ratio / delivery ratio where available.
        if "Volume" in x.columns:
            vol=pd.to_numeric(x["Volume"],errors="coerce")
            x["Volume x"]=vol/vol.rolling(20,min_periods=20).mean()
        else:
            x["Volume x"]=np.nan

        # Phase-2 indicators.
        mid,upper,lower,width,pctb=_options_bollinger_bands(x,20,2.0)
        adx,plus_di,minus_di=_options_adx(x,14)
        atr=_options_atr(x,14)

        x["BB Middle"]=mid
        x["BB Upper"]=upper
        x["BB Lower"]=lower
        x["BB Width"]=width
        x["BB %B"]=pctb
        x["ADX14"]=adx
        x["+DI14"]=plus_di
        x["-DI14"]=minus_di
        x["ATR14"]=atr
        x["ADX Rising"]=adx.diff()>0
        x["BB Squeeze"]=width<=width.rolling(120,min_periods=40).quantile(.20)

        # We validate the technical/regime portion here. Full OI/PCR
        # validation requires historical derivatives columns for each date.
        for i in range(205,len(x)-1):
            row=x.iloc[i]
            nxt=x.iloc[i+1]

            # Build the technical signal/regime without future data.
            bb_state=""
            if row["Close"]>row["BB Upper"]:
                bb_state="Upper Band Breakout"
            elif row["Close"]<row["BB Lower"]:
                bb_state="Lower Band Breakdown"
            elif row["Close"]>row["BB Middle"]:
                bb_state="Above BB Midline"
            else:
                bb_state="Below BB Midline"

            if bool(row["BB Squeeze"]) and i>0 and row["BB Width"]>x["BB Width"].iloc[i-1]:
                bb_state+=" + Squeeze Release"

            trend=(
                "Bullish" if row["Close"]>row["EMA200"] and row["EMA20"]>row["EMA50"]
                else "Bearish" if row["Close"]<row["EMA200"] and row["EMA20"]<row["EMA50"]
                else "Neutral"
            )

            # Technical-only Phase-3 direction score.
            call=0.0; put=0.0
            if trend=="Bullish": call+=15
            elif trend=="Bearish": put+=15

            if "Upper Band Breakout" in bb_state: call+=10
            elif "Lower Band Breakdown" in bb_state: put+=10
            elif "Above BB Midline" in bb_state: call+=5
            elif "Below BB Midline" in bb_state: put+=5

            if np.isfinite(row["ADX14"]) and row["ADX14"]>=25:
                if row["+DI14"]>row["-DI14"]: call+=10
                elif row["-DI14"]>row["+DI14"]: put+=10

            support=np.nan
            resistance=np.nan
            # Rolling swing levels are used only as a conservative proxy when
            # historical Fib levels are not supplied.
            prior_low=x["Low"].iloc[max(0,i-60):i].min()
            prior_high=x["High"].iloc[max(0,i-60):i].max()
            if np.isfinite(prior_low): support=float(prior_low)
            if np.isfinite(prior_high): resistance=float(prior_high)

            # Require meaningful technical confluence.
            if call>=25 and call>put+5:
                signal="🟢 Strong Call Candidate"
            elif put>=25 and put>call+5:
                signal="🔴 Strong Put Candidate"
            else:
                continue

            plan=_options_phase4_signal_plan(
                signal,float(row["Close"]),float(row["ATR14"]),
                support,resistance
            )
            if plan is None:
                continue

            direction,trigger,sl,t1,t2=plan
            result=_options_phase4_evaluate_path(
                direction,trigger,sl,t1,t2,
                float(nxt["Open"]),float(nxt["High"]),
                float(nxt["Low"]),float(nxt["Close"])
            )

            records.append({
                "Symbol":symbol,
                "Signal Date":x.index[i],
                "Next Date":x.index[i+1],
                "Signal":signal,
                "Technical Score":max(call,put),
                "Regime":(
                    "🚀 Trending Bull" if call>=25 and row["ADX14"]>=25
                    else "🔻 Trending Bear" if put>=25 and row["ADX14"]>=25
                    else "🟡 Other"
                ),
                "Close":float(row["Close"]),
                "Trigger":trigger,
                "Stop":sl,
                "Target 1":t1,
                "Target 2":t2,
                **result
            })

    return pd.DataFrame(records)


def _options_phase4_metrics(bt):
    if bt is None or bt.empty:
        return {}

    triggered=bt[bt["Triggered"]==True].copy()
    if triggered.empty:
        return {
            "Signals":len(bt),
            "Triggered":0
        }

    ret=pd.to_numeric(triggered["Return %"],errors="coerce").dropna()
    wins=ret[ret>0]
    losses=ret[ret<0]

    gross_profit=wins.sum()
    gross_loss=abs(losses.sum())

    equity=(1+ret/100).cumprod()
    running_max=equity.cummax()
    drawdown=(equity/running_max-1)*100

    return {
        "Signals":len(bt),
        "Triggered":len(triggered),
        "Win Rate %":len(wins)/len(ret)*100,
        "Average Return %":ret.mean(),
        "Median Return %":ret.median(),
        "Best Return %":ret.max(),
        "Worst Return %":ret.min(),
        "Profit Factor":gross_profit/gross_loss if gross_loss>0 else np.inf,
        "Max Drawdown %":drawdown.min(),
        "Cumulative Return %":(equity.iloc[-1]-1)*100
    }



def _options_normalize_columns(df):
    """Normalize yfinance/CSV columns without assuming MultiIndex."""
    x=df.copy()

    # A true MultiIndex can safely be flattened.
    if isinstance(x.columns, pd.MultiIndex):
        names=[]
        for tup in x.columns.to_list():
            vals=[str(v).strip() for v in tup if str(v).strip().lower() not in ("nan","none")]
            names.append(vals[0] if vals else "")
        x.columns=names
    else:
        # Some CSVs contain tuple-looking column labels as strings.
        cleaned=[]
        for c in x.columns:
            s=str(c).strip()
            # Remove common tuple wrappers only when they are textual.
            if len(s)>=4 and s[0]=="(" and s[-1]==")":
                parts=[z.strip().strip("'\"") for z in s[1:-1].split(",")]
                s=parts[0] if parts else s
            cleaned.append(s)
        x.columns=cleaned

    # Normalize common OHLC names.
    aliases={
        "adj close":"Adj Close",
        "adj_close":"Adj Close",
        "open":"Open",
        "high":"High",
        "low":"Low",
        "close":"Close",
        "volume":"Volume",
        "date":"Date",
        "datetime":"Date",
        "timestamp":"Date"
    }
    x.columns=[
        aliases.get(str(c).strip().lower(),str(c).strip())
        for c in x.columns
    ]
    return x




# ============================================================
# MOMENTUM CATALYST SCANNER
# Price + Volume + Trend + Breakout + Relative Strength + Fundamentals
# ============================================================
def _mcs_prepare(df):
    if df is None or df.empty: return pd.DataFrame()
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=[next((str(v).strip() for v in tup if str(v).strip().lower() not in ("nan","none")),"") for tup in x.columns.to_list()]
    else: x.columns=[str(c).strip() for c in x.columns]
    ren={}
    for c in x.columns:
        k=str(c).lower().replace('_',' ').strip()
        if k in ('open','high','low','close','volume'): ren[c]=k.title()
        elif k in ('date','datetime','timestamp'): ren[c]='Date'
    x=x.rename(columns=ren)
    if 'Date' in x.columns:
        x['Date']=pd.to_datetime(x['Date'],errors='coerce'); x=x.dropna(subset=['Date']).set_index('Date')
    elif not isinstance(x.index,pd.DatetimeIndex):
        x.index=pd.to_datetime(x.index,errors='coerce'); x=x[~x.index.isna()]
    need=['Open','High','Low','Close']
    if any(c not in x.columns for c in need): return pd.DataFrame()
    for c in need+(['Volume'] if 'Volume' in x.columns else []): x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=need).sort_index()
    if len(x)<205: return pd.DataFrame()
    c=x['Close']; x['SMA20']=c.rolling(20).mean(); x['SMA50']=c.rolling(50).mean(); x['SMA200']=c.rolling(200).mean()
    x['Return 1D %']=c.pct_change()*100; x['Return 20D %']=c.pct_change(20)*100; x['Return 60D %']=c.pct_change(60)*100
    x['High20Prev']=x['High'].shift(1).rolling(20).max(); x['High50Prev']=x['High'].shift(1).rolling(50).max()
    x['Volume Ratio']=x['Volume']/x['Volume'].rolling(20).mean() if 'Volume' in x.columns else np.nan
    x['52WHigh']=x['High'].rolling(252,min_periods=120).max(); x['Distance 52W High %']=(c/x['52WHigh']-1)*100
    tr=pd.concat([x['High']-x['Low'],(x['High']-c.shift()).abs(),(x['Low']-c.shift()).abs()],axis=1).max(axis=1)
    x['ATR14']=tr.rolling(14).mean()
    return x.dropna(subset=['SMA50','SMA200'])

def _mcs_scan_one(symbol,df,benchmark=None,fundamentals=None):
    x=_mcs_prepare(df)
    if x.empty: return None
    r=x.iloc[-1]; f=fundamentals or {}
    def num(k):
        v=pd.to_numeric(f.get(k),errors='coerce') if f.get(k) is not None else np.nan
        return float(v) if pd.notna(v) else np.nan
    annual=num('annual'); pat=num('pat'); ebitda=num('ebitda'); order=num('order'); catalyst=str(f.get('catalyst') or '').strip()
    score=0; reasons=[]; warnings=[]
    ret1=float(r['Return 1D %']); ret20=float(r['Return 20D %'])
    if ret1>=4: score+=10; reasons.append(f'1D +{ret1:.1f}%')
    elif ret1>=3: score+=7; reasons.append(f'1D +{ret1:.1f}%')
    if ret20>=10: score+=10; reasons.append(f'20D +{ret20:.1f}%')
    elif ret20>=5: score+=6
    vr=float(r['Volume Ratio']) if np.isfinite(r['Volume Ratio']) else np.nan
    if np.isfinite(vr):
        if vr>=2: score+=20; reasons.append(f'Volume {vr:.1f}x 20D avg')
        elif vr>=1.5: score+=15; reasons.append(f'Volume {vr:.1f}x 20D avg')
        elif vr>=1.2: score+=8
        else: warnings.append('No strong volume expansion')
    else: warnings.append('Volume unavailable')
    trend=0
    if r['Close']>r['SMA20']: trend+=5
    if r['Close']>r['SMA50']: trend+=5
    if r['Close']>r['SMA200']: trend+=5
    if r['SMA50']>r['SMA200']: trend+=5
    score+=trend
    if trend>=15: reasons.append('Bullish 20/50/200 structure')
    breakout='None'
    if r['Close']>r['High50Prev']: breakout='50D Breakout'; score+=15; reasons.append('50-day high breakout')
    elif r['Close']>r['High20Prev']: breakout='20D Breakout'; score+=10; reasons.append('20-day high breakout')
    elif r['Close']>r['SMA20'] and ret20>0: breakout='Emerging'; score+=4
    else: warnings.append('No price breakout')
    rs20=rs60=np.nan
    if benchmark is not None and not benchmark.empty:
        b=_mcs_prepare(benchmark)
        if not b.empty:
            rs20=ret20-float(b.iloc[-1]['Return 20D %']); rs60=float(r['Return 60D %'])-float(b.iloc[-1]['Return 60D %'])
    if np.isfinite(rs20):
        if rs20>=5: score+=6; reasons.append(f'20D RS +{rs20:.1f}% vs Nifty')
        elif rs20>=2: score+=4
    if np.isfinite(rs60) and rs60>=5: score+=4
    fund=0
    if np.isfinite(annual):
        if annual>=20: fund+=4; reasons.append(f'Revenue growth {annual:.1f}%')
        elif annual>=15: fund+=2
    if np.isfinite(pat):
        if pat>=50: fund+=3; reasons.append(f'PAT growth {pat:.1f}%')
        elif pat>=25: fund+=2
    if np.isfinite(ebitda) and ebitda>=20: fund+=2; reasons.append(f'EBITDA growth {ebitda:.1f}%')
    if np.isfinite(order) and order>=15: fund+=1; reasons.append(f'Order book growth {order:.1f}%')
    score+=min(10,fund)
    if catalyst: score+=5; reasons.append('Catalyst supplied')
    else: warnings.append('Catalyst/news not supplied')
    score=min(100,int(round(score)))
    grade='🟢 Explosive Momentum' if score>=80 else '🟢 Strong Momentum' if score>=70 else '🟡 Emerging Momentum' if score>=60 else '⚪ Watchlist'
    ready=bool(score>=75 and r['Close']>r['SMA50']>r['SMA200'] and np.isfinite(vr) and vr>=1.2)
    entry=float(r['Close']); atr=float(r['ATR14']) if np.isfinite(r['ATR14']) else entry*.02; sl=max(.01,entry-1.5*atr); risk=entry-sl
    return {'Symbol':symbol,'Score':score,'Rating':grade,'Trade Ready':ready,'Close':entry,'1D %':ret1,'20D %':ret20,'60D %':float(r['Return 60D %']),'Volume Ratio':vr,'Breakout':breakout,'RS 20D %':rs20,'RS 60D %':rs60,'SMA20':float(r['SMA20']),'SMA50':float(r['SMA50']),'SMA200':float(r['SMA200']),'SMA50>SMA200':bool(r['SMA50']>r['SMA200']),'52W High Distance %':float(r['Distance 52W High %']),'Revenue Growth %':annual,'PAT Growth %':pat,'EBITDA Growth %':ebitda,'Order Book Growth %':order,'Catalyst':catalyst or 'Not supplied','Entry Reference':entry,'ATR14':atr,'Suggested SL':sl,'Target 1':entry+2*risk,'Target 2':entry+3*risk,'Reasons':' | '.join(reasons),'Warnings':' | '.join(warnings)}

def _mcs_read_fundamentals(uploaded_file):
    try:
        f=pd.read_csv(uploaded_file); cols={str(c).strip().lower():c for c in f.columns}
        sym=next((cols[k] for k in ('symbol','ticker','stock','code') if k in cols),None)
        if sym is None: return {}
        def find(keys): return next((cols[k] for k in keys if k in cols),None)
        ac=find(('annual revenue growth %','revenue growth %','revenue growth')); pc=find(('pat growth %','pat growth','profit growth %','profit growth')); ec=find(('ebitda growth %','ebitda growth')); oc=find(('order book growth %','order book growth','order growth %')); cc=find(('catalyst','news catalyst','event catalyst'))
        out={}
        for _,r in f.iterrows():
            s=str(r[sym]).strip().upper()
            if not s or s=='NAN': continue
            def num(c):
                if c is None: return None
                v=pd.to_numeric(r[c],errors='coerce'); return float(v) if pd.notna(v) else None
            out[s]={'annual':num(ac),'pat':num(pc),'ebitda':num(ec),'order':num(oc),'catalyst':str(r[cc]).strip() if cc and pd.notna(r[cc]) else ''}
        return out
    except Exception: return {}

# ============================================================
# KRATTER MOMENTUM SCANNER
# Based mechanically on "Learn to Trade Momentum Stocks"
# by Matthew R. Kratter.
# ============================================================

def _kratter_prepare_ohlcv(df):
    """Normalize and prepare daily OHLCV data for the book rules."""
    if df is None or df.empty:
        return pd.DataFrame()

    x=df.copy()

    if isinstance(x.columns,pd.MultiIndex):
        cols=[]
        for tup in x.columns.to_list():
            vals=[str(v).strip() for v in tup
                  if str(v).strip().lower() not in ("nan","none")]
            cols.append(vals[-1] if vals else "")
        x.columns=cols
    else:
        x.columns=[str(c).strip() for c in x.columns]

    rename={}
    for c in x.columns:
        key=str(c).strip().lower()
        if key in ("open","high","low","close","volume"):
            rename[c]=key.title()
        elif key in ("adj close","adj_close"):
            rename[c]="Adj Close"
        elif key in ("date","datetime","timestamp"):
            rename[c]="Date"
    x=x.rename(columns=rename)

    if "Date" in x.columns:
        x["Date"]=pd.to_datetime(x["Date"],errors="coerce")
        x=x.dropna(subset=["Date"]).set_index("Date")
    elif not isinstance(x.index,pd.DatetimeIndex):
        try:
            x.index=pd.to_datetime(x.index)
        except Exception:
            return pd.DataFrame()

    needed=["Open","High","Low","Close"]
    if any(c not in x.columns for c in needed):
        return pd.DataFrame()

    for c in needed+([ "Volume" ] if "Volume" in x.columns else []):
        x[c]=pd.to_numeric(x[c],errors="coerce")

    x=x.dropna(subset=needed).sort_index()

    if len(x)<205:
        return pd.DataFrame()

    x["SMA50"]=x["Close"].rolling(50,min_periods=50).mean()
    x["SMA200"]=x["Close"].rolling(200,min_periods=200).mean()

    # Exact book buy trigger:
    # previous SMA50 <= previous SMA200 AND
    # latest SMA50 > latest SMA200 AND
    # latest close > latest SMA50.
    x["Golden Cross"]=(
        (x["SMA50"].shift(1)<=x["SMA200"].shift(1)) &
        (x["SMA50"]>x["SMA200"])
    )
    x["Price Above SMA50"]=x["Close"]>x["SMA50"]

    # Useful context, not an extra book rule.
    x["SMA50 Slope %"]=(x["SMA50"]/x["SMA50"].shift(20)-1)*100
    x["SMA200 Slope %"]=(x["SMA200"]/x["SMA200"].shift(20)-1)*100
    x["52W High"]=x["Close"].rolling(252,min_periods=120).max()
    x["Distance from 52W High %"]=(x["Close"]/x["52W High"]-1)*100

    return x.dropna(subset=["SMA50","SMA200"])


def _kratter_scan_one(symbol, df, revenue_growth=None,
                      revenue_3y_growth=None, category=None):
    x=_kratter_prepare_ohlcv(df)
    if x.empty:
        return None

    row=x.iloc[-1]
    signal=bool(row["Golden Cross"] and row["Price Above SMA50"])

    annual_ok=(
        revenue_growth is not None and
        np.isfinite(revenue_growth) and
        revenue_growth>20
    )
    three_year_ok=(
        revenue_3y_growth is not None and
        np.isfinite(revenue_3y_growth) and
        revenue_3y_growth>20
    )

    # The book describes >20% annual revenue growth as the rule of thumb,
    # with 3-year average growth >20% as ideal. These are kept separate.
    fundamental_status=(
        "Confirmed" if annual_ok and three_year_ok
        else "Annual >20%" if annual_ok
        else "Not supplied"
    )

    # Exact book candidate requires the technical buy signal.
    if signal:
        if annual_ok:
            grade="🟢 BOOK MOMENTUM BUY"
        else:
            grade="🟡 TECHNICAL BUY — FUNDAMENTAL CHECK NEEDED"
    else:
        grade="—"

    return {
        "Symbol":symbol,
        "Signal":grade,
        "Signal Date":x.index[-1],
        "Close":float(row["Close"]),
        "SMA50":float(row["SMA50"]),
        "SMA200":float(row["SMA200"]),
        "50/200 Cross":bool(row["Golden Cross"]),
        "Close > SMA50":bool(row["Price Above SMA50"]),
        "Annual Revenue Growth %":(
            float(revenue_growth) if revenue_growth is not None
            and np.isfinite(revenue_growth) else np.nan
        ),
        "3Y Avg Revenue Growth %":(
            float(revenue_3y_growth) if revenue_3y_growth is not None
            and np.isfinite(revenue_3y_growth) else np.nan
        ),
        "Fundamental Status":fundamental_status,
        "Company Type":category if category else "Not supplied",
        "SMA50 Slope %":float(row["SMA50 Slope %"]),
        "SMA200 Slope %":float(row["SMA200 Slope %"]),
        "Distance from 52W High %":float(row["Distance from 52W High %"]),
        "Technical Buy":signal,
    }


def _kratter_position_plan(entry_price, account_size, risk_pct=2.0):
    """Book-style position sizing: risk a fixed % of account with a 15% stop."""
    if not np.isfinite(entry_price) or entry_price<=0:
        return {}
    stop=entry_price*0.85
    target=entry_price*4.0
    risk_per_share=entry_price-stop
    risk_amount=account_size*(risk_pct/100)
    shares=int(np.floor(risk_amount/risk_per_share))
    allocation=shares*entry_price
    return {
        "Entry":float(entry_price),
        "Stop (-15%)":float(stop),
        "Target (+300%)":float(target),
        "Risk/Share":float(risk_per_share),
        "Account Risk":float(risk_amount),
        "Shares":max(0,shares),
        "Capital Required":float(allocation),
    }


@st.cache_data(ttl=900, show_spinner=False)
def _kratter_download_batches(tickers, batch_size=40):
    """Serial/batched Yahoo download with graceful failure."""
    result={}
    ticker_list=list(dict.fromkeys([str(x).strip() for x in tickers if str(x).strip()]))

    for start in range(0,len(ticker_list),batch_size):
        batch=ticker_list[start:start+batch_size]
        yahoo=[s if s.endswith(".NS") or s.startswith("^") else s+".NS" for s in batch]
        try:
            d=yf.download(
                tickers=yahoo,
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker"
            )
        except Exception:
            continue

        if d is None or d.empty:
            continue

        # Single ticker response.
        if len(batch)==1:
            symbol=batch[0]
            stock=d.copy()
            if isinstance(stock.columns,pd.MultiIndex):
                level0=stock.columns.get_level_values(0)
                level1=stock.columns.get_level_values(1)
                if "Close" in level0:
                    stock.columns=level0
                elif "Close" in level1:
                    stock.columns=level1
            result[symbol]=stock
            continue

        # Multi-ticker response: extract each symbol independently.
        if isinstance(d.columns,pd.MultiIndex):
            for symbol,yahoo_symbol in zip(batch,yahoo):
                candidates=[yahoo_symbol,symbol]
                stock=None
                for key in candidates:
                    try:
                        if key in d.columns.get_level_values(0):
                            stock=d[key].copy()
                            break
                        if key in d.columns.get_level_values(1):
                            stock=d.xs(key,axis=1,level=1).copy()
                            break
                    except Exception:
                        pass
                if stock is not None and not stock.empty:
                    result[symbol]=stock
        else:
            # Rare fallback: only useful if one unnamed block was returned.
            if len(batch)==1:
                result[batch[0]]=d.copy()

    return result


def _kratter_read_fundamental_csv(uploaded_file):
    """Optional CSV: Symbol, Annual Revenue Growth %, 3Y Avg Revenue Growth %, Company Type."""
    try:
        f=pd.read_csv(uploaded_file)
        cols={str(c).strip().lower():c for c in f.columns}

        sym_col=next((cols[k] for k in
                      ["symbol","ticker","stock","code"] if k in cols),None)
        if sym_col is None:
            return {}

        def find_col(keys):
            for k in keys:
                if k in cols:
                    return cols[k]
            return None

        annual_col=find_col([
            "annual revenue growth %",
            "annual revenue growth",
            "revenue growth %",
            "revenue growth"
        ])
        avg3_col=find_col([
            "3y avg revenue growth %",
            "3 year average revenue growth %",
            "3y revenue growth %",
            "3-year average revenue growth"
        ])
        type_col=find_col(["company type","category","type"])

        out={}
        for _,r in f.iterrows():
            sym=str(r[sym_col]).strip().upper()
            if not sym or sym=="NAN":
                continue
            def num(col):
                if col is None:
                    return None
                v=pd.to_numeric(r[col],errors="coerce")
                return float(v) if pd.notna(v) else None
            out[sym]={
                "annual":num(annual_col),
                "avg3":num(avg3_col),
                "category":str(r[type_col]).strip() if type_col is not None
                           and pd.notna(r[type_col]) else None
            }
        return out
    except Exception:
        return {}



if module == "🚀 Smart Breakout Scanner":

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

        nifty_midcap100 = (
            load_nifty_midcap100()
        )

        nifty_smallcap250 = (
            load_nifty_smallcap250()
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
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ]
    )

    stocks = resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

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
        universe == "Nifty Midcap 100"
        and not stocks
    ):

        st.error(
            """
            Nifty Midcap 100 list could not be loaded.

            Please try again later.
            """
        )

        st.stop()

    if (
        universe == "Nifty Smallcap 250"
        and not stocks
    ):

        st.error(
            """
            Nifty Smallcap 250 list could not be loaded.

            Please try again later.
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

    if universe == "Nifty Midcap 100":

        st.caption(
            "Nifty Midcap 100 = 100 tradable NSE stocks in "
            "the midcap segment, sourced from Nifty Indices."
        )

    if universe == "Nifty Smallcap 250":

        st.caption(
            "Nifty Smallcap 250 = 250 small-cap NSE stocks, "
            "sourced from Nifty Indices."
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
# 120-DAY HIGH BREAKOUT SCANNER
# ============================================================

elif module == "🎯 Buy / Sell Signal Engine":

    st.header(
        "🎯 Scanner Behaviour → BUY / SELL Signal Engine"
    )

    st.caption(
        "Select a scanner and universe. The engine evaluates "
        "the selected scanner's behaviour and produces a "
        "BUY / SELL / WATCH signal."
    )

    scanner_name=st.sidebar.selectbox(
        "Scanner Behaviour",
        [
            "Smart Breakout",
            "120-Day High Breakout",
            "Hourly Donchian Breakout",
            "Daily RSI(9)/WMA(21)",
            "Weekly Trend",
            "Daily Trend",
            "Multi-Timeframe",
            "Top 20 Momentum"
        ],
        key="signal_engine_scanner"
    )

    st.sidebar.subheader("Signal Engine Universe")

    st.caption("Loading stock universes...")

    nse_stocks=load_nse_equity_universe()
    nifty500=load_nifty500()
    fno_stocks=load_fno_stocks()
    nifty_midcap100=load_nifty_midcap100()
    nifty_smallcap250=load_nifty_smallcap250()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="signal_engine_universe"
    )

    stocks=resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    max_stocks=st.sidebar.slider(
        "Maximum Stocks",
        10,
        min(500,max(10,len(stocks))),
        min(100,max(10,len(stocks))),
        10,
        key="signal_engine_max_stocks"
    )

    run_signal=st.sidebar.button(
        "🎯 GENERATE BUY / SELL SIGNALS",
        type="primary",
        key="signal_engine_run"
    )

    if not stocks:
        st.error(
            f"No stocks available for {universe}."
        )
        st.stop()

    if run_signal:

        selected=stocks[:max_stocks]

        with st.spinner(
            "Downloading daily market data..."
        ):

            daily_market=download_batches(
                selected,
                "1y",
                50
            )

        hourly_market={}

        if scanner_name=="Hourly Donchian Breakout":

            with st.spinner(
                "Downloading recent hourly data..."
            ):

                hourly_market=download_rsi_wma_batches(
                    selected,
                    "Hourly",
                    50
                )

        rows=[]

        for symbol in selected:

            daily_data=daily_market.get(symbol)

            if daily_data is None or daily_data.empty:
                continue

            result=generate_scanner_signal(
                daily_data,
                scanner_name,
                hourly_market.get(symbol)
            )

            if result is None:
                continue

            trade=result["Trade Plan"]

            rows.append(
                {
                    "Stock":symbol,
                    "Signal":result["Signal"],
                    "Strength":result["Signal Strength"],
                    "Bullish":result["Bullish Score"],
                    "Bearish":result["Bearish Score"],
                    "Net":result["Net Score"],
                    "Close":round(
                        float(daily_data["Close"].iloc[-1]),
                        2
                    ),
                    "RSI(9)":result["RSI9"],
                    "RSI WMA21":result["WMA21"],
                    "Entry":(
                        round(trade["Entry"],2)
                        if trade else np.nan
                    ),
                    "Stop Loss":(
                        round(trade["Stop Loss"],2)
                        if trade else np.nan
                    ),
                    "Target 1":(
                        round(trade["Target 1"],2)
                        if trade else np.nan
                    ),
                    "Target 2":(
                        round(trade["Target 2"],2)
                        if trade else np.nan
                    ),
                    "R:R":"1:2 / 1:3" if trade else "—"
                }
            )

        result_df=pd.DataFrame(rows)

        if result_df.empty:

            st.warning(
                "No usable market data was returned."
            )

        else:

            counts=result_df["Signal"].value_counts()

            a,b,c,d,e=st.columns(5)

            a.metric(
                "🟢 Strong Buy",
                int(counts.get("🟢 STRONG BUY",0))
            )

            b.metric(
                "🟢 Buy",
                int(counts.get("🟢 BUY",0))
            )

            c.metric(
                "🟡 Watch",
                int(counts.get("🟡 WATCH / HOLD",0))
            )

            d.metric(
                "🔴 Sell",
                int(counts.get("🔴 SELL",0))
            )

            e.metric(
                "🔴 Strong Sell",
                int(counts.get("🔴 STRONG SELL",0))
            )

            actionable=result_df[
                result_df["Signal"].isin(
                    [
                        "🟢 STRONG BUY",
                        "🟢 BUY",
                        "🔴 SELL",
                        "🔴 STRONG SELL"
                    ]
                )
            ].sort_values(
                "Strength",
                ascending=False
            )

            st.subheader(
                "🎯 Current BUY / SELL Signals"
            )

            if actionable.empty:
                st.info(
                    "No actionable BUY/SELL signal currently."
                )
            else:
                st.dataframe(
                    actionable,
                    width="stretch",
                    hide_index=True
                )

            st.download_button(
                "⬇️ Download All Signals",
                result_df.to_csv(index=False),
                "Buy_Sell_Signal_Engine.csv",
                "text/csv"
            )

            with st.expander(
                "📋 All Stocks"
            ):
                st.dataframe(
                    result_df.sort_values(
                        "Strength",
                        ascending=False
                    ),
                    width="stretch",
                    hide_index=True
                )



# ============================================================
# SMART BREAKOUT DRAWDOWN OPTIMIZER MODULE
# ============================================================

elif module == "🧪 Smart Breakout Drawdown Optimizer":

    st.header(
        "🧪 Smart Breakout — Drawdown Reduction Optimizer"
    )

    st.write(
        """
        Automatically tests combinations of Smart Breakout
        filters to identify configurations that reduce
        drawdown while preserving a reasonable number of trades.
        """
    )

    st.warning(
        """
        The optimizer does NOT replace your original Smart
        Breakout strategy. It creates additional filtered
        variants for comparison. Results are historical and
        should be validated on an out-of-sample period before
        live use.
        """
    )

    with st.expander(
        "🔧 Filters being optimized",
        expanded=True
    ):

        st.markdown(
            """
            **Always retained:** existing Smart Breakout C1–C5 /
            technical scoring logic.

            Additional filters tested:

            • Minimum Smart Breakout score  
            • RSI(9) minimum and maximum  
            • Volume ratio minimum  
            • ATR(14) as % of price maximum  
            • Entry buffer  
            • Optional rising SMA(200)

            **Optimization objective:** reduce maximum drawdown
            while maintaining trade count, profit factor and
            risk-adjusted return.
            """
        )

    st.sidebar.subheader(
        "🧪 Optimizer Settings"
    )

    st.caption(
        "Loading stock universes..."
    )

    nse_stocks=load_nse_equity_universe()
    nifty500=load_nifty500()
    fno_stocks=load_fno_stocks()
    nifty_midcap100=load_nifty_midcap100()
    nifty_smallcap250=load_nifty_smallcap250()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="optimizer_universe"
    )

    stocks=resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    max_stocks=st.sidebar.slider(
        "Maximum Stocks",
        20,
        min(500,max(20,len(stocks))),
        min(100,max(20,len(stocks))),
        10,
        key="optimizer_max_stocks"
    )

    period=st.sidebar.selectbox(
        "Historical Period",
        [
            "2y",
            "3y",
            "5y",
            "10y"
        ],
        index=1,
        key="optimizer_period"
    )

    holding_days=st.sidebar.slider(
        "Maximum Holding Days",
        5,
        60,
        20,
        5,
        key="optimizer_holding_days"
    )

    st.sidebar.markdown(
        "### Filter Ranges"
    )

    score_values=st.sidebar.multiselect(
        "Smart Breakout Score Minimum",
        [5,6,7,8,9,10],
        default=[7,8,9],
        key="optimizer_scores"
    )

    rsi_min_values=st.sidebar.multiselect(
        "RSI(9) Minimum",
        [50,55,60,65],
        default=[55,60],
        key="optimizer_rsi_min"
    )

    rsi_max_values=st.sidebar.multiselect(
        "RSI(9) Maximum",
        [65,70,75,80],
        default=[70,75],
        key="optimizer_rsi_max"
    )

    volume_values=st.sidebar.multiselect(
        "Minimum Volume Ratio",
        [1.0,1.25,1.5,1.8,2.0,2.5],
        default=[1.5,1.8,2.0],
        key="optimizer_volume"
    )

    atr_values=st.sidebar.multiselect(
        "Maximum ATR % of Price",
        [3.0,4.0,5.0,6.0,8.0],
        default=[4.0,5.0,6.0],
        key="optimizer_atr"
    )

    entry_values=st.sidebar.multiselect(
        "Entry Buffer %",
        [0.25,0.50,0.75],
        default=[0.25,0.50],
        key="optimizer_entry"
    )

    sma_rising=st.sidebar.checkbox(
        "Require SMA(200) rising",
        value=True,
        key="optimizer_sma_rising"
    )

    min_trades=st.sidebar.number_input(
        "Minimum Trades for Ranking",
        min_value=10,
        max_value=200,
        value=20,
        step=5,
        key="optimizer_min_trades"
    )

    run_optimizer=st.sidebar.button(
        "🧪 RUN OPTIMIZATION",
        type="primary",
        key="optimizer_run"
    )

    if not stocks:
        st.error(
            f"No stocks available for {universe}."
        )
        st.stop()

    combinations=(
        len(score_values)
        *len(rsi_min_values)
        *len(rsi_max_values)
        *len(volume_values)
        *len(atr_values)
        *len(entry_values)
    )

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks: **{len(stocks)}** | "
        f"Parameter combinations: **{combinations:,}**"
    )

    if combinations>500:
        st.error(
            "Too many combinations. Reduce the selected "
            "parameter ranges to 500 or fewer combinations."
        )
        st.stop()

    if run_optimizer:

        selected=stocks[:max_stocks]

        with st.spinner(
            f"Downloading {period} historical data "
            f"for {len(selected)} stocks..."
        ):

            historical=download_batches(
                selected,
                period,
                50
            )

        with st.spinner(
            f"Testing {combinations:,} configurations..."
        ):

            optimization=run_smart_breakout_optimizer(
                historical,
                score_values,
                rsi_min_values,
                rsi_max_values,
                volume_values,
                atr_values,
                entry_values,
                sma_rising,
                holding_days
            )

        if optimization.empty:

            st.warning(
                "No configuration generated enough trades. "
                "Relax the filters or reduce the minimum "
                "trade requirement."
            )

        else:

            ranked=optimization[
                optimization["Trades"]>=int(
                    min_trades
                )
            ].copy()

            if ranked.empty:
                st.warning(
                    "No configuration met the minimum trade "
                    "requirement."
                )
                ranked=optimization.copy()

            # Primary ranking: Calmar, then drawdown,
            # then profit factor and trade count.
            ranked=ranked.sort_values(
                [
                    "Calmar",
                    "Max Drawdown %",
                    "Profit Factor",
                    "Trades"
                ],
                ascending=[
                    False,
                    True,
                    False,
                    False
                ]
            )

            st.subheader(
                "🏆 Top Drawdown-Adjusted Configurations"
            )

            st.dataframe(
                ranked.head(20),
                width="stretch",
                hide_index=True
            )

            best=ranked.iloc[0]

            c1,c2,c3,c4,c5=st.columns(5)

            c1.metric(
                "Best Max DD",
                f"{best['Max Drawdown %']:.2f}%"
            )

            c2.metric(
                "Calmar",
                f"{best['Calmar']:.2f}"
            )

            c3.metric(
                "Profit Factor",
                (
                    f"{best['Profit Factor']:.2f}"
                    if np.isfinite(
                        best["Profit Factor"]
                    )
                    else "∞"
                )
            )

            c4.metric(
                "Win Rate",
                f"{best['Win Rate %']:.1f}%"
            )

            c5.metric(
                "Trades",
                int(best["Trades"])
            )

            st.success(
                "Recommended configuration for further "
                "out-of-sample testing:"
            )

            st.code(
                f"""
Smart Breakout Score >= {best['Score Min']}
RSI(9): {best['RSI Min']} to {best['RSI Max']}
Volume Ratio >= {best['Volume Ratio Min']}
ATR(14) % <= {best['ATR % Max']}%
Entry Buffer = {best['Entry Buffer %']}%
SMA(200) Rising = {best['SMA200 Rising']}

Trades = {int(best['Trades'])}
Win Rate = {best['Win Rate %']:.2f}%
Profit Factor = {best['Profit Factor']}
Max Drawdown = {best['Max Drawdown %']:.2f}%
CAGR = {best['CAGR %']:.2f}%
Calmar = {best['Calmar']:.2f}
Net R = {best['Net R']:.2f}
                """.strip()
            )

            st.download_button(
                "⬇️ Download Full Optimization Results",
                optimization.to_csv(
                    index=False
                ),
                "Smart_Breakout_Drawdown_Optimization.csv",
                "text/csv"
            )

            st.caption(
                """
                Ranking is a research aid, not a guarantee.
                Always validate the selected configuration on
                an out-of-sample period to reduce overfitting.
                """
            )



# ============================================================
# CHART PATTERN SCANNER MODULE
# ============================================================

elif module == "📐 Chart Pattern Scanner":

    st.header(
        "📐 Chart Pattern Recognition Scanner"
    )

    st.write(
        """
        Quantitative recognition of classical price structures
        using swing highs/lows, geometry, breakout levels and
        volume behaviour.
        """
    )

    st.info(
        """
        This module identifies patterns from OHLCV price data.
        It is a rule-based detector, not an image classifier.
        Pattern recognition is approximate by nature, so use the
        confidence score and breakout stage as confirmation.
        """
    )

    with st.expander(
        "📋 Supported Patterns",
        expanded=True
    ):

        st.markdown(
            """
            **Reversal**
            - Head & Shoulders
            - Inverse Head & Shoulders
            - Double Top
            - Double Bottom

            **Continuation / consolidation**
            - Cup & Handle
            - Ascending Triangle
            - Descending Triangle
            - Symmetrical Triangle

            **Pattern stages**
            - FORMING
            - NEAR BREAKOUT
            - BREAKOUT
            - CONFIRMED BREAKOUT
            """
        )

    st.sidebar.subheader(
        "📐 Pattern Scanner Settings"
    )

    st.caption(
        "Loading stock universes..."
    )

    nse_stocks=load_nse_equity_universe()
    nifty500=load_nifty500()
    fno_stocks=load_fno_stocks()
    nifty_midcap100=load_nifty_midcap100()
    nifty_smallcap250=load_nifty_smallcap250()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="pattern_universe"
    )

    timeframe=st.sidebar.selectbox(
        "Timeframe",
        [
            "Daily",
            "Weekly"
        ],
        key="pattern_timeframe"
    )

    pattern_filter=st.sidebar.multiselect(
        "Patterns",
        [
            "Head & Shoulders",
            "Inverse Head & Shoulders",
            "Double Top",
            "Double Bottom",
            "Cup & Handle",
            "Ascending Triangle",
            "Descending Triangle",
            "Symmetrical Triangle"
        ],
        default=[
            "Head & Shoulders",
            "Inverse Head & Shoulders",
            "Double Top",
            "Double Bottom",
            "Cup & Handle",
            "Ascending Triangle",
            "Descending Triangle",
            "Symmetrical Triangle"
        ],
        key="pattern_filter"
    )

    stage_filter=st.sidebar.multiselect(
        "Pattern Stage",
        [
            "FORMING",
            "NEAR BREAKOUT",
            "BREAKOUT",
            "CONFIRMED BREAKOUT"
        ],
        default=[
            "FORMING",
            "NEAR BREAKOUT",
            "BREAKOUT",
            "CONFIRMED BREAKOUT"
        ],
        key="pattern_stage_filter"
    )

    min_confidence=st.sidebar.slider(
        "Minimum Confidence",
        50,
        95,
        65,
        5,
        key="pattern_min_confidence"
    )

    max_stocks=st.sidebar.slider(
        "Maximum Stocks",
        10,
        min(500,max(10,len(
            resolve_stock_universe(
                universe,
                nse_stocks,
                nifty500,
                fno_stocks,
                nifty_midcap100,
                nifty_smallcap250
            )
        ))),
        min(100,max(10,len(
            resolve_stock_universe(
                universe,
                nse_stocks,
                nifty500,
                fno_stocks,
                nifty_midcap100,
                nifty_smallcap250
            )
        ))),
        10,
        key="pattern_max_stocks"
    )

    run_patterns=st.sidebar.button(
        "📐 SCAN CHART PATTERNS",
        type="primary",
        key="run_pattern_scanner"
    )

    stocks=resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    if not stocks:
        st.error(
            f"No stocks available for {universe}."
        )
        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Timeframe: **{timeframe}** | "
        f"Stocks: **{len(stocks)}**"
    )

    if run_patterns:

        selected=stocks[:max_stocks]

        if timeframe=="Daily":

            period="2y"

            with st.spinner(
                f"Downloading daily data for {len(selected)} stocks..."
            ):

                market=download_batches(
                    selected,
                    period,
                    50
                )

        else:

            # Weekly data is derived from a longer daily history.
            with st.spinner(
                f"Downloading weekly-source data for {len(selected)} stocks..."
            ):

                market=download_batches(
                    selected,
                    "5y",
                    50
                )

                for symbol in list(market.keys()):

                    d=market[symbol]

                    if d is None or d.empty:
                        continue

                    d=d.copy()

                    if isinstance(
                        d.columns,
                        pd.MultiIndex
                    ):
                        d.columns=d.columns.get_level_values(0)

                    d.index=pd.to_datetime(d.index)

                    market[symbol]=(
                        d.resample("W-FRI")
                        .agg(
                            {
                                "Open":"first",
                                "High":"max",
                                "Low":"min",
                                "Close":"last",
                                "Volume":"sum"
                            }
                        )
                        .dropna()
                    )

        rows=[]

        for symbol in selected:

            data=market.get(symbol)

            if data is None or data.empty:
                continue

            patterns=detect_chart_patterns(
                data
            )

            for item in patterns:

                if pattern_filter and (
                    item["Pattern"]
                    not in pattern_filter
                ):
                    continue

                if item["Stage"] not in stage_filter:
                    continue

                if item["Confidence"]<min_confidence:
                    continue

                rows.append(
                    {
                        "Stock":symbol,
                        "Pattern":item["Pattern"],
                        "Direction":item["Direction"],
                        "Stage":item["Stage"],
                        "Confidence":
                            item["Confidence"],
                        "Current Price":
                            round(
                                item["Current Price"],
                                2
                            ),
                        "Breakout Level":
                            round(
                                item["Breakout Level"],
                                2
                            ),
                        "Distance %":
                            round(
                                item["Distance %"],
                                2
                            ),
                        "Entry":
                            round(
                                item["Entry"],
                                2
                            ),
                        "Stop Loss":
                            round(
                                item["Stop Loss"],
                                2
                            ),
                        "Target 1":
                            round(
                                item["Target 1"],
                                2
                            ),
                        "Target 2":
                            round(
                                item["Target 2"],
                                2
                            ),
                        "R:R":"1:2 / 1:3",
                        "Volume Ratio":
                            round(
                                item["Volume Ratio"],
                                2
                            ),
                        "Description":
                            item["Details"]
                    }
                )

        result_df=pd.DataFrame(rows)

        if result_df.empty:

            st.warning(
                "No chart patterns matched the selected "
                "filters."
            )

        else:

            result_df=result_df.sort_values(
                [
                    "Stage",
                    "Confidence"
                ],
                ascending=[
                    True,
                    False
                ]
            )

            st.success(
                f"Found **{len(result_df)} pattern signals** "
                f"across the selected universe."
            )

            # Strongest confirmed/near-breakout setups first.
            priority={
                "CONFIRMED BREAKOUT":0,
                "BREAKOUT":1,
                "NEAR BREAKOUT":2,
                "FORMING":3
            }

            display_df=result_df.copy()

            display_df["_priority"]=[
                priority.get(x,9)
                for x in display_df["Stage"]
            ]

            display_df=display_df.sort_values(
                [
                    "_priority",
                    "Confidence"
                ],
                ascending=[
                    True,
                    False
                ]
            ).drop(
                columns=["_priority"]
            )

            st.subheader(
                "📐 Detected Chart Patterns"
            )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Pattern Results",
                display_df.to_csv(index=False),
                "Chart_Pattern_Scanner.csv",
                "text/csv"
            )

            confirmed=display_df[
                display_df["Stage"].isin(
                    [
                        "BREAKOUT",
                        "CONFIRMED BREAKOUT"
                    ]
                )
            ]

            if not confirmed.empty:

                st.subheader(
                    "🚀 Breakout / Confirmation Candidates"
                )

                st.dataframe(
                    confirmed.head(20),
                    width="stretch",
                    hide_index=True
                )




# ============================================================
# MINERVINI SEPA + VCP SCANNER MODULE
# ============================================================



elif module == "📚 Kratter Momentum Scanner":

    st.header("📚 Kratter Momentum Scanner")
    st.caption(
        "Mechanical implementation of the trend-following rules in "
        "Matthew R. Kratter's *Learn to Trade Momentum Stocks*."
    )

    st.info(
        "Core book setup: the 50-day moving average closes above the "
        "200-day moving average, while the stock is trading above its "
        "50-day moving average. The book then buys the next morning."
    )

    with st.expander("📖 Book rules used by this scanner", expanded=False):
        st.markdown("""
        **Universe / watchlist**
        - Focus on companies with rapidly growing revenues.
        - The book gives **>20% annual revenue growth** as a rule of thumb.
        - **3-year average revenue growth >20%** is described as ideal.
        - It discusses New Technology Companies and Formula Companies.

        **Buy**
        1. 50-day SMA closes above 200-day SMA.
        2. Stock is above its 50-day SMA when the crossover occurs.
        3. Buy the next morning/open.

        **Exit**
        1. Emergency stop: **-15% from actual entry price**.
        2. Exit when 50-day SMA closes below 200-day SMA.
        3. Profit-taking option: **+300% from entry**.

        **Position sizing**
        - The book's example risks **2% of account equity** per trade.
        - Shares = (account size × risk %) / (entry − stop).
        """)
        st.caption(
            "Source: Matthew R. Kratter, 2nd edition; see the exact "
            "buy/sell rules and risk example in the uploaded book."
        )

    st.subheader("1️⃣ Select universe")

    universe_options={
        "Nifty 50":"nifty50",
        "Nifty 500":"nifty500",
        "Nifty Midcap 100":"midcap100",
        "Nifty Smallcap 250":"smallcap250",
        "F&O Stocks":"fno",
        "NSE Equity":"nse"
    }

    selected_universe=st.selectbox(
        "Stock Universe",
        list(universe_options.keys()),
        key="kratter_universe"
    )

    universe_key=universe_options[selected_universe]

    # Reuse the app's existing universe loaders.
    try:
        if universe_key=="nifty50":
            stocks=load_nifty50()
        elif universe_key=="nifty500":
            stocks=load_nifty500()
        elif universe_key=="midcap100":
            stocks=load_nifty_midcap100()
        elif universe_key=="smallcap250":
            stocks=load_nifty_smallcap250()
        elif universe_key=="fno":
            stocks=load_fno_stocks()
        else:
            stocks=load_nse_equity_universe()
    except Exception:
        stocks=[]

    stocks=list(stocks) if stocks is not None else []

    if not stocks:
        st.warning("No stocks could be loaded for this universe.")
    else:
        c1,c2,c3=st.columns(3)
        c1.metric("Universe Stocks",len(stocks))

        account_size=st.number_input(
            "Account Size (₹)",
            min_value=10000.0,
            value=100000.0,
            step=10000.0,
            key="kratter_account"
        )
        risk_pct=st.number_input(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="kratter_risk"
        )

        st.subheader("2️⃣ Optional fundamental confirmation")

        fund_file=st.file_uploader(
            "Upload fundamentals CSV (optional)",
            type=["csv"],
            key="kratter_fundamentals"
        )

        st.caption(
            "Recommended columns: Symbol, Annual Revenue Growth %, "
            "3Y Avg Revenue Growth %, Company Type. Without this file, "
            "the scanner can identify the exact technical crossover but "
            "cannot claim the book's revenue-growth filter is satisfied."
        )

        fundamentals=_kratter_read_fundamental_csv(fund_file) if fund_file else {}

        if st.button(
            "🔎 RUN KRATTER MOMENTUM SCAN",
            type="primary",
            key="kratter_run"
        ):
            with st.spinner("Scanning for fresh 50/200 SMA momentum signals..."):
                data_map=_kratter_download_batches(stocks)

            rows=[]
            failed=0

            for symbol in stocks:
                d=data_map.get(symbol)
                if d is None or d.empty:
                    failed+=1
                    continue

                f=fundamentals.get(str(symbol).upper(),{})
                result=_kratter_scan_one(
                    symbol,
                    d,
                    f.get("annual"),
                    f.get("avg3"),
                    f.get("category")
                )
                if result is not None:
                    rows.append(result)

            result_df=pd.DataFrame(rows)

            if result_df.empty:
                st.warning(
                    "No technically valid stocks were returned. "
                    "The scanner requires at least 200 daily bars."
                )
            else:
                tech=result_df[result_df["Technical Buy"]==True].copy()

                st.subheader("🎯 Fresh Book Buy Signals")

                if tech.empty:
                    st.info(
                        "No fresh 50/200 SMA crossover signals today. "
                        "This scanner is intentionally selective."
                    )
                else:
                    # Add next-entry risk plan using the latest close as a
                    # planning reference; actual book entry is next morning.
                    plans=[]
                    for _,r in tech.iterrows():
                        plan=_kratter_position_plan(
                            float(r["Close"]),
                            account_size,
                            risk_pct
                        )
                        plans.append(plan)

                    plan_df=pd.DataFrame(plans,index=tech.index)
                    display=tech.join(plan_df)

                    cols=[
                        "Symbol","Signal","Signal Date","Close","SMA50",
                        "SMA200","Annual Revenue Growth %",
                        "3Y Avg Revenue Growth %","Fundamental Status",
                        "Company Type","Entry","Stop (-15%)",
                        "Target (+300%)","Shares","Capital Required",
                        "SMA50 Slope %","SMA200 Slope %",
                        "Distance from 52W High %"
                    ]
                    cols=[c for c in cols if c in display.columns]
                    st.dataframe(
                        display[cols].sort_values(
                            by=["Fundamental Status","SMA50 Slope %"],
                            ascending=[True,False]
                        ),
                        width="stretch",
                        hide_index=True
                    )

                    st.success(
                        f"{len(tech)} fresh technical buy signal(s) found. "
                        f"Stocks with annual revenue growth >20% are marked "
                        f"as **🟢 BOOK MOMENTUM BUY**."
                    )

                st.subheader("📊 Scan diagnostics")
                d1,d2,d3=st.columns(3)
                d1.metric("Technical Signals",len(tech))
                d2.metric("Data Loaded",len(data_map))
                d3.metric("Data Failures",failed)

                with st.expander("All scanned stocks"):
                    allcols=[
                        "Symbol","Signal","Close","SMA50","SMA200",
                        "50/200 Cross","Close > SMA50",
                        "Annual Revenue Growth %",
                        "3Y Avg Revenue Growth %",
                        "Fundamental Status","SMA50 Slope %",
                        "SMA200 Slope %","Distance from 52W High %"
                    ]
                    allcols=[c for c in allcols if c in result_df.columns]
                    st.dataframe(
                        result_df[allcols],
                        width="stretch",
                        hide_index=True
                    )

                csv=result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Kratter Scan CSV",
                    csv,
                    "kratter_momentum_scan.csv",
                    "text/csv",
                    key="kratter_download"
                )

        st.subheader("🧮 Position-size calculator")
        st.caption(
            "The book's example uses a 15% stop and 2% account risk. "
            "This calculator follows that formula; it does not guarantee "
            "the actual next-open fill price."
        )

        calc_entry=st.number_input(
            "Reference Entry Price (₹)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            key="kratter_calc_entry"
        )
        calc=_kratter_position_plan(
            calc_entry,
            account_size,
            risk_pct
        )
        if calc:
            a,b,c,d=st.columns(4)
            a.metric("Stop",f"₹{calc['Stop (-15%)']:.2f}")
            b.metric("Target",f"₹{calc['Target (+300%)']:.2f}")
            c.metric("Shares",f"{calc['Shares']:,}")
            d.metric("Capital",f"₹{calc['Capital Required']:,.0f}")


elif module == "🔥 Momentum Catalyst Scanner":

    st.header("🔥 Momentum Catalyst Scanner")
    st.caption("Price acceleration + volume expansion + trend + breakout + relative strength + optional fundamental acceleration")
    st.info("Designed from the common pattern observed in your high-momentum stock list. Fundamentals and catalysts are optional inputs; the scanner never fabricates them.")

    universe_options={"Nifty 50":"nifty50","Nifty 500":"nifty500","Nifty Midcap 100":"midcap100","Nifty Smallcap 250":"smallcap250","F&O Stocks":"fno","NSE Equity":"nse"}
    selected_universe=st.selectbox("Stock Universe",list(universe_options.keys()),key="mcs_universe")
    uk=universe_options[selected_universe]
    try:
        if uk=="nifty50": stocks=load_nifty50()
        elif uk=="nifty500": stocks=load_nifty500()
        elif uk=="midcap100": stocks=load_nifty_midcap100()
        elif uk=="smallcap250": stocks=load_nifty_smallcap250()
        elif uk=="fno": stocks=load_fno_stocks()
        else: stocks=load_nse_equity_universe()
    except Exception: stocks=[]
    stocks=list(stocks or [])
    st.sidebar.markdown("### Momentum thresholds")
    min_score=st.sidebar.slider("Minimum score",50,90,70,5,key="mcs_min_score")
    min_vol=st.sidebar.slider("Minimum volume ratio",1.0,3.0,1.5,0.1,key="mcs_min_vol")
    if not stocks: st.warning("No stocks could be loaded for this universe.")
    else:
        fund_file=st.file_uploader("Optional fundamentals/catalyst CSV",type=["csv"],key="mcs_fundamentals")
        st.caption("Columns: Symbol, Annual Revenue Growth %, PAT Growth %, EBITDA Growth %, Order Book Growth %, Catalyst")
        fundamentals=_mcs_read_fundamentals(fund_file) if fund_file else {}
        if st.button("🔎 RUN MOMENTUM CATALYST SCAN",type="primary",key="mcs_run"):
            with st.spinner("Scanning momentum, volume, trend and breakout structure..."):
                data_map=_kratter_download_batches(stocks)
                benchmark=_download_nifty50_history("2y") if '_download_nifty50_history' in globals() else pd.DataFrame()
            rows=[]; failed=0
            for symbol in stocks:
                d=data_map.get(symbol)
                if d is None or d.empty: failed+=1; continue
                res=_mcs_scan_one(symbol,d,benchmark,fundamentals.get(str(symbol).upper(),{}))
                if res is not None: rows.append(res)
            result_df=pd.DataFrame(rows)
            if result_df.empty:
                st.warning("No valid stocks were returned. The scanner requires at least 205 daily bars.")
            else:
                result_df=result_df[result_df["Volume Ratio"].fillna(0)>=min_vol].copy()
                result_df=result_df[result_df["Score"]>=min_score].sort_values(["Score","Volume Ratio"],ascending=False)
                if result_df.empty: st.info("No stocks met the selected score and volume thresholds.")
                else:
                    a,b,c,d=st.columns(4)
                    a.metric("Qualified",len(result_df)); b.metric("Explosive ≥80",int((result_df.Score>=80).sum())); c.metric("Strong 70–79",int(((result_df.Score>=70)&(result_df.Score<80)).sum())); d.metric("Trade Ready",int(result_df["Trade Ready"].sum()))
                    st.subheader("🏆 Highest-Probability Momentum Stocks")
                    cols=["Symbol","Score","Rating","Trade Ready","Close","1D %","20D %","Volume Ratio","Breakout","RS 20D %","SMA50>SMA200","52W High Distance %","Revenue Growth %","PAT Growth %","EBITDA Growth %","Order Book Growth %","Entry Reference","Suggested SL","Target 1","Target 2","Reasons","Warnings"]
                    cols=[c for c in cols if c in result_df.columns]
                    st.dataframe(result_df[cols],width="stretch",hide_index=True)
                    st.subheader("🧠 Why are these stocks rising?")
                    for _,r in result_df.head(10).iterrows():
                        st.markdown(f"**{r['Symbol']} — {r['Rating']} — {r['Score']}/100**")
                        st.write(r["Reasons"] or "No strong explanatory factors available.")
                        if r["Warnings"]: st.caption("⚠️ "+r["Warnings"])
                    csv=result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Momentum Catalyst Scan CSV",csv,"momentum_catalyst_scan.csv","text/csv",key="mcs_download")

elif module == "📊 Options Next-Day Analyzer":

    st.header("📊 Options Next-Day Analyzer")

    st.markdown(
        """
        Upload your **end-of-day derivatives CSV** after market close.
        The analyzer combines futures/OI positioning with the latest
        underlying price/volume data and Fibonacci support/resistance
        to generate a plan for the **next trading session**.

        **This module is independent of the Minervini SEPA/VCP scanner.**
        """
    )

    st.sidebar.subheader("📊 Options Analyzer Settings")

    options_universe=st.sidebar.selectbox(
        "Underlying Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="options_universe"
    )

    options_period=st.sidebar.selectbox(
        "Underlying Market Data Period",
        ["1y","2y","3y","5y"],
        index=1,
        key="options_period"
    )

    options_fib_lookback=st.sidebar.select_slider(
        "Fibonacci lookback (days)",
        options=[40,50,60,80,100],
        value=60,
        key="options_fib_lookback"
    )

    min_call_score=st.sidebar.slider(
        "Strong Call threshold",
        70,95,80,1,
        key="options_call_threshold_independent"
    )

    min_put_score=st.sidebar.slider(
        "Strong Put threshold",
        70,95,80,1,
        key="options_put_threshold_independent"
    )

    options_file=st.file_uploader(
        "📤 Upload EOD derivatives CSV",
        type=["csv"],
        key="options_eod_csv_independent"
    )

    if options_file is not None:
        try:
            option_df,option_meta=_read_options_eod_csv(options_file)

            if option_df.empty:
                st.warning("The uploaded CSV contains no stock rows.")
            else:
                c1,c2,c3,c4=st.columns(4)

                c1.metric("Stocks in CSV",len(option_df))
                c2.metric(
                    "EOD Date",
                    str(option_meta.get(
                        "Date",
                        option_df["Date"].iloc[0]
                        if "Date" in option_df.columns else ""
                    ))
                )
                c3.metric(
                    "OI Trend",
                    "Available"
                    if "OI Trend" in option_df.columns
                    else "Missing"
                )
                c4.metric(
                    "PCR",
                    "Available"
                    if "Put Call Ratio (PCR)" in option_df.columns
                    else "Missing"
                )

                st.info(
                    "The uploaded EOD file is treated as information "
                    "available after the market close. Signals are "
                    "intended for the next trading session."
                )

                if st.button(
                    "🔍 ANALYZE FOR NEXT TRADING DAY",
                    type="primary",
                    key="options_analyze_independent"
                ):
                    # Load only stocks present in the uploaded CSV.
                    symbols=[
                        str(s).strip().upper()
                        for s in option_df["Symbol"].dropna().unique()
                    ]

                    with st.spinner(
                        "Analyzing underlying trend, Fibonacci levels "
                        "and derivatives positioning..."
                    ):
                        options_market=download_batches(
                            symbols,
                            options_period,
                            50
                        )

                        option_result=run_options_next_day_analysis(
                            option_df,
                            options_market,
                            options_fib_lookback
                        )

                        # Phase 2: Bollinger Bands + ADX + ATR.
                        phase2_rows=[]
                        for _, _orow in option_result.iterrows():
                            _sym=str(_orow["Symbol"]).strip().upper()
                            _snap=_options_phase2_snapshot(
                                options_market.get(_sym)
                            )
                            phase2_rows.append(_snap)

                        phase2_df=pd.DataFrame(phase2_rows)

                        if not phase2_df.empty:
                            for _col in phase2_df.columns:
                                option_result[_col]=phase2_df[_col].values

                            # Keep the original score intact and create
                            # a separate Phase-2 technical confluence score.
                            option_result["Technical Confluence Score"]=(
                                option_result["BB Bullish Points"].fillna(0)
                                +option_result["BB Bearish Points"].fillna(0)
                                +option_result["BB/ADX Selling Points"].fillna(0)
                            )

                            # Directional confirmation: only reward BB/ADX
                            # points that agree with the existing signal.
                            option_result["Phase 2 Bullish Confirm"]=(
                                option_result["BB Bullish Points"].fillna(0)
                            )
                            option_result["Phase 2 Bearish Confirm"]=(
                                option_result["BB Bearish Points"].fillna(0)
                            )
                            option_result["Phase 2 Selling Confirm"]=(
                                option_result["BB/ADX Selling Points"].fillna(0)
                            )

                            # Add Phase-2 points to the relevant score with
                            # a conservative cap. This avoids letting one
                            # indicator dominate the original OI/Fib model.
                            option_result["Call Score"]=np.minimum(
                                100,
                                option_result["Call Score"].astype(float)
                                +option_result["Phase 2 Bullish Confirm"].astype(float)
                            ).round(1)

                            option_result["Put Score"]=np.minimum(
                                100,
                                option_result["Put Score"].astype(float)
                                +option_result["Phase 2 Bearish Confirm"].astype(float)
                            ).round(1)

                            option_result["Selling Score"]=np.minimum(
                                100,
                                option_result["Selling Score"].astype(float)
                                +option_result["Phase 2 Selling Confirm"].astype(float)
                            ).round(1)

                        # Phase 3: confluence, conflict detection and regime.
                        phase3_df=pd.DataFrame([
                            _options_phase3_confluence(r)
                            for _,r in option_result.iterrows()
                        ])
                        if not phase3_df.empty:
                            for col in phase3_df.columns:
                                option_result[col]=phase3_df[col].values
                            option_result["Phase 3 Signal"]=option_result.apply(
                                _options_phase3_final_signal,axis=1
                            )
                            option_result["Final Confluence Score"]=option_result[
                                ["Phase 3 Call Score","Phase 3 Put Score","Phase 3 Selling Score"]
                            ].max(axis=1).round(1)

                        # Phase 1: explain the signal and build
                        # underlying-based entry/SL/target levels.
                        plan_df=pd.DataFrame(
                            option_result.apply(
                                _options_phase1_plan, axis=1
                            ).tolist()
                        )
                        for col in plan_df.columns:
                            option_result[col]=plan_df[col].values

                        rr=option_result.apply(
                            lambda r: _options_phase1_rr(
                                r["Direction"],
                                r["Underlying Trigger"],
                                r["Underlying Invalidation"],
                                r["Underlying Target 1"],
                                r["Underlying Target 2"]
                            ),
                            axis=1
                        )
                        option_result["RR Target 1"]=[x[0] for x in rr]
                        option_result["RR Target 2"]=[x[1] for x in rr]
                        option_result["Contract Guidance"]=option_result[
                            "Direction"
                        ].map(_options_phase1_contract_note)

                    if option_result.empty:
                        st.warning(
                            "No candidates could be calculated. "
                            "Check the CSV symbols and market-data availability."
                        )
                    else:
                        def _options_final_signal(r):
                            call=float(r["Call Score"])
                            put=float(r["Put Score"])
                            sell=float(r["Selling Score"])

                            if (
                                call>=min_call_score
                                and call>put
                                and call>sell
                            ):
                                return "🟢 Strong Call Candidate"

                            if (
                                put>=min_put_score
                                and put>call
                                and put>sell
                            ):
                                return "🔴 Strong Put Candidate"

                            if sell>=75 and sell>call and sell>put:
                                return "🟡 Option Selling Candidate"

                            if max(call,put)>=65:
                                return "🔵 Option Buying Candidate"

                            return "⚪ Avoid / Wait"

                        option_result["Signal"]=option_result.apply(
                            _options_final_signal,
                            axis=1
                        )

                        calls=option_result[
                            option_result["Signal"]==
                            "🟢 Strong Call Candidate"
                        ]
                        puts=option_result[
                            option_result["Signal"]==
                            "🔴 Strong Put Candidate"
                        ]
                        sellers=option_result[
                            option_result["Signal"]==
                            "🟡 Option Selling Candidate"
                        ]
                        buyers=option_result[
                            option_result["Signal"]==
                            "🔵 Option Buying Candidate"
                        ]

                        a,b,c,d=st.columns(4)
                        a.metric("🟢 Strong Calls",len(calls))
                        b.metric("🔴 Strong Puts",len(puts))
                        c.metric("🟡 Selling",len(sellers))
                        d.metric("🔵 Buying",len(buyers))

                        st.subheader("📈 Next-Day Options Analysis")

                        display_cols=[
                            "Stock","Symbol","Close","Trend",
                            "Call Score","Put Score","Selling Score",
                            "Signal","Position","OI Trend","PCR",
                            "Future OI Chg %","Fib Support Price",
                            "Fib Resistance Price"
                        ]
                        display_cols=[
                            c for c in display_cols
                            if c in option_result.columns
                        ]

                        st.dataframe(
                            option_result[display_cols],
                            width="stretch",
                            hide_index=True
                        )

                        explanation_pool=option_result[
                            option_result["Signal"].isin([
                                "🟢 Strong Call Candidate",
                                "🔴 Strong Put Candidate",
                                "🟡 Option Selling Candidate",
                                "🔵 Option Buying Candidate"
                            ])
                        ].head(20)

                        st.subheader("📈 Phase 4 — Backtesting & Validation")

                        st.caption(
                            "Phase 4 validates the technical/regime component using "
                            "the next trading session. It does not invent option "
                            "premium or IV data. Full derivatives validation requires "
                            "historical OI/PCR fields for each past date."
                        )

                        with st.expander("Run historical validation",expanded=False):
                            st.info(
                                "Upload one or more historical OHLC CSV files. "
                                "Each file should contain Date, Open, High, Low and Close. "
                                "The filename is used as the stock symbol."
                            )

                            bt_files=st.file_uploader(
                                "Historical OHLC CSV files",
                                type=["csv"],
                                accept_multiple_files=True,
                                key="options_phase4_files"
                            )

                            if bt_files:
                                bt_history={}
                                for bf in bt_files:
                                    try:
                                        bdf=pd.read_csv(bf)
                                        date_col=next(
                                            (c for c in bdf.columns
                                             if str(c).strip().lower() in
                                             ["date","datetime","timestamp"]),
                                            None
                                        )
                                        if date_col:
                                            bdf[date_col]=pd.to_datetime(
                                                bdf[date_col],errors="coerce"
                                            )
                                            bdf=bdf.dropna(subset=[date_col]).set_index(date_col).sort_index()

                                        bdf=_options_normalize_columns(bdf)

                                        symbol=Path(bf.name).stem.upper()
                                        bt_history[symbol]=bdf
                                    except Exception as exc:
                                        st.error(
                                            f"Could not read {bf.name}: {exc}"
                                        )

                                if bt_history:
                                    bt_result=_options_phase4_build_backtest(bt_history)

                                    if bt_result.empty:
                                        st.warning(
                                            "No historical signals were generated. "
                                            "Provide sufficiently long daily OHLC "
                                            "history (preferably 1+ year)."
                                        )
                                    else:
                                        metrics=_options_phase4_metrics(bt_result)

                                        m1,m2,m3,m4,m5=st.columns(5)
                                        m1.metric("Signals",int(metrics.get("Signals",0)))
                                        m2.metric("Triggered",int(metrics.get("Triggered",0)))
                                        m3.metric("Win Rate",f'{metrics.get("Win Rate %",0):.1f}%')
                                        m4.metric("Avg Return",f'{metrics.get("Average Return %",0):.2f}%')
                                        m5.metric("Max DD",f'{metrics.get("Max Drawdown %",0):.2f}%')

                                        st.dataframe(
                                            bt_result,
                                            width="stretch",
                                            hide_index=True
                                        )

                                        csv_data=bt_result.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            "⬇️ Download Backtest Results",
                                            csv_data,
                                            "options_phase4_backtest.csv",
                                            "text/csv",
                                            key="options_phase4_download"
                                        )

                                        st.markdown("**Validation metrics**")
                                        st.write({
                                            k:round(v,3) if isinstance(v,(float,np.floating)) and np.isfinite(v) else v
                                            for k,v in metrics.items()
                                        })

                        st.subheader("🧠 Phase 3 — Confluence & Market Regime")

                        regime_counts=option_result["Market Regime"].value_counts()
                        d1,d2,d3,d4=st.columns(4)
                        d1.metric("🚀 Trending Bull",int(regime_counts.get("🚀 Trending Bull",0)))
                        d2.metric("🔻 Trending Bear",int(regime_counts.get("🔻 Trending Bear",0)))
                        d3.metric("🟡 Range / Compression",int(regime_counts.get("🟡 Range / Compression",0)+regime_counts.get("🟡 Weak / Range",0)))
                        d4.metric("⚠️ Conflicted",int((option_result["Conflict Count"]>=1).sum()))
                        st.caption("Phase 3 combines derivatives, trend, Bollinger, ADX/DI, Fibonacci and participation. Conflicts reduce directional confidence.")

                        phase3_cols=[
                            "Stock","Symbol","Market Regime",
                            "Phase 3 Call Score","Phase 3 Put Score","Phase 3 Selling Score",
                            "Conflict Status","Phase 3 Signal",
                            "Bullish Factors","Bearish Factors","Conflict Details"
                        ]
                        phase3_cols=[c for c in phase3_cols if c in option_result.columns]
                        st.dataframe(option_result[phase3_cols].head(50),width="stretch",hide_index=True)

                        st.subheader("📐 Bollinger + ADX + ATR Analysis")

                        if not explanation_pool.empty:
                            _bb_row=explanation_pool.iloc[0]

                            bc1,bc2,bc3,bc4,bc5=st.columns(5)
                            bc1.metric(
                                "BB State",
                                str(_bb_row.get("BB State","N/A"))
                            )
                            bc2.metric(
                                "BB Width",
                                f'{float(_bb_row.get("BB Width",np.nan)):.4f}'
                                if np.isfinite(float(_bb_row.get("BB Width",np.nan)))
                                else "N/A"
                            )
                            bc3.metric(
                                "ADX(14)",
                                f'{float(_bb_row.get("ADX14",np.nan)):.1f}'
                                if np.isfinite(float(_bb_row.get("ADX14",np.nan)))
                                else "N/A"
                            )
                            bc4.metric(
                                "ATR(14)",
                                f'{float(_bb_row.get("ATR14",np.nan)):.2f}'
                                if np.isfinite(float(_bb_row.get("ATR14",np.nan)))
                                else "N/A"
                            )
                            bc5.metric(
                                "ATR %",
                                f'{float(_bb_row.get("ATR %",np.nan)):.2f}%'
                                if np.isfinite(float(_bb_row.get("ATR %",np.nan)))
                                else "N/A"
                            )

                            st.caption(
                                "Phase 2 adds Bollinger Band state, ADX trend "
                                "strength and ATR volatility confirmation to "
                                "the existing derivatives + Fibonacci model."
                            )

                        st.subheader("🔎 Why This Signal?")

                        if not explanation_pool.empty:
                            selected_symbol=st.selectbox(
                                "Select candidate",
                                explanation_pool["Symbol"].tolist(),
                                key="options_phase1_candidate"
                            )
                            selected=explanation_pool[
                                explanation_pool["Symbol"]==selected_symbol
                            ].iloc[0]

                            e1,e2,e3,e4=st.columns(4)
                            e1.metric("Signal",selected["Signal"])
                            e2.metric("Call Score",f'{selected["Call Score"]:.1f}')
                            e3.metric("Put Score",f'{selected["Put Score"]:.1f}')
                            e4.metric("Selling Score",f'{selected["Selling Score"]:.1f}')

                            st.markdown("**Phase 3 confluence**")
                            st.write(f'**Market regime:** {selected.get("Market Regime","N/A")}')
                            st.write(f'**Phase 3 signal:** {selected.get("Phase 3 Signal","N/A")}')
                            st.write(f'**Conflict:** {selected.get("Conflict Status","N/A")}')
                            if str(selected.get("Conflict Details","None"))!="None":
                                st.warning(selected.get("Conflict Details",""))

                            st.markdown("**Contributing factors**")
                            for reason in str(selected["Why Signal"]).split(" | "):
                                if reason:
                                    st.write("• "+reason)

                            p1,p2=st.columns(2)
                            with p1:
                                st.markdown("**Next-day plan**")
                                st.write(f'**Direction:** {selected[ "Market Regime","Phase 3 Call Score","Phase 3 Put Score",
                            "Phase 3 Selling Score","Conflict Status","Phase 3 Signal",
                            "Direction"]}')
                                st.write(f'**Preferred:** {selected["Preferred Contract"]}')
                                if np.isfinite(selected["Underlying Trigger"]):
                                    st.write(f'**Trigger:** {selected["Underlying Trigger"]:.2f}')
                                    st.write(f'**Invalidation / SL:** {selected["Underlying Invalidation"]:.2f}')
                            with p2:
                                st.markdown("**Targets**")
                                if np.isfinite(selected["Underlying Target 1"]):
                                    st.write(f'**Target 1:** {selected["Underlying Target 1"]:.2f}')
                                if np.isfinite(selected["Underlying Target 2"]):
                                    st.write(f'**Target 2:** {selected["Underlying Target 2"]:.2f}')
                                if np.isfinite(selected["RR Target 1"]):
                                    st.write(f'**R:R to T1:** {selected["RR Target 1"]:.2f}')
                                if np.isfinite(selected["RR Target 2"]):
                                    st.write(f'**R:R to T2:** {selected["RR Target 2"]:.2f}')

                            st.caption(selected["Contract Guidance"])
                            st.warning(
                                "Phase 1 uses underlying-stock levels. "
                                "It does not invent option premiums or IV. "
                                "Use the underlying trigger/invalidation to manage "
                                "the selected ATM/near-ITM option."
                            )

                        st.download_button(
                            "⬇️ Download Next-Day Options Report",
                            option_result.to_csv(index=False),
                            "Options_Next_Day_Analysis.csv",
                            "text/csv",
                            key="options_download_independent"
                        )

                        st.subheader("⭐ Strong Candidates")

                        top=option_result[
                            option_result["Signal"].isin([
                                "🟢 Strong Call Candidate",
                                "🔴 Strong Put Candidate",
                                "🟡 Option Selling Candidate",
                                "🔵 Option Buying Candidate"
                            ])
                        ].head(30)

                        top_cols=[
                            "Stock","Symbol","Close","Call Score",
                            "Put Score","Selling Score","Signal",
                            "Trend","Position","OI Trend","PCR",
                            "Fib Support Price","Fib Resistance Price"
                        ]
                        top_cols=[
                            c for c in top_cols
                            if c in top.columns
                        ]

                        st.dataframe(
                            top[top_cols],
                            width="stretch",
                            hide_index=True
                        )

                        st.info(
                            "IV, option premium, bid/ask spread and "
                            "strike-specific analysis are not included "
                            "unless those fields are present in your CSV."
                        )

        except Exception as exc:
            st.error(
                f"Could not analyze the uploaded derivatives CSV: {exc}"
            )


elif module == "🏆 Minervini SEPA + VCP Scanner":

    st.header("🏆 Minervini SEPA + VCP Leadership Scanner")

    st.markdown(
        """
        **Mechanical technical implementation of the Minervini framework:**
        Trend Template → Relative Strength → VCP/base quality → volume
        contraction → pivot → breakout.

        **Important:** the current app's market-data layer does not provide
        reliable historical quarterly fundamentals. Therefore this version
        does **not invent EPS/sales/margin values**. Fundamental SEPA scoring
        is intentionally left out until a reliable fundamentals source is
        connected. The scanner below is the technical/price-volume portion.
        """
    )

    st.sidebar.subheader("🏆 Minervini SEPA + VCP Settings")

    nse_stocks=load_nse_equity_universe()
    nifty500=load_nifty500()
    fno_stocks=load_fno_stocks()
    nifty_midcap100=load_nifty_midcap100()
    nifty_smallcap250=load_nifty_smallcap250()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="minervini_universe"
    )

    stocks=resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    period=st.sidebar.selectbox(
        "Market Data Period",
        ["1y","2y","3y","5y"],
        index=2,
        key="minervini_period"
    )

    vcp_window=st.sidebar.select_slider(
        "VCP analysis window (days)",
        options=[40,50,60,70,80,90],
        value=60,
        key="minervini_vcp_window"
    )

    breakout_volume_mult=st.sidebar.slider(
        "Breakout volume / SMA50",
        1.0,2.5,1.2,0.1,
        key="minervini_breakout_volume"
    )

    chase_pct=st.sidebar.slider(
        "Maximum distance above pivot (%)",
        1.0,5.0,3.0,0.5,
        key="minervini_chase_pct"
    )

    min_liquidity=st.sidebar.slider(
        "Minimum daily traded value (₹ lakh)",
        1.0,100.0,5.0,1.0,
        key="minervini_liquidity"
    )

    st.sidebar.info(
        "Trend Template is a hard gate. A BUY requires Trend PASS + "
        "valid VCP + pivot breakout + volume confirmation + score ≥80 "
        "and price within the no-chase zone."
    )

    st.sidebar.subheader("📊 Minervini Backtest")

    backtest_stop=st.sidebar.slider(
        "Risk stop (%)",
        5.0,10.0,8.0,0.5,
        key="minervini_bt_stop"
    )

    backtest_max_hold=st.sidebar.slider(
        "Maximum holding days",
        20,120,60,10,
        key="minervini_bt_max_hold"
    )

    backtest_capital=st.sidebar.number_input(
        "Starting capital ₹",
        min_value=100000.0,
        value=1000000.0,
        step=100000.0,
        key="minervini_bt_capital"
    )

    backtest_max_positions=st.sidebar.slider(
        "Maximum simultaneous positions",
        1,20,10,1,
        key="minervini_bt_max_positions"
    )

    backtest_risk=st.sidebar.slider(
        "Risk per position (%)",
        0.25,2.0,1.0,0.25,
        key="minervini_bt_risk"
    )

    backtest_max_stocks=st.sidebar.slider(
        "Maximum stocks for backtest",
        25,500,100,25,
        key="minervini_bt_max_stocks"
    )

    scan_col, bt_col, opt_col = st.columns(3)

    with scan_col:
        run=st.button(
            "🔎 RUN MINERVINI SEPA + VCP SCANNER",
            type="primary",
            key="minervini_run"
        )

    with bt_col:
        run_backtest=st.button(
            "📊 RUN MINERVINI BACKTEST",
            type="secondary",
            key="minervini_backtest_run"
        )

    with opt_col:
        run_options=st.button(
            "📤 OPTIONS NEXT-DAY ANALYZER",
            type="secondary",
            key="minervini_options_analyzer_run"
        )


    # ========================================================
    # OPTIONS NEXT-DAY ANALYZER
    # ========================================================
    st.markdown("---")
    st.subheader("📤 Options Next-Day Analyzer")

    st.caption(
        "Upload the EOD derivatives CSV after market close. "
        "The analyzer combines that day's futures/OI positioning "
        "with the latest underlying OHLCV and Fibonacci levels "
        "to create the next trading day's directional candidates."
    )

    options_file=st.file_uploader(
        "Upload EOD derivatives CSV",
        type=["csv"],
        key="minervini_options_csv"
    )

    oc1,oc2,oc3=st.columns(3)

    with oc1:
        options_fib_lookback=st.select_slider(
            "Fibonacci lookback (days)",
            options=[40,50,60,80,100],
            value=60,
            key="options_fib_lookback"
        )

    with oc2:
        min_call_score=st.slider(
            "Strong Call threshold",
            70,95,80,1,
            key="options_call_threshold"
        )

    with oc3:
        min_put_score=st.slider(
            "Strong Put threshold",
            70,95,80,1,
            key="options_put_threshold"
        )

    if options_file is not None:
        try:
            option_df,option_meta=_read_options_eod_csv(options_file)

            if option_df.empty:
                st.warning("The uploaded CSV contains no stock rows.")
            else:
                d1,d2,d3,d4=st.columns(4)
                d1.metric("Stocks",len(option_df))
                d2.metric(
                    "EOD Date",
                    str(option_meta.get(
                        "Date",
                        option_df["Date"].iloc[0]
                        if "Date" in option_df.columns else ""
                    ))
                )
                d3.metric(
                    "OI Trend",
                    "Available"
                    if "OI Trend" in option_df.columns
                    else "Missing"
                )
                d4.metric(
                    "PCR",
                    "Available"
                    if "Put Call Ratio (PCR)" in option_df.columns
                    else "Missing"
                )

                if st.button(
                    "🔍 ANALYZE FOR NEXT TRADING DAY",
                    type="primary",
                    key="options_analyze_button"
                ):
                    symbols=[
                        str(s).strip().upper()
                        for s in option_df["Symbol"].dropna().unique()
                    ]

                    with st.spinner(
                        "Downloading underlying charts and calculating "
                        "Fibonacci + option positioning..."
                    ):
                        options_market=download_batches(
                            symbols,
                            period,
                            50
                        )

                        option_result=run_options_next_day_analysis(
                            option_df,
                            options_market,
                            options_fib_lookback
                        )

                    if option_result.empty:
                        st.warning(
                            "No technical analysis could be calculated. "
                            "Try a longer market-data period."
                        )
                    else:
                        def _threshold_signal(r):
                            c=float(r["Call Score"])
                            p=float(r["Put Score"])
                            s=float(r["Selling Score"])

                            if c>=min_call_score and c>p and c>s:
                                return "🟢 Strong Call Candidate"
                            if p>=min_put_score and p>c and p>s:
                                return "🔴 Strong Put Candidate"
                            if s>=75 and s>c and s>p:
                                return "🟡 Option Selling Candidate"
                            if max(c,p)>=65:
                                return "🔵 Option Buying Candidate"
                            return "⚪ Avoid / Wait"

                        option_result["Signal"]=option_result.apply(
                            _threshold_signal,axis=1
                        )

                        st.session_state[
                            "minervini_options_result"
                        ]=option_result

                        calls=option_result[
                            option_result["Signal"]==
                            "🟢 Strong Call Candidate"
                        ]
                        puts=option_result[
                            option_result["Signal"]==
                            "🔴 Strong Put Candidate"
                        ]
                        sellers=option_result[
                            option_result["Signal"]==
                            "🟡 Option Selling Candidate"
                        ]
                        buyers=option_result[
                            option_result["Signal"]==
                            "🔵 Option Buying Candidate"
                        ]

                        a,b,c,d=st.columns(4)
                        a.metric("🟢 Strong Calls",len(calls))
                        b.metric("🔴 Strong Puts",len(puts))
                        c.metric("🟡 Selling",len(sellers))
                        d.metric("🔵 Buying",len(buyers))

                        st.subheader(
                            "📈 Next-Day Options Candidates"
                        )

                        display=[
                            "Stock","Symbol","Close","Trend",
                            "Call Score","Put Score","Selling Score",
                            "Signal","Position","OI Trend","PCR",
                            "Future OI Chg %","Fib Support Price",
                            "Fib Resistance Price"
                        ]
                        display=[
                            c for c in display
                            if c in option_result.columns
                        ]

                        st.dataframe(
                            option_result[display],
                            width="stretch",
                            hide_index=True
                        )

                        st.download_button(
                            "⬇️ Download Next-Day Options Report",
                            option_result.to_csv(index=False),
                            "Options_Next_Day_Analysis.csv",
                            "text/csv"
                        )

                        st.subheader("⭐ Strong Candidates")

                        top_cols=[
                            "Stock","Symbol","Close",
                            "Call Score","Put Score","Selling Score",
                            "Signal","Trend","Position","OI Trend","PCR",
                            "Fib Support Price","Fib Resistance Price"
                        ]
                        top_cols=[
                            c for c in top_cols
                            if c in option_result.columns
                        ]

                        top=option_result[
                            option_result["Signal"].isin([
                                "🟢 Strong Call Candidate",
                                "🔴 Strong Put Candidate",
                                "🟡 Option Selling Candidate",
                                "🔵 Option Buying Candidate"
                            ])
                        ].head(30)

                        st.dataframe(
                            top[top_cols],
                            width="stretch",
                            hide_index=True
                        )

                        st.info(
                            "The uploaded file is treated as EOD information "
                            "for the next trading session. This is a research "
                            "signal, not an automatic order. IV, option premium "
                            "and bid/ask spread are not scored unless those "
                            "fields are present in the CSV."
                        )

        except Exception as exc:
            st.error(
                f"Could not analyze the uploaded derivatives CSV: {exc}"
            )

    if run:
        with st.spinner("Scanning Minervini Trend Template + VCP setups..."):
            market=download_batches(stocks,period,50)
            benchmark=download_nifty50_benchmark(period)
            result=run_minervini_scanner(
                market,
                benchmark=benchmark,
                window_days=vcp_window,
                min_volume_lakhs=min_liquidity,
                breakout_volume_mult=breakout_volume_mult,
                chase_pct=chase_pct
            )

        if result.empty:
            st.warning(
                "No valid technical candidates could be calculated. "
                "Try a longer data period or another universe."
            )
        else:
            buys=result[result["Status"]=="🚀 BUY"]
            watches=result[result["Status"]=="🟡 VCP WATCH"]
            qualified=result[result["Status"]=="🟢 SEPA QUALIFIED"]

            c1,c2,c3,c4=st.columns(4)
            c1.metric("🚀 BUY",len(buys))
            c2.metric("🟡 VCP WATCH",len(watches))
            c3.metric("🟢 SEPA QUALIFIED",len(qualified))
            c4.metric("Stocks Scanned",len(result))

            st.subheader("🏆 Minervini Scanner Results")

            display_cols=[
                "Stock","Score","Trend Template","RS Rank","RS Line",
                "6M Return %","12M Return %","52W High Distance %",
                "VCP","VCP Score","Contractions","Final Tightness %",
                "Pivot","Pivot Distance %","Volume / SMA50",
                "Liquidity ₹L","Status"
            ]
            display_cols=[c for c in display_cols if c in result.columns]

            st.dataframe(
                result[display_cols],
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Minervini Results",
                result.to_csv(index=False),
                "Minervini_SEPA_VCP_Scanner.csv",
                "text/csv"
            )

            st.subheader("📌 Signal Logic")
            st.markdown(
                """
                **🚀 BUY:** Trend Template PASS + valid VCP + price above pivot + """
                f"breakout volume ≥ {breakout_volume_mult:.1f}× SMA50 + price no more than {chase_pct:.1f}% above pivot + score ≥80."
                """
                **🟡 VCP WATCH:** Trend Template PASS + valid VCP + waiting for a clean pivot breakout.

                **🟢 SEPA QUALIFIED:** Trend Template PASS and technical leadership score ≥70, but VCP/breakout confirmation is incomplete.

                **⚠️ BREAKOUT — DON'T CHASE:** price has broken the pivot but is already beyond the configured chase zone.
                """
            )

            st.subheader("📖 Technical Score Breakdown")
            st.caption(
                "Technical 100-point implementation used by this module: "
                "Trend Template 25 + RS/Leadership 20 + VCP 25 + Pivot/Breakout 15 + "
                "Market/Liquidity 15. Fundamental points are not fabricated."
            )

            selected=st.selectbox(
                "Inspect a stock",
                result["Stock"].tolist(),
                key="minervini_inspect_stock"
            )
            row=result[result["Stock"]==selected].iloc[0]

            a,b,c,d,e=st.columns(5)
            a.metric("Score",f'{row["Score"]:.0f}/100')
            b.metric("RS Rank",f'{row["RS Rank"]:.1f}')
            c.metric("VCP",row["VCP"])
            d.metric("Pivot",f'₹{row["Pivot"]:.2f}' if pd.notna(row["Pivot"]) else "-")
            e.metric("Status",row["Status"])

            st.write(
                f"**Contractions:** {row['Contractions']}  |  "
                f"**Final tightness:** {row['Final Tightness %']}%  |  "
                f"**Volume/SMA50:** {row['Volume / SMA50']}×  |  "
                f"**Pivot distance:** {row['Pivot Distance %']}%"
            )

    if run_backtest:

        if not stocks:
            st.error("No stocks available for the Minervini backtest.")
            st.stop()

        st.subheader("📊 Minervini SEPA + VCP Backtest")

        st.info(
            "The historical scanner is evaluated bar-by-bar using only "
            "information available up to each completed day. A BUY signal "
            "is entered at the following day's open. This prevents look-ahead bias."
        )

        progress=st.progress(
            0,
            text="Downloading historical data..."
        )

        bt_stocks=stocks[:int(backtest_max_stocks)]

        st.caption(
            f"Backtest universe: {len(bt_stocks)} of {len(stocks)} stocks."
        )

        market=download_batches(
            bt_stocks,
            period,
            50
        )

        benchmark=download_nifty50_benchmark(period)

        progress.progress(
            25,
            text="Preparing historical Minervini signals..."
        )

        bt_result=backtest_minervini_sepa_vcp(
            market,
            benchmark=benchmark,
            window_days=vcp_window,
            min_volume_lakhs=min_liquidity,
            breakout_volume_mult=breakout_volume_mult,
            chase_pct=chase_pct,
            stop_loss_pct=backtest_stop,
            max_holding_days=backtest_max_hold,
            starting_capital=backtest_capital,
            max_positions=backtest_max_positions,
            position_risk_pct=backtest_risk
        )

        progress.progress(
            100,
            text="Backtest complete"
        )

        summary=bt_result["summary"]
        equity=bt_result["equity"]
        trades=bt_result["trades_df"]

        c1,c2,c3,c4,c5=st.columns(5)

        c1.metric(
            "Return",
            f'{summary.get("Return %",0):.2f}%'
        )

        c2.metric(
            "Max Drawdown",
            f'{summary.get("Max Drawdown %",0):.2f}%'
        )

        c3.metric(
            "Trades",
            int(summary.get("Trades",0))
        )

        c4.metric(
            "Win Rate",
            f'{summary.get("Win Rate %",0):.1f}%'
        )

        pf=summary.get("Profit Factor",0)
        pf_text=(
            "∞" if pf==np.inf
            else f"{pf:.2f}"
        )

        c5.metric(
            "Profit Factor",
            pf_text
        )

        st.subheader("📈 Portfolio Equity Curve")

        if not equity.empty:

            fig=go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=equity["Date"],
                    y=equity["Equity"],
                    mode="lines",
                    name="Portfolio Equity"
                )
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Portfolio Value ₹",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if not trades.empty:

            st.subheader("📋 Trade Log")

            trade_cols=[
                "Stock","Entry Date","Exit Date",
                "Entry","Exit","Shares","Pivot","Stop",
                "PnL","Return %","Holding Days",
                "Score","RS Rank","VCP Score","Exit Reason"
            ]

            trade_cols=[
                c for c in trade_cols
                if c in trades.columns
            ]

            st.dataframe(
                trades[trade_cols],
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Minervini Backtest Trades",
                trades.to_csv(index=False),
                "Minervini_SEPA_VCP_Backtest_Trades.csv",
                "text/csv"
            )

        st.subheader("📐 Backtest Rules")

        st.markdown(
            f"""
            **Entry:** exactly the scanner's historical BUY condition:
            Trend Template PASS + valid VCP + pivot breakout +
            breakout volume ≥ {breakout_volume_mult:.1f}× SMA50 +
            price within {chase_pct:.1f}% of pivot + score ≥80.

            **Execution:** next day's open.

            **Risk stop:** {backtest_stop:.1f}% below entry.

            **Trend failure:** profitable position closing below SMA50.

            **Breakout failure:** close below the original pivot.

            **Maximum holding period:** {backtest_max_hold} days.

            **Position sizing:** maximum {backtest_max_positions} positions,
            with approximately {backtest_risk:.2f}% of current equity at risk per position.
            """
        )


# ============================================================
# CCI + EMA9/21/200 + RSI9/WMA21 STRATEGY MODULE
# ============================================================

elif module == "🎯 CCI + EMA + RSI Strategy":

    st.header(
        "🎯 CCI + EMA9/21/200 + RSI9/WMA21 Strategy"
    )

    st.markdown(
        """
        ### Entry — ALL 5 conditions required

        1. **EMA(9) and EMA(21) are above EMA(200)** and
           each is within the configured 1–2% upper range.
           If either EMA is below EMA(200), there is **NO ENTRY**.
        2. **Close > EMA(200)**
        3. **Daily CCI(20) > 100**
        4. **RSI(9) > 60 and < 70**
        5. **RSI(9) > WMA(21)**

        ### Exit — ANY 2 OF 3 conditions required

        1. **EMA(9) crosses below EMA(200)** from above
        2. **WMA(21) > RSI(9)**
        3. **CCI(20) < 100**

        The core exit is triggered only when **at least 2 of the
        3 exit conditions are true on the same completed daily bar**.
        """
    )

    st.sidebar.subheader(
        "🎯 Strategy Settings"
    )

    nse_stocks=load_nse_equity_universe()
    nifty500=load_nifty500()
    fno_stocks=load_fno_stocks()
    nifty_midcap100=load_nifty_midcap100()
    nifty_smallcap250=load_nifty_smallcap250()

    universe=st.sidebar.selectbox(
        "Stock Universe",
        [
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "NSE F&O Stocks",
            "Full NSE"
        ],
        key="cci_ema_rsi_universe"
    )

    stocks=resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    timeframe=st.sidebar.selectbox(
        "Timeframe",
        ["Daily"],
        key="cci_ema_rsi_timeframe"
    )

    ema200_near_pct=st.sidebar.select_slider(
        "EMA9 & EMA21 range above EMA200 (%)",
        options=[1.0,2.0],
        value=2.0,
        help=(
            "Both EMA9 and EMA21 must be above EMA200 and "
            "no more than the selected percentage above it. "
            "If either EMA is below EMA200, entry is rejected."
        ),
        key="cci_ema_rsi_near_pct"
    )

    st.sidebar.info(
        "Exit = at least 2 of 3 conditions: "
        "EMA9 cross below EMA200, WMA21 > RSI9, CCI < 100."
    )

    max_holding_days=st.sidebar.slider(
        "Maximum holding days",
        20,
        250,
        120,
        10,
        key="cci_ema_rsi_max_hold"
    )

    st.sidebar.markdown("### 🛡️ Risk Management")

    st.sidebar.caption(
        "Trailing profit is optional. Turning it OFF does not "
        "disable the maximum-loss stop."
    )

    max_loss_pct=st.sidebar.slider(
        "Maximum loss per trade (%)",
        5.0,
        20.0,
        20.0,
        1.0,
        help=(
            "Hard maximum loss from the entry price. "
            "Default is 20%."
        ),
        key="cci_ema_rsi_max_loss"
    )

    trailing_enabled=st.sidebar.checkbox(
        "Enable trailing profit",
        value=False,
        help=(
            "When OFF, the trail activation and trailing "
            "stop are completely disabled. The original "
            "strategy exit and maximum-loss stop remain active."
        ),
        key="cci_ema_rsi_trailing_enabled"
    )

    trail_activation_pct=st.sidebar.slider(
        "Trail activates after profit (%)",
        5.0,
        30.0,
        10.0,
        1.0,
        help=(
            "Trailing stop starts after the trade reaches "
            "this profit from entry."
        ),
        key="cci_ema_rsi_trail_activation",
        disabled=not trailing_enabled
    )

    trailing_stop_pct=st.sidebar.slider(
        "Trailing stop distance (%)",
        3.0,
        20.0,
        10.0,
        1.0,
        help=(
            "Trailing stop remains this percentage below "
            "the highest price reached after entry."
        ),
        key="cci_ema_rsi_trailing_distance",
        disabled=not trailing_enabled
    )

    st.sidebar.markdown("### 💰 Profit Booking")

    profit_booking_enabled=st.sidebar.checkbox(
        "Enable partial profit booking",
        value=False,
        help=(
            "Book part of the position at Target 1 and "
            "Target 2 while allowing the remaining position "
            "to continue running."
        ),
        key="cci_ema_rsi_profit_booking_enabled"
    )

    target1_pct=st.sidebar.slider(
        "Target 1 profit (%)",
        5.0,30.0,15.0,1.0,
        key="cci_ema_rsi_target1",
        disabled=not profit_booking_enabled
    )

    target1_booking_pct=st.sidebar.slider(
        "Target 1: book position (%)",
        5.0,50.0,25.0,5.0,
        key="cci_ema_rsi_target1_booking",
        disabled=not profit_booking_enabled
    )

    target2_pct=st.sidebar.slider(
        "Target 2 profit (%)",
        10.0,60.0,25.0,1.0,
        key="cci_ema_rsi_target2",
        disabled=not profit_booking_enabled
    )

    target2_booking_pct=st.sidebar.slider(
        "Target 2: book position (%)",
        5.0,50.0,25.0,5.0,
        key="cci_ema_rsi_target2_booking",
        disabled=not profit_booking_enabled
    )

    st.sidebar.markdown("### 🛡️ Improved Exit")

    improved_exit_enabled=st.sidebar.checkbox(
        "Enable additional profit-protection exit",
        value=False,
        help=(
            "After the trade first reaches the selected profit "
            "threshold, exit early if EMA9 < EMA21 OR RSI9 < WMA21. "
            "The original technical exit remains active."
        ),
        key="cci_ema_rsi_improved_exit_enabled"
    )

    improved_exit_activation_pct=st.sidebar.slider(
        "Profit-protection activates after (%)",
        3.0,30.0,5.0,1.0,
        key="cci_ema_rsi_improved_exit_activation",
        disabled=not improved_exit_enabled
    )

    ema200_breakdown_exit_enabled=st.sidebar.checkbox(
        "Enable additional EMA200 breakdown exit",
        value=False,
        help=(
            "Exit when WMA21 > RSI9 AND either EMA9 or EMA21 "
            "has crossed below EMA200 AND price closes below "
            "EMA200 AND both EMA9 and EMA21 are sloping down."
        ),
        key="cci_ema_rsi_ema200_breakdown_exit"
    )

    # --------------------------------------------
    # Safe stock-count selector
    #
    # The previous slider reused one widget value across
    # different universes. For example, switching from a
    # 100-stock universe to a smaller universe could leave
    # the old value (e.g. 100) in session state while the
    # new slider maximum was smaller, causing:
    # StreamlitAPIException from st.slider().
    #
    # Each universe now gets its own widget state and the
    # slider bounds are always valid.
    # --------------------------------------------
    stock_count=len(stocks)

    if stock_count<=0:

        st.sidebar.warning(
            "No stocks are currently available for this universe."
        )

        max_stocks=0

    else:

        max_stock_limit=min(
            500,
            stock_count
        )

        default_max_stocks=min(
            100,
            max_stock_limit
        )

        slider_step=(
            10
            if max_stock_limit>=10
            else 1
        )

        max_stocks=st.sidebar.slider(
            "Maximum Stocks",
            min_value=1,
            max_value=max_stock_limit,
            value=default_max_stocks,
            step=slider_step,
            key=f"cci_ema_rsi_max_stocks_{universe}"
        )

    if stock_count<=0:
        st.error(
            f"No stocks could be loaded for **{universe}**. "
            "The universe loader could not reach any configured "
            "constituent source. Please reload the app; if the "
            "problem persists, check the universe loader status "
            "shown below."
        )


    # Strategy comparison controls MUST be outside the
    # stock_count<=0 block. Otherwise these variables do not
    # exist when a normal universe loads.
    st.sidebar.markdown("### 🔬 Strategy Comparison")

    previous_baseline_enabled=st.sidebar.checkbox(
        "Enable Previous vs New comparison",
        value=True,
        help=(
            "Runs both strategies on the same stocks and "
            "period with trailing/profit booking disabled, "
            "so the entry/exit rule change can be isolated."
        ),
        key="cci_ema_rsi_compare_enabled"
    )

    previous_ema_range=st.sidebar.slider(
        "Previous strategy EMA range (%)",
        1.0,5.0,3.0,0.5,
        help=(
            "Previous-style baseline uses absolute distance "
            "from EMA200, allowing EMA9/EMA21 either above "
            "or below EMA200."
        ),
        key="cci_ema_rsi_previous_ema_range"
    )

    previous_rsi50_tolerance=st.sidebar.slider(
        "Previous exit RSI/WMA tolerance around 50",
        0.5,5.0,2.0,0.5,
        key="cci_ema_rsi_previous_rsi50_tolerance"
    )

    scan_tab, backtest_tab = st.tabs(
        [
            "🔎 Live Scanner",
            "📊 Backtest"
        ]
    )

    with scan_tab:

        run_scan=st.button(
            "🔎 SCAN ENTRY / EXIT SIGNALS",
            type="primary",
            key="cci_ema_rsi_scan"
        )

        if run_scan:

            selected=stocks[:max_stocks]

            with st.spinner(
                "Downloading daily data..."
            ):

                market=download_batches(
                    selected,
                    "2y",
                    50
                )

            rows=[]

            for symbol in selected:

                data=market.get(symbol)

                if data is None or data.empty:
                    continue

                prepared=prepare_cci_ema_rsi_strategy(
                    data
                )

                if prepared.empty:
                    continue

                prepared=add_cci_ema_rsi_conditions(
                    prepared,
                    ema200_near_pct,
                    None
                )

                last=prepared.iloc[-1]

                entry=bool(last["ENTRY_SIGNAL"])
                exit_signal=bool(last["EXIT_SIGNAL"])

                if entry:
                    signal="🟢 BUY"
                elif exit_signal:
                    signal="🔴 EXIT / SELL"
                else:
                    signal="⚪ NO SIGNAL"

                rows.append(
                    {
                        "Stock":symbol,
                        "Signal":signal,
                        "Close":round(
                            float(last["Close"]),
                            2
                        ),
                        "CCI(20)":round(
                            float(last["CCI20"]),
                            2
                        ),
                        "EMA9":round(
                            float(last["EMA9"]),
                            2
                        ),
                        "EMA21":round(
                            float(last["EMA21"]),
                            2
                        ),
                        "EMA200":round(
                            float(last["EMA200"]),
                            2
                        ),
                        "EMA9-EMA200 %":round(
                            float(
                                last[
                                    "EMA9_DISTANCE_EMA200_%"
                                ]
                            ),
                            2
                        ),
                        "EMA21-EMA200 %":round(
                            float(
                                last[
                                    "EMA21_DISTANCE_EMA200_%"
                                ]
                            ),
                            2
                        ),
                        "RSI(9)":round(
                            float(last["RSI9"]),
                            2
                        ),
                        "WMA(21)":round(
                            float(last["RSI9_WMA21"]),
                            2
                        ),
                        "EMA9 Down":(
                            float(last["EMA9"])
                            <
                            float(last["EMA9_PREV"])
                        ),
                        "EMA21 Down":(
                            float(last["EMA21"])
                            <
                            float(last["EMA21_PREV"])
                        )
                    }
                )

            result=pd.DataFrame(rows)

            if result.empty:
                st.warning(
                    "No usable daily data was returned."
                )
            else:

                buys=result[
                    result["Signal"]=="🟢 BUY"
                ]

                exits=result[
                    result["Signal"]=="🔴 EXIT / SELL"
                ]

                c1,c2,c3=st.columns(3)

                c1.metric(
                    "🟢 BUY",
                    len(buys)
                )

                c2.metric(
                    "🔴 EXIT / SELL",
                    len(exits)
                )

                c3.metric(
                    "⚪ No Signal",
                    len(result)
                    -len(buys)
                    -len(exits)
                )

                st.dataframe(
                    result.sort_values(
                        "Signal"
                    ),
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Scanner Results",
                    result.to_csv(index=False),
                    "CCI_EMA_RSI_Scanner.csv",
                    "text/csv"
                )

    with backtest_tab:

        st.subheader(
            "📊 Strategy-Specific Backtest"
        )

        st.caption(
            "Only the entry and exit rules shown above "
            "are used. Smart Breakout, chart-pattern and "
            "other scanner conditions are NOT included."
        )

        years=st.sidebar.selectbox(
            "Backtest Period",
            ["2y","3y","5y","10y"],
            index=1,
            key="cci_ema_rsi_bt_period"
        )

        if previous_baseline_enabled:

            run_compare=st.button(
                "🔬 COMPARE PREVIOUS vs NEW STRATEGY",
                key="cci_ema_rsi_compare_button"
            )

            if run_compare:

                selected=stocks[:max_stocks]

                with st.spinner(
                    "Running Previous vs New comparison..."
                ):

                    market=download_batches(
                        selected,
                        years,
                        50
                    )

                comparison=compare_cci_ema_rsi_strategies(
                    market,
                    new_ema_range_pct=ema200_near_pct,
                    legacy_ema_range_pct=previous_ema_range,
                    legacy_rsi_50_tolerance=previous_rsi50_tolerance,
                    max_holding_days=max_holding_days,
                    max_loss_pct=max_loss_pct
                )

                new_s=comparison["New Summary"]
                old_s=comparison["Previous Summary"]

                st.subheader(
                    "🔬 Previous vs New Strategy"
                )

                comparison_summary=pd.DataFrame([
                    {
                        "Metric":"Trades",
                        "Previous":old_s["Trades"],
                        "New":new_s["Trades"],
                        "Change":new_s["Trades"]-old_s["Trades"]
                    },
                    {
                        "Metric":"Net Return %",
                        "Previous":old_s["Net Return %"],
                        "New":new_s["Net Return %"],
                        "Change":round(
                            new_s["Net Return %"]-
                            old_s["Net Return %"],2
                        )
                    },
                    {
                        "Metric":"Max Drawdown %",
                        "Previous":old_s["Max Drawdown %"],
                        "New":new_s["Max Drawdown %"],
                        "Change":round(
                            new_s["Max Drawdown %"]-
                            old_s["Max Drawdown %"],2
                        )
                    },
                    {
                        "Metric":"Win Rate %",
                        "Previous":old_s["Win Rate %"],
                        "New":new_s["Win Rate %"],
                        "Change":round(
                            new_s["Win Rate %"]-
                            old_s["Win Rate %"],2
                        )
                    },
                    {
                        "Metric":"Profit Factor",
                        "Previous":old_s["Profit Factor"],
                        "New":new_s["Profit Factor"],
                        "Change":(
                            round(
                                new_s["Profit Factor"]-
                                old_s["Profit Factor"],2
                            )
                            if (
                                np.isfinite(
                                    new_s["Profit Factor"]
                                )
                                and
                                np.isfinite(
                                    old_s["Profit Factor"]
                                )
                            )
                            else np.nan
                        )
                    },
                    {
                        "Metric":"Average Trade %",
                        "Previous":old_s["Average Trade %"],
                        "New":new_s["Average Trade %"],
                        "Change":round(
                            new_s["Average Trade %"]-
                            old_s["Average Trade %"],2
                        )
                    }
                ])

                st.dataframe(
                    comparison_summary,
                    width="stretch",
                    hide_index=True
                )

                st.info(
                    "Comparison uses identical stock universe, "
                    "historical period, maximum holding period and "
                    "20% maximum-loss setting. Trailing, partial "
                    "profit booking and additional exits are OFF. "
                    "This isolates the effect of the strategy rules."
                )

                stock_compare=comparison[
                    "Stock Comparison"
                ]

                if not stock_compare.empty:

                    st.subheader(
                        "📊 Stock-by-Stock Comparison"
                    )

                    st.dataframe(
                        stock_compare.sort_values(
                            "New Return %",
                            ascending=False
                        ),
                        width="stretch",
                        hide_index=True
                    )

                    st.download_button(
                        "⬇️ Download Strategy Comparison",
                        stock_compare.to_csv(
                            index=False
                        ),
                        "CCI_EMA_RSI_Previous_vs_New.csv",
                        "text/csv"
                    )

        run_bt=st.button(
            "📊 RUN CCI/EMA/RSI BACKTEST",
            type="primary",
            key="cci_ema_rsi_backtest"
        )

        if run_bt:

            selected=stocks[:max_stocks]

            with st.spinner(
                "Downloading historical daily data..."
            ):

                market=download_batches(
                    selected,
                    years,
                    50
                )

            all_trades=[]
            stock_stats=[]

            for symbol in selected:

                data=market.get(symbol)

                if data is None or data.empty:
                    continue

                bt=backtest_cci_ema_rsi_strategy(
                    data,
                    ema200_near_pct,
                    None,
                    max_holding_days,
                    max_loss_pct,
                    trailing_enabled,
                    trail_activation_pct,
                    trailing_stop_pct,
                    profit_booking_enabled,
                    target1_pct,
                    target1_booking_pct,
                    target2_pct,
                    target2_booking_pct,
                    improved_exit_enabled,
                    improved_exit_activation_pct,
                    ema200_breakdown_exit_enabled
                )

                trades=bt["Trades"]

                for trade in trades:
                    trade2=trade.copy()
                    trade2["Stock"]=symbol
                    all_trades.append(trade2)

                if trades:

                    s=summarize_cci_ema_rsi_backtest(
                        trades
                    )

                    s["Stock"]=symbol
                    stock_stats.append(s)

            summary=summarize_cci_ema_rsi_backtest(
                all_trades
            )

            a,b,c,d,e,f=st.columns(6)

            a.metric(
                "Trades",
                summary["Trades"]
            )

            b.metric(
                "Win Rate",
                f"{summary['Win Rate %']:.1f}%"
            )

            c.metric(
                "Profit Factor",
                (
                    f"{summary['Profit Factor']:.2f}"
                    if np.isfinite(
                        summary["Profit Factor"]
                    )
                    else "∞"
                )
            )

            d.metric(
                "Net Return",
                f"{summary['Net Return %']:.2f}%"
            )

            e.metric(
                "Max Drawdown",
                f"{summary['Max Drawdown %']:.2f}%"
            )

            f.metric(
                "Avg Trade",
                f"{summary['Average Trade %']:.2f}%"
            )

            if all_trades:

                trade_df=pd.DataFrame(
                    all_trades
                ).sort_values(
                    "Entry Date"
                )

                st.subheader(
                    "📋 Trade-by-Trade Results"
                )

                st.dataframe(
                    trade_df,
                    width="stretch",
                    hide_index=True
                )

                # --------------------------------------------
                # PORTFOLIO EQUITY CURVE
                # --------------------------------------------
                # Builds a sequential portfolio curve from the
                # realized return of each completed trade.
                # This is an unleveraged, one-position-at-a-time
                # equity curve based on the backtest trade list.
                equity_df=trade_df.copy()

                equity_df["Entry Date"]=pd.to_datetime(
                    equity_df["Entry Date"]
                )
                equity_df["Exit Date"]=pd.to_datetime(
                    equity_df["Exit Date"]
                )

                equity_df=equity_df.sort_values(
                    ["Exit Date","Entry Date"]
                ).reset_index(drop=True)

                starting_capital=st.number_input(
                    "Portfolio starting capital",
                    min_value=10000.0,
                    value=100000.0,
                    step=10000.0,
                    key="cci_ema_rsi_starting_capital"
                )

                equity_df["Equity"] = starting_capital

                current_equity=float(starting_capital)
                equity_values=[]

                for pnl in equity_df["P&L %"].fillna(0):
                    current_equity *= (
                        1.0 + float(pnl)/100.0
                    )
                    equity_values.append(current_equity)

                equity_df["Equity"]=equity_values

                portfolio_chart=equity_df[
                    ["Exit Date","Equity"]
                ].copy()

                portfolio_chart=portfolio_chart.rename(
                    columns={
                        "Exit Date":"Date"
                    }
                )

                st.subheader(
                    "📈 Portfolio Equity Curve"
                )

                st.line_chart(
                    portfolio_chart.set_index("Date")["Equity"],
                    height=400,
                    use_container_width=True
                )

                # Portfolio drawdown calculated from the
                # same sequential equity curve.
                equity_series=pd.Series(
                    [starting_capital] + equity_values,
                    dtype="float64"
                )

                running_peak=equity_series.cummax()

                drawdown_pct=(
                    equity_series-running_peak
                )/running_peak*100.0

                max_portfolio_drawdown=float(
                    drawdown_pct.min()
                )

                total_return_pct=(
                    current_equity-starting_capital
                )/starting_capital*100.0

                p1,p2,p3=st.columns(3)

                p1.metric(
                    "Starting Capital",
                    f"₹{starting_capital:,.0f}"
                )

                p2.metric(
                    "Final Equity",
                    f"₹{current_equity:,.0f}"
                )

                p3.metric(
                    "Portfolio Max Drawdown",
                    f"{max_portfolio_drawdown:.2f}%"
                )

                st.caption(
                    f"Portfolio return: {total_return_pct:.2f}%. "
                    "The curve compounds each completed trade sequentially "
                    "and therefore represents a strategy-level equity curve, "
                    "not simultaneous multi-position capital allocation."
                )

                st.download_button(
                    "⬇️ Download Backtest Trades",
                    trade_df.to_csv(
                        index=False
                    ),
                    "CCI_EMA_RSI_Backtest_Trades.csv",
                    "text/csv"
                )

            if stock_stats:

                st.subheader(
                    "📊 Stock-Level Performance"
                )

                stock_df=pd.DataFrame(
                    stock_stats
                ).sort_values(
                    "Net Return %",
                    ascending=False
                )

                st.dataframe(
                    stock_df,
                    width="stretch",
                    hide_index=True
                )



# ============================================================
# RSI / WMA TIMEFRAME SCANNER
# ============================================================

elif module == "🏆 Top 20 Stocks":

    st.header(
        "🏆 Top 20 Stocks"
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

        nifty_midcap100 = (
            load_nifty_midcap100()
        )

        nifty_smallcap250 = (
            load_nifty_smallcap250()
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
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "Full NSE"
        ],
        index=0
    )

    stocks = resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
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

    # Do not wrap these cached loaders in st.spinner. Streamlit's
    # spinner starts a helper thread; on Streamlit Cloud a process
    # that has accumulated many downloader/BLAS threads can fail
    # here with: RuntimeError: can't start new thread.
    st.caption("Loading stock universes...")

    fno_stocks = load_fno_stocks()
    nifty500 = load_nifty500()
    nse_stocks = load_nse_equity_universe()
    nifty_midcap100 = load_nifty_midcap100()
    nifty_smallcap250 = load_nifty_smallcap250()

    universe = st.sidebar.selectbox(
        "Stock Universe",
        [
            "NSE F&O Stocks",
            "Nifty 50",
            "Nifty 500",
            "Nifty Midcap 100",
            "Nifty Smallcap 250",
            "Full NSE"
        ],
        index=0,
        key="backtest_universe"
    )

    stocks = resolve_stock_universe(
        universe,
        nse_stocks,
        nifty500,
        fno_stocks,
        nifty_midcap100,
        nifty_smallcap250
    )

    if not stocks:
        st.error(
            f"No stocks are available for **{universe}**. "
            "Please try again later."
        )
        st.stop()

    strategy = st.sidebar.selectbox(
        "Strategy to Backtest",
        [
            "Smart Breakout",
            "120-Day High Breakout",
            "Hourly Donchian Breakout",
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
        "120-Day High Breakout":
            "Only the four supplied 120-day breakout/liquidity conditions are tested.",
        "Hourly Donchian Breakout":
            "Only the six supplied hourly Donchian/SMA200/RSI(9) conditions are tested.",
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
        "Hourly Donchian Breakout",
        "Multi-Timeframe RSI/WMA"
    ] and (
        strategy in [
            "Hourly RSI(9)/WMA(21)",
            "Hourly Donchian Breakout"
        ]
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

            **120-Day High Breakout**
            - ONLY these four supplied conditions are tested:
              1. Daily Close > 1 day ago Max(120, Daily High)
              2. 1 day ago Close < 2 days ago Max(120, Daily High)
              3. NSE Value in lakhs > 50
              4. Daily Close > 1 day ago Close × 1.03
            - No Smart Breakout, RSI/WMA, Weekly Trend, or Daily Trend
              condition is added.

            **Hourly Donchian Breakout**
            - ONLY these six hourly conditions are tested:
              1. [0] 1-hour Close > [-1] 1-hour High
              2. [-1] 1-hour High < [-2] 1-hour High
              3. [0] 1-hour Close > [0] 1-hour SMA(Close, 200)
              4. [-1] 1-hour Low > [0] 1-hour Donchian Lower Band(5)
              5. [-1] 1-hour High < [0] 1-hour Donchian Upper Band(5)
              6. [0] 1-hour RSI(9) >= 55
            - No Smart Breakout, Daily RSI, Weekly Trend, or Daily Trend
              condition is added.
            - Because 60-minute history is limited, historical testing
              is limited to the available recent intraday dataset.

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
                strategy in [
                    "Hourly RSI(9)/WMA(21)",
                    "Hourly Donchian Breakout"
                ]
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
                    breakout_120_result = None
                    hourly_breakout = None

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
                    # 2. 120-DAY HIGH BREAKOUT ONLY
                    # --------------------------------------------
                    elif strategy == "120-Day High Breakout":

                        breakout_120_result = (
                            calculate_120day_breakout_asof(
                                data,
                                signal_date
                            )
                        )

                        signal = (
                            breakout_120_result is not None
                            and breakout_120_result[
                                "Pass"
                            ]
                        )

                    # --------------------------------------------
                    # 3. HOURLY DONCHIAN BREAKOUT ONLY
                    # --------------------------------------------
                    elif strategy == "Hourly Donchian Breakout":

                        hourly_breakout = (
                            hourly_donchian_breakout_asof(
                                hourly_history.get(symbol),
                                signal_date
                            )
                        )

                        signal = (
                            hourly_breakout is not None
                            and hourly_breakout["Pass"]
                        )

                    # --------------------------------------------
                    # 4. DAILY RSI/WMA ONLY
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

                    elif strategy == "120-Day High Breakout":

                        grade = "120-Day Breakout Qualified"

                    elif strategy == "Hourly Donchian Breakout":

                        grade = "Hourly Donchian Breakout Qualified"

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

                        "120D Breakout Close":
                            (
                                round(
                                    breakout_120_result[
                                        "Close"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "120-Day High Breakout"
                                    and breakout_120_result is not None
                                )
                                else np.nan
                            ),

                        "120D Prior High":
                            (
                                round(
                                    breakout_120_result[
                                        "120D High 1D Ago"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "120-Day High Breakout"
                                    and breakout_120_result is not None
                                )
                                else np.nan
                            ),

                        "NSE Value Lakhs":
                            (
                                round(
                                    breakout_120_result[
                                        "NSE Value Lakhs"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "120-Day High Breakout"
                                    and breakout_120_result is not None
                                )
                                else np.nan
                            ),

                        "120D Breakout 3pct":
                            (
                                (
                                    breakout_120_result[
                                        "Close"
                                    ]
                                    >
                                    breakout_120_result[
                                        "Previous Close"
                                    ] * 1.03
                                )
                                if (
                                    strategy
                                    == "120-Day High Breakout"
                                    and breakout_120_result is not None
                                )
                                else np.nan
                            ),

                        "Hourly Breakout RSI9":
                            (
                                round(
                                    hourly_breakout["RSI9"],
                                    2
                                )
                                if (
                                    strategy
                                    == "Hourly Donchian Breakout"
                                    and hourly_breakout is not None
                                )
                                else np.nan
                            ),

                        "Hourly SMA200":
                            (
                                round(
                                    hourly_breakout["SMA200"],
                                    2
                                )
                                if (
                                    strategy
                                    == "Hourly Donchian Breakout"
                                    and hourly_breakout is not None
                                )
                                else np.nan
                            ),

                        "Hourly Donchian Upper5":
                            (
                                round(
                                    hourly_breakout[
                                        "Donchian Upper5"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "Hourly Donchian Breakout"
                                    and hourly_breakout is not None
                                )
                                else np.nan
                            ),

                        "Hourly Donchian Lower5":
                            (
                                round(
                                    hourly_breakout[
                                        "Donchian Lower5"
                                    ],
                                    2
                                )
                                if (
                                    strategy
                                    == "Hourly Donchian Breakout"
                                    and hourly_breakout is not None
                                )
                                else np.nan
                            ),

                        "Hourly Signal Time":
                            (
                                str(
                                    hourly_breakout["Date"]
                                )
                                if (
                                    strategy
                                    == "Hourly Donchian Breakout"
                                    and hourly_breakout is not None
                                )
                                else ""
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


elif module == "🤖 AI Analyst":

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

