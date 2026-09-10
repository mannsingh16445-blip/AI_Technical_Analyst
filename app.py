import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import io
import time
import math
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='NSE Multi-Timeframe EMA Scanner', page_icon='📈', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1rem;}
.small {font-size: 0.85rem; color: #666;}
.badge {padding: 0.18rem 0.45rem; border-radius: 0.4rem; font-weight: 700;}
</style>
''', unsafe_allow_html=True)

st.title('📈 NSE Multi-Timeframe EMA 9/21/200 + RSI(9) + CCI(20) Scanner')
st.caption('Research scanner based on your 3-timeframe chart structure: Monthly → Weekly → Daily, with early, confirmed and strong-momentum stages.')

DEFAULT_SYMBOLS = '''KAVDEFENCE
RELIANCE
TCS
INFY
HDFCBANK
ICICIBANK
SBIN
AXISBANK
LT
BHARTIARTL
SUNPHARMA
TATAMOTORS
TATASTEEL
BEL
HAL
BHEL
RVNL
MAZDOCK
CDSL
IRFC'''


def clean_symbol(x: str) -> str:
    x = str(x).strip().upper()
    x = x.replace('NSE:', '').replace('.NS', '')
    return ''.join(ch for ch in x if ch.isalnum() or ch in ('_', '-'))


def load_universe(uploaded, pasted: str):
    symbols = []
    names = {}
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            cols = {str(c).strip().lower(): c for c in raw.columns}
            sym_col = None
            for k in ('ticker', 'symbol', 'tradingsymbol', 'nse symbol', 'stock'):
                if k in cols:
                    sym_col = cols[k]; break
            name_col = None
            for k in ('name', 'stock name', 'company', 'company name'):
                if k in cols:
                    name_col = cols[k]; break
            if sym_col:
                for _, r in raw.iterrows():
                    s = clean_symbol(r[sym_col])
                    if s:
                        symbols.append(s)
                        if name_col and pd.notna(r[name_col]):
                            names[s] = str(r[name_col]).strip()
        except Exception as e:
            st.error(f'Could not read universe CSV: {e}')
    else:
        for line in pasted.replace(',', '\n').splitlines():
            s = clean_symbol(line)
            if s:
                symbols.append(s)
    seen = set(); out = []
    for s in symbols:
        if s not in seen:
            seen.add(s); out.append(s)
    return out, names


def yf_download(symbol, interval, period):
    ticker = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={c: c.title() for c in df.columns})
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep].dropna(subset=['Close'])
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index)
        try:
            if getattr(df.index, 'tz', None) is not None:
                df.index = df.index.tz_localize(None)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi_wilder(close, n=9):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ag = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    r = r.mask(al.eq(0) & ag.gt(0), 100)
    return r.fillna(50)


def cci(df, n=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def add_indicators(df):
    x = df.copy()
    x['EMA9'] = ema(x['Close'], 9)
    x['EMA21'] = ema(x['Close'], 21)
    x['EMA200'] = ema(x['Close'], 200)
    x['RSI9'] = rsi_wilder(x['Close'], 9)
    x['CCI20'] = cci(x, 20)
    x['VolSMA20'] = x['Volume'].rolling(20).mean()
    x['VolRatio'] = x['Volume'] / x['VolSMA20'].replace(0, np.nan)
    x['EMA9_Gap_Pct'] = (x['EMA9'] - x['EMA21']).abs() / x['Close'] * 100
    x['EMA200_Slope_Pct'] = x['EMA200'].pct_change(10) * 100
    x['RSI9_MA3'] = x['RSI9'].rolling(3).mean()
    x['RecentHigh20'] = x['High'].shift(1).rolling(20).max()
    x['RecentHigh10'] = x['High'].shift(1).rolling(10).max()
    return x.dropna(subset=['EMA200','RSI9','CCI20'])


def latest_features(df):
    if df.empty or len(df) < 220:
        return None
    x = add_indicators(df)
    if len(x) < 2:
        return None
    a = x.iloc[-1]; p = x.iloc[-2]
    # Structure
    price = float(a.Close)
    trend = price > a.EMA200
    ema_stack = a.EMA9 > a.EMA21 > a.EMA200
    ema_bull = a.EMA9 > a.EMA21
    ema_cross_recent = (p.EMA9 <= p.EMA21) and (a.EMA9 > a.EMA21)
    ema_cross_5 = bool((x['EMA9'].tail(5).shift(1) <= x['EMA21'].tail(5).shift(1)) & (x['EMA9'].tail(5) > x['EMA21'].tail(5))).any()
    rsi_rising = a.RSI9 > p.RSI9
    rsi_above50 = a.RSI9 >= 50
    rsi_ma_bull = a.RSI9 > a.RSI9_MA3
    cci_above100 = a.CCI20 >= 100
    cci_cross100 = (p.CCI20 < 100) and (a.CCI20 >= 100)
    cci_rising = a.CCI20 > p.CCI20
    vol_ok = (a.VolRatio >= 1.2) if pd.notna(a.VolRatio) else False
    breakout = pd.notna(a.RecentHigh20) and price > a.RecentHigh20
    near_ema = a.EMA9_Gap_Pct <= 2.5
    # Expansion after compression: current gap larger than its recent median, while recent median itself is modest
    gap_med_20 = x['EMA9_Gap_Pct'].tail(20).median()
    expansion = pd.notna(gap_med_20) and a.EMA9_Gap_Pct > gap_med_20 * 1.25
    long_slope = a.EMA200_Slope_Pct > 0

    # Stages: intentionally score-based; exact labels are not trading guarantees.
    early = trend and a.EMA21 > a.EMA200 and near_ema and rsi_above50 and rsi_rising and cci_rising
    confirmed = trend and ema_bull and rsi_above50 and (cci_above100 or cci_cross100) and (vol_ok or breakout)
    strong = ema_stack and rsi_above50 and cci_above100 and breakout and vol_ok

    score = 0
    score += 15 if trend else 0
    score += 10 if a.EMA21 > a.EMA200 else 0
    score += 10 if ema_bull else 0
    score += 5 if ema_cross_recent or ema_cross_5 else 0
    score += 10 if near_ema else 0
    score += 10 if rsi_above50 else 0
    score += 5 if rsi_rising else 0
    score += 5 if rsi_ma_bull else 0
    score += 10 if cci_above100 else (5 if cci_rising else 0)
    score += 8 if vol_ok else 0
    score += 7 if breakout else 0
    score += 3 if long_slope else 0
    score += 2 if expansion else 0
    score = min(100, int(score))

    if strong:
        stage = 'C — Strong Momentum'
    elif confirmed:
        stage = 'B — Confirmed Momentum'
    elif early:
        stage = 'A — Early Setup'
    else:
        stage = 'Watch / Developing'

    return {
        'Close': price, 'EMA9': float(a.EMA9), 'EMA21': float(a.EMA21), 'EMA200': float(a.EMA200),
        'RSI9': float(a.RSI9), 'CCI20': float(a.CCI20), 'VolRatio': float(a.VolRatio) if pd.notna(a.VolRatio) else np.nan,
        'EMA9-21 Gap %': float(a.EMA9_Gap_Pct), 'EMA200 Slope 10-bar %': float(a.EMA200_Slope_Pct),
        'Trend': trend, 'EMA Stack': ema_stack, 'EMA9>EMA21': ema_bull, 'Recent Cross': ema_cross_recent or ema_cross_5,
        'Near EMA9/21': near_ema, 'RSI Rising': rsi_rising, 'RSI > 50': rsi_above50, 'RSI > RSI-MA3': rsi_ma_bull,
        'CCI > 100': cci_above100, 'CCI Rising': cci_rising, 'Volume > 1.2x': vol_ok, '20D Breakout': breakout,
        'EMA Expansion': expansion, 'EMA200 Rising': long_slope, 'Stage': stage, 'Score': score,
        'Date': x.index[-1]
    }


def analyze_symbol(symbol, names):
    # 3 timeframes matching the user's chart set
    specs = [('Monthly','1mo','10y'), ('Weekly','1wk','5y'), ('Daily','1d','2y')]
    feats = {}
    for label, interval, period in specs:
        df = yf_download(symbol, interval, period)
        f = latest_features(df)
        if f is None:
            return None
        feats[label] = f
        time.sleep(0.03)
    m, w, d = feats['Monthly'], feats['Weekly'], feats['Daily']

    mtf_bull = (m['Trend'] and w['Trend'] and d['Trend'])
    mtf_stack = (m['EMA Stack'] and w['EMA Stack'] and d['EMA Stack'])
    mtf_rsi = (m['RSI > 50'] and w['RSI > 50'] and d['RSI > 50'])
    mtf_cci = (m['CCI > 100'] and w['CCI > 100'] and d['CCI > 100'])
    daily_setup = d['Stage'] != 'Watch / Developing'
    # A normalized MTF score: monthly 25, weekly 30, daily 45
    score = 0
    score += 25 if m['Trend'] else 0
    score += 8 if m['EMA Stack'] else (5 if m['EMA9>EMA21'] else 0)
    score += 7 if m['RSI > 50'] else 0
    score += 30 if w['Trend'] else 0
    score += 10 if w['EMA Stack'] else (7 if w['EMA9>EMA21'] else 0)
    score += 8 if w['RSI > 50'] else 0
    score += 20 if d['Trend'] else 0
    score += 8 if d['EMA Stack'] else (6 if d['EMA9>EMA21'] else 0)
    score += 5 if d['RSI > 50'] else 0
    score += 5 if d['CCI > 100'] else (2 if d['CCI Rising'] else 0)
    score += 5 if d['Volume > 1.2x'] else 0
    score += 5 if d['20D Breakout'] else 0
    score += 4 if d['EMA Expansion'] else 0
    # cap
    score = min(100, int(score))

    if mtf_stack and d['20D Breakout'] and d['Volume > 1.2x'] and d['RSI > 50'] and d['CCI > 100']:
        verdict = '🔥 High-Confluence Breakout'
    elif mtf_bull and d['EMA9>EMA21'] and d['RSI > 50'] and (d['CCI Rising'] or d['CCI > 100']):
        verdict = '✅ Bullish Alignment'
    elif mtf_bull and (d['Near EMA9/21'] or d['Recent Cross']):
        verdict = '🟡 Early / Building'
    elif mtf_bull:
        verdict = '🔵 Trend Positive'
    else:
        verdict = '⚪ Mixed'

    return {
        'Stock': names.get(symbol, symbol), 'Symbol': symbol,
        'MTF Score': score, 'Verdict': verdict, 'Daily Stage': d['Stage'],
        'Monthly Trend': 'Yes' if m['Trend'] else 'No', 'Weekly Trend': 'Yes' if w['Trend'] else 'No', 'Daily Trend': 'Yes' if d['Trend'] else 'No',
        'Monthly EMA9>21': 'Yes' if m['EMA9>EMA21'] else 'No', 'Weekly EMA9>21': 'Yes' if w['EMA9>EMA21'] else 'No', 'Daily EMA9>21': 'Yes' if d['EMA9>EMA21'] else 'No',
        'Daily Close': round(d['Close'], 2), 'Daily EMA9': round(d['EMA9'], 2), 'Daily EMA21': round(d['EMA21'], 2), 'Daily EMA200': round(d['EMA200'], 2),
        'Daily RSI9': round(d['RSI9'], 2), 'Daily CCI20': round(d['CCI20'], 2), 'Daily Vol x20': round(d['VolRatio'], 2) if pd.notna(d['VolRatio']) else np.nan,
        'Daily EMA9-21 Gap %': round(d['EMA9-21 Gap %'], 2), 'Daily Breakout': 'Yes' if d['20D Breakout'] else 'No',
        'Daily EMA200 Rising': 'Yes' if d['EMA200 Rising'] else 'No', 'Monthly RSI9': round(m['RSI9'],2), 'Weekly RSI9': round(w['RSI9'],2),
        'Analysis Date': str(d['Date'].date())
    }


def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar:
    st.header('Universe')
    uploaded = st.file_uploader('Upload CSV (Ticker/Symbol + optional Name)', type=['csv'])
    pasted = st.text_area('Or paste NSE symbols (one per line)', DEFAULT_SYMBOLS, height=260)
    max_stocks = st.number_input('Maximum stocks to scan', min_value=1, max_value=1000, value=50, step=10)
    st.divider()
    st.header('Ranking filters')
    min_score = st.slider('Minimum MTF Score', 0, 100, 60)
    show_mixed = st.checkbox('Show mixed / developing stocks', value=False)
    st.caption('Important: this is a research scanner, not a guarantee of future returns.')

symbols, names = load_universe(uploaded, pasted)
symbols = symbols[:int(max_stocks)]

col1, col2, col3 = st.columns([1,1,2])
with col1:
    run = st.button('🚀 Scan Now', type='primary', use_container_width=True)
with col2:
    st.metric('Stocks loaded', len(symbols))
with col3:
    st.info('Best use: run after market close so Monthly/Weekly/Daily bars are complete.')

if run:
    results = []
    progress = st.progress(0)
    status = st.empty()
    for i, s in enumerate(symbols, start=1):
        status.write(f'Scanning {i}/{len(symbols)} — {s}')
        r = analyze_symbol(s, names)
        if r is not None:
            results.append(r)
        progress.progress(i / max(1, len(symbols)))
    progress.empty(); status.empty()
    if not results:
        st.error('No stocks could be analyzed. Check symbols/network or upload a valid NSE universe CSV.')
        st.stop()

    out = pd.DataFrame(results).sort_values(['MTF Score','Daily RSI9'], ascending=[False,False])
    out = out[out['MTF Score'] >= min_score]
    if not show_mixed:
        out = out[out['Verdict'] != '⚪ Mixed']

    st.session_state['mtf_results'] = out

if 'mtf_results' in st.session_state:
    out = st.session_state['mtf_results']
    st.subheader('Top Ranked Stocks')
    if out.empty:
        st.warning('No stocks met the selected filters. Lower the Minimum MTF Score or enable mixed/developing stocks.')
    else:
        top = out.head(10)
        cols = st.columns(min(5, len(top)))
        for i, (_, row) in enumerate(top.iterrows()):
            with cols[i % len(cols)]:
                st.metric(row['Symbol'], f"{int(row['MTF Score'])}/100", row['Verdict'])

        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button('⬇️ Download ranked results CSV', csv_bytes(out), 'NSE_MultiTF_EMA_Ranked.csv', 'text/csv')

        st.subheader('How the scanner classifies the setup')
        st.markdown('''
**A — Early Setup:** daily trend is above EMA200, EMA9/EMA21 are close or recovering, RSI(9) is above 50 and rising, and CCI(20) is improving.  
**B — Confirmed Momentum:** bullish EMA9/EMA21 structure plus RSI/CCI confirmation, with breakout or volume participation.  
**C — Strong Momentum:** Monthly + Weekly + Daily trend alignment, EMA9 > EMA21 > EMA200 on the daily chart, 20-day breakout, RSI(9) > 50, CCI(20) > +100 and volume ≥ 1.2× its 20-day average.
''')
        st.warning('The thresholds are intentionally transparent and editable. They should be validated with out-of-sample backtesting before being used as a trading system.')
else:
    st.markdown('### Scanner logic')
    st.markdown('''
**Primary trend:** Price above EMA200 on Monthly, Weekly and Daily.  
**Preferred structure:** EMA9 > EMA21 > EMA200.  
**Momentum:** RSI(9) > 50 and rising; CCI(20) rising, with +100 supporting confirmed momentum.  
**Participation:** Volume ≥ 1.2× 20-day average for stronger setups.  
**Breakout:** Close above the prior 20-day high upgrades the signal.  
**Compression:** Small EMA9/EMA21 gap followed by expansion is treated as an acceleration clue.
''')
