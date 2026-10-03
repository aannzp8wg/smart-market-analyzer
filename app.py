import streamlit as st

st.set_page_config(page_title="محلل السوق الذكي", page_icon="🤖", layout="wide")

import json
import math
import os
import re
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import yfinance as yf
from google import genai
from google.genai import types
from groq import Groq

# ---------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------
# قوائم احتياطية فقط — القوائم الفعلية تُجلب ديناميكياً من حسابك (انظر get_gemini_models / get_groq_models)
GEMINI_FALLBACK = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"]
GROQ_FALLBACK = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
DATA_PERIOD = "1y"

AUTO = "Auto (اختيار تلقائي حسب السوق)"
BREAKOUT = "Breakout (اختراق)"
PULLBACK = "Trend + Pullback (اتجاه+ارتداد)"
MEAN_REV = "Mean Reversion (عودة للمتوسط)"

TF_D1 = "يومي (1D)"
TF_H4 = "4 ساعات (4H)"
TF_DUAL = "مزدوج (اتجاه يومي + دخول 4H)"

SL_ATR = 1.5   # وقف الخسارة = 1.5 ATR
TP1_ATR = 1.5  # الهدف الأول = 1R
TP2_ATR = 3.0  # الهدف الثاني = 2R

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"خطأ: المفتاح {e} غير موجود في Streamlit Secrets.")
    st.stop()

st.title("محلل السوق الذكي")
st.markdown("### فريق وكلاء: فني + أخبار + مخاطر")

# ---------------------------------------------------------------------
# تهيئة الجلسة
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# السجل الدائم للصفقات (GitHub Gist، أو ملف مؤقت إن لم يُضبط)
# ---------------------------------------------------------------------
JOURNAL_FILE = os.path.join(tempfile.gettempdir(), "trading_journal.json")


def _gist_cfg():
    try:
        tok, gid = st.secrets.get("GITHUB_TOKEN"), st.secrets.get("GIST_ID")
    except Exception:
        return None
    return (str(tok), str(gid)) if tok and gid else None


def storage_mode():
    return "gist" if _gist_cfg() else "local"


def _jdump(obj):
    return json.dumps(obj, ensure_ascii=False,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def _gist_req(url, tok, body=None, method="GET"):
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
        "User-Agent": "trading-journal", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _scrub(msg, cfg):
    return (msg.replace(cfg[0], "***") if cfg else msg)[:160]


def journal_load():
    """يرجع (dict فيه trades وposition، رسالة خطأ). أي خطأ يوقف الحفظ لاحقاً حمايةً للبيانات."""
    empty = {"trades": [], "position": None}
    cfg = _gist_cfg()
    try:
        if cfg:
            data = _gist_req(f"https://api.github.com/gists/{cfg[1]}", cfg[0])
            f = (data.get("files") or {}).get("trades.json")
            if not f:
                return empty, "لا يوجد ملف باسم trades.json داخل الـ Gist (أنشئه بمحتوى {})."
            txt = (f.get("content") or "").strip()
            obj = json.loads(txt) if txt else {}
        else:
            if not os.path.exists(JOURNAL_FILE):
                return empty, ""
            with open(JOURNAL_FILE, encoding="utf-8") as fh:
                obj = json.load(fh)
        if isinstance(obj, list):
            obj = {"trades": obj}
        if not isinstance(obj, dict):
            obj = {}
        trades = obj.get("trades")
        return {"trades": trades if isinstance(trades, list) else [], "position": obj.get("position")}, ""
    except Exception as e:
        return empty, _scrub(f"{type(e).__name__}: {e}", cfg)


def journal_save(trades, position):
    obj = {"version": 1, "trades": trades, "position": position}
    cfg = _gist_cfg()
    try:
        if cfg:
            body = json.dumps({"files": {"trades.json": {"content": _jdump(obj)}}}).encode()
            _gist_req(f"https://api.github.com/gists/{cfg[1]}", cfg[0], body, "PATCH")
        else:
            with open(JOURNAL_FILE, "w", encoding="utf-8") as fh:
                fh.write(_jdump(obj))
        return ""
    except Exception as e:
        return _scrub(f"{type(e).__name__}: {e}", cfg)


def persist():
    """يحفظ السجل والصفقة المفتوحة (ولا يحفظ إذا فشل التحميل الأول، لئلا يُمسح سجلك الموجود)."""
    if not st.session_state.get("journal_ok", True):
        st.session_state["journal_save_err"] = "الحفظ متوقف لأن تحميل السجل فشل — أصلح الخطأ ثم أعد تحميل الصفحة."
        return
    st.session_state["journal_save_err"] = journal_save(st.session_state.trades, st.session_state.current_position)


def journal_groups(tdf):
    """إحصاءات الأداء مجمّعة (استراتيجية/إطار/نقاط/مصدر...). ترجع [(العنوان، DataFrame)]."""
    d = tdf.copy()
    d["win"] = d["pnl_percent"] > 0
    for c in ("r_multiple", "score"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in ("used", "tf", "source", "direction", "regime"):
        if c not in d:
            d[c] = None
    first = lambda x: str(x).split(" (")[0] if isinstance(x, str) and x else None
    d["الاستراتيجية"] = d["used"].map(first)
    d["الإطار"] = d["tf"].map(first)
    d["المصدر"] = d["source"].map({"plan": "حسب الخطة", "manual": "يدوي"})
    d["الاتجاه"] = d["direction"]
    d["حالة السوق"] = d["regime"].map(lambda x: x if isinstance(x, str) and x else None)
    d["نقاط الجودة"] = pd.cut(d["score"], [-1, 69, 79, 84, 100], labels=["أقل من 70", "70–79", "80–84", "85 فأكثر"]).astype(object)
    out = []
    for col in ("نقاط الجودة", "الاستراتيجية", "الإطار", "المصدر", "الاتجاه", "حالة السوق"):
        sub = d.dropna(subset=[col])
        if sub.empty:
            continue
        g = sub.groupby(col).agg(n=("win", "size"), wr=("win", "mean"), r=("r_multiple", "mean"),
                                 tot=("pnl_percent", "sum")).reset_index()
        g["wr"] = (g["wr"] * 100).round(1)
        g["r"] = g["r"].round(2)
        g["tot"] = g["tot"].round(2)
        g.columns = [col, "الصفقات", "نسبة الربح %", "متوسط R", "الإجمالي %"]
        out.append((col, g))
    return out


defaults = {
    "trades": [],
    "current_position": None,
    "analysis": None,
    "live_price": None,
    "notice": None,
    "backtest": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "journal_loaded" not in st.session_state:   # تحميل السجل مرة واحدة لكل جلسة
    _j, _jerr = journal_load()
    st.session_state.trades = _j["trades"]
    st.session_state.current_position = _j["position"]
    st.session_state.journal_ok = not _jerr
    st.session_state.journal_load_err = _jerr
    st.session_state.journal_loaded = True


# ---------------------------------------------------------------------
# البيانات والمؤشرات
# ---------------------------------------------------------------------
class DataError(Exception):
    """فشل جلب البيانات (رسالة جاهزة للعرض)."""


def _fetch_yf(ticker, period, interval):
    """جلب من Yahoo مع إعادة محاولة، وتشخيص السبب الحقيقي عند الفشل. يرفع DataError."""
    last = ""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
            last = "empty"
        except Exception as e:
            last = str(e)
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    if last == "empty":  # اكتشف السبب: حظر أم رمز خاطئ؟
        try:
            yf.Ticker(ticker).history(period=period, interval=interval, raise_errors=True)
        except TypeError:
            pass
        except Exception as e:
            last = str(e)
    low = last.lower()
    if any(k in low for k in ("rate", "too many", "429")):
        raise DataError("Yahoo Finance حجب الطلبات مؤقتاً (Rate limit) — شائع على Streamlit Cloud. "
                        "انتظر 5–10 دقائق ثم أعد المحاولة.")
    if last == "empty" or any(k in low for k in ("no data", "delisted", "not found")):
        raise DataError(f"لا توجد بيانات لـ {ticker} ({interval}) — تحقق من الرمز.")
    raise DataError(f"فشل جلب البيانات ({interval}): {last[:150]}")


# ملاحظة: الأخطاء تُرفع (ولا تُخزَّن)، فلا يُحفظ الفشل في الكاش لـ5 دقائق.
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, period=DATA_PERIOD):
    df = _fetch_yf(ticker, period, "1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_data_h4(ticker):
    err = None
    df = None
    for per in ("700d", "365d", "180d"):
        try:
            df = _fetch_yf(ticker, per, "1h")
            break
        except DataError as e:
            err = e
            if "Rate limit" in str(e) or "لا توجد بيانات" in str(e):
                break
    if df is None:
        raise err
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    agg = {c: f for c, f in {"Open": "first", "High": "max", "Low": "min",
                             "Close": "last", "Volume": "sum"}.items() if c in df.columns}
    return df.resample("4h").agg(agg).dropna(subset=["Open"])


def drop_incomplete_candle(df, hours=None):
    """حذف الشمعة الجارية (غير المكتملة) لتفادي إشارات تتغير. hours=4 لإطار 4H."""
    if len(df) <= 1:
        return df
    if hours:
        tz = df.index.tz
        now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
        if df.index[-1] + pd.Timedelta(hours=hours) > now:
            return df.iloc[:-1]
    elif df.index[-1].date() >= datetime.now(timezone.utc).date():
        return df.iloc[:-1]
    return df


def calculate_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Donchian_Upper"] = df["High"].rolling(20).max()
    df["Donchian_Lower"] = df["Low"].rolling(20).min()

    # ATR (Wilder)
    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # RSI (Wilder)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ADX (Wilder) — لقياس قوة الاتجاه وتحديد حالة السوق
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / df["ATR14"]
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / df["ATR14"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # بولنجر (SMA20 ± 2 انحراف معياري)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    return df


# ---------------------------------------------------------------------
# الأطر الزمنية واتجاه اليومي
# ---------------------------------------------------------------------
def daily_bias_series(daily_df):
    """اتجاه اليومي: UP إذا الإغلاق فوق EMA200، DOWN إذا تحتها."""
    c, e = daily_df["Close"], daily_df["EMA200"]
    return pd.Series(np.select([c > e, c < e], ["UP", "DOWN"], default="NEUTRAL"), index=daily_df.index)


def align_bias(df_lower, bias_daily):
    """يربط اتجاه اليومي بكل شمعة 4H باستخدام آخر يومية *مكتملة* (حتى أمس) — بدون نظر للمستقبل."""
    d_idx = pd.DatetimeIndex(bias_daily.index)
    if d_idx.tz is not None:
        d_idx = d_idx.tz_localize(None)
    b = pd.Series(bias_daily.values, index=d_idx.normalize())
    b = b[~b.index.duplicated(keep="last")].sort_index()

    l_idx = pd.DatetimeIndex(df_lower.index)
    if l_idx.tz is not None:
        l_idx = l_idx.tz_localize(None)
    target = l_idx.normalize() - pd.Timedelta(days=1)
    out = b.reindex(target, method="ffill").fillna("NEUTRAL")
    return pd.Series(out.values, index=df_lower.index)


def apply_bias_filter(sigs, bias):
    """يرفض الإشارات المعاكسة لاتجاه اليومي: شراء فقط في UP، بيع فقط في DOWN."""
    b = bias.values
    out = {}
    for k, sr in sigs.items():
        v = sr.values.copy()
        v[(v == 1) & (b != "UP")] = 0
        v[(v == -1) & (b != "DOWN")] = 0
        out[k] = pd.Series(v, index=sr.index)
    return out


def load_prepared(ticker, tf, use_closed=True, daily_period=DATA_PERIOD):
    try:
        return _load_prepared(ticker, tf, use_closed, daily_period)
    except DataError as e:
        return None, None, str(e)


def _load_prepared(ticker, tf, use_closed=True, daily_period=DATA_PERIOD):
    """يرجع (df الرئيسي بالمؤشرات، اتجاه اليومي لكل شمعة أو None، رسالة خطأ أو None)."""
    if tf == TF_D1:
        raw = load_data(ticker, daily_period)
        if raw.empty:
            return None, None, f"لم يتم العثور على بيانات يومية لـ {ticker}"
        df = drop_incomplete_candle(raw) if use_closed else raw
        return calculate_indicators(df), None, None

    raw4 = load_data_h4(ticker)
    if raw4.empty:
        return None, None, f"لا تتوفر بيانات 4 ساعات لـ {ticker}"
    df4 = drop_incomplete_candle(raw4, 4) if use_closed else raw4
    df4 = calculate_indicators(df4)
    if tf == TF_H4:
        return df4, None, None

    rawd = load_data(ticker, "5y")  # 5 سنوات ليستقر EMA200 اليومي قبل بداية بيانات 4H
    if rawd.empty:
        return None, None, f"لم يتم العثور على بيانات يومية لـ {ticker}"
    dfd = calculate_indicators(rawd)
    return df4, align_bias(df4, daily_bias_series(dfd)), None


# ---------------------------------------------------------------------
# الاستراتيجيات + الاختيار التلقائي
# ---------------------------------------------------------------------
def generate_signal(df, strategy):
    latest = df.iloc[-1]
    signal, reason = "No Signal", ""

    if strategy == BREAKOUT:
        if latest["Close"] > df["Donchian_Upper"].iloc[-2] and latest["Close"] > latest["EMA200"]:
            signal, reason = "BUY", "اختراق صعودي فوق قمة 20 يوم + اتجاه صاعد"
        elif latest["Close"] < df["Donchian_Lower"].iloc[-2] and latest["Close"] < latest["EMA200"]:
            signal, reason = "SELL", "كسر هبوطي تحت قاع 20 يوم + اتجاه هابط"
        else:
            reason = "لا يوجد اختراق واضح حاليا"

    elif strategy == PULLBACK:
        if latest["Close"] > latest["EMA200"] and latest["Low"] <= latest["EMA50"] and latest["Close"] > latest["Open"]:
            signal, reason = "BUY", "ارتداد من EMA50 في اتجاه صاعد"
        elif latest["Close"] < latest["EMA200"] and latest["High"] >= latest["EMA50"] and latest["Close"] < latest["Open"]:
            signal, reason = "SELL", "ارتداد من EMA50 في اتجاه هابط"
        else:
            reason = "لا يوجد ارتداد واضح"

    elif strategy == MEAN_REV:
        if latest["Close"] < latest["BB_Lower"] and latest["RSI14"] < 35:
            signal, reason = "BUY", "تشبع بيعي: السعر تحت بولنجر السفلي + RSI < 35"
        elif latest["Close"] > latest["BB_Upper"] and latest["RSI14"] > 65:
            signal, reason = "SELL", "تشبع شرائي: السعر فوق بولنجر العلوي + RSI > 65"
        else:
            reason = "السعر في المنطقة الطبيعية"

    return signal, reason


def select_strategy(df, mode, bias=None):
    """يحدد حالة السوق ويختار الاستراتيجية المناسبة (أو يستخدم اختيار المستخدم)."""
    adx = float(df["ADX14"].iloc[-1])
    if adx >= 25:
        regime, order = "اتجاه قوي 📈", [BREAKOUT, PULLBACK]
    elif adx < 20:
        regime, order = "سوق عرضي ↔️", [MEAN_REV]
    else:
        regime, order = "انتقالي (اتجاه ضعيف) 🔄", [PULLBACK, BREAKOUT, MEAN_REV]

    if mode != AUTO:
        order = [mode]

    checks = []
    for strat in order:
        sig, reason = generate_signal(df, strat)
        if bias is not None and sig != "No Signal":
            allowed = (sig == "BUY" and bias == "UP") or (sig == "SELL" and bias == "DOWN")
            if not allowed:
                reason = f"{reason} — ❌ مرفوضة: تعاكس اتجاه اليومي"
                sig = "No Signal"
        checks.append((strat, sig, reason))
    chosen = next((c for c in checks if c[1] != "No Signal"), None)
    return regime, adx, checks, chosen


# ---------------------------------------------------------------------
# الرسم
# ---------------------------------------------------------------------
def create_chart(df, ticker, signal=None, sl=None, tp1=None, tp2=None, tf_label=""):
    chart_df = df.tail(100)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name="السعر"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], name="EMA20", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], name="EMA50", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200"], name="EMA200", line=dict(color="red", width=1.5)))

    if signal in ("BUY", "SELL"):
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1]], y=[chart_df["Close"].iloc[-1]], mode="markers", name=signal,
            marker=dict(size=14, symbol="triangle-up" if signal == "BUY" else "triangle-down",
                        color="green" if signal == "BUY" else "red")))
        if sl is not None:
            fig.add_hline(y=sl, line_dash="dash", line_color="red", annotation_text="SL")
        if tp1 is not None:
            fig.add_hline(y=tp1, line_dash="dot", line_color="green", annotation_text="TP1")
        if tp2 is not None:
            fig.add_hline(y=tp2, line_dash="dash", line_color="green", annotation_text="TP2")

    fig.update_layout(title=f"{ticker} - {tf_label}", template="plotly_white",
                      height=500, xaxis_rangeslider_visible=False)
    return fig


# ---------------------------------------------------------------------
# أدوات مساعدة
# ---------------------------------------------------------------------
def P(x):
    """تنسيق السعر: خانتان للذهب/الأسهم، 4-5 خانات للفوركس."""
    x = float(x)
    if abs(x) >= 10:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def truthy(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1", "نعم")


def as_list(v):
    if isinstance(v, list):
        return v
    return [v] if v else []


# ---------------------------------------------------------------------
# شارت TradingView الحي + شارت الصفقات
# ---------------------------------------------------------------------
TV_MAP = {"GC=F": "OANDA:XAUUSD", "SI=F": "OANDA:XAGUSD", "CL=F": "TVC:USOIL", "BZ=F": "TVC:UKOIL",
          "NG=F": "CAPITALCOM:NATURALGAS", "BTC-USD": "BINANCE:BTCUSDT", "ETH-USD": "BINANCE:ETHUSDT",
          "^GSPC": "SP:SPX", "^IXIC": "NASDAQ:IXIC", "^DJI": "DJ:DJI", "DX-Y.NYB": "TVC:DXY"}


def guess_tv_symbol(t):
    """رمز TradingView المقابل لرمز Yahoo (قابل للتعديل من الواجهة)."""
    u = t.strip().upper()
    if u in TV_MAP:
        return TV_MAP[u]
    if u.endswith("=X") and len(u) == 8:
        return f"FX:{u[:6]}"
    if u.endswith("-USD"):
        return f"BINANCE:{u[:-4]}USDT"
    return u


def tradingview_html(symbol, interval, height=520):
    cfg = {"autosize": True, "symbol": symbol, "interval": interval, "timezone": "Etc/UTC",
           "theme": "dark", "style": "1", "locale": "en", "allow_symbol_change": True,
           "withdateranges": True, "support_host": "https://www.tradingview.com"}
    payload = json.dumps(cfg).replace("</", "<\\/")
    return (f'<div class="tradingview-widget-container" style="height:{height}px;width:100%">'
            f'<div class="tradingview-widget-container__widget" style="height:{height - 32}px;width:100%"></div>'
            '<script type="text/javascript" '
            'src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'
            f'{payload}</script></div>')


def render_tv_chart(a=None):
    sym_default = guess_tv_symbol(ticker)
    with st.expander("📺 الشارت الحي (TradingView)", expanded=True):
        c1, c2 = st.columns([2, 1])
        sym = c1.text_input("رمز TradingView", value=sym_default, key=f"tv_sym_{ticker.strip().upper()}")
        opts = ["15", "60", "240", "D", "W"]
        labels = {"15": "15 دقيقة", "60": "ساعة", "240": "4 ساعات", "D": "يومي", "W": "أسبوعي"}
        iv = c2.selectbox("الفريم", opts, index=opts.index("D") if timeframe == TF_D1 else opts.index("240"),
                          format_func=lambda x: labels[x], key=f"tv_iv_{timeframe}")
        components.html(tradingview_html(sym.strip() or sym_default, iv, 520), height=530)
        pl = a.get("plan") if a else None
        if pl and pl.get("direction") and a.get("ticker") == ticker.strip():
            st.markdown(f"**مستويات الخطة:** الدخول `{P(pl['entry'])}` | الوقف `{P(pl['sl'])}` | "
                        f"الهدف 1 `{P(pl['tp1'])}` | الهدف 2 `{P(pl['tp2'])}`")
            st.caption("الودجت لا يرسم المستويات تلقائياً: أضف خطاً أفقياً عند كل سعر. وسعر TradingView (فوري) "
                       "قد يختلف قليلاً عن سعر العقود الآجلة الذي يعتمد عليه التحليل، فاعتمد الفرق لا الرقم المطلق.")
        else:
            st.caption("إذا لم يظهر الشارت فاكتب رمزاً من TradingView (مثال: OANDA:XAUUSD أو BINANCE:BTCUSDT).")


def create_trades_chart(df, tr, title, max_trades=25, max_bars=700):
    """شموع + دخول (مثلث) وخروج (×) وخط بينهما لكل صفقة: أخضر رابحة، أحمر خاسرة."""
    t = tr.tail(max_trades)
    i0 = i1 = 0
    while True:
        i0 = max(int(df.index.searchsorted(t["entry_date"].iloc[0])) - 15, 0)
        i1 = min(int(df.index.searchsorted(t["exit_date"].iloc[-1])) + 10, len(df))
        if i1 - i0 <= max_bars or len(t) <= 1:
            break
        t = t.iloc[1:]
    d = df.iloc[i0:i1]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="السعر", opacity=0.55))
    for win, color, name in ((True, "#2ecc71", "رابحة"), (False, "#e74c3c", "خاسرة")):
        part = t[(t["pnl"] > 0) == win]
        xs, ys = [], []
        for _, r in part.iterrows():
            xs += [r["entry_date"], r["exit_date"], None]
            ys += [r["entry"], r["exit"], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2), name=name))
    for direction, sym_, col in (("BUY", "triangle-up", "#2ecc71"), ("SELL", "triangle-down", "#e74c3c")):
        part = t[t["direction"] == direction]
        if len(part):
            fig.add_trace(go.Scatter(x=part["entry_date"], y=part["entry"], mode="markers",
                                     marker=dict(symbol=sym_, size=12, color=col, line=dict(width=1, color="white")),
                                     name=f"دخول {direction}",
                                     text=[f"{direction} @ {P(e)}" for e in part["entry"]], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=t["exit_date"], y=t["exit"], mode="markers",
                             marker=dict(symbol="x", size=10, color=["#2ecc71" if v > 0 else "#e74c3c" for v in t["pnl"]]),
                             name="خروج",
                             text=[f"{r} @ {P(x)} | {p_:+.2f}%" for r, x, p_ in zip(t["reason"], t["exit"], t["pnl"])],
                             hoverinfo="text"))
    fig.update_layout(title=title, template="plotly_white", height=520, xaxis_rangeslider_visible=False)
    return fig


def guess_contract_size(ticker):
    """عدد الوحدات في 1 لوت (تقريبي — تحقق من وسيطك)."""
    t = ticker.strip().upper()
    known = {"GC=F": 100.0, "SI=F": 5000.0, "CL=F": 1000.0, "NG=F": 10000.0, "BTC-USD": 1.0, "ETH-USD": 1.0}
    if t in known:
        return known[t]
    if t.endswith("=X"):
        return 100000.0
    return 1.0  # أسهم: اللوت = سهم واحد


def trade_levels(signal, price, atr):
    sign = 1 if signal == "BUY" else -1
    return (price - sign * SL_ATR * atr,
            price + sign * TP1_ATR * atr,
            price + sign * TP2_ATR * atr)


# ---------------------------------------------------------------------
# استدعاء النماذج
# ---------------------------------------------------------------------
def extract_json(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def rank_gemini_models(names):
    """يرتّب نماذج Gemini المتاحة: الأحدث المستقر أولاً (flash قبل pro)، ثم التجريبي، ثم الاحتياطي."""
    skip = ("image", "live", "audio", "tts", "embedding", "aqa", "vision", "robotics",
            "computer", "native", "imagen", "veo", "lite")

    def ver(n):
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)

    cands = [n for n in names if n.startswith("gemini") and ("flash" in n or "pro" in n)
             and not any(x in n for x in skip) and ver(n)[0] >= 3]
    key = lambda n: (0 if "flash" in n else 1, -ver(n)[0], -ver(n)[1])
    stable = sorted([n for n in cands if not any(x in n for x in ("preview", "exp"))], key=key)
    preview = sorted([n for n in cands if n not in stable], key=key)
    ranked = stable + preview
    out = ranked[:4] + [f for f in GEMINI_FALLBACK if f not in ranked[:4]]
    return out[:5]


def rank_groq_models(ids):
    skip = ("whisper", "tts", "guard", "embedding", "vision", "orpheus", "playai", "compound", "distil")
    chat = [i for i in ids if not any(x in i.lower() for x in skip)]
    pri = sorted([i for i in chat if "gpt-oss" in i.lower()], key=lambda i: 0 if "120b" in i else 1)
    rest = [i for i in chat if i not in pri and any(k in i.lower() for k in ("llama", "qwen", "kimi", "deepseek"))]
    out = pri + rest
    out += [f for f in GROQ_FALLBACK if f not in out]
    return out[:3]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_gemini():
    # يرفع استثناء عند الفشل ← لا يُخزَّن الفشل
    client = genai.Client(api_key=GEMINI_API_KEY)  # إبقاء المرجع أثناء التكرار، وإلا يُغلق العميل (RuntimeError)
    return [m.name.replace("models/", "") for m in client.models.list()]


@st.cache_data(ttl=1800, show_spinner=False)
def _list_groq():
    client = Groq(api_key=GROQ_API_KEY)
    return [m.id for m in client.models.list().data]


def get_gemini_models():
    """استدعِها من الخيط الرئيسي فقط (قبل تشغيل الخيوط)."""
    try:
        return rank_gemini_models(_list_gemini())
    except Exception:
        return list(GEMINI_FALLBACK)


def get_groq_models():
    try:
        return rank_groq_models(_list_groq())
    except Exception:
        return list(GROQ_FALLBACK)


def _gemini_client():
    try:
        return genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=30000))
    except Exception:
        return genai.Client(api_key=GEMINI_API_KEY)


def _groq_client():
    try:
        return Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=1)
    except TypeError:
        return Groq(api_key=GROQ_API_KEY)


def call_groq(system, user, temperature=0.2, models=None, budget=45):
    """يرجع (النص، رسالة الخطأ، اسم النموذج). مهلة كلية = budget ثانية."""
    models = models or GROQ_FALLBACK
    t0, last_err = time.time(), ""
    for model_name in models:
        if time.time() - t0 > budget:
            last_err = last_err or "انتهت المهلة"
            break
        try:
            c = _groq_client().chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
            )
            return c.choices[0].message.content or "", "", model_name
        except Exception as e:
            low = str(e).lower()
            last_err = f"{model_name}: {str(e)[:140]}"
            if "401" in low or "invalid_api_key" in low or "invalid api key" in low:
                return "", "Groq: مفتاح API غير صالح — تحقق من Streamlit Secrets.", ""
    return "", f"Groq: {last_err}", ""


def _grounding_sources(resp):
    out, seen = [], set()
    try:
        for ch in (resp.candidates[0].grounding_metadata.grounding_chunks or []):
            w = ch.web
            if w and w.uri and w.uri not in seen:
                seen.add(w.uri)
                out.append({"title": w.title or w.uri, "uri": w.uri})
    except Exception:
        pass
    return out[:6]


def call_gemini(prompt, search=False, models=None, budget=45):
    """يرجع (النص، المصادر، رسالة الخطأ، اسم النموذج). لا إعادة محاولة عند نفاد الحصة (429)."""
    models = models or GEMINI_FALLBACK
    t0, err = time.time(), ""
    try:
        client = _gemini_client()
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]) if search else None
        for model_name in models:
            if time.time() - t0 > budget:
                err = err or "انتهت المهلة"
                break
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
                    return (resp.text or ""), (_grounding_sources(resp) if search else []), "", model_name
                except Exception as e:
                    low = str(e).lower()
                    err = f"{model_name}: {str(e)[:140]}"
                    if any(k in low for k in ("api key not valid", "api_key_invalid", "unauthenticated")):
                        return "", [], "Gemini: مفتاح API غير صالح — تحقق من Streamlit Secrets.", ""
                    if attempt == 0 and any(k in low for k in ("503", "unavailable", "overloaded")):
                        time.sleep(2)
                        continue
                    break  # 429/404/403/غيرها → النموذج التالي فوراً
    except Exception as e:
        err = f"Gemini: {str(e)[:140]}"
    if search and any(k in err.lower() for k in ("grounding", "google_search", "permission", "403")):
        err += " — قد لا يدعم حسابك/نموذجك بحث جوجل الحي."
    if "429" in err or "resource_exhausted" in err.lower():
        err += " — نفدت الحصة المجانية لهذا النموذج."
    return "", [], err, ""


# ---------------------------------------------------------------------
# سياق السوق الغني الذي يستلمه الوكلاء
# ---------------------------------------------------------------------
def swing_levels(df, window=5, n=3):
    d = df.tail(150)
    hi, lo = d["High"], d["Low"]
    k = 2 * window + 1
    highs = hi[hi == hi.rolling(k, center=True).max()].tolist()
    lows = lo[lo == lo.rolling(k, center=True).min()].tolist()
    uniq = lambda xs: list(dict.fromkeys(xs))[-n:]
    return uniq(highs), uniq(lows)


def build_context(t, df, tf, bias, regime, adx, used, signal, reason, checks):
    latest = df.iloc[-1]
    price, atr = float(latest["Close"]), float(latest["ATR14"])
    dec = 2 if price >= 10 else 5

    def r(x):
        return round(float(x), dec)

    def chg(n):
        return (price / float(df["Close"].iloc[-1 - n]) - 1) * 100 if len(df) > n else float("nan")

    highs, lows = swing_levels(df)
    candles = df.tail(8)[["Open", "High", "Low", "Close"]].round(dec).to_string()
    candle_kind = "daily" if tf == TF_D1 else "4-hour"
    return "\n".join([
        f"Asset (Yahoo Finance symbol): {t}",
        f"Timeframe: {tf} (each candle = {candle_kind})",
        f"Daily trend filter (daily close vs daily EMA200): {bias or 'not used'}",
        f"Market regime: {regime}, ADX14 = {adx:.1f}",
        f"Last close: {r(price)} | ATR14: {r(atr)} ({atr / price * 100:.2f}% of price)",
        f"EMA20 {r(latest['EMA20'])} | EMA50 {r(latest['EMA50'])} | EMA200 {r(latest['EMA200'])} "
        f"(price is {(price / float(latest['EMA200']) - 1) * 100:+.2f}% from EMA200)",
        f"RSI14: {float(latest['RSI14']):.1f}",
        f"20-bar high / low: {r(latest['Donchian_Upper'])} / {r(latest['Donchian_Lower'])}",
        f"Recent swing highs (resistance): {[r(x) for x in highs]}",
        f"Recent swing lows (support): {[r(x) for x in lows]}",
        f"Change: 5 bars {chg(5):+.2f}% | 20 bars {chg(20):+.2f}%",
        "Last 8 candles:",
        candles,
        f"Strategy used: {used}",
        f"Technical signal: {signal} — {reason}",
        "All strategies checked: " + "; ".join(f"{n.split(' (')[0]}={sg}" for n, sg, _ in checks),
    ])


# ---------------------------------------------------------------------
# الأخبار: عناوين حيّة من Google News (RSS) — بدون مفتاح ولا حصة بحث
# ---------------------------------------------------------------------
NEWS_NAMES = {"GC=F": "gold price", "SI=F": "silver price", "CL=F": "crude oil price",
              "NG=F": "natural gas price", "BTC-USD": "bitcoin", "ETH-USD": "ethereum",
              "^GSPC": "S&P 500", "^IXIC": "Nasdaq"}


def news_queries(t):
    """يرجع (استعلام الأصل، استعلام الأحداث الكبرى)."""
    u = t.strip().upper()
    if u in NEWS_NAMES:
        return NEWS_NAMES[u], "Fed OR FOMC OR CPI OR payrolls OR OPEC OR ECB OR tariffs"
    if u.endswith("=X") and len(u) == 8:
        return f"{u[:3]}/{u[3:6]} forex", "Fed OR ECB OR CPI OR payrolls OR central bank"
    return f"{u} stock", f"{u} earnings OR guidance OR downgrade OR upgrade"


def fetch_google_news(query, days=3, limit=8):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        root = ET.fromstring(r.read())
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        items.append({"title": title, "uri": (it.findtext("link") or "").strip(),
                      "date": (it.findtext("pubDate") or "").strip()[:16],
                      "source": (it.findtext("source") or "").strip()})
        if len(items) >= limit:
            break
    return items


def fetch_news(t):
    """يرجع (قائمة العناوين، رسالة خطأ إن فشل الكل)."""
    q1, q2 = news_queries(t)
    items, errs, seen = [], [], set()
    for q, days, lim in ((q1, 3, 8), (q2, 2, 5)):
        try:
            for it in fetch_google_news(q, days, lim):
                if it["title"] not in seen:
                    seen.add(it["title"])
                    items.append(it)
        except Exception as e:
            errs.append(str(e)[:100])
    return items, ("; ".join(errs) if not items else "")


NEWS_SYSTEM = (
    "You are a markets news analyst. Use ONLY the headlines provided; never invent facts or events. "
    "Reply with ONE JSON object only — no markdown. Text values must be in Arabic."
)


def news_agent(t, signal, g_models, q_models):
    """يرجع (النص، العناوين كمصادر، رسالة الخطأ، اسم النموذج) — نفس شكل call_gemini."""
    items, ferr = fetch_news(t)
    if not items:
        return "", [], f"لا عناوين أخبار متاحة ({ferr or 'نتيجة فارغة'})", ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = "\n".join(f"{i}. [{x['date']}] {x['title']}" for i, x in enumerate(items, 1))
    prompt = f"""Today is {today}. Asset: {t}. Technical signal under review: {signal}.
Latest headlines from Google News (newest first). Base your analysis ONLY on them:
{lines}

Reply with ONE JSON object only (no markdown). Text values in Arabic:
{{"sentiment":"bullish"|"bearish"|"neutral","event_risk":"high"|"medium"|"low"|"unknown",
 "impact_on_signal":"supports"|"conflicts"|"neutral","summary":"<2-3 sentences>",
 "key_events":["<max 4 short items taken from the headlines>"]}}
"event_risk" = "high" only if the headlines show a major scheduled or ongoing event/shock likely within ~24h
(central bank decision, key data release, geopolitical shock). If the headlines are insufficient, use
sentiment "neutral" and event_risk "unknown"."""
    raw, _, err, model = call_gemini(prompt, False, g_models, budget=35)
    if not (raw and extract_json(raw)):
        raw2, err2, model2 = call_groq(NEWS_SYSTEM, prompt, 0.2, q_models, budget=35)
        if raw2 and extract_json(raw2):
            raw, err, model = raw2, "", model2
        else:
            err = err or err2
    return raw, [{"title": x["title"], "uri": x["uri"]} for x in items[:8]], err, model


# ---------------------------------------------------------------------
# فريق الوكلاء: فني + أخبار + مخاطر
# ---------------------------------------------------------------------
TECH_SYSTEM = (
    "You are a senior technical analyst. Judge ONLY from the data provided (price action, levels, "
    "indicators, multi-timeframe bias); never invent data. Be sceptical: if the evidence is mixed, say WAIT. "
    "Reply with ONE JSON object only — no markdown, no text outside the JSON. Text values must be in Arabic."
)

RISK_SYSTEM = (
    "You are a strict risk manager acting as devil's advocate on a trading desk. Your job is to look for reasons "
    "NOT to take the trade, but be fair: approve when the risk is acceptable. Judge ONLY from the data provided. "
    "Reply with ONE JSON object only — no markdown, no text outside the JSON. Text values must be in Arabic."
)


def run_agents(ctx, t, signal, price, atr):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    has_signal = signal in ("BUY", "SELL")

    tech_user = ctx + """

Return this JSON:
{"verdict":"BUY"|"SELL"|"WAIT","confidence":0-100,"supports_signal":true|false,
 "market_context":"<2 sentences>","key_levels":{"support":[numbers],"resistance":[numbers]},
 "reasons":["<max 3 short reasons>"],"invalidation":"<price condition that cancels the idea>"}
"supports_signal" is true only if you agree with the direction of the technical signal right now.
If the technical signal is "No Signal", set it to false."""

    g_models, q_models = get_gemini_models(), get_groq_models()  # في الخيط الرئيسي
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_tech = ex.submit(call_groq, TECH_SYSTEM, tech_user, 0.2, q_models)
        f_news = ex.submit(news_agent, t, signal, g_models, q_models)
        try:
            tech_raw, tech_err, tech_model = f_tech.result(timeout=100)
        except Exception as e:
            tech_raw, tech_err, tech_model = "", f"انتهت المهلة/خطأ: {str(e)[:80]}", ""
        try:
            news_raw, news_sources, news_err, news_model = f_news.result(timeout=100)
        except Exception as e:
            news_raw, news_sources, news_err, news_model = "", [], f"انتهت المهلة/خطأ: {str(e)[:80]}", ""

    tech, news = extract_json(tech_raw), extract_json(news_raw)
    tech_err = tech_err or ("" if tech else "تعذّر قراءة رد الوكيل (ليس JSON صالحاً)")
    news_err = news_err or ("" if news else "تعذّر قراءة رد الوكيل (ليس JSON صالحاً)")

    risk, risk_raw, risk_err, risk_model, skipped = None, "", "", "", not has_signal
    if has_signal:
        sl, tp1, tp2 = trade_levels(signal, price, atr)
        risk_user = f"""{ctx}

Proposed trade: {signal} at {P(price)} | SL {P(sl)} ({SL_ATR} x ATR) | TP1 {P(tp1)} | TP2 {P(tp2)} (2R).
Technical analyst report: {json.dumps(tech, ensure_ascii=False) if tech else "unavailable"}
News analyst report: {json.dumps(news, ensure_ascii=False) if news else "unavailable"}

Return this JSON:
{{"decision":"APPROVE"|"REDUCE"|"VETO","risk_score":1-10,"concerns":["<max 3 short items>"],
 "comment":"<1-2 sentences about stop/target sizing versus volatility and timing>"}}
VETO if the technical analyst does not support the signal, a high-risk event is imminent, or the stop is
unreasonably tight/wide versus volatility. REDUCE if there are concerns but the trade is still defensible.
Otherwise APPROVE."""
        risk_raw, risk_err, risk_model = call_groq(RISK_SYSTEM, risk_user, 0.2, q_models)
        risk = extract_json(risk_raw)
        risk_err = risk_err or ("" if risk else "تعذّر قراءة رد الوكيل (ليس JSON صالحاً)")

    return {"tech": tech, "tech_raw": tech_raw, "tech_err": tech_err,
            "news": news, "news_raw": news_raw, "news_err": news_err, "news_sources": news_sources,
            "risk": risk, "risk_raw": risk_raw, "risk_err": risk_err, "risk_skipped": skipped,
            "tech_model": tech_model, "news_model": news_model, "risk_model": risk_model}


def arbitrate(signal, tech, news, risk):
    """قواعد ثابتة تجمع الوكلاء الثلاثة في قرار واحد (وليس تصويتاً نصياً)."""
    if signal not in ("BUY", "SELL"):
        return {"action": "انتظر — لا توجد إشارة مناسبة الآن", "level": "info", "mult": 0.0, "notes": []}

    notes = []
    tech_ok = None if tech is None else (
        truthy(tech.get("supports_signal")) and str(tech.get("verdict", "")).upper() == signal)
    risk_dec = None if risk is None else str(risk.get("decision", "")).upper()
    news_conflict = news is not None and str(news.get("impact_on_signal", "")).lower() == "conflicts"
    event_high = news is not None and str(news.get("event_risk", "")).lower() == "high"

    if tech is None and risk is None:
        return {"action": "لا تدخل — الوكلاء الفني والمخاطر غير متاحين", "level": "error", "mult": 0.0,
                "notes": ["تحقق من مفاتيح API وأسماء النماذج."]}
    if risk_dec == "VETO":
        concerns = [str(c) for c in as_list(risk.get("concerns"))]
        return {"action": "لا تدخل — مدير المخاطر اعترض", "level": "error", "mult": 0.0, "notes": concerns}
    if tech_ok is False:
        return {"action": "لا تدخل — المحلل الفني لا يؤيد الإشارة", "level": "error", "mult": 0.0,
                "notes": [str(tech.get("market_context", ""))] if tech.get("market_context") else []}

    if tech_ok is None:
        notes.append("الوكيل الفني غير متاح")
    if risk_dec is None:
        notes.append("وكيل المخاطر غير متاح")
    elif risk_dec == "REDUCE":
        notes += [f"مدير المخاطر: {c}" for c in as_list(risk.get("concerns"))] or ["مدير المخاطر يطلب تخفيض الحجم"]
    if news_conflict:
        notes.append("الأخبار تعاكس اتجاه الإشارة")
    if event_high:
        notes.append("حدث اقتصادي كبير قريب (مخاطرة مرتفعة)")
    if news is None:
        notes.append("الأخبار غير متاحة — راجع الأجندة الاقتصادية يدوياً")

    reduce = (tech_ok is None or risk_dec in (None, "REDUCE") or news_conflict or event_high or news is None)
    if reduce:
        return {"action": "دخول بحذر — نصف الحجم", "level": "warning", "mult": 0.5, "notes": notes}
    label = ("ادخل الآن — الوكلاء الثلاثة متفقون" if news is not None
             else "ادخل الآن — الفني والمخاطر متفقان (الأخبار غير متاحة)")
    return {"action": label, "level": "success", "mult": 1.0, "notes": notes}


# ---------------------------------------------------------------------
# خطة الصفقة
# ---------------------------------------------------------------------
def build_plan(signal, price, atr, arb, balance, risk_pct, contract_size=1.0):
    plan = {"direction": None, "entry": price, "sl": None, "tp1": None, "tp2": None,
            "size": None, "lots": None, "risk_amount": 0.0,
            "action": arb["action"], "level": arb["level"]}
    if signal not in ("BUY", "SELL"):
        return plan
    plan["direction"] = signal
    plan["sl"], plan["tp1"], plan["tp2"] = trade_levels(signal, price, atr)
    sl_dist = SL_ATR * atr
    risk_amount = balance * risk_pct / 100 * arb["mult"]
    plan["risk_amount"] = risk_amount
    if sl_dist > 0 and risk_amount > 0:
        plan["size"] = risk_amount / sl_dist                      # بالوحدات
        plan["lots"] = math.floor(plan["size"] / contract_size * 100) / 100  # تقريب لأسفل (أكثر أماناً)
    return plan


# ---------------------------------------------------------------------
# الأدلة التاريخية + نقاط الجودة (0-100) + التنبيه
# ---------------------------------------------------------------------
def history_evidence(t, tf, df, bias_s, used, use_closed, cost=0.1, hold=30):
    """يختبر الاستراتيجية نفسها تاريخياً على الأصل نفسه وبنفس الفلاتر. يرجع إحصاءات أو None."""
    try:
        if tf == TF_D1:
            dfh, _, err = load_prepared(t, TF_D1, use_closed, "5y")
            bias_h = None
            if err or dfh is None:
                return None
        else:
            dfh, bias_h = df, bias_s
        if len(dfh) < 300:
            return None
        sigs = strategy_signals(dfh)
        if bias_h is not None:
            sigs = apply_bias_filter(sigs, bias_h)
        stats = backtest_stats(run_backtest(dfh, sigs[used], cost, hold))
        stats["bars"] = len(dfh) - 200
        return stats
    except Exception:
        return None


def compute_quality(signal, df, adx, used, bias, tech, news, risk, ev):
    """
    نقاط جودة من 100 (وليست احتمال ربح):
      الفلاتر 30 + الوكلاء 40 + الدليل التاريخي 30.
    يرجع (المجموع، قائمة [(البند، النقاط، الحد الأقصى، ملاحظة)]).
    """
    parts = []
    latest = df.iloc[-1]
    close, e50, e200 = float(latest["Close"]), float(latest["EMA50"]), float(latest["EMA200"])
    buy = signal == "BUY"

    # --- الفلاتر (30)
    side = (close > e200) if buy else (close < e200)
    slope = (e50 > e200) if buy else (e50 < e200)
    pts = 6 * side + 4 * slope
    parts.append(("توافق الاتجاه (EMA)", pts, 10, "السعر وEMA50 في اتجاه الصفقة" if pts == 10 else "الصفقة ضد الاتجاه العام جزئياً أو كلياً"))

    if used in (BREAKOUT, PULLBACK):
        pts = 10 if adx >= 25 else 5 if adx >= 20 else 0
    else:
        pts = 10 if adx < 20 else 5 if adx < 25 else 0
    parts.append(("ملاءمة الاستراتيجية لحالة السوق", pts, 10, f"ADX = {adx:.1f}"))

    if bias is None:
        parts.append(("اتجاه اليومي", 5, 10, "غير مستخدم في هذا الإطار (نقاط جزئية)"))
    else:
        ok = (bias == "UP" and buy) or (bias == "DOWN" and not buy)
        parts.append(("اتجاه اليومي", 10 if ok else 0, 10, "متوافق" if ok else "معاكس"))

    # --- الوكلاء (40)
    if tech is None:
        parts.append(("الوكيل الفني", 0, 18, "غير متاح"))
    else:
        supports = truthy(tech.get("supports_signal")) and str(tech.get("verdict", "")).upper() == signal
        try:
            conf = float(tech.get("confidence", 50))
        except (TypeError, ValueError):
            conf = 50.0
        pts = round(18 * min(max(conf, 0), 90) / 90, 1) if supports else 0
        parts.append(("الوكيل الفني", pts, 18, f"يؤيد (ثقة {conf:.0f}%)" if supports else "لا يؤيد"))

    if news is None:
        parts.append(("الأخبار", 3, 10, "غير متاحة (نقاط جزئية)"))
    else:
        impact = str(news.get("impact_on_signal", "")).lower()
        pts = {"supports": 10, "neutral": 6, "conflicts": 0}.get(impact, 4)
        ev_risk = str(news.get("event_risk", "")).lower()
        if ev_risk == "high":
            pts = 0
        elif ev_risk == "medium":
            pts = max(pts - 2, 0)
        parts.append(("الأخبار", pts, 10, f"الأثر: {impact or '—'} | مخاطر الأحداث: {ev_risk or '—'}"))

    dec = None if risk is None else str(risk.get("decision", "")).upper()
    pts = {"APPROVE": 12, "REDUCE": 6}.get(dec, 0)
    parts.append(("وكيل المخاطر", pts, 12, {"APPROVE": "موافقة", "REDUCE": "تخفيض الحجم"}.get(dec, "اعتراض/غير متاح")))

    # --- الدليل التاريخي (30)
    if not ev or ev["trades"] == 0:
        parts.append(("الأداء التاريخي لهذه الاستراتيجية", 0, 30, "لا بيانات كافية"))
    else:
        f = min(ev["trades"], 30) / 30
        exp_pts = 20 * min(max(ev["avg_r"], 0) / 0.3, 1)
        pf = ev["pf"] if np.isfinite(ev["pf"]) else 3.0
        pf_pts = 10 * min(max((pf - 1) / 0.5, 0), 1)
        pts = round((exp_pts + pf_pts) * f, 1)
        parts.append(("الأداء التاريخي لهذه الاستراتيجية", pts, 30,
                      f"{ev['trades']} صفقة | متوسط R = {ev['avg_r']:.2f} | PF = {ev['pf']:.2f}"
                      + (" | عينة صغيرة (<30)" if ev["trades"] < 30 else "")))

    total = int(round(sum(p[1] for p in parts)))
    return total, parts


def final_decision(signal, arb, score, threshold):
    """يدمج قرار الوكلاء مع حد النقاط. enter=True فقط إذا نجح الاثنان."""
    if signal not in ("BUY", "SELL") or arb["mult"] <= 0:
        return {**arb, "enter": False}
    if score >= threshold:
        return {**arb, "enter": True, "action": f"🔔 {arb['action']} — الجودة {score}/100"}
    return {"action": f"لا تدخل الآن — الجودة {score}/100 أقل من الحد {threshold}", "level": "warning",
            "mult": 0.0, "enter": False,
            "notes": arb["notes"] + ["الوكلاء لا يعترضون، لكن نقاط الجودة لم تبلغ الحد المطلوب."]}


def format_alert(t, tf, used, plan, score):
    d = "شراء 🟢" if plan["direction"] == "BUY" else "بيع 🔴"
    lots = f"{plan['lots']:.2f} لوت" if plan.get("lots") else "أقل من 0.01 لوت"
    return (f"🔔 إشارة دخول — {t}\n"
            f"الاتجاه: {d}\nالجودة: {score}/100\n"
            f"الدخول: {P(plan['entry'])}\nالوقف: {P(plan['sl'])}\n"
            f"الهدف 1: {P(plan['tp1'])}\nالهدف 2: {P(plan['tp2'])}\n"
            f"الحجم: {lots} (مخاطرة {plan['risk_amount']:,.2f})\n"
            f"{used.split(' (')[0]} | {tf.split(' (')[0]}\n"
            f"⚠️ لأغراض تعليمية، وليست نصيحة مالية.")


def send_telegram(text):
    """يرجع None إذا لم يُضبط تيليجرام، وإلا 'ok' أو رسالة خطأ."""
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return None
    if not token or not chat:
        return None
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        with urllib.request.urlopen(req, timeout=10) as r:
            return "ok" if r.status == 200 else f"فشل الإرسال ({r.status})"
    except Exception as e:
        return f"خطأ تيليجرام: {str(e).replace(str(token), '***')[:120]}"


# ---------------------------------------------------------------------
# المحاكي
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# محرك الـ Backtest
# (منطق الإشارات هنا نسخة متجهة من generate_signal/select_strategy — عدّلهما معاً)
# ---------------------------------------------------------------------
def strategy_signals(df):
    """إشارات كل استراتيجية لكل شمعة: +1 شراء، -1 بيع، 0 لا شيء."""
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]

    def to_sig(buy, sell):
        return pd.Series(np.select([buy, sell], [1, -1], default=0), index=df.index)

    return {
        BREAKOUT: to_sig(
            (c > df["Donchian_Upper"].shift(1)) & (c > df["EMA200"]),
            (c < df["Donchian_Lower"].shift(1)) & (c < df["EMA200"])),
        PULLBACK: to_sig(
            (c > df["EMA200"]) & (l <= df["EMA50"]) & (c > o),
            (c < df["EMA200"]) & (h >= df["EMA50"]) & (c < o)),
        MEAN_REV: to_sig(
            (c < df["BB_Lower"]) & (df["RSI14"] < 35),
            (c > df["BB_Upper"]) & (df["RSI14"] > 65)),
    }


def auto_signals(df, sigs):
    """نفس منطق الاختيار التلقائي حسب ADX."""
    adx = df["ADX14"]
    b, p, m = sigs[BREAKOUT], sigs[PULLBACK], sigs[MEAN_REV]
    trending_sig = b.where(b != 0, p)                       # ADX >= 25
    transition_sig = p.where(p != 0, b).where(lambda s: s != 0, m)  # 20 <= ADX < 25
    out = np.where(adx >= 25, trending_sig, np.where(adx < 20, m, transition_sig))
    return pd.Series(out, index=df.index).astype(int)


def run_backtest(df, signals, cost_pct=0.05, max_hold=30, start=200):
    """
    الإشارة تُحسب عند إغلاق الشمعة i، والدخول عند افتتاح الشمعة i+1 (بدون نظر للمستقبل).
    SL = 1.5 ATR ، TP = 3 ATR. إذا لمست الشمعة الاثنين معاً يُفترض ضرب الـ SL أولاً (متحفظ).
    إذا لم يتحقق شيء خلال max_hold شمعة يُغلق عند الإغلاق.
    """
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    atr, sig, n = df["ATR14"].values, signals.values, len(df)
    trades, i = [], start

    while i < n - 1:
        s = int(sig[i])
        if s == 0 or np.isnan(atr[i]):
            i += 1
            continue

        entry = o[i + 1]
        sd = SL_ATR * atr[i]
        sl = entry - s * sd
        tp = entry + s * TP2_ATR * atr[i]
        exit_price, reason = None, "Time"

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if s == 1:
                hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
            else:
                hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
            if hit_sl:
                # فجوة افتتاح أسوأ من الوقف → الخروج عند الافتتاح
                exit_price = min(sl, o[j]) if s == 1 else max(sl, o[j])
                reason = "SL"
                break
            if hit_tp:
                exit_price, reason = tp, "TP"
                break
        else:
            exit_price = c[j]

        pnl = s * (exit_price - entry) / entry * 100 - cost_pct
        r = pnl / (sd / entry * 100)
        trades.append({
            "entry_date": df.index[i + 1], "exit_date": df.index[j],
            "direction": "BUY" if s == 1 else "SELL",
            "entry": entry, "exit": exit_price, "reason": reason,
            "pnl": pnl, "r": r, "bars": j - i,
        })
        i = j  # الإشارة التالية تُفحص من شمعة الخروج

    return pd.DataFrame(trades)


def backtest_stats(tr):
    if tr.empty:
        return {"trades": 0, "win_rate": 0.0, "total": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd": 0.0, "avg_bars": 0.0}
    gross_win = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(tr.loc[tr["pnl"] <= 0, "pnl"].sum())
    cum = tr["pnl"].cumsum()
    return {
        "trades": len(tr),
        "win_rate": (tr["pnl"] > 0).mean() * 100,
        "total": tr["pnl"].sum(),
        "avg_r": tr["r"].mean(),
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "max_dd": (cum - cum.cummax()).min(),
        "avg_bars": tr["bars"].mean(),
    }


def get_live_price(ticker, fallback):
    try:
        h = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return fallback


def calc_pnl(pos, price):
    if pos["direction"] == "BUY":
        return (price - pos["entry_price"]) / pos["entry_price"] * 100
    return (pos["entry_price"] - price) / pos["entry_price"] * 100


def meta_from(a, source):
    """سياق الصفقة وقت الدخول (للتعلم لاحقاً)."""
    if not a:
        return {"source": source}
    return {"source": source, "used": a.get("used"), "tf": a.get("tf"), "score": a.get("score"),
            "adx": round(float(a["adx"]), 1) if a.get("adx") is not None else None,
            "regime": a.get("regime"), "signal": a.get("signal"),
            "decision": (a.get("arb") or {}).get("action", "")[:70]}


def open_position(direction, price, ticker, atr, meta=None):
    sign = 1 if direction == "BUY" else -1
    st.session_state.current_position = {
        "direction": direction, "entry_price": price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "sl": price - sign * SL_ATR * atr,
        "tp": price + sign * TP2_ATR * atr,
        "atr": atr,
        "meta": meta or {},
    }
    persist()


def close_trade(price, exit_reason):
    if st.session_state.current_position is None:
        return
    pos = dict(st.session_state.current_position)
    meta = pos.pop("meta", None) or {}
    pnl = calc_pnl(pos, price)
    risk_pct = abs(pos["entry_price"] - pos["sl"]) / pos["entry_price"] * 100
    st.session_state.trades.append({
        **pos, **meta, "exit_price": price,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exit_reason": exit_reason, "pnl_percent": pnl,
        "r_multiple": (pnl / risk_pct) if risk_pct else None,
    })
    st.session_state.current_position = None
    why = {"SL": "ضُرب وقف الخسارة", "TP": "تحقق الهدف", "Manual": "أُغلقت يدوياً"}[exit_reason]
    kind = "success" if pnl > 0 else "error"
    st.session_state.notice = (kind, f"{'✅' if pnl > 0 else '❌'} {why} عند {P(price)} | النتيجة: {pnl:+.2f}%")
    persist()


def check_exit(price):
    pos = st.session_state.current_position
    if pos is None:
        return
    if pos["direction"] == "BUY":
        hit_sl, hit_tp = price <= pos["sl"], price >= pos["tp"]
    else:
        hit_sl, hit_tp = price >= pos["sl"], price <= pos["tp"]
    if hit_sl:
        close_trade(price, "SL")
    elif hit_tp:
        close_trade(price, "TP")


# ---------------------------------------------------------------------
# التشخيص الشامل
# ---------------------------------------------------------------------
def run_diagnostics(t):
    out = []

    def step(title, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
            out.append((ok, title, f"{detail}  ({time.time() - t0:.1f}s)"))
        except Exception as e:
            out.append((False, title, f"{type(e).__name__}: {str(e)[:300]}"))

    def secrets():
        have = {k: bool(st.secrets.get(k)) for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
        return have["GEMINI_API_KEY"] and have["GROQ_API_KEY"], str(have)

    def yd():
        df = load_data(t, "1y")
        return True, f"{len(df)} شمعة يومية، آخر تاريخ: {df.index[-1]}"

    def yh():
        df = load_data_h4(t)
        return True, f"{len(df)} شمعة 4H، آخر تاريخ: {df.index[-1]}"

    def ind():
        df = calculate_indicators(load_data(t, "1y"))
        l = df.iloc[-1]
        return (not l[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any(),
                f"ATR={l['ATR14']:.4f} ADX={l['ADX14']:.1f} RSI={l['RSI14']:.1f}")

    def gl():
        raw = _list_gemini()
        return True, f"{len(raw)} نموذجاً في حسابك | سيُجرَّب: {get_gemini_models()}"

    def gc():
        txt, _, err, model = call_gemini("Reply with the single word OK", False, get_gemini_models(), budget=25)
        return (not err and bool(txt)), (f"{model}: {txt.strip()[:40]}" if not err else err)

    def rss():
        items, ferr = fetch_news(t)
        return bool(items), (f"{len(items)} عنواناً | أولها: {items[0]['title'][:80]}" if items else f"فشل: {ferr}")

    def nag():
        raw, src_, err, model = news_agent(t, "BUY", get_gemini_models(), get_groq_models())
        ok = bool(raw and extract_json(raw))
        return ok, (f"{model}: فُهم الرد، العناوين = {len(src_)}" if ok else (err or "الرد ليس JSON"))

    def ql():
        raw = _list_groq()
        return True, f"{len(raw)} نموذجاً | سيُجرَّب: {get_groq_models()}"

    def qc():
        txt, err, model = call_groq("Reply briefly.", "Reply with the single word OK", 0.0, get_groq_models(), budget=25)
        return (not err and bool(txt.strip())), (f"{model}: {txt.strip()[:40]}" if not err else err)

    def jr():
        obj, err = journal_load()
        if err:
            return False, f"({storage_mode()}) {err}"
        e2 = journal_save(obj["trades"], obj["position"])  # كتابة اختبارية بنفس المحتوى
        note = "" if storage_mode() == "gist" else " — مؤقت (أضف GITHUB_TOKEN وGIST_ID للدوام)"
        return (not e2), (f"{storage_mode()}: {len(obj['trades'])} صفقة | الكتابة " + ("نجحت" if not e2 else f"فشلت: {e2}") + note)

    step("1) المفاتيح في Secrets", secrets)
    step(f"2) Yahoo يومي — {t}", yd)
    step(f"3) Yahoo ساعة (لإطار 4H) — {t}", yh)
    step("4) حساب المؤشرات", ind)
    step("5) قائمة نماذج Gemini", gl)
    step("6) نداء Gemini عادي", gc)
    step("7) أخبار Google News (RSS)", rss)
    step("8) قائمة نماذج Groq", ql)
    step("9) نداء Groq", qc)
    step("10) وكيل الأخبار كاملاً (عناوين + تحليل)", nag)
    step("11) التخزين الدائم للسجل", jr)
    return out


# ---------------------------------------------------------------------
# الشريط الجانبي
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("الإعدادات")
    ticker = st.text_input("رمز الأصل", value="GC=F")
    strategy = st.selectbox("الاستراتيجية", [AUTO, BREAKOUT, PULLBACK, MEAN_REV])
    timeframe = st.selectbox("الإطار الزمني", [TF_DUAL, TF_D1, TF_H4])
    st.caption("المزدوج: لا شراء إلا إذا كان اليومي فوق EMA200، ولا بيع إلا إذا كان تحتها. بيانات 4H متاحة لحوالي سنتين.")
    use_closed = st.checkbox("استخدم آخر شمعة مكتملة فقط", value=True)
    balance = st.number_input("رأس المال (للحجم المقترح)", min_value=100.0, value=10000.0, step=100.0)
    risk_pct = st.slider("المخاطرة لكل صفقة %", 0.25, 3.0, 1.0, 0.25)
    alert_threshold = st.slider("حد التنبيه (نقاط الجودة)", 60, 95, 85, 5)
    contract_size = st.number_input("وحدات لكل 1 لوت", min_value=0.0001, value=guess_contract_size(ticker),
                                    key=f"cs_{ticker.strip().upper()}")
    st.caption("تحقق من حجم العقد عند وسيطك. يُفترض أن عملة التسعير = عملة حسابك.")
    analyze_btn = st.button("ابدأ التحليل", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### رموز مقترحة")
    st.markdown("- GC=F — الذهب\n- BTC-USD — البيتكوين\n- EURUSD=X — يورو/دولار\n"
                "- AAPL — أبل\n- TSLA — تسلا\n- CL=F — النفط")
    st.caption("تنبيه تيليجرام (اختياري): أضف TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID في Secrets.")
    if st.button("📨 اختبر تيليجرام", use_container_width=True):
        res = send_telegram("✅ اختبار: التطبيق متصل بتيليجرام.")
        if res is None:
            st.warning("لم تُضبط TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID في Secrets.")
        elif res == "ok":
            st.success("وصلت رسالة الاختبار ✅")
        else:
            st.error(res)
    st.caption("لأغراض تعليمية فقط")

    with st.expander("🩺 تشخيص شامل"):
        st.caption("يختبر كل مكوّن على حدة ويعرض الخطأ الحرفي. شغّله أولاً إذا لم يعمل التحليل.")
        if st.button("شغّل التشخيص", use_container_width=True):
            with st.spinner("جاري الفحص..."):
                st.session_state["diag"] = run_diagnostics(ticker.strip() or "GC=F")
        for ok_, title_, detail_ in st.session_state.get("diag", []):
            (st.success if ok_ else st.error)(f"{title_}")
            st.code(str(detail_)[:400])


# ---------------------------------------------------------------------
# تشغيل التحليل
# ---------------------------------------------------------------------
if analyze_btn:
    pos = st.session_state.current_position
    t = ticker.strip()
    if not t:
        st.error("الرجاء إدخال رمز الأصل")
    elif pos is not None and pos["ticker"] != t:
        st.warning(f"⚠️ لديك صفقة مفتوحة على {pos['ticker']}. أغلقها أولاً أو حلل نفس الأصل.")
    else:
        with st.spinner(f"جاري تحليل {t}..."):
            try:
                df, bias_s, err = load_prepared(t, timeframe, use_closed)
                if err:
                    st.error(err)
                else:
                    bias = bias_s.iloc[-1] if bias_s is not None else None
                    latest = df.iloc[-1]
                    if len(df) < 60 or latest[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any():
                        st.error("البيانات غير كافية لحساب المؤشرات.")
                    else:
                        regime, adx, checks, chosen = select_strategy(df, strategy, bias)
                        if chosen:
                            used, signal, reason = chosen
                        else:
                            used, signal, reason = checks[0][0], "No Signal", "لا توجد استراتيجية مناسبة تعطي إشارة الآن"

                        price, atr = float(latest["Close"]), float(latest["ATR14"])
                        ctx = build_context(t, df, timeframe, bias, regime, adx, used, signal, reason, checks)
                        agents = run_agents(ctx, t, signal, price, atr)
                        arb = arbitrate(signal, agents["tech"], agents["news"], agents["risk"])

                        evidence, score, parts = None, 0, []
                        if signal in ("BUY", "SELL"):
                            evidence = history_evidence(t, timeframe, df, bias_s, used, use_closed)
                            score, parts = compute_quality(signal, df, adx, used, bias, agents["tech"],
                                                           agents["news"], agents["risk"], evidence)
                        dec = final_decision(signal, arb, score, alert_threshold)
                        plan_obj = build_plan(signal, price, atr, dec, balance, risk_pct, contract_size)
                        tg = None
                        if dec.get("enter"):
                            st.toast("🔔 إشارة دخول جاهزة!", icon="🔔")
                            tg = send_telegram(format_alert(t, timeframe, used, plan_obj, score))

                        st.session_state.analysis = {
                            "ticker": t, "mode": strategy, "used": used, "df": df, "tf": timeframe, "bias": bias,
                            "regime": regime, "adx": adx, "checks": checks,
                            "signal": signal, "reason": reason,
                            "price": price, "atr": atr,
                            "rsi": float(latest["RSI14"]), "ema200": float(latest["EMA200"]),
                            "agents": agents, "arb": dec, "score": score, "parts": parts,
                            "threshold": alert_threshold, "evidence": evidence, "tg": tg,
                            "plan": plan_obj,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.session_state.live_price = None
            except Exception as e:
                st.error(f"خطأ: {type(e).__name__}: {e}")
                st.code(traceback.format_exc()[-1800:])


# ---------------------------------------------------------------------
# عرض النتائج
# ---------------------------------------------------------------------
a = st.session_state.analysis
if a is None:
    render_tv_chart(None)
if a is not None:
    plan = a["plan"]

    # ---- خطة الصفقة (أهم شيء في الأعلى) ----
    st.markdown("## 📋 خطة الصفقة")
    getattr(st, plan["level"])(f"**{plan['action']}**")
    if plan["direction"]:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("الاتجاه", "شراء 🟢" if plan["direction"] == "BUY" else "بيع 🔴")
        p2.metric("الدخول", f"{P(plan['entry'])}")
        p3.metric("وقف الخسارة", f"{P(plan['sl'])}")
        p4.metric("الهدف 1", f"{P(plan['tp1'])}")
        p5.metric("الهدف 2", f"{P(plan['tp2'])}")
        if plan.get("size"):
            p6.metric("الحجم", f"{plan['lots']:.2f} لوت" if plan.get("lots") else "أقل من 0.01 لوت")
            st.caption(f"≈ {plan['size']:,.2f} وحدة | المخاطرة {plan['risk_amount']:,.2f} من رأس المال")
        else:
            p6.metric("الحجم", "—")
        st.caption("الهدف 1 = 1R، الهدف 2 = 2R. عند الهدف 1 يُنصح بإغلاق نصف الصفقة ونقل الوقف لنقطة الدخول.")

    if a.get("score") is not None and a["signal"] in ("BUY", "SELL"):
        st.markdown("## 🎯 نقاط الجودة")
        q1, q2 = st.columns([1, 2])
        q1.metric("الجودة", f"{a['score']}/100")
        q2.progress(min(max(a["score"], 0), 100) / 100)
        q2.caption(f"الحد المطلوب للتنبيه: {a['threshold']}")
        with st.expander("تفاصيل النقاط"):
            for label, pts_, mx, note in a["parts"]:
                st.markdown(f"- **{label}:** {pts_:g}/{mx} — {note}")
            st.caption("هذه نقاط جودة تجمع الفلاتر والوكلاء والأداء التاريخي، وليست احتمال ربح. "
                       "لا تُفسَّر كنسبة نجاح إلا بعد معايرتها على نتائج صفقاتك الفعلية.")

    render_tv_chart(a)

    st.markdown(f"**الرمز:** {a['ticker']} | **الاستراتيجية المستخدمة:** {a['used']} | **الإطار:** {a.get('tf', TF_D1)} | **وقت التحليل:** {a['time']}")

    # ---- حالة السوق ----
    st.markdown("## 🧭 حالة السوق")
    r1, r2, r3 = st.columns(3)
    r1.metric("الحالة", a["regime"])
    r2.metric("ADX14 (قوة الاتجاه)", f"{a['adx']:.1f}")
    r3.metric("اتجاه اليومي", {"UP": "صاعد 🟢", "DOWN": "هابط 🔴", "NEUTRAL": "محايد ⚪"}.get(a.get("bias"), "غير مستخدم"))
    for name, sig, reason in a["checks"]:
        icon = {"BUY": "🟢", "SELL": "🔴"}.get(sig, "⚪")
        st.markdown(f"- {icon} **{name}**: {reason}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سعر الإغلاق", f"{P(a['price'])}")
    c2.metric("RSI14", f"{a['rsi']:.1f}")
    c3.metric("ATR14", f"{P(a['atr'])}")
    c4.metric("EMA200", f"{P(a['ema200'])}")

    st.markdown("## 📈 شارت الخطة (الدخول والوقف والأهداف)")
    st.plotly_chart(create_chart(a["df"], a["ticker"], plan["direction"], plan["sl"], plan["tp1"], plan["tp2"], a.get("tf", TF_D1)),
                    use_container_width=True)

    st.markdown("## 🧠 فريق الوكلاء")
    ag = a.get("agents")
    if not ag:
        st.info("أعد تشغيل التحليل لإظهار نتائج الوكلاء.")
    else:
        tech, news, risk = ag["tech"], ag["news"], ag["risk"]
        vmap = {"BUY": "شراء 🟢", "SELL": "بيع 🔴", "WAIT": "انتظار ⚪"}

        with st.expander("📈 الوكيل الفني", expanded=True):
            if ag.get("tech_model"):
                st.caption(f"النموذج: {ag['tech_model']}")
            if tech is None:
                st.error(f"غير متاح: {ag['tech_err']}")
                if ag["tech_raw"]:
                    st.caption(ag["tech_raw"][:500])
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("الرأي", vmap.get(str(tech.get("verdict", "")).upper(), "—"))
                m2.metric("الثقة", f"{tech.get('confidence', '—')}%")
                m3.metric("يؤيد الإشارة", "نعم ✅" if truthy(tech.get("supports_signal")) else "لا ❌")
                if tech.get("market_context"):
                    st.markdown(str(tech["market_context"]))
                kl = tech.get("key_levels") if isinstance(tech.get("key_levels"), dict) else {}
                st.markdown(f"**دعم:** {as_list(kl.get('support'))} | **مقاومة:** {as_list(kl.get('resistance'))}")
                for rsn in as_list(tech.get("reasons")):
                    st.markdown(f"- {rsn}")
                if tech.get("invalidation"):
                    st.caption(f"يُلغى التحليل إذا: {tech['invalidation']}")

        with st.expander("📰 وكيل الأخبار (عناوين Google News)", expanded=True):
            if ag.get("news_model"):
                st.caption(f"النموذج: {ag['news_model']}")
            if news is None:
                st.error(f"غير متاح: {ag['news_err']}")
                if ag["news_raw"]:
                    st.caption(ag["news_raw"][:500])
            else:
                sent = {"bullish": "صاعد 🟢", "bearish": "هابط 🔴", "neutral": "محايد ⚪"}
                evr = {"high": "مرتفعة 🔴", "medium": "متوسطة 🟠", "low": "منخفضة 🟢", "unknown": "غير معروفة ⚪"}
                imp = {"supports": "يدعم الإشارة", "conflicts": "يعاكس الإشارة", "neutral": "محايد"}
                m1, m2, m3 = st.columns(3)
                m1.metric("المزاج", sent.get(str(news.get("sentiment", "")).lower(), "—"))
                m2.metric("مخاطر الأحداث", evr.get(str(news.get("event_risk", "")).lower(), "—"))
                m3.metric("الأثر على الإشارة", imp.get(str(news.get("impact_on_signal", "")).lower(), "—"))
                if news.get("summary"):
                    st.markdown(str(news["summary"]))
                for ev in as_list(news.get("key_events")):
                    st.markdown(f"- {ev}")
                if ag["news_sources"]:
                    st.markdown("**المصادر:**")
                    for src_ in ag["news_sources"]:
                        _t = str(src_["title"]).replace("[", "(").replace("]", ")")   # لا تكسر عناوين الأخبار تنسيق الماركداون
                        st.markdown(f"- [{_t}]({src_['uri']})")
                else:
                    st.warning("⚠️ لا توجد عناوين مصدرية — تحقق من الأخبار يدوياً.")

        with st.expander("🛡️ وكيل المخاطر", expanded=True):
            if ag.get("risk_model"):
                st.caption(f"النموذج: {ag['risk_model']}")
            if ag["risk_skipped"]:
                st.info("لم يُشغَّل: لا توجد إشارة للتقييم.")
            elif risk is None:
                st.error(f"غير متاح: {ag['risk_err']}")
                if ag["risk_raw"]:
                    st.caption(ag["risk_raw"][:500])
            else:
                dmap = {"APPROVE": "موافقة ✅", "REDUCE": "تخفيض الحجم ⚠️", "VETO": "اعتراض 🛑"}
                m1, m2 = st.columns(2)
                m1.metric("القرار", dmap.get(str(risk.get("decision", "")).upper(), "—"))
                m2.metric("درجة المخاطرة", f"{risk.get('risk_score', '—')}/10")
                for c_ in as_list(risk.get("concerns")):
                    st.markdown(f"- {c_}")
                if risk.get("comment"):
                    st.caption(str(risk["comment"]))

    arb = a.get("arb")
    if arb:
        st.markdown("## القرار النهائي")
        getattr(st, arb["level"])(f"**{arb['action']}**")
        for n_ in arb["notes"]:
            st.markdown(f"- {n_}")
        if a.get("tg") == "ok":
            st.success("📨 أُرسل تنبيه تيليجرام.")
        elif a.get("tg"):
            st.warning(a["tg"])
        st.caption("القرار بقواعد ثابتة: اعتراض المخاطر أو رفض الفني = لا دخول، وأي تحفظ = نصف الحجم، "
                   "ولا تنبيه دخول إلا إذا بلغت نقاط الجودة الحد المطلوب.")


# ---------------------------------------------------------------------
# محاكي التداول
# ---------------------------------------------------------------------
pos = st.session_state.current_position
if st.session_state.notice:
    kind, msg = st.session_state.notice
    getattr(st, kind)(msg)
    st.session_state.notice = None

if a is not None or pos is not None:
    sim_ticker = pos["ticker"] if pos else a["ticker"]
    atr = pos["atr"] if pos else a["atr"]
    last_close = a["price"] if (a and a["ticker"] == sim_ticker) else pos["entry_price"]
    price = st.session_state.live_price or last_close

    st.markdown("---")
    st.header("🎯 محاكي التداول (Paper Trading)")

    src = "لحظي" if st.session_state.live_price else ("آخر إغلاق" if a else "سعر الدخول — اضغط تحديث السعر")
    st.markdown(f"**الأصل:** `{sim_ticker}` | **السعر ({src}):** `{P(price)}` | **ATR:** `{P(atr)}`")

    if st.button("🔄 تحديث السعر", use_container_width=True):
        st.session_state.live_price = get_live_price(sim_ticker, last_close)
        check_exit(st.session_state.live_price)
        st.rerun()

    plan_dir = a["plan"]["direction"] if (a and a["plan"]["level"] != "error") else None
    if st.button("📌 افتح الصفقة حسب الخطة", use_container_width=True, type="primary",
                 disabled=(pos is not None or plan_dir is None)):
        open_position(plan_dir, price, sim_ticker, atr, meta_from(a, "plan"))
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 شراء يدوي", use_container_width=True, disabled=(pos is not None or a is None)):
            open_position("BUY", price, sim_ticker, atr, meta_from(a, "manual"))
            st.rerun()
    with col2:
        if st.button("🔴 بيع يدوي", use_container_width=True, disabled=(pos is not None or a is None)):
            open_position("SELL", price, sim_ticker, atr, meta_from(a, "manual"))
            st.rerun()
    with col3:
        if st.button("⏹️ إغلاق الصفقة", use_container_width=True, disabled=pos is None):
            close_trade(price, "Manual")
            st.rerun()

    if pos is not None:
        cur = calc_pnl(pos, price)
        st.markdown("### الصفقة الحالية")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("الاتجاه", pos["direction"])
        c2.metric("سعر الدخول", f"{P(pos['entry_price'])}")
        c3.metric("وقف الخسارة", f"{P(pos['sl'])}")
        c4.metric("الهدف", f"{P(pos['tp'])}")
        c5.metric("الربح الحالي", f"{cur:+.2f}%")


# ---------------------------------------------------------------------
# 📒 السجل الدائم + إحصاءات التعلم
# ---------------------------------------------------------------------
st.markdown("---")
st.header("📒 سجل الصفقات الدائم")
if storage_mode() == "gist":
    st.caption("💾 التخزين: GitHub Gist (دائم) ✅")
else:
    st.warning("💾 التخزين مؤقت (ملف محلي يُمسح عند إعادة تشغيل التطبيق). للحفظ الدائم أضف GITHUB_TOKEN و GIST_ID في Secrets.")
if st.session_state.get("journal_load_err"):
    st.error(f"تعذّر تحميل السجل: {st.session_state.journal_load_err} — لن يُحفظ أي شيء حتى يُحل الخطأ.")
if st.session_state.get("journal_save_err"):
    st.error(f"تعذّر حفظ السجل: {st.session_state.journal_save_err}")

trades = st.session_state.trades
if not trades:
    st.info("لا صفقات بعد. افتح صفقة من المحاكي ثم أغلقها لتظهر هنا.")
else:
    tdf = pd.DataFrame(trades)
    tdf["pnl_percent"] = pd.to_numeric(tdf["pnl_percent"], errors="coerce")
    names = {"entry_time": "وقت الدخول", "ticker": "الأصل", "direction": "الاتجاه", "entry_price": "سعر الدخول",
             "exit_price": "سعر الخروج", "exit_reason": "سبب الخروج", "pnl_percent": "الربح %",
             "r_multiple": "R", "score": "النقاط", "used": "الاستراتيجية", "tf": "الإطار", "source": "المصدر"}
    cols = [c for c in names if c in tdf.columns]
    disp = tdf[cols].copy()
    for c in ("used", "tf"):
        if c in disp:
            disp[c] = disp[c].map(lambda x: x.split(" (")[0] if isinstance(x, str) else x)
    for c in ("pnl_percent", "r_multiple"):
        if c in disp:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)
    st.dataframe(disp.rename(columns=names), use_container_width=True, hide_index=True)

    wins = tdf["pnl_percent"] > 0
    gw, gl = tdf.loc[wins, "pnl_percent"].sum(), abs(tdf.loc[~wins, "pnl_percent"].sum())
    rr = pd.to_numeric(tdf["r_multiple"], errors="coerce").mean() if "r_multiple" in tdf else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("عدد الصفقات", len(tdf))
    m2.metric("نسبة الربح", f"{wins.mean() * 100:.1f}%")
    m3.metric("متوسط R", f"{rr:.2f}" if rr == rr else "—")
    m4.metric("PF", f"{gw / gl:.2f}" if gl > 0 else "∞")
    st.caption("نتائج ورقية بدون تكلفة أو سبريد، فهي أفضل قليلاً من الواقع.")

    with st.expander("📚 إحصاءات التعلم (أي الظروف تنجح؟)"):
        if len(tdf) < 20:
            st.warning("أقل من 20 صفقة: الإحصاءات للاطلاع فقط ولا يُبنى عليها قرار.")
        groups = journal_groups(tdf)
        for title, g in groups:
            st.markdown(f"**حسب {title}**")
            st.dataframe(g, use_container_width=True, hide_index=True)
        top = tdf[pd.to_numeric(tdf["score"], errors="coerce") >= 85] if "score" in tdf else tdf.iloc[0:0]
        if len(top) >= 10:
            st.success(f"صفقات النقاط 85+: {len(top)} صفقة، نسبة ربحها {(top['pnl_percent'] > 0).mean() * 100:.1f}%.")
        else:
            st.info("لا توجد بعد عينة كافية (10 صفقات بنقاط 85+ على الأقل) لمعايرة النقاط بنتائجك الفعلية.")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ تنزيل السجل CSV", tdf.to_csv(index=False).encode("utf-8-sig"),
                       "trades.csv", "text/csv", use_container_width=True)
    with st.expander("⚠️ مسح السجل"):
        sure = st.checkbox("أؤكد حذف كل الصفقات نهائياً", key="confirm_clear")
        if st.button("🗑️ مسح السجل", disabled=not sure):
            st.session_state.trades = []
            persist()
            st.rerun()

# ---------------------------------------------------------------------
# 📊 Backtest
# ---------------------------------------------------------------------
st.markdown("---")
st.header("📊 اختبار الاستراتيجيات (Backtest)")
st.caption("يقارن الاستراتيجيات والأطر الزمنية على بيانات تاريخية للرمز المكتوب في الشريط الجانبي. "
           "الإشارة عند الإغلاق والدخول عند افتتاح الشمعة التالية. لا يشمل رأي الذكاء الاصطناعي. "
           "بيانات 4H والمزدوج تغطي حوالي سنتين فقط مهما كانت المدة المختارة.")

bc1, bc2, bc3 = st.columns(3)
bt_period = bc1.selectbox("مدة البيانات اليومية", ["3y", "5y", "10y"], index=1)
bt_cost = bc2.number_input("تكلفة الصفقة % (سبريد+عمولة)", 0.0, 2.0, 0.1, 0.01)
bt_hold = bc3.number_input("أقصى مدة (شموع)", 5, 200, 30)
bt_tfs = st.multiselect("الأطر المطلوب مقارنتها", [TF_D1, TF_H4, TF_DUAL], default=[TF_D1, TF_H4, TF_DUAL])


def short(label):
    return label.split(" (")[0]


if st.button("▶️ شغّل الاختبار", use_container_width=True):
    bt_ticker = ticker.strip()
    if not bt_ticker:
        st.error("الرجاء إدخال رمز الأصل")
    elif not bt_tfs:
        st.error("اختر إطاراً واحداً على الأقل")
    else:
        with st.spinner("جاري الاختبار..."):
            try:
                results, bh, bars, notes, prices = {}, {}, {}, [], {}
                for tf in bt_tfs:
                    dfb, bias_b, err = load_prepared(bt_ticker, tf, True, bt_period)
                    if err:
                        notes.append(f"{short(tf)}: {err}")
                        continue
                    if len(dfb) < 300:
                        notes.append(f"{short(tf)}: البيانات غير كافية ({len(dfb)} شمعة، يلزم 300+)")
                        continue
                    sigs = strategy_signals(dfb)
                    if bias_b is not None:
                        sigs = apply_bias_filter(sigs, bias_b)
                    sigs[AUTO] = auto_signals(dfb, sigs)
                    for name in [AUTO, BREAKOUT, PULLBACK, MEAN_REV]:
                        tr = run_backtest(dfb, sigs[name], bt_cost, int(bt_hold))
                        results[(tf, name)] = {"trades": tr, "stats": backtest_stats(tr)}
                    bh[tf] = float((dfb["Close"].iloc[-1] / dfb["Close"].iloc[200] - 1) * 100)
                    bars[tf] = len(dfb) - 200
                    prices[tf] = dfb[["Open", "High", "Low", "Close"]]
                st.session_state.backtest = {
                    "ticker": bt_ticker, "period": bt_period, "cost": bt_cost,
                    "results": results, "buy_hold": bh, "bars": bars, "notes": notes, "prices": prices,
                }
            except Exception as e:
                st.error(f"خطأ في الاختبار: {e}")

bt = st.session_state.get("backtest")
if bt and "results" in bt and isinstance(bt.get("buy_hold"), dict):
    for n in bt["notes"]:
        st.warning(n)
    if not bt["results"]:
        st.error("لا توجد نتائج. تأكد من الرمز وتوفر البيانات.")
    else:
        st.markdown(f"**{bt['ticker']}** | تكلفة {bt['cost']}% | " +
                    " | ".join(f"{short(tf)}: {n} شمعة" for tf, n in bt["bars"].items()))

        rows = []
        for (tf, name), res in bt["results"].items():
            s_ = res["stats"]
            rows.append({
                "الإطار": short(tf),
                "الاستراتيجية": short(name),
                "الصفقات": s_["trades"],
                "نسبة الربح %": round(s_["win_rate"], 1),
                "الإجمالي %": round(s_["total"], 1),
                "متوسط R": round(s_["avg_r"], 2),
                "PF": round(s_["pf"], 2) if np.isfinite(s_["pf"]) else "∞",
                "أقصى تراجع %": round(s_["max_dd"], 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("الشراء والاحتفاظ في نفس الفترة: " +
                   " | ".join(f"{short(tf)}: {v:+.1f}%" for tf, v in bt["buy_hold"].items()) +
                   ". «متوسط R» = متوسط الربح لكل صفقة بوحدة المخاطرة؛ فوق 0.2 بعد التكلفة يُعتبر مقبولاً. "
                   "PF = Profit Factor.")

        valid = {k: r["stats"] for k, r in bt["results"].items() if r["stats"]["trades"] >= 30}
        if valid:
            best = max(valid, key=lambda k: valid[k]["avg_r"])
            if valid[best]["avg_r"] > 0:
                st.success(f"الأفضل (عينة ≥ 30 صفقة): **{short(best[1])}** على إطار **{short(best[0])}** "
                           f"بمتوسط {valid[best]['avg_r']:.2f}R لكل صفقة.")
            else:
                st.warning("لا توجد تركيبة رابحة بعد التكلفة على هذا الرمز والمدة.")
        else:
            st.warning("كل التركيبات أقل من 30 صفقة — العينة صغيرة ولا يُعتمد على النتائج.")

        tf_keys = list(bt["bars"].keys())
        tf_pick = st.selectbox("منحنى الأرباح للإطار:", tf_keys, format_func=short, key="bt_tf_pick")
        fig_bt = go.Figure()
        for (tf, name), res in bt["results"].items():
            if tf == tf_pick and not res["trades"].empty:
                tr = res["trades"]
                fig_bt.add_trace(go.Scatter(x=tr["exit_date"], y=tr["pnl"].cumsum(), mode="lines", name=short(name)))
        fig_bt.update_layout(title="الأرباح التراكمية % (بدون إعادة استثمار)", template="plotly_white", height=400)
        st.plotly_chart(fig_bt, use_container_width=True)

        pick = st.selectbox("عرض صفقات:", list(bt["results"].keys()), key="bt_pick",
                            format_func=lambda k: f"{short(k[0])} | {short(k[1])}")
        tr_pick = bt["results"][pick]["trades"]
        if tr_pick.empty:
            st.info("لا صفقات لهذه التركيبة.")
        else:
            show = tr_pick.copy()
            for col in ("entry_date", "exit_date"):
                show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M")
            show[["entry", "exit", "pnl", "r"]] = show[["entry", "exit", "pnl", "r"]].round(2)
            st.dataframe(show, use_container_width=True, hide_index=True)
            px_ = bt.get("prices", {}).get(pick[0])
            if px_ is not None:
                st.markdown("#### 📍 الدخول والخروج على الشارت")
                st.plotly_chart(create_trades_chart(px_, tr_pick, f"{short(pick[0])} | {short(pick[1])} — آخر الصفقات"),
                                use_container_width=True)
                st.caption("▲ شراء ▼ بيع ✕ خروج. الخط الأخضر صفقة رابحة والأحمر خاسرة (يظهر حتى 25 صفقة أخيرة).")

        st.caption("⚠️ الأداء السابق لا يضمن المستقبل. اختبر عدة رموز ومدد، وتجنّب تعديل الأرقام حتى تعطي أفضل نتيجة (Overfitting). "
                   "على 4H تأكل التكلفة نسبة أكبر من الربح، فلا تجعلها أقل من الواقع.")

st.markdown("---")
st.caption("تنبيه: هذا التطبيق لأغراض تعليمية فقط، ولا يعد نصيحة مالية.")
