import streamlit as st

st.set_page_config(page_title="محلل السوق الذكي", page_icon="🤖", layout="wide")

import json
import math
import re
import time
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from google import genai
from google.genai import types
from groq import Groq

# ---------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------
DATA_PERIOD = "1y"

AUTO = "Auto (اختيار تلقائي حسب السوق)"
BREAKOUT = "Breakout (اختراق)"
PULLBACK = "Trend + Pullback (اتجاه+ارتداد)"
MEAN_REV = "Mean Reversion (عودة للمتوسط)"

TF_D1 = "يومي (1D)"
TF_H4 = "4 ساعات (4H)"
TF_DUAL = "مزدوج (اتجاه يومي + دخول 4H)"

SL_ATR = 1.5
TP1_ATR = 1.5
TP2_ATR = 3.0

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"خطأ: المفتاح {e} غير موجود في Streamlit Secrets.")
    st.stop()

st.title("محلل السوق الذكي")
st.markdown("### فريق وكلاء: فني + أخبار + مخاطر")


# =====================================================================
# 🔄 دوال جلب النماذج ديناميكياً (تعمل مع أي إصدار مستقبلي)
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_gemini_models():
    """جلب نماذج Gemini المتاحة لحسابك تلقائياً، مرتّبة بالأولوية."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        names = [m.name.replace("models/", "") for m in client.models.list()]
        banned = ("image", "live", "audio", "tts", "lite", "embedding",
                  "aqa", "vision", "exp", "preview", "thinking")
        cands = [n for n in names
                 if "gemini" in n.lower()
                 and ("flash" in n.lower() or "pro" in n.lower())
                 and not any(x in n.lower() for x in banned)]

        def ver_key(n):
            m = re.search(r"gemini-(\d+)\.(\d+)", n)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        cands.sort(key=ver_key, reverse=True)
        cands.sort(key=lambda n: 0 if "flash" in n.lower() else 1)
        return cands[:5] if cands else ["gemini-3.8-flash"]
    except Exception:
        return ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_groq_models():
    """جلب نماذج Groq المتاحة، مرتّبة بالأولوية."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        names = [m.id for m in client.models.list().data]
        priority = [n for n in names if "gpt-oss" in n.lower()]
        priority += [n for n in names
                     if any(k in n.lower() for k in ("llama", "qwen", "deepseek"))
                     and n not in priority]
        priority += [n for n in names if n not in priority]
        banned = ("whisper", "tts", "vision", "guard", "prompt-guard", "embedding")
        priority = [n for n in priority if not any(x in n.lower() for x in banned)]
        return priority[:4] if priority else ["openai/gpt-oss-120b"]
    except Exception:
        return ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


# ---------------------------------------------------------------------
# تهيئة الجلسة
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# البيانات والمؤشرات
# ---------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, period=DATA_PERIOD):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_data_h4(ticker):
    df = yf.download(ticker, period="700d", interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    agg = {c: f for c, f in {"Open": "first", "High": "max", "Low": "min",
                             "Close": "last", "Volume": "sum"}.items() if c in df.columns}
    return df.resample("4h").agg(agg).dropna(subset=["Open"])


def drop_incomplete_candle(df, hours=None):
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

    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / df["ATR14"]
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / df["ATR14"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    return df


# ---------------------------------------------------------------------
# الأطر الزمنية واتجاه اليومي
# ---------------------------------------------------------------------
def daily_bias_series(daily_df):
    c, e = daily_df["Close"], daily_df["EMA200"]
    return pd.Series(np.select([c > e, c < e], ["UP", "DOWN"], default="NEUTRAL"), index=daily_df.index)


def align_bias(df_lower, bias_daily):
    try:
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
    except Exception:
        return pd.Series("NEUTRAL", index=df_lower.index)


def apply_bias_filter(sigs, bias):
    b = bias.values
    out = {}
    for k, sr in sigs.items():
        v = sr.values.copy()
        v[(v == 1) & (b != "UP")] = 0
        v[(v == -1) & (b != "DOWN")] = 0
        out[k] = pd.Series(v, index=sr.index)
    return out


def load_prepared(ticker, tf, use_closed=True, daily_period=DATA_PERIOD):
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

    rawd = load_data(ticker, "5y")
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


def guess_contract_size(ticker):
    t = ticker.strip().upper()
    known = {"GC=F": 100.0, "SI=F": 5000.0, "CL=F": 1000.0, "NG=F": 10000.0,
             "BTC-USD": 1.0, "ETH-USD": 1.0}
    if t in known:
        return known[t]
    if t.endswith("=X"):
        return 100000.0
    return 1.0


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


def call_groq(system, user, temperature=0.2):
    """يرجع (النص، رسالة الخطأ، اسم النموذج الذي أجاب)."""
    last_err = ""
    for model_name in get_groq_models():
        try:
            client = Groq(api_key=GROQ_API_KEY)
            c = client.chat.completions.create(
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
            continue
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


def call_gemini(prompt, search=False):
    """يرجع (النص، المصادر، رسالة الخطأ، اسم النموذج الذي أجاب)."""
    err = ""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]) if search else None
        for model_name in get_gemini_models():
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
                    return (resp.text or ""), (_grounding_sources(resp) if search else []), "", model_name
                except Exception as e:
                    low = str(e).lower()
                    err = f"{model_name}: {str(e)[:140]}"
                    if any(k in low for k in ("api key not valid", "api_key_invalid", "unauthenticated")):
                        return "", [], "Gemini: مفتاح API غير صالح — تحقق من Streamlit Secrets.", ""
                    if any(k in low for k in ("503", "unavailable", "429", "resource_exhausted")):
                        time.sleep((2 ** attempt) + random.uniform(0, 1))
                        continue
                    break
    except Exception as e:
        err = f"Gemini: {str(e)[:140]}"
    if search and any(k in err.lower() for k in ("grounding", "google_search", "permission", "403")):
        err += " — قد لا يدعم حسابك/نموذجك بحث جوجل الحي."
    return "", [], err, ""


# ---------------------------------------------------------------------
# سياق السوق
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
# فريق الوكلاء
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

    news_prompt = f"""Today is {today}. Asset: {t} (a Yahoo Finance symbol — identify what asset it is).
Technical signal under review: {signal}.
Use Google Search to find (1) the most important news of the last 3 days affecting this asset and
(2) high-impact scheduled events in the next 48 hours (e.g. FOMC, CPI, NFP, central banks, OPEC, earnings).
Then reply with ONE JSON object only (no markdown). Text values in Arabic:
{{"sentiment":"bullish"|"bearish"|"neutral","event_risk":"high"|"medium"|"low"|"unknown",
 "impact_on_signal":"supports"|"conflicts"|"neutral","summary":"<2-3 sentences>",
 "key_events":["<max 4 short items with dates>"]}}
"event_risk" = "high" only for a major event or shock within ~24h that could cause violent moves.
If you could not find reliable information, use sentiment "neutral" and event_risk "unknown"."""

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_tech = ex.submit(call_groq, TECH_SYSTEM, tech_user, 0.2)
        f_news = ex.submit(call_gemini, news_prompt, True)
        tech_raw, tech_err, tech_model = f_tech.result()
        news_raw, news_sources, news_err, news_model = f_news.result()

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
        risk_raw, risk_err, risk_model = call_groq(RISK_SYSTEM, risk_user, 0.2)
        risk = extract_json(risk_raw)
        risk_err = risk_err or ("" if risk else "تعذّر قراءة رد الوكيل (ليس JSON صالحاً)")

    return {"tech": tech, "tech_raw": tech_raw, "tech_err": tech_err,
            "news": news, "news_raw": news_raw, "news_err": news_err, "news_sources": news_sources,
            "risk": risk, "risk_raw": risk_raw, "risk_err": risk_err, "risk_skipped": skipped,
            "tech_model": tech_model, "news_model": news_model, "risk_model": risk_model}


def arbitrate(signal, tech, news, risk):
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

    reduce = (tech_ok is None or risk_dec in (None, "REDUCE") or news_conflict or event_high)
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
        plan["size"] = risk_amount / sl_dist
        plan["lots"] = math.floor(plan["size"] / contract_size * 100) / 100
    return plan


# ---------------------------------------------------------------------
# الأدلة التاريخية + نقاط الجودة + التنبيه
# ---------------------------------------------------------------------
def history_evidence(t, tf, df, bias_s, used, use_closed, cost=0.1, hold=30):
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
    parts = []
    latest = df.iloc[-1]
    close, e50, e200 = float(latest["Close"]), float(latest["EMA50"]), float(latest["EMA200"])
    buy = signal == "BUY"

    side = (close > e200) if buy else (close < e200)
    slope = (e50 > e200) if buy else (e50 < e200)
    pts = 6 * side + 4 * slope
    parts.append(("توافق الاتجاه (EMA)", pts, 10,
                  "السعر وEMA50 في اتجاه الصفقة" if pts == 10 else "الصفقة ضد الاتجاه العام جزئياً أو كلياً"))

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
    parts.append(("وكيل المخاطر", pts, 12,
                  {"APPROVE": "موافقة", "REDUCE": "تخفيض الحجم"}.get(dec, "اعتراض/غير متاح")))

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
# محرك الـ Backtest
# ---------------------------------------------------------------------
def strategy_signals(df):
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
    adx = df["ADX14"]
    b, p, m = sigs[BREAKOUT], sigs[PULLBACK], sigs[MEAN_REV]
    trending_sig = b.where(b != 0, p)
    transition_sig = p.where(p != 0, b).where(lambda s: s != 0, m)
    out = np.where(adx >= 25, trending_sig, np.where(adx < 20, m, transition_sig))
    return pd.Series(out, index=df.index).astype(int)


def run_backtest(df, signals, cost_pct=0.05, max_hold=30, start=200):
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
        i = j

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


def open_position(direction, price, ticker, atr):
    sign = 1 if direction == "BUY" else -1
    st.session_state.current_position = {
        "direction": direction, "entry_price": price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "sl": price - sign * SL_ATR * atr,
        "tp": price + sign * TP2_ATR * atr,
        "atr": atr,
    }


def close_trade(price, exit_reason):
    pos = st.session_state.current_position
    pnl = calc_pnl(pos, price)
    st.session_state.trades.append({
        **pos, "exit_price": price,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exit_reason": exit_reason, "pnl_percent": pnl,
    })
    st.session_state.current_position = None
    why = {"SL": "ضُرب وقف الخسارة", "TP": "تحقق الهدف", "Manual": "أُغلقت يدوياً"}[exit_reason]
    kind = "success" if pnl > 0 else "error"
    st.session_state.notice = (kind, f"{'✅' if pnl > 0 else '❌'} {why} عند {P(price)} | النتيجة: {pnl:+.2f}%")


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
    contract_size = st.number_input("وحدات لكل 1 لوت", min_value=0.0001,
                                    value=guess_contract_size(ticker),
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

    with st.expander("🔧 فحص النماذج المتاحة"):
        st.caption("يعرض النماذج التي سيستخدمها الكود فعلياً الآن.")
        if st.button("افحص الآن", use_container_width=True):
            st.markdown("**Gemini (سيُجرَّب بهذا الترتيب):**")
            st.code("\n".join(get_gemini_models()) or "لا شيء")
            st.markdown("**Groq (سيُجرَّب بهذا الترتيب):**")
            st.code("\n".join(get_groq_models()) or "لا شيء")
            st.caption("إذا ظهرت القائمة فارغة، اضغط الزر مرة أخرى بعد دقيقة (قد يكون API مشغولاً).")


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
                st.error(f"خطأ: {e}")


# ---------------------------------------------------------------------
# عرض النتائج
# ---------------------------------------------------------------------
a = st.session_state.analysis
if a is not None:
    plan = a["plan"]

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
            st.caption("هذه نقاط جودة تجمع الفلاتر والوكلاء والأداء التاريخي، وليست احتمال ربح.")

    st.markdown(f"**الرمز:** {a['ticker']} | **الاستراتيجية المستخدمة:** {a['used']} | **الإطار:** {a.get('tf', TF_D1)} | **وقت التحليل:** {a['time']}")

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

    st.markdown("## الرسم البياني")
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

        with st.expander("📰 وكيل الأخبار (بحث جوجل الحي)", expanded=True):
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
                        st.markdown(f"- [{src_['title']}]({src_['uri']})")
                else:
                    st.warning("⚠️ لم يُرجع البحث أي مصادر.")

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
        st.caption("القرار بقواعد ثابتة: اعتراض المخاطر أو رفض الفني = لا دخول، وأي تحفظ = نصف الحجم.")


# ---------------------------------------------------------------------
# محاكي التداول
# ---------------------------------------------------------------------
if a is not None:
    pos = st.session_state.current_position
    sim_ticker = pos["ticker"] if pos else a["ticker"]
    atr = pos["atr"] if pos else a["atr"]
    price = st.session_state.live_price or a["price"]

    st.markdown("---")
    st.header("🎯 محاكي التداول (Paper Trading)")

    if st.session_state.notice:
        kind, msg = st.session_state.notice
        getattr(st, kind)(msg)
        st.session_state.notice = None

    src = "لحظي" if st.session_state.live_price else "آخر إغلاق"
    st.markdown(f"**الأصل:** `{sim_ticker}` | **السعر ({src}):** `{P(price)}` | **ATR:** `{P(atr)}`")

    if st.button("🔄 تحديث السعر", use_container_width=True):
        st.session_state.live_price = get_live_price(sim_ticker, a["price"])
        check_exit(st.session_state.live_price)
        st.rerun()

    plan_dir = a["plan"]["direction"] if a["plan"]["level"] != "error" else None
    if st.button("📌 افتح الصفقة حسب الخطة", use_container_width=True, type="primary",
                 disabled=(pos is not None or plan_dir is None)):
        open_position(plan_dir, price, sim_ticker, atr)
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 شراء يدوي", use_container_width=True, disabled=pos is not None):
            open_position("BUY", price, sim_ticker, atr)
            st.rerun()
    with col2:
        if st.button("🔴 بيع يدوي", use_container_width=True, disabled=pos is not None):
            open_position("SELL", price, sim_ticker, atr)
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

    if st.session_state.trades:
        st.markdown("### سجل الصفقات")
        tdf = pd.DataFrame(st.session_state.trades)
        disp = tdf[["entry_time", "ticker", "direction", "entry_price", "exit_price",
                    "exit_reason", "pnl_percent"]].copy()
        disp.columns = ["وقت الدخول", "الأصل", "الاتجاه", "سعر الدخول", "سعر الخروج", "سبب الخروج", "الربح %"]
        st.dataframe(disp, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("عدد الصفقات", len(tdf))
        c2.metric("إجمالي الربح", f"{tdf['pnl_percent'].sum():+.2f}%")
        c3.metric("نسبة الربح", f"{(tdf['pnl_percent'] > 0).mean() * 100:.1f}%")
        c4.metric("رابحة", int((tdf["pnl_percent"] > 0).sum()))
        c5.metric("خاسرة", int((tdf["pnl_percent"] <= 0).sum()))

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ تنزيل السجل CSV", tdf.to_csv(index=False).encode("utf-8-sig"),
                           "trades.csv", "text/csv", use_container_width=True)
        if d2.button("🗑️ مسح السجل", use_container_width=True):
            st.session_state.trades = []
            st.rerun()


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------
st.markdown("---")
st.header("📊 اختبار الاستراتيجيات (Backtest)")
st.caption("يقارن الاستراتيجيات والأطر الزمنية على بيانات تاريخية للرمز المكتوب في الشريط الجانبي.")

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
                results, bh, bars, notes = {}, {}, {}, []
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
                st.session_state.backtest = {
                    "ticker": bt_ticker, "period": bt_period, "cost": bt_cost,
                    "results": results, "buy_hold": bh, "bars": bars, "notes": notes,
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
                   " | ".join(f"{short(tf)}: {v:+.1f}%" for tf, v in bt["buy_hold"].items()))

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

        st.caption("⚠️ الأداء السابق لا يضمن المستقبل.")

st.markdown("---")
st.caption("تنبيه: هذا التطبيق لأغراض تعليمية فقط، ولا يعد نصيحة مالية.")