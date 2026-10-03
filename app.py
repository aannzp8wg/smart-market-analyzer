import streamlit as st

st.set_page_config(page_title="محلل السوق الذكي", page_icon="🤖", layout="wide")

import re
import time
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from google import genai
from groq import Groq

# ---------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-pro"]
GROQ_MODEL = "openai/gpt-oss-120b"
DATA_PERIOD = "1y"

AUTO = "Auto (اختيار تلقائي حسب السوق)"
BREAKOUT = "Breakout (اختراق)"
PULLBACK = "Trend + Pullback (اتجاه+ارتداد)"
MEAN_REV = "Mean Reversion (عودة للمتوسط)"

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
st.markdown("### نظام تحليل متعدد العقول (Gemini + GPT-OSS 120B)")

# ---------------------------------------------------------------------
# تهيئة الجلسة
# ---------------------------------------------------------------------
defaults = {
    "trades": [],
    "current_position": None,
    "analysis": None,
    "live_price": None,
    "notice": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------
# البيانات والمؤشرات
# ---------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker):
    df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def drop_incomplete_candle(df):
    """حذف شمعة اليوم الجارية (غير المكتملة) لتفادي إشارات تتغير."""
    if len(df) > 1:
        if df.index[-1].date() >= datetime.now(timezone.utc).date():
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


def select_strategy(df, mode):
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

    checks = [(s, *generate_signal(df, s)) for s in order]
    chosen = next((c for c in checks if c[1] != "No Signal"), None)
    return regime, adx, checks, chosen


# ---------------------------------------------------------------------
# الرسم
# ---------------------------------------------------------------------
def create_chart(df, ticker, signal=None, sl=None, tp1=None, tp2=None):
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

    fig.update_layout(title=f"{ticker} - Daily Chart", template="plotly_white",
                      height=500, xaxis_rangeslider_visible=False)
    return fig


# ---------------------------------------------------------------------
# الذكاء الاصطناعي
# ---------------------------------------------------------------------
def get_gemini_analysis(prompt_text):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model_name in GEMINI_MODELS:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt_text)
                    return response.text, True
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        time.sleep((2 ** attempt) + random.uniform(0, 1))
                    else:
                        break
        return "Gemini: فشلت كل المحاولات.", False
    except Exception as e:
        return f"Gemini Error: {e}", False


def get_groq_analysis(prompt_text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst. Reply in Arabic."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content, True
    except Exception as e:
        return f"Groq Error: {e}", False


def parse_decision(text, ok):
    """True / False / None (None = خطأ أو لم يُفهم الرد)."""
    if not ok:
        return None
    m = re.findall(r"DECISION:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m[-1].upper() == "YES"


def decision_label(d):
    return "نعم" if d is True else "لا" if d is False else "غير متاح ⚠️"


# ---------------------------------------------------------------------
# خطة الصفقة
# ---------------------------------------------------------------------
def build_plan(signal, price, atr, g, q, balance, risk_pct):
    """يجمع الإشارة الفنية + رأي النموذجين في خطة نهائية."""
    plan = {"direction": None, "entry": price, "sl": None, "tp1": None, "tp2": None,
            "size": None, "action": "", "level": "info"}
    if signal not in ("BUY", "SELL"):
        plan["action"] = "انتظر — لا توجد إشارة مناسبة الآن"
        return plan

    sign = 1 if signal == "BUY" else -1
    plan["direction"] = signal
    plan["sl"] = price - sign * SL_ATR * atr
    plan["tp1"] = price + sign * TP1_ATR * atr
    plan["tp2"] = price + sign * TP2_ATR * atr

    votes = sum(1 for d in (g, q) if d is True)
    if votes == 2:
        plan["action"], plan["level"], mult = "ادخل الآن (إجماع النموذجين)", "success", 1.0
    elif votes == 1:
        plan["action"], plan["level"], mult = "دخول بحذر — نصف الحجم (نموذج واحد فقط يؤيد)", "warning", 0.5
    else:
        plan["action"], plan["level"], mult = "لا تدخل — النماذج لا تؤيد الإشارة", "error", 0.0
        plan["direction"] = None if False else plan["direction"]  # نُبقي الاتجاه للعرض فقط

    risk_amount = balance * risk_pct / 100 * mult
    sl_dist = SL_ATR * atr
    plan["size"] = risk_amount / sl_dist if sl_dist > 0 else None
    return plan


# ---------------------------------------------------------------------
# المحاكي
# ---------------------------------------------------------------------
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
    st.session_state.notice = (kind, f"{'✅' if pnl > 0 else '❌'} {why} عند {price:.2f} | النتيجة: {pnl:+.2f}%")


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
    use_closed = st.checkbox("استخدم آخر شمعة مكتملة فقط", value=True)
    balance = st.number_input("رأس المال (للحجم المقترح)", min_value=100.0, value=10000.0, step=100.0)
    risk_pct = st.slider("المخاطرة لكل صفقة %", 0.25, 3.0, 1.0, 0.25)
    analyze_btn = st.button("ابدأ التحليل", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### رموز مقترحة")
    st.markdown("- GC=F — الذهب\n- BTC-USD — البيتكوين\n- EURUSD=X — يورو/دولار\n"
                "- AAPL — أبل\n- TSLA — تسلا\n- CL=F — النفط")
    st.caption("لأغراض تعليمية فقط")


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
                raw = load_data(t)
                if raw.empty:
                    st.error(f"لم يتم العثور على بيانات لـ {t}")
                else:
                    df = drop_incomplete_candle(raw) if use_closed else raw
                    df = calculate_indicators(df)
                    latest = df.iloc[-1]
                    if len(df) < 60 or latest[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any():
                        st.error("البيانات غير كافية لحساب المؤشرات.")
                    else:
                        regime, adx, checks, chosen = select_strategy(df, strategy)
                        if chosen:
                            used, signal, reason = chosen
                        else:
                            used, signal, reason = checks[0][0], "No Signal", "لا توجد استراتيجية مناسبة تعطي إشارة الآن"

                        price, atr = float(latest["Close"]), float(latest["ATR14"])

                        data_summary = df.tail(5)[["Open", "High", "Low", "Close", "EMA20", "EMA50",
                                                   "EMA200", "RSI14", "ATR14", "ADX14"]].round(2).to_string()
                        prompt = f"""
You are an expert financial analyst. Please write your entire response in Arabic.

Asset: {t}
Market regime: {regime} (ADX14 = {adx:.1f})
Strategy used: {used}

Last 5 daily candles with indicators:
{data_summary}

Technical Signal: {signal}
Signal Reason: {reason}
Proposed SL: {SL_ATR} x ATR, TP1: {TP1_ATR} x ATR, TP2: {TP2_ATR} x ATR

Required:
1. Analyze the general context (Uptrend, Downtrend or Range?).
2. Do you support the technical signal and direction? Mention the reason.
3. Are the proposed Stop Loss and Take Profit levels sensible?
4. Rate the risk from 1 to 10.
5. Do you recommend entering now in the signal's direction? The very last line of your
   response MUST be exactly "DECISION: YES" or "DECISION: NO" (in English, nothing after it).
"""
                        g_text, g_ok = get_gemini_analysis(prompt)
                        q_text, q_ok = get_groq_analysis(prompt)
                        g_dec = parse_decision(g_text, g_ok)
                        q_dec = parse_decision(q_text, q_ok)

                        st.session_state.analysis = {
                            "ticker": t, "mode": strategy, "used": used, "df": df,
                            "regime": regime, "adx": adx, "checks": checks,
                            "signal": signal, "reason": reason,
                            "price": price, "atr": atr,
                            "rsi": float(latest["RSI14"]), "ema200": float(latest["EMA200"]),
                            "gemini": g_text, "groq": q_text, "g_dec": g_dec, "q_dec": q_dec,
                            "plan": build_plan(signal, price, atr, g_dec, q_dec, balance, risk_pct),
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

    # ---- خطة الصفقة (أهم شيء في الأعلى) ----
    st.markdown("## 📋 خطة الصفقة")
    getattr(st, plan["level"])(f"**{plan['action']}**")
    if plan["direction"]:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("الاتجاه", "شراء 🟢" if plan["direction"] == "BUY" else "بيع 🔴")
        p2.metric("الدخول", f"{plan['entry']:.2f}")
        p3.metric("وقف الخسارة", f"{plan['sl']:.2f}")
        p4.metric("الهدف 1", f"{plan['tp1']:.2f}")
        p5.metric("الهدف 2", f"{plan['tp2']:.2f}")
        p6.metric("الحجم المقترح", f"{plan['size']:.4g}" if plan["size"] else "—")
        st.caption("الهدف 1 = 1R، الهدف 2 = 2R. عند الهدف 1 يُنصح بإغلاق نصف الصفقة ونقل الوقف لنقطة الدخول.")

    st.markdown(f"**الرمز:** {a['ticker']} | **الاستراتيجية المستخدمة:** {a['used']} | **وقت التحليل:** {a['time']}")

    # ---- حالة السوق ----
    st.markdown("## 🧭 حالة السوق")
    r1, r2 = st.columns(2)
    r1.metric("الحالة", a["regime"])
    r2.metric("ADX14 (قوة الاتجاه)", f"{a['adx']:.1f}")
    for name, sig, reason in a["checks"]:
        icon = {"BUY": "🟢", "SELL": "🔴"}.get(sig, "⚪")
        st.markdown(f"- {icon} **{name}**: {reason}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سعر الإغلاق", f"{a['price']:.2f}")
    c2.metric("RSI14", f"{a['rsi']:.1f}")
    c3.metric("ATR14", f"{a['atr']:.2f}")
    c4.metric("EMA200", f"{a['ema200']:.2f}")

    st.markdown("## الرسم البياني")
    st.plotly_chart(create_chart(a["df"], a["ticker"], plan["direction"], plan["sl"], plan["tp1"], plan["tp2"]),
                    use_container_width=True)

    st.markdown("## تحليل الذكاء الاصطناعي")
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("تحليل Gemini", expanded=True):
            st.markdown(a["gemini"])
    with col_b:
        with st.expander("تحليل GPT-OSS 120B", expanded=True):
            st.markdown(a["groq"])

    st.markdown("## القرار النهائي")
    g, q = a["g_dec"], a["q_dec"]
    if g is None or q is None:
        st.warning("أحد النموذجين لم يعطِ قرارًا واضحًا (خطأ أو صيغة غير متوقعة). راجع النصوص أعلاه.")
    elif g and q:
        st.success("إجماع ثنائي: العقلان يتفقان على الدخول - إشارة قوية")
    elif not g and not q:
        st.error("إجماع ثنائي: العقلان يتفقان على عدم الدخول - انتظر")
    else:
        st.warning("انقسام: عقل يدخل وعقل ينتظر - توخَّ الحذر")
    st.markdown(f"- Gemini: {decision_label(g)}")
    st.markdown(f"- GPT-OSS 120B: {decision_label(q)}")


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
    st.markdown(f"**الأصل:** `{sim_ticker}` | **السعر ({src}):** `{price:.2f}` | **ATR:** `{atr:.2f}`")

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
        c2.metric("سعر الدخول", f"{pos['entry_price']:.2f}")
        c3.metric("وقف الخسارة", f"{pos['sl']:.2f}")
        c4.metric("الهدف", f"{pos['tp']:.2f}")
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

st.markdown("---")
st.caption("تنبيه: هذا التطبيق لأغراض تعليمية فقط، ولا يعد نصيحة مالية.")
