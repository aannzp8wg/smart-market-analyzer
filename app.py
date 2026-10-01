import streamlit as st

st.set_page_config(
    page_title="محلل السوق الذكي",
    page_icon="🤖",
    layout="wide"
)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import random
from datetime import datetime
from google import genai
from groq import Groq

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"خطأ: المفتاح {e} غير موجود في Streamlit Secrets.")
    st.stop()

st.title("محلل السوق الذكي")
st.markdown("### نظام تحليل متعدد العقول (Gemini + GPT-OSS 120B)")

with st.sidebar:
    st.header("الإعدادات")
    ticker = st.text_input("رمز الأصل", value="GC=F")
    strategy = st.selectbox(
        "الاستراتيجية",
        [
            "Breakout (اختراق)",
            "Trend + Pullback (اتجاه+ارتداد)",
            "Mean Reversion (عودة للمتوسط)"
        ]
    )
    analyze_btn = st.button("ابدأ التحليل", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### رموز مقترحة")
    st.markdown("- GC=F — الذهب")
    st.markdown("- BTC-USD — البيتكوين")
    st.markdown("- EURUSD=X — يورو/دولار")
    st.markdown("- AAPL — أبل")
    st.markdown("- TSLA — تسلا")
    st.markdown("- CL=F — النفط")
    st.caption("لأغراض تعليمية فقط")


def calculate_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["Donchian_Upper"] = df["High"].rolling(20).max()
    df["Donchian_Lower"] = df["Low"].rolling(20).min()

    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR14'] = ranges.max(axis=1).rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    return df


def generate_signal(df, strategy):
    latest = df.iloc[-1]
    signal = "No Signal"
    reason = ""

    if strategy == "Breakout (اختراق)":
        if latest["Close"] > df["Donchian_Upper"].iloc[-2] and latest["Close"] > latest["EMA200"]:
            signal = "BUY"
            reason = "اختراق صعودي فوق قمة 20 يوم + اتجاه صاعد"
        elif latest["Close"] < df["Donchian_Lower"].iloc[-2] and latest["Close"] < latest["EMA200"]:
            signal = "SELL"
            reason = "كسر هبوطي تحت قاع 20 يوم + اتجاه هابط"
        else:
            reason = "لا يوجد اختراق واضح حاليا"

    elif strategy == "Trend + Pullback (اتجاه+ارتداد)":
        if latest["Close"] > latest["EMA200"] and latest["Low"] <= latest["EMA50"] and latest["Close"] > latest["Open"]:
            signal = "BUY"
            reason = "ارتداد من EMA50 في اتجاه صاعد"
        elif latest["Close"] < latest["EMA200"] and latest["High"] >= latest["EMA50"] and latest["Close"] < latest["Open"]:
            signal = "SELL"
            reason = "ارتداد من EMA50 في اتجاه هابط"
        else:
            reason = "لا يوجد ارتداد واضح"

    elif strategy == "Mean Reversion (عودة للمتوسط)":
        df["BB_Std"] = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["EMA20"] + (2 * df["BB_Std"])
        df["BB_Lower"] = df["EMA20"] - (2 * df["BB_Std"])
        latest = df.iloc[-1]

        if latest["Close"] < latest["BB_Lower"] and latest["RSI14"] < 35:
            signal = "BUY"
            reason = "تشبع بيعي: السعر تحت بولنجر السفلي + RSI < 35"
        elif latest["Close"] > latest["BB_Upper"] and latest["RSI14"] > 65:
            signal = "SELL"
            reason = "تشبع شرائي: السعر فوق بولنجر العلوي + RSI > 65"
        else:
            reason = "السعر في المنطقة الطبيعية"

    return signal, reason, latest


def create_chart(df, ticker):
    chart_df = df.tail(100)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'], high=chart_df['High'],
        low=chart_df['Low'], close=chart_df['Close'],
        name='السعر'
    ))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA20'],
                             name='EMA20', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA50'],
                             name='EMA50', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA200'],
                             name='EMA200', line=dict(color='red', width=1.5)))

    fig.update_layout(
        title=f'{ticker} - Daily Chart',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )
    return fig


def get_gemini_analysis(prompt_text):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model_name in ['gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-2.5-pro']:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt_text)
                    return response.text
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        time.sleep((2 ** attempt) + random.uniform(0, 1))
                    else:
                        break
        return "Gemini: فشلت المحاولات."
    except Exception as e:
        return f"Gemini Error: {e}"


def get_groq_analysis(prompt_text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst. Reply in Arabic."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"


if analyze_btn:
    if not ticker or ticker.strip() == "":
        st.error("الرجاء إدخال رمز الأصل")
    else:
        with st.spinner(f"جاري تحليل {ticker}..."):
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)

                if df.empty:
                    st.error(f"لم يتم العثور على بيانات لـ {ticker}")
                else:
                    df.columns = df.columns.get_level_values(0)
                    df = calculate_indicators(df)
                    signal, reason, latest = generate_signal(df, strategy)

                    st.markdown("## نتائج التحليل")
                    st.markdown(f"**الرمز:** {ticker} | **الاستراتيجية:** {strategy}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("السعر الحالي", f"{float(latest['Close']):.2f}")
                    col2.metric("RSI14", f"{float(latest['RSI14']):.1f}")
                    col3.metric("ATR14", f"{float(latest['ATR14']):.2f}")
                    col4.metric("EMA200", f"{float(latest['EMA200']):.2f}")

                    if signal == "BUY":
                        st.success(f"إشارة شراء | {reason}")
                    elif signal == "SELL":
                        st.error(f"إشارة بيع | {reason}")
                    else:
                        st.info(f"لا توجد إشارة | {reason}")

                    atr = float(latest['ATR14'])
                    price = float(latest['Close'])

                    if signal == "BUY":
                        sl = price - (1.5 * atr)
                        tp = price + (3.0 * atr)
                        st.markdown(f"**إدارة المخاطر:** شراء | SL: {sl:.2f} | TP: {tp:.2f} | R:R 1:2")
                    elif signal == "SELL":
                        sl = price + (1.5 * atr)
                        tp = price - (3.0 * atr)
                        st.markdown(f"**إدارة المخاطر:** بيع | SL: {sl:.2f} | TP: {tp:.2f} | R:R 1:2")

                    st.markdown("## الرسم البياني")
                    fig = create_chart(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("## تحليل الذكاء الاصطناعي")

                    data_summary = df.tail(5)[['Open', 'High', 'Low', 'Close', 'EMA20', 'EMA50', 'EMA200', 'RSI14', 'ATR14']].to_string()

                    prompt = f"""
You are an expert financial analyst. Please write your entire response in Arabic.

Asset: {ticker}
Strategy: {strategy}

Last 5 daily candles with indicators:
{data_summary}

Technical Signal: {signal}
Signal Reason: {reason}

Required:
1. Analyze the general context (Uptrend or Downtrend?).
2. Do you support the technical signal? Mention the reason.
3. What is the suggested Stop Loss and Take Profit (using ATR)?
4. Rate the risk from 1 to 10.
5. Do you recommend entering now? Write "Yes" (نعم) or "No" (لا) clearly at the end.
"""

                    col_a, col_b = st.columns(2)

                    with col_a:
                        with st.expander("تحليل Gemini", expanded=True):
                            with st.spinner("Gemini يحلل..."):
                                gemini_result = get_gemini_analysis(prompt)
                                st.markdown(gemini_result)

                    with col_b:
                        with st.expander("تحليل GPT-OSS 120B", expanded=True):
                            with st.spinner("GPT-OSS يحلل..."):
                                groq_result = get_groq_analysis(prompt)
                                st.markdown(groq_result)

                    st.markdown("## القرار النهائي")

                    g_yes = "نعم" in gemini_result
                    q_yes = "نعم" in groq_result
                    yes_count = sum([g_yes, q_yes])

                    if yes_count == 2:
                        st.success("إجماع ثنائي: العقلان يتفقان على الدخول - إشارة قوية")
                    elif yes_count == 0:
                        st.error("إجماع ثنائي: العقلان يتفقان على عدم الدخول - انتظر")
                    else:
                        st.warning("انقسام: عقل يدخل وعقل ينتظر - توخ الحذر")

                    st.markdown(f"- Gemini: {'نعم' if g_yes else 'لا'}")
                    st.markdown(f"- GPT-OSS 120B: {'نعم' if q_yes else 'لا'}")

            except Exception as e:
                st.error(f"خطأ: {e}")

else:
    st.info("أدخل رمز الأصل في الشريط الجانبي، ثم اضغط ابدأ التحليل")

st.markdown("---")
st.caption("تنبيه: هذا التطبيق لأغراض تعليمية فقط، ولا يعد نصيحة مالية.")