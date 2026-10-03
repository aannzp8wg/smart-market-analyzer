#!/usr/bin/env python3
"""
الفاحص التلقائي — يعمل على GitHub Actions كل ساعة.

* يقرأ منطق التحليل من app.py نفسه (لا نسخة ثانية) فيبقى القراران متطابقين.
* يفحص قائمة الرموز في scanner_config.json.
* لا يستدعي الذكاء الاصطناعي إلا إذا وُجدت إشارة فنية.
* يرسل تيليجرام فقط عند بلوغ نقاط الجودة الحد المطلوب وموافقة الوكلاء.
* يمنع تكرار التنبيه لنفس الشمعة عبر ملف الحالة في الـ Gist (إن وُجد).
"""
import ast
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types as gtypes
from groq import Groq

HERE = os.path.dirname(os.path.abspath(__file__))
APP_FILE = os.path.join(HERE, "app.py")
CONFIG_FILE = os.path.join(HERE, "scanner_config.json")
STATE_NAME = "scanner_state.json"

DEFAULT_CFG = {
    "symbols": ["GC=F"], "timeframe": "dual", "strategy": "auto", "threshold": 85,
    "balance": 10000, "risk_pct": 1.0, "max_signal_age_hours": 6,
    "daily_summary": True, "summary_hour_utc": 6,
}

WANTED = {
    "DATA_PERIOD", "AUTO", "BREAKOUT", "PULLBACK", "MEAN_REV", "TF_D1", "TF_H4", "TF_DUAL",
    "SL_ATR", "TP1_ATR", "TP2_ATR", "GEMINI_FALLBACK", "GROQ_FALLBACK", "NEWS_NAMES",
    "TECH_SYSTEM", "RISK_SYSTEM", "NEWS_SYSTEM",
    "DataError", "_fetch_yf", "load_data", "load_data_h4", "drop_incomplete_candle", "calculate_indicators",
    "daily_bias_series", "align_bias", "apply_bias_filter", "load_prepared", "_load_prepared",
    "generate_signal", "select_strategy",
    "P", "truthy", "as_list", "guess_contract_size", "trade_levels", "extract_json",
    "rank_gemini_models", "rank_groq_models", "_list_gemini", "_list_groq", "get_gemini_models",
    "get_groq_models", "_gemini_client", "_groq_client", "call_groq", "_grounding_sources", "call_gemini",
    "swing_levels", "build_context", "news_queries", "fetch_google_news", "fetch_news", "news_agent",
    "run_agents", "arbitrate",
    "build_plan", "history_evidence", "compute_quality", "final_decision", "format_alert", "send_telegram",
    "strategy_signals", "auto_signals", "run_backtest", "backtest_stats",
    "_jdump", "_gist_req",
}


class _StubSt:
    """بديل مصغّر لـ streamlit: مفاتيح من متغيرات البيئة + cache_data بلا تخزين."""

    def __init__(self):
        env = os.environ.get
        self.secrets = {
            "GEMINI_API_KEY": env("GEMINI_API_KEY", ""),
            "GROQ_API_KEY": env("GROQ_API_KEY", ""),
            "TELEGRAM_BOT_TOKEN": env("TELEGRAM_BOT_TOKEN", ""),
            "TELEGRAM_CHAT_ID": env("TELEGRAM_CHAT_ID", ""),
        }

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


def load_app_logic():
    """ينفّذ الدوال والثوابت المطلوبة فقط من app.py (بدون واجهة streamlit)."""
    with open(APP_FILE, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in WANTED for t in node.targets):
            keep.append(node)
    st = _StubSt()
    ns = {
        "st": st, "json": json, "math": math, "os": os, "re": re, "time": time, "random": random,
        "urllib": urllib, "ET": ET, "ThreadPoolExecutor": ThreadPoolExecutor,
        "datetime": datetime, "timezone": timezone,
        "np": np, "pd": pd, "yf": yf, "genai": genai, "types": gtypes, "Groq": Groq,
        "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"],
        "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
        "__name__": "app_logic",
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), APP_FILE, "exec"), ns)
    missing = sorted(n for n in WANTED if n not in ns)
    if missing:
        raise SystemExit(f"app.py لا يحتوي هذه العناصر التي يحتاجها الفاحص: {missing}")
    return ns


def _gist_env():
    tok, gid = os.environ.get("GIST_TOKEN", ""), os.environ.get("GIST_ID", "")
    return (tok, gid) if tok and gid else None


def state_load(A):
    cfg = _gist_env()
    if not cfg:
        return {}, False
    try:
        data = A["_gist_req"](f"https://api.github.com/gists/{cfg[1]}", cfg[0])
        f = (data.get("files") or {}).get(STATE_NAME)
        txt = ((f or {}).get("content") or "").strip()
        state = json.loads(txt) if txt else {}
        return (state if isinstance(state, dict) else {}), True
    except Exception as e:
        print(f"⚠️ تعذّر تحميل الحالة: {type(e).__name__}: {str(e)[:100]}")
        return {}, False


def state_save(A, state):
    cfg = _gist_env()
    if not cfg:
        return
    try:
        state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        body = json.dumps({"files": {STATE_NAME: {"content": A["_jdump"](state)}}}).encode()
        A["_gist_req"](f"https://api.github.com/gists/{cfg[1]}", cfg[0], body, "PATCH")
    except Exception as e:
        print(f"⚠️ تعذّر حفظ الحالة: {type(e).__name__}: {str(e)[:100]}")


def bar_age_hours(df, tf, A):
    """عمر آخر شمعة مكتملة بالساعات (منذ إغلاقها)."""
    length = pd.Timedelta(days=1) if tf == A["TF_D1"] else pd.Timedelta(hours=4)
    end = df.index[-1] + length
    now = (pd.Timestamp.now(tz=df.index.tz)
           if df.index.tz is not None
           else pd.Timestamp.now(tz="UTC").tz_localize(None))
    return (now - end).total_seconds() / 3600


def scan_symbol(A, t, cfg, state, state_ok):
    tf = {"dual": A["TF_DUAL"], "1d": A["TF_D1"], "4h": A["TF_H4"]}[str(cfg["timeframe"]).lower()]
    strategy = A["AUTO"] if str(cfg["strategy"]).lower() == "auto" else cfg["strategy"]

    df, bias_s, err = A["load_prepared"](t, tf, True)
    if err:
        return {"symbol": t, "status": "error", "msg": err}
    latest = df.iloc[-1]
    if len(df) < 60 or latest[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any():
        return {"symbol": t, "status": "error", "msg": "بيانات غير كافية للمؤشرات"}

    bias = bias_s.iloc[-1] if bias_s is not None else None
    regime, adx, checks, chosen = A["select_strategy"](df, strategy, bias)
    if not chosen:
        return {"symbol": t, "status": "no_signal", "msg": "لا إشارة فنية"}
    used, signal, reason = chosen

    max_age = float(cfg["max_signal_age_hours"]) if state_ok else min(float(cfg["max_signal_age_hours"]), 1.2)
    age = bar_age_hours(df, tf, A)
    if age > max_age:
        return {"symbol": t, "status": "stale", "msg": f"إشارة {signal} قديمة (عمر الشمعة {age:.1f} س)"}

    bar_id = df.index[-1].isoformat()
    key = f"{t}|{signal}"
    if state.get("alerts", {}).get(key) == bar_id:
        return {"symbol": t, "status": "dup", "msg": f"{signal} — سبق تنبيهها لهذه الشمعة"}

    price, atr = float(latest["Close"]), float(latest["ATR14"])
    ctx = A["build_context"](t, df, tf, bias, regime, adx, used, signal, reason, checks)
    agents = A["run_agents"](ctx, t, signal, price, atr)
    arb = A["arbitrate"](signal, agents["tech"], agents["news"], agents["risk"])
    evidence = A["history_evidence"](t, tf, df, bias_s, used, True)
    score, _parts = A["compute_quality"](signal, df, adx, used, bias, agents["tech"],
                                         agents["news"], agents["risk"], evidence)
    dec = A["final_decision"](signal, arb, score, cfg["threshold"])
    plan = A["build_plan"](signal, price, atr, dec, float(cfg["balance"]), float(cfg["risk_pct"]),
                           A["guess_contract_size"](t))

    res = {"symbol": t, "signal": signal, "score": score, "used": used, "bar_id": bar_id, "key": key}
    if dec.get("enter"):
        res.update(status="alert", msg=f"{signal} — الجودة {score}/100",
                   text=A["format_alert"](t, tf, used, plan, score))
    elif arb["mult"] <= 0:
        res.update(status="blocked", msg=f"{signal} — {arb['action']} (النقاط {score})")
    else:
        res.update(status="below", msg=f"{signal} — الجودة {score}/100 أقل من الحد {cfg['threshold']}")
    return res


ICON = {"alert": "🔔", "blocked": "🛑", "below": "🟡", "no_signal": "⚪", "stale": "⏳",
        "dup": "✔️", "error": "❌"}


def main(cfg_override=None):
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    cfg.update(cfg_override or {})
    if os.environ.get("SCAN_SYMBOLS"):
        cfg["symbols"] = [s.strip() for s in os.environ["SCAN_SYMBOLS"].split(",") if s.strip()]

    A = load_app_logic()
    state, state_ok = state_load(A)
    if not state_ok:
        print("⚠️ لا حالة محفوظة (لم يُضبط GIST_TOKEN/GIST_ID أو فشل التحميل): قد تتكرر بعض التنبيهات.")
    state.setdefault("alerts", {})

    results = []
    for t in cfg["symbols"]:
        try:
            res = scan_symbol(A, t, cfg, state, state_ok)
        except Exception as e:
            res = {"symbol": t, "status": "error", "msg": f"{type(e).__name__}: {str(e)[:120]}"}
        results.append(res)
        print(f"{ICON.get(res['status'], '?')} {t:<10} {res['status']:<10} {res.get('msg', '')}")

        if res["status"] == "alert":
            sent = A["send_telegram"](res["text"])
            print(f"   تيليجرام: {sent if sent else 'غير مُعدّ'}")
            if sent == "ok" or sent is None:
                state["alerts"][res["key"]] = res["bar_id"]

    now = datetime.now(timezone.utc)
    all_failed = bool(results) and all(r["status"] == "error" for r in results)

    if state_ok and cfg.get("daily_summary") and now.hour >= int(cfg["summary_hour_utc"]) \
            and state.get("summary_date") != now.strftime("%Y-%m-%d"):
        lines = [f"{ICON.get(r['status'], '?')} {r['symbol']}: {r.get('msg', '')}"[:110] for r in results]
        txt = f"🩺 الفاحص يعمل — {now:%Y-%m-%d}\nفُحصت {len(results)} رموز:\n" + "\n".join(lines)
        if A["send_telegram"](txt) == "ok":
            state["summary_date"] = now.strftime("%Y-%m-%d")

    if all_failed and state_ok and state.get("error_date") != now.strftime("%Y-%m-%d"):
        msg = "⚠️ فشل الفاحص التلقائي لكل الرموز:\n" + "\n".join(r["msg"][:100] for r in results)
        if A["send_telegram"](msg) == "ok":
            state["error_date"] = now.strftime("%Y-%m-%d")

    if state_ok:
        state_save(A, state)

    print(f"\nالملخص: {len(results)} رموز | تنبيهات: {sum(r['status'] == 'alert' for r in results)} "
          f"| أخطاء: {sum(r['status'] == 'error' for r in results)}")
    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())