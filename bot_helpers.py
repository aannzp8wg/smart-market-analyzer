#!/usr/bin/env python3
"""
bot_helpers.py — دوال مساعدة للبوت.

* يقرأ منطق التحليل من app_multi_corrected.py عبر AST (نفس منطق الفاحص).
* يحفظ حالة البوت والصفقات في Gist.
* يعيد استخدام نفس دوال التحليل والفلترة.
"""
import ast
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
APP_FILE = os.path.join(HERE, "app_multi_corrected.py")
CONFIG_FILE = os.path.join(HERE, "scanner_config.json")
BOT_STATE_FILE = "bot_state.json"
PAPER_TRADES_FILE = "paper_trades.json"

# اسم الملف الرئيسي — نستخدم app_multi_corrected.py لأن هذا هو الذي يحتوي كل المنطق
if not os.path.exists(APP_FILE):
    APP_FILE = os.path.join(HERE, "app.py")

# متغيرات البيئة
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GIST_TOKEN = os.environ.get("GIST_TOKEN", "").strip()
GIST_ID = os.environ.get("GIST_ID", "").strip()


# =====================================================================
# Gist
# =====================================================================
def _gist_ready():
    return bool(GIST_TOKEN and GIST_ID)


def _gist_req(url, body=None, method="GET"):
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {GIST_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "telegram-bot",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _gist_read(filename):
    if not _gist_ready():
        return None
    try:
        data = _gist_req(f"https://api.github.com/gists/{GIST_ID}")
        f = (data.get("files") or {}).get(filename)
        if not f:
            return None
        txt = (f.get("content") or "").strip()
        return json.loads(txt) if txt else {}
    except Exception as e:
        print(f"⚠️ Gist read error ({filename}): {e}")
        return None


def _gist_write(filename, obj):
    if not _gist_ready():
        return False
    try:
        body = json.dumps({"files": {filename: {"content": json.dumps(obj, ensure_ascii=False, default=str)}}}).encode()
        _gist_req(f"https://api.github.com/gists/{GIST_ID}", body, "PATCH")
        return True
    except Exception as e:
        print(f"⚠️ Gist write error ({filename}): {e}")
        return False


# =====================================================================
# حالة البوت
# =====================================================================
def load_bot_state():
    return _gist_read(BOT_STATE_FILE) or {}


def save_bot_state(state):
    _gist_write(BOT_STATE_FILE, state)


# =====================================================================
# الصفقات الورقية
# =====================================================================
def _load_paper_trades():
    data = _gist_read(PAPER_TRADES_FILE) or {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("open", [])
    data.setdefault("closed", [])
    data.setdefault("next_id", 1)
    return data


def _save_paper_trades(data):
    _gist_write(PAPER_TRADES_FILE, data)


# =====================================================================
# الإعدادات (من scanner_config.json)
# =====================================================================
def load_config():
    default = {
        "symbols": ["GC=F", "EURUSD=X", "BTC-USD"],
        "timeframe": "dual",
        "strategy": "auto",
        "threshold": 85,
        "balance": 10000,
        "risk_pct": 1.0,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as fh:
                cfg = json.load(fh)
            default.update(cfg)
        except Exception:
            pass
    return default


def save_config(cfg):
    # نكتب على Gist إن لم يكن محلياً قابلاً للكتابة (GitHub Actions لا يدعم git push دائماً)
    _gist_write("scanner_config.json", cfg)
    # نحاول الكتابة محلياً أيضاً (للاستخدام المحلي)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =====================================================================
# استيراد منطق التطبيق عبر AST
# =====================================================================
class _StubSt:
    def __init__(self):
        env = os.environ.get
        self.secrets = {
            "GEMINI_API_KEY": env("GEMINI_API_KEY", ""),
            "GROQ_API_KEY": env("GROQ_API_KEY", ""),
            "TELEGRAM_BOT_TOKEN": env("TELEGRAM_BOT_TOKEN", ""),
            "TELEGRAM_CHAT_ID": env("TELEGRAM_CHAT_ID", ""),
            "GITHUB_TOKEN": env("GIST_TOKEN", ""),
            "GIST_ID": env("GIST_ID", ""),
        }

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


WANTED = {
    "DATA_PERIOD", "AUTO", "BREAKOUT", "PULLBACK", "MEAN_REV", "TREND",
    "TF_D1", "TF_H4", "TF_DUAL", "CONTRARIAN",
    "GEMINI_FALLBACK", "GROQ_FALLBACK",
    "DataError", "_fetch_yf", "load_data", "load_data_h4", "drop_incomplete_candle",
    "calculate_indicators", "daily_bias_series", "align_bias", "apply_bias_filter",
    "load_prepared", "_load_prepared", "generate_signal", "select_strategy",
    "P", "truthy", "as_list", "clean_price_levels",
    "guess_contract_size", "trade_levels",
    "extract_json", "validate_tech_json", "validate_news_json", "validate_risk_json",
    "rank_gemini_models", "rank_groq_models", "_list_gemini", "_list_groq",
    "get_gemini_models", "get_groq_models", "_gemini_client", "_groq_client",
    "call_groq", "_grounding_sources", "call_gemini",
    "swing_levels", "build_context", "news_queries", "fetch_google_news",
    "fetch_news", "news_agent", "run_agents", "news_policy", "arbitrate",
    "build_plan", "history_evidence", "compute_quality", "final_decision",
    "format_alert", "send_telegram",
    "strategy_signals", "auto_signals", "run_backtest", "backtest_stats",
    "journal_groups",
    "guess_tv_symbol",  # قد لا يكون موجوداً، لكن لا بأس
}

_APP_NS = None


def _load_app_logic():
    global _APP_NS
    if _APP_NS is not None:
        return _APP_NS

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
        "st": st, "json": json, "os": os, "re": __import__("re"), "time": time,
        "math": __import__("math"), "random": __import__("random"),
        "urllib": urllib, "ET": __import__("xml.etree.ElementTree", fromlist=["ElementTree"]),
        "ThreadPoolExecutor": __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor,
        "datetime": datetime, "timezone": timezone,
        "np": np, "pd": pd, "yf": __import__("yfinance"),
        "genai": __import__("google.genai", fromlist=["genai"]),
        "types": __import__("google.genai", fromlist=["types"]).types,
        "Groq": __import__("groq", fromlist=["Groq"]).Groq,
        "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"],
        "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
        "__name__": "app_logic",
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), APP_FILE, "exec"), ns)
    _APP_NS = ns
    return ns


# =====================================================================
# التحليل
# =====================================================================
def analyze_symbol(symbol: str) -> dict:
    """يحلل أصلاً واحداً ويعيد dict كامل."""
    A = _load_app_logic()
    cfg = load_config()
    tf_map = {"dual": A["TF_DUAL"], "1d": A["TF_D1"], "4h": A["TF_H4"]}
    tf = tf_map.get(cfg.get("timeframe", "dual"), A["TF_DUAL"])
    strategy = A["AUTO"] if cfg.get("strategy", "auto") == "auto" else cfg["strategy"]

    df, bias_s, err = A["load_prepared"](symbol, tf, True)
    if err:
        return {"symbol": symbol, "error": err}

    latest = df.iloc[-1]
    if len(df) < 60 or latest[["EMA200", "ATR14", "ADX14", "RSI14"]].isna().any():
        return {"symbol": symbol, "error": "بيانات غير كافية للمؤشرات"}

    bias = bias_s.iloc[-1] if bias_s is not None else None
    regime, adx, checks, chosen = A["select_strategy"](df, strategy, bias)
    if not chosen:
        return {
            "symbol": symbol, "signal": "No Signal",
            "reason": "لا توجد استراتيجية مناسبة تعطي إشارة الآن",
            "regime": regime, "adx": adx, "checks": checks,
            "price": float(latest["Close"]), "atr": float(latest["ATR14"]),
            "rsi": float(latest["RSI14"]), "ema200": float(latest["EMA200"]),
        }

    used, signal, reason = chosen
    price = float(latest["Close"])
    atr = float(latest["ATR14"])
    mean_rev = used in (A["MEAN_REV"], A.get("RSI2", "RSI2"))

    sl = tp1 = tp2 = None
    if signal in ("BUY", "SELL"):
        try:
            from gold_engine import smart_levels
            levels = smart_levels(signal, price, df, mean_rev=mean_rev)
            if levels["valid"]:
                sl, tp1, tp2 = levels["sl"], levels["tp1"], levels["tp2"]
        except Exception:
            pass

    ctx = A["build_context"](symbol, df, tf, bias, regime, adx, used, signal, reason, checks)
    agents = A["run_agents"](ctx, symbol, signal, price, atr, df, mean_rev)
    arb = A["arbitrate"](signal, agents["tech"], agents["news"], agents["risk"])
    evidence = A["history_evidence"](symbol, tf, df, bias_s, used, True)
    score, parts = A["compute_quality"](signal, df, adx, used, bias,
                                        agents["tech"], agents["news"], agents["risk"], evidence)
    dec = A["final_decision"](signal, arb, score, cfg.get("threshold", 85))

    return {
        "symbol": symbol, "signal": signal, "reason": reason,
        "used": used, "regime": regime, "adx": adx, "bias": bias,
        "price": price, "atr": atr,
        "rsi": float(latest["RSI14"]), "ema200": float(latest["EMA200"]),
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "agents": agents, "arb": dec, "score": score, "parts": parts,
        "checks": checks,
    }


def format_analysis_for_telegram(r: dict) -> str:
    """تنسيق التحليل كرسالة Telegram."""
    if r.get("error"):
        return f"❌ *{r['symbol']}*: {r['error']}"

    lines = [f"📊 *تحليل {r['symbol']}*\n"]

    signal = r.get("signal", "No Signal")
    if signal == "BUY":
        lines.append("🟢 *إشارة شراء*")
    elif signal == "SELL":
        lines.append("🔴 *إشارة بيع*")
    else:
        lines.append("⚪ *لا توجد إشارة*")

    lines.append(f"_{r.get('reason', '')}_\n")

    if signal in ("BUY", "SELL"):
        lines.append("💰 *خطة الصفقة:*")
        lines.append(f"  الدخول: `{r['price']:.5g}`")
        if r.get("sl") is not None:
            lines.append(f"  الوقف: `{r['sl']:.5g}`")
        if r.get("tp1") is not None:
            lines.append(f"  الهدف 1: `{r['tp1']:.5g}`")
        if r.get("tp2") is not None:
            lines.append(f"  الهدف 2: `{r['tp2']:.5g}`")
        lines.append("")

    lines.append("🎯 *الجودة:*")
    score = r.get("score", 0)
    stars = "⭐" * min(3, max(0, score // 30))
    lines.append(f"  {score}/100 {stars}")
    lines.append("")

    lines.append("🧭 *حالة السوق:*")
    lines.append(f"  الحالة: {r.get('regime', '—')}")
    lines.append(f"  ADX: `{r.get('adx', 0):.1f}`")
    bias_map = {"UP": "صاعد 🟢", "DOWN": "هابط 🔴", "NEUTRAL": "محايد ⚪"}
    lines.append(f"  اليومي: {bias_map.get(r.get('bias'), '—')}")
    lines.append(f"  RSI: `{r.get('rsi', 0):.1f}`")
    lines.append("")

    if r.get("arb"):
        lines.append("🎯 *القرار النهائي:*")
        lines.append(f"  {r['arb'].get('action', '—')}")
        for n in r["arb"].get("notes", [])[:3]:
            lines.append(f"  • {n}")
        lines.append("")

    lines.append("⚠️ لأغراض تعليمية فقط.")
    return "\n".join(lines)


def scan_symbols(symbols) -> list:
    """فحص سريع لكل الرموز."""
    results = []
    for s in symbols:
        try:
            r = analyze_symbol(s)
            results.append(r)
        except Exception as e:
            results.append({"symbol": s, "error": f"{type(e).__name__}: {str(e)[:100]}"})
    return results


def format_scan_summary(results: list) -> str:
    """ملخص الفحص."""
    lines = [f"🔍 *فحص {len(results)} أصول*\n"]
    for r in results:
        s = r["symbol"]
        if r.get("error"):
            lines.append(f"❌ `{s}`: {r['error'][:80]}")
            continue
        sig = r.get("signal", "No Signal")
        score = r.get("score", 0)
        emoji = {"BUY": "🟢", "SELL": "🔴"}.get(sig, "⚪")
        lines.append(f"{emoji} `{s}`: {sig} | جودة {score}/100")
    return "\n".join(lines)


# =====================================================================
# الصفقات الورقية
# =====================================================================
def open_paper_trade(symbol: str, direction: str) -> dict:
    A = _load_app_logic()
    r = analyze_symbol(symbol)
    if r.get("error"):
        return {"error": r["error"]}
    if r.get("signal") not in ("BUY", "SELL"):
        return {"error": "لا توجد إشارة دخول حالياً."}

    price = r["price"]
    atr = r["atr"]
    sl = r.get("sl")
    tp = r.get("tp2") or r.get("tp1")
    if sl is None or tp is None:
        return {"error": "لم يتم حساب مستويات الوقف والهدف."}

    data = _load_paper_trades()
    trade = {
        "id": data["next_id"],
        "ticker": symbol,
        "direction": direction,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "opened": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signal": r["signal"],
        "score": r["score"],
    }
    data["open"].append(trade)
    data["next_id"] += 1
    _save_paper_trades(data)
    return {"position": trade}


def close_paper_trade(trade_id: int) -> dict:
    data = _load_paper_trades()
    trade = next((t for t in data["open"] if t["id"] == trade_id), None)
    if not trade:
        return {"error": f"الصفقة #{trade_id} غير موجودة."}

    # سعر الإغلاق = آخر سعر معروف
    try:
        import yfinance as yf
        h = yf.Ticker(trade["ticker"]).history(period="1d", interval="1m")
        exit_price = float(h["Close"].iloc[-1]) if not h.empty else trade["entry"]
    except Exception:
        exit_price = trade["entry"]

    if trade["direction"] == "BUY":
        pnl = (exit_price - trade["entry"]) / trade["entry"] * 100
    else:
        pnl = (trade["entry"] - exit_price) / trade["entry"] * 100

    trade["exit_price"] = exit_price
    trade["closed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trade["pnl_pct"] = pnl

    data["open"].remove(trade)
    data["closed"].append(trade)
    _save_paper_trades(data)
    return {"exit_price": exit_price, "pnl_pct": pnl}


def get_open_positions() -> list:
    data = _load_paper_trades()
    out = []
    for t in data["open"]:
        try:
            import yfinance as yf
            h = yf.Ticker(t["ticker"]).history(period="1d", interval="1m")
            cur = float(h["Close"].iloc[-1]) if not h.empty else t["entry"]
        except Exception:
            cur = t["entry"]
        if t["direction"] == "BUY":
            pnl = (cur - t["entry"]) / t["entry"] * 100
        else:
            pnl = (t["entry"] - cur) / t["entry"] * 100
        p = dict(t)
        p["pnl_pct"] = pnl
        p["current_price"] = cur
        out.append(p)
    return out


def get_stats() -> dict:
    data = _load_paper_trades()
    closed = data.get("closed", [])
    if not closed:
        return {"total": 0, "win_rate": 0, "total_pnl": 0, "avg_r": 0,
                "pf": 0, "best": 0, "worst": 0}
    pnls = [t.get("pnl_pct", 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    return {
        "total": len(closed),
        "win_rate": len(wins) / len(closed) * 100,
        "total_pnl": sum(pnls),
        "avg_r": sum(pnls) / len(closed) / 2,  # تقديري
        "pf": gw / gl if gl > 0 else float("inf"),
        "best": max(pnls),
        "worst": min(pnls),
    }


# =====================================================================
# Telegram
# =====================================================================
def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except Exception as e:
        print(f"⚠️ send error: {e}")
        return False


def send_telegram_chat_action(action: str = "typing"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "action": action}).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass