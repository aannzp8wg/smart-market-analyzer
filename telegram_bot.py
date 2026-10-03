#!/usr/bin/env python3
"""
telegram_bot.py — بوت Telegram يتحكم بالتطبيق بالكامل.

يعمل على GitHub Actions كل 5 دقائق.
يقرأ الرسائل الجديدة، يعالجها، يرسل الردود.
الحالة محفوظة في Gist (bot_state.json).
"""
import os
import sys
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bot_helpers import (
    load_bot_state, save_bot_state,
    analyze_symbol, format_analysis_for_telegram,
    scan_symbols, format_scan_summary,
    open_paper_trade, close_paper_trade, get_open_positions, get_stats,
    load_config, save_config,
    send_telegram_message, send_telegram_chat_action,
)

# ---------------------------------------------------------------------
# الاستعدادات
# ---------------------------------------------------------------------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if not TOKEN or not CHAT_ID:
    print("❌ TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID غير مضبوط.")
    sys.exit(0)


# ---------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------
def get_updates(offset: int = 0, timeout: int = 0):
    """جلب الرسائل الجديدة من Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"offset": offset + 1, "timeout": timeout, "allowed_updates": json.dumps(["message"])}
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return data.get("result", []) if data.get("ok") else []
    except Exception as e:
        print(f"⚠️ getUpdates error: {e}")
        return []


# ---------------------------------------------------------------------
# الأوامر
# ---------------------------------------------------------------------
HELP_TEXT = """🤖 *محلل السوق الذكي* — الأوامر:

*التحليل:*
`/analyze GC=F` — تحليل أصل واحد (ذهبي، بيتكوين، يورو...)
`/scan` — فحص كل الأصول في القائمة

*الصفقات:*
`/open GC=F BUY` — فتح صفقة ورقية
`/status` — الصفقات المفتوحة
`/close 1` — إغلاق الصفقة رقم 1
`/stats` — الإحصاءات

*الإعدادات:*
`/settings` — عرض الإعدادات
`/threshold 80` — تغيير حد الجودة
`/symbols GC=F,BTC-USD` — تغيير قائمة الأصول

*أخرى:*
`/help` — قائمة الأوامر
`/start` — رسالة ترحيبية

⚠️ لأغراض تعليمية فقط.
"""

WELCOME_TEXT = """👋 مرحباً بك في *محلل السوق الذكي*!

اكتب `/help` لعرض الأوامر المتاحة.

🎯 *جرب الآن:*
`/analyze GC=F` — لتحليل الذهب
`/scan` — لفحص كل الأصول

⚠️ لأغراض تعليمية فقط، ليست نصيحة مالية.
"""


def cmd_analyze(args):
    """`/analyze GC=F`"""
    if not args:
        return "❌ مثال: `/analyze GC=F`"
    symbol = args[0].strip().upper()
    send_telegram_chat_action("typing")
    try:
        result = analyze_symbol(symbol)
        return format_analysis_for_telegram(result)
    except Exception as e:
        return f"❌ خطأ أثناء التحليل: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_scan(args):
    """`/scan`"""
    send_telegram_chat_action("typing")
    try:
        cfg = load_config()
        symbols = cfg.get("symbols", ["GC=F", "EURUSD=X", "BTC-USD"])
        results = scan_symbols(symbols)
        return format_scan_summary(results)
    except Exception as e:
        return f"❌ خطأ في الفحص: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_status(args):
    """`/status`"""
    try:
        positions = get_open_positions()
        if not positions:
            return "📭 لا توجد صفقات مفتوحة."
        lines = ["📊 *الصفقات المفتوحة:*\n"]
        for p in positions:
            emoji = "🟢" if p["direction"] == "BUY" else "🔴"
            pnl = p.get("pnl_pct", 0)
            lines.append(
                f"{emoji} *{p['ticker']}* | {p['direction']}\n"
                f"  دخول: `{p['entry']:.5g}` | SL: `{p['sl']:.5g}` | TP: `{p['tp']:.5g}`\n"
                f"  الربح الحالي: `{pnl:+.2f}%`\n"
                f"  ID: `{p['id']}`"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_open(args):
    """`/open GC=F BUY`"""
    if len(args) < 2:
        return "❌ مثال: `/open GC=F BUY`"
    symbol = args[0].strip().upper()
    direction = args[1].strip().upper()
    if direction not in ("BUY", "SELL"):
        return "❌ الاتجاه يجب أن يكون `BUY` أو `SELL`."
    try:
        result = open_paper_trade(symbol, direction)
        if result.get("error"):
            return f"❌ {result['error']}"
        p = result["position"]
        emoji = "🟢" if direction == "BUY" else "🔴"
        return (
            f"{emoji} *تم فتح صفقة ورقية*\n\n"
            f"الأصل: `{symbol}`\n"
            f"الاتجاه: `{direction}`\n"
            f"الدخول: `{p['entry']:.5g}`\n"
            f"وقف الخسارة: `{p['sl']:.5g}`\n"
            f"الهدف: `{p['tp']:.5g}`\n"
            f"ID: `{p['id']}`"
        )
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_close(args):
    """`/close 1`"""
    if not args:
        return "❌ مثال: `/close 1` (استخدم `/status` لمعرفة الأرقام)"
    try:
        trade_id = int(args[0])
        result = close_paper_trade(trade_id)
        if result.get("error"):
            return f"❌ {result['error']}"
        pnl = result["pnl_pct"]
        emoji = "✅" if pnl > 0 else "❌"
        return (
            f"{emoji} *تم إغلاق الصفقة #{trade_id}*\n\n"
            f"النتيجة: `{pnl:+.2f}%`\n"
            f"سعر الخروج: `{result['exit_price']:.5g}`"
        )
    except ValueError:
        return "❌ الرقم يجب أن يكون صحيحاً."
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_stats(args):
    """`/stats`"""
    try:
        stats = get_stats()
        if stats["total"] == 0:
            return "📊 لا توجد صفقات مغلقة بعد."
        return (
            f"📊 *إحصاءاتك:*\n\n"
            f"عدد الصفقات: `{stats['total']}`\n"
            f"نسبة الربح: `{stats['win_rate']:.1f}%`\n"
            f"إجمالي الربح: `{stats['total_pnl']:+.2f}%`\n"
            f"متوسط R: `{stats['avg_r']:+.2f}`\n"
            f"Profit Factor: `{stats['pf']:.2f}`\n"
            f"أفضل صفقة: `{stats['best']:+.2f}%`\n"
            f"أسوأ صفقة: `{stats['worst']:+.2f}%`"
        )
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_settings(args):
    """`/settings`"""
    try:
        cfg = load_config()
        return (
            f"⚙️ *الإعدادات الحالية:*\n\n"
            f"الأصول: `{', '.join(cfg['symbols'])}`\n"
            f"الإطار: `{cfg.get('timeframe', 'dual')}`\n"
            f"الاستراتيجية: `{cfg.get('strategy', 'auto')}`\n"
            f"حد الجودة: `{cfg.get('threshold', 85)}`\n"
            f"رأس المال: `{cfg.get('balance', 10000)}`\n"
            f"المخاطرة: `{cfg.get('risk_pct', 1.0)}%`"
        )
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_threshold(args):
    """`/threshold 80`"""
    if not args:
        return "❌ مثال: `/threshold 80`"
    try:
        value = int(args[0])
        if not 40 <= value <= 95:
            return "❌ القيمة يجب أن تكون بين 40 و 95."
        cfg = load_config()
        cfg["threshold"] = value
        save_config(cfg)
        return f"✅ تم تحديث حد الجودة إلى `{value}`."
    except ValueError:
        return "❌ الرقم غير صحيح."
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


def cmd_symbols(args):
    """`/symbols GC=F,BTC-USD`"""
    if not args:
        return "❌ مثال: `/symbols GC=F,BTC-USD,EURUSD=X`"
    try:
        symbols = [s.strip().upper() for s in args[0].split(",") if s.strip()]
        if not symbols:
            return "❌ قائمة فارغة."
        if len(symbols) > 10:
            return "❌ الحد الأقصى 10 رموز."
        cfg = load_config()
        cfg["symbols"] = symbols
        save_config(cfg)
        return f"✅ تم تحديث الأصول إلى: `{', '.join(symbols)}`"
    except Exception as e:
        return f"❌ خطأ: `{type(e).__name__}: {str(e)[:200]}`"


COMMANDS = {
    "/start": lambda args: WELCOME_TEXT,
    "/help": lambda args: HELP_TEXT,
    "/analyze": cmd_analyze,
    "/scan": cmd_scan,
    "/status": cmd_status,
    "/positions": cmd_status,
    "/open": cmd_open,
    "/close": cmd_close,
    "/stats": cmd_stats,
    "/settings": cmd_settings,
    "/threshold": cmd_threshold,
    "/symbols": cmd_symbols,
}


def process_command(text: str) -> str:
    """معالجة الرسالة وإرجاع الرد."""
    if not text:
        return ""
    parts = text.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in COMMANDS:
        try:
            return COMMANDS[cmd](args)
        except Exception as e:
            return f"❌ خطأ غير متوقع: `{type(e).__name__}: {str(e)[:200]}`"

    return (
        "❓ أمر غير معروف.\n\n"
        "اكتب `/help` لعرض الأوامر المتاحة."
    )


# ---------------------------------------------------------------------
# الحلقة الرئيسية
# ---------------------------------------------------------------------
def main():
    state = load_bot_state()
    offset = int(state.get("last_update_id", 0) or 0)
    print(f"📨 جلب الرسائل من offset={offset}...")

    updates = get_updates(offset)
    if not updates:
        print("✅ لا رسائل جديدة.")
        return

    print(f"📩 وصلت {len(updates)} رسالة.")

    for upd in updates:
        msg = upd.get("message", {})
        text = msg.get("text", "")
        chat_id = str(msg.get("chat", {}).get("id", ""))
        from_user = msg.get("from", {}).get("first_name", "")
        update_id = int(upd.get("update_id", 0))

        # فقط من المستخدم المصرح
        if chat_id != CHAT_ID:
            print(f"⏭️ تجاهل رسالة من chat_id={chat_id} (غير مصرح).")
            offset = max(offset, update_id)
            continue

        print(f"👤 {from_user}: {text}")

        reply = process_command(text)
        if reply:
            send_telegram_message(reply)
            print(f"✅ تم الرد على {from_user}.")

        offset = max(offset, update_id)

    state["last_update_id"] = offset
    state["last_run"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_bot_state(state)
    print(f"💾 تم حفظ آخر offset: {offset}")


if __name__ == "__main__":
    main()