"""
gold_engine.py — محرك الاستراتيجيات الذهبية + مستويات ذكية + Backtest مع TP1/نقل الوقف.
يُستورد من التطبيق الأصلي. يعتمد على أعمدة calculate_indicators الموجودة
(EMA20/50/200, ATR14, ADX14, RSI14, BB_Upper/Lower, Donchian_Upper/Lower).
"""
import numpy as np
import pandas as pd

# أسماء الاستراتيجيات الجديدة
SQUEEZE = "Squeeze Breakout (انفجار بعد الانضغاط)"
TURTLE = "Turtle 55 (اختراق السلاحف)"
RSI2 = "RSI2 Reversion (ارتداد كونورز)"
CONFLUENCE = "Confluence (تلاقي الاستراتيجيات)"
NEW_STRATEGIES = [CONFLUENCE, SQUEEZE, TURTLE, RSI2]

SL_MIN_ATR, SL_MAX_ATR, SL_BUFFER_ATR = 1.0, 2.5, 0.25
TP1_R, TP2_R = 1.5, 3.0
MIN_RR_TP1 = 1.0


def _safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- مؤشرات إضافية
def add_extra_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["SMA5"] = c.rolling(5).mean()
    df["SMA20"] = c.rolling(20).mean()
    df["SMA200"] = c.rolling(200).mean()

    # RSI(2)
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    df["RSI2"] = 100 - 100 / (1 + g / l.replace(0, np.nan))

    # Keltner + Squeeze
    kel_u = df["EMA20"] + 1.5 * df["ATR14"]
    kel_l = df["EMA20"] - 1.5 * df["ATR14"]
    df["SQUEEZE_ON"] = (df["BB_Upper"] < kel_u) & (df["BB_Lower"] > kel_l)

    # Donchian 55 / 10
    df["Don55_Up"] = df["High"].rolling(55).max()
    df["Don55_Dn"] = df["Low"].rolling(55).min()
    df["Don10_Up"] = df["High"].rolling(10).max()
    df["Don10_Dn"] = df["Low"].rolling(10).min()

    # MACD histogram
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()

    # قمم/قيعان قريبة لوضع الوقف
    df["Low5"] = df["Low"].rolling(5).min()
    df["High5"] = df["High"].rolling(5).max()
    return df


# ---------------------------------------------------------------- الإشارات
def extra_signals(df, base_sigs):
    """base_sigs = خرج strategy_signals الأصلي. يرجع قاموساً بالإشارات الجديدة (1/-1/0)."""
    c = df["Close"]

    def to_sig(buy, sell):
        return pd.Series(np.select([buy.fillna(False), sell.fillna(False)], [1, -1], default=0), index=df.index)

    # انفجار بعد انضغاط
    was_sq = df["SQUEEZE_ON"].astype(float).shift(1).rolling(5).max().fillna(0).astype(bool)
    squeeze = to_sig(
        was_sq & (c > df["BB_Upper"]) & (df["MACD_H"] > 0) & (c > df["EMA50"]),
        was_sq & (c < df["BB_Lower"]) & (df["MACD_H"] < 0) & (c < df["EMA50"]))

    # السلاحف: اختراق 55 + فلتر EMA200 + ADX > 20
    turtle = to_sig(
        (c > df["Don55_Up"].shift(1)) & (c > df["EMA200"]) & (df["ADX14"] > 20),
        (c < df["Don55_Dn"].shift(1)) & (c < df["EMA200"]) & (df["ADX14"] > 20))

    # RSI2: شراء الهبوط الحاد داخل اتجاه صاعد (فوق SMA200) والعكس
    rsi2 = to_sig(
        (c > df["SMA200"]) & (df["RSI2"] < 10) & (c < df["SMA5"]),
        (c < df["SMA200"]) & (df["RSI2"] > 90) & (c > df["SMA5"]))

    # التلاقي: استخدم فقط الاستراتيجيات الأساسية الأربع + الإشارات الجديدة الثلاث
    base_keys = [k for k in (
        "Breakout (اختراق)",
        "Trend + Pullback (اتجاه+ارتداد)",
        "Mean Reversion (عودة للمتوسط)",
        "Trend Follow (تتبع الاتجاه)",
    ) if k in base_sigs]
    votes = [base_sigs[k] for k in base_keys] + [squeeze, turtle, rsi2]
    allv = pd.concat(votes, axis=1)
    ups, dns = (allv == 1).sum(axis=1), (allv == -1).sum(axis=1)
    conf = pd.Series(np.select([(ups >= 2) & (dns == 0), (dns >= 2) & (ups == 0)], [1, -1], default=0),
                     index=df.index)
    return {SQUEEZE: squeeze, TURTLE: turtle, RSI2: rsi2, CONFLUENCE: conf}


# ---------------------------------------------------------------- مستويات ذكية
def smart_levels(signal, entry, df, i=-1, mean_rev=False):
    """Return deterministic SL/TP levels from completed-bar data only."""
    if signal not in ("BUY", "SELL"):
        return {"sl": np.nan, "tp1": np.nan, "tp2": np.nan, "risk_dist": np.nan,
                "rr1": np.nan, "rr2": np.nan, "valid": False}
    row = df.iloc[i]
    atr = _safe_float(row.get("ATR14"))
    entry = _safe_float(entry)
    low5 = _safe_float(row.get("Low5"))
    high5 = _safe_float(row.get("High5"))
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or not np.isfinite(low5) or not np.isfinite(high5):
        return {"sl": np.nan, "tp1": np.nan, "tp2": np.nan, "risk_dist": np.nan,
                "rr1": np.nan, "rr2": np.nan, "valid": False}

    s = 1 if signal == "BUY" else -1
    swing = low5 - SL_BUFFER_ATR * atr if s == 1 else high5 + SL_BUFFER_ATR * atr
    sd = min(max(abs(entry - swing), SL_MIN_ATR * atr), SL_MAX_ATR * atr)
    if not np.isfinite(sd) or sd <= 0:
        return {"sl": np.nan, "tp1": np.nan, "tp2": np.nan, "risk_dist": np.nan,
                "rr1": np.nan, "rr2": np.nan, "valid": False}

    sl = entry - s * sd
    tp1 = entry + s * TP1_R * sd
    tp2 = entry + s * TP2_R * sd

    if mean_rev:
        mean = _safe_float(row.get("SMA20"))
        if np.isfinite(mean) and s * (mean - entry) >= sd * MIN_RR_TP1:
            tp1 = mean
        tp2 = entry + s * 2.5 * sd
        if s * (tp2 - tp1) <= 0:
            tp2 = tp1 + s * 0.5 * sd

    rr1 = abs(tp1 - entry) / sd
    rr2 = abs(tp2 - entry) / sd
    return {"sl": sl, "tp1": tp1, "tp2": tp2, "risk_dist": sd,
            "rr1": rr1, "rr2": rr2, "valid": bool(rr1 >= MIN_RR_TP1)}


# ---------------------------------------------------------------- Backtest v2
def run_backtest_v2(df, signals, cost_pct=0.05, max_hold=30, start=200, mean_rev=False, partial=0.5):
    """
    Backtest with next-bar entry, smart SL, TP1 partial close and BE.

    ``pnl`` is net of the configured percentage cost. ``r`` is calculated from
    the gross trade result relative to the initial price risk.
    """
    required = ["Open", "High", "Low", "Close", "ATR14", "Low5", "High5", "SMA20"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"أعمدة الباكتيست مفقودة: {missing}")
    if len(df) == 0 or len(signals) != len(df):
        return pd.DataFrame()
    partial = min(max(float(partial), 0.0), 1.0)
    cost_pct = max(float(cost_pct), 0.0)
    max_hold = max(int(max_hold), 1)
    start = max(int(start), 1)

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    atr = df["ATR14"].to_numpy(dtype=float)
    low5 = df["Low5"].to_numpy(dtype=float)
    high5 = df["High5"].to_numpy(dtype=float)
    sma20 = df["SMA20"].to_numpy(dtype=float)
    sig = pd.Series(signals, index=df.index).fillna(0).to_numpy(dtype=int)
    n = len(df)
    trades, i = [], min(start, n - 1)

    while i < n - 1:
        s = int(sig[i])
        if s not in (-1, 1) or not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(low5[i]) or not np.isfinite(high5[i]):
            i += 1
            continue

        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        swing = low5[i] - SL_BUFFER_ATR * atr[i] if s == 1 else high5[i] + SL_BUFFER_ATR * atr[i]
        sd = min(max(abs(entry - swing), SL_MIN_ATR * atr[i]), SL_MAX_ATR * atr[i])
        if not np.isfinite(sd) or sd <= 0:
            i += 1
            continue

        sl = entry - s * sd
        tp1 = entry + s * TP1_R * sd
        tp2 = entry + s * TP2_R * sd
        if mean_rev and np.isfinite(sma20[i]):
            if s * (sma20[i] - entry) >= sd * MIN_RR_TP1:
                tp1 = sma20[i]
            tp2 = entry + s * 2.5 * sd
            if s * (tp2 - tp1) <= 0:
                tp2 = tp1 + s * 0.5 * sd
        if abs(tp1 - entry) / sd < MIN_RR_TP1:
            i += 1
            continue

        stage = 0
        realized_gross = 0.0
        reason = "Time"
        exit_price = c[min(i + max_hold, n - 1)]
        exit_idx = min(i + max_hold, n - 1)

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            hit_sl = (l[j] <= sl) if s == 1 else (h[j] >= sl)
            if hit_sl:
                exit_price = min(sl, o[j]) if s == 1 else max(sl, o[j])
                reason = "BE" if stage == 1 else "SL"
                exit_idx = j
                break

            hit_tp1 = (h[j] >= tp1) if s == 1 else (l[j] <= tp1)
            if stage == 0 and hit_tp1:
                realized_gross = partial * s * (tp1 - entry) / entry * 100.0
                stage = 1
                sl = entry

            hit_tp2 = (h[j] >= tp2) if s == 1 else (l[j] <= tp2)
            if stage == 1 and hit_tp2:
                exit_price, reason, exit_idx = tp2, "TP", j
                break
        else:
            exit_idx = min(i + max_hold, n - 1)
            exit_price = c[exit_idx]

        rem = (1.0 - partial) if stage == 1 else 1.0
        remaining_gross = rem * s * (exit_price - entry) / entry * 100.0
        gross_pnl = realized_gross + remaining_gross
        net_pnl = gross_pnl - cost_pct
        risk_pct = sd / entry * 100.0
        r_multiple = gross_pnl / risk_pct if risk_pct > 0 else np.nan
        trades.append({
            "entry_date": df.index[i + 1], "exit_date": df.index[exit_idx],
            "direction": "BUY" if s == 1 else "SELL", "entry": entry,
            "exit": exit_price, "reason": reason, "pnl": net_pnl,
            "gross_pnl": gross_pnl, "cost_pct": cost_pct, "r": r_multiple,
            "bars": exit_idx - i,
        })
        i = exit_idx

    return pd.DataFrame(trades)