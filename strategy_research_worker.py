import argparse
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


APP_TIMEZONE = ZoneInfo("America/Los_Angeles")
STRATEGY_REGISTRY_FILE = "strategy_registry.json"
BACKTEST_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "IWM", "DIA"]
STARTING_EQUITY = 10000.0
APPROVED_EMA_SHORT = [10, 12, 20]
APPROVED_EMA_LONG = [34, 50, 60]
APPROVED_SCORE_THRESHOLDS = [60, 65, 70, 75]
APPROVED_RSI_LONG = [53, 55, 58, 60]
APPROVED_RSI_SHORT = [47, 45, 42, 40]
APPROVED_REL_VOL = [1.4, 1.5, 1.6, 1.8]
APPROVED_STOP_MULTIPLIERS = [1.0, 1.1, 1.2]
APPROVED_TP1_MULTIPLIERS = [1.3, 1.5, 1.7]
APPROVED_TP2_MULTIPLIERS = [2.0, 2.3, 2.6]
APPROVED_TREND_WEIGHTS = [24, 30, 34]
APPROVED_MOMENTUM_WEIGHTS = [14, 18, 22]
APPROVED_VOLUME_WEIGHTS = [10, 12, 16]
APPROVED_STRUCTURE_WEIGHTS = [12, 18, 24]
APPROVED_RR_WEIGHTS = [10, 20, 24]
MIN_PROMOTION_TRADES = 30
MAX_DRAWDOWN_DEGRADATION = 0.10
RESEARCH_LOOP_MINUTES = 20


def now_str():
    return datetime.now(APP_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def default_strategy_parameters():
    return {
        "score_threshold": 65,
        "rsi_long_min": 53,
        "rsi_short_max": 47,
        "rel_vol_min": 1.4,
        "ema_short_len": 20,
        "ema_long_len": 50,
        "stop_multiplier": 1.0,
        "tp1_multiplier": 1.7,
        "tp2_multiplier": 2.6,
        "trend_weight": 34,
        "momentum_weight": 18,
        "volume_weight": 16,
        "structure_weight": 24,
        "rr_weight": 24,
    }


def sanitize_strategy_parameters(params):
    defaults = default_strategy_parameters()
    merged = {**defaults, **(params or {})}
    merged["score_threshold"] = int(merged["score_threshold"]) if int(merged["score_threshold"]) in APPROVED_SCORE_THRESHOLDS else defaults["score_threshold"]
    merged["rsi_long_min"] = int(merged["rsi_long_min"]) if int(merged["rsi_long_min"]) in APPROVED_RSI_LONG else defaults["rsi_long_min"]
    merged["rsi_short_max"] = int(merged["rsi_short_max"]) if int(merged["rsi_short_max"]) in APPROVED_RSI_SHORT else defaults["rsi_short_max"]
    merged["rel_vol_min"] = float(merged["rel_vol_min"]) if float(merged["rel_vol_min"]) in APPROVED_REL_VOL else defaults["rel_vol_min"]
    merged["ema_short_len"] = int(merged["ema_short_len"]) if int(merged["ema_short_len"]) in APPROVED_EMA_SHORT else defaults["ema_short_len"]
    merged["ema_long_len"] = int(merged["ema_long_len"]) if int(merged["ema_long_len"]) in APPROVED_EMA_LONG else defaults["ema_long_len"]
    merged["stop_multiplier"] = float(merged["stop_multiplier"]) if float(merged["stop_multiplier"]) in APPROVED_STOP_MULTIPLIERS else defaults["stop_multiplier"]
    merged["tp1_multiplier"] = float(merged["tp1_multiplier"]) if float(merged["tp1_multiplier"]) in APPROVED_TP1_MULTIPLIERS else defaults["tp1_multiplier"]
    merged["tp2_multiplier"] = float(merged["tp2_multiplier"]) if float(merged["tp2_multiplier"]) in APPROVED_TP2_MULTIPLIERS else defaults["tp2_multiplier"]
    merged["trend_weight"] = int(merged["trend_weight"]) if int(merged["trend_weight"]) in APPROVED_TREND_WEIGHTS else defaults["trend_weight"]
    merged["momentum_weight"] = int(merged["momentum_weight"]) if int(merged["momentum_weight"]) in APPROVED_MOMENTUM_WEIGHTS else defaults["momentum_weight"]
    merged["volume_weight"] = int(merged["volume_weight"]) if int(merged["volume_weight"]) in APPROVED_VOLUME_WEIGHTS else defaults["volume_weight"]
    merged["structure_weight"] = int(merged["structure_weight"]) if int(merged["structure_weight"]) in APPROVED_STRUCTURE_WEIGHTS else defaults["structure_weight"]
    merged["rr_weight"] = int(merged["rr_weight"]) if int(merged["rr_weight"]) in APPROVED_RR_WEIGHTS else defaults["rr_weight"]
    return merged


def strategy_to_label(strategy):
    return f"{strategy['id']} (v{strategy['version']})"


def load_strategy_registry():
    with open(STRATEGY_REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    registry.setdefault("last_promotion_at", "")
    registry.setdefault("last_rejection_reason", "")
    registry.setdefault("promotion_history", [])
    registry.setdefault("research_worker_status", "offline")
    registry.setdefault("research_worker_last_seen", "")
    registry.setdefault("experiments", [])
    registry.setdefault("experiment_index", len(registry.get("experiments", [])))
    return registry


def save_strategy_registry(registry):
    with open(STRATEGY_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def parse_strategy_datetime(value: str):
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is None:
        return parsed.tz_localize(APP_TIMEZONE)
    return parsed.tz_convert(APP_TIMEZONE)


def fetch_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()
    df = df[needed].copy().dropna()
    df.index = pd.to_datetime(df.index)
    return df


def get_market_bias(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 60:
        return "neutral"
    market_df = df.copy()
    market_df["ema50"] = market_df["Close"].ewm(span=50, adjust=False).mean()
    last = market_df.iloc[-1]
    prev = market_df.iloc[-2]
    last_close = float(last["Close"])
    ema50 = float(last["ema50"]) if pd.notna(last["ema50"]) else last_close
    prev_ema50 = float(prev["ema50"]) if pd.notna(prev["ema50"]) else ema50
    if last_close > ema50 and ema50 >= prev_ema50:
        return "bullish"
    if last_close < ema50 and ema50 <= prev_ema50:
        return "bearish"
    return "neutral"


def build_trade_reason(signal: str, trend_ok: bool, momentum_ok: bool, volume_ok: bool, structure_ok: bool, rr_ok: bool) -> str:
    parts = []
    if trend_ok:
        parts.append("Trend")
    if momentum_ok:
        parts.append("RSI")
    if volume_ok:
        parts.append("Volume")
    if structure_ok:
        parts.append("Structure")
    if rr_ok:
        parts.append("R/R")
    return " + ".join(parts) if parts else "Scanner"


def build_signal_snapshot(df: pd.DataFrame, symbol: str, market_bias: str = "neutral", strategy_params=None):
    if df.empty or len(df) < 60:
        return None

    params = sanitize_strategy_parameters(strategy_params)
    work_df = df.copy()
    work_df["ema_short"] = work_df["Close"].ewm(span=params["ema_short_len"], adjust=False).mean()
    work_df["ema_long"] = work_df["Close"].ewm(span=params["ema_long_len"], adjust=False).mean()
    work_df["vol_avg20"] = work_df["Volume"].rolling(20).mean()
    work_df["rel_vol"] = work_df["Volume"] / work_df["vol_avg20"]
    delta = work_df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    work_df["rsi14"] = 100 - (100 / (1 + rs))

    last = work_df.iloc[-1]
    prev = work_df.iloc[-2]
    recent_window = work_df.iloc[-21:-1].copy()
    if recent_window.empty:
        return None

    close_price = float(last["Close"])
    ema_short = float(last["ema_short"]) if pd.notna(last["ema_short"]) else close_price
    ema_long = float(last["ema_long"]) if pd.notna(last["ema_long"]) else close_price
    rel_vol = float(last["rel_vol"]) if pd.notna(last["rel_vol"]) else 0.0
    rsi14 = float(last["rsi14"]) if pd.notna(last["rsi14"]) else 50.0
    prev_rsi = float(prev["rsi14"]) if pd.notna(prev["rsi14"]) else rsi14
    change_pct = ((float(last["Close"]) - float(prev["Close"])) / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0.0
    recent_support = float(recent_window["Low"].min())
    recent_resistance = float(recent_window["High"].max())
    recent_range = max(recent_resistance - recent_support, close_price * 0.01)
    ema_distance_pct = abs(close_price - ema_long) / ema_long if ema_long else 0.0
    structure_buffer = max(close_price * 0.0015, recent_range * 0.05, 0.05)
    confirmed_breakout = float(last["High"]) > recent_resistance and close_price > recent_resistance
    confirmed_breakdown = float(last["Low"]) < recent_support and close_price < recent_support

    long_score = 0
    short_score = 0
    trend_up = ema_short > ema_long and close_price > ema_long and ema_distance_pct >= 0.005
    trend_down = ema_short < ema_long and close_price < ema_long and ema_distance_pct >= 0.005
    if trend_up:
        long_score += params["trend_weight"]
    if trend_down:
        short_score += params["trend_weight"]
    if params["rsi_long_min"] <= rsi14 <= 72 and rsi14 >= prev_rsi:
        long_score += params["momentum_weight"]
    if 28 <= rsi14 <= params["rsi_short_max"] and rsi14 <= prev_rsi:
        short_score += params["momentum_weight"]
    if rel_vol >= 1.8:
        long_score += params["volume_weight"] + 8
        short_score += params["volume_weight"] + 8
    elif rel_vol >= params["rel_vol_min"]:
        long_score += params["volume_weight"]
        short_score += params["volume_weight"]
    if confirmed_breakout:
        long_score += params["structure_weight"]
    if confirmed_breakdown:
        short_score += params["structure_weight"]
    if change_pct >= 0.6:
        long_score += 10
    elif change_pct >= 0.3:
        long_score += 5
    if change_pct <= -0.6:
        short_score += 10
    elif change_pct <= -0.3:
        short_score += 5

    long_stop_base = recent_support - structure_buffer
    short_stop_base = recent_resistance + structure_buffer
    long_risk = max(close_price - long_stop_base, close_price * 0.004, 0.25)
    short_risk = max(short_stop_base - close_price, close_price * 0.004, 0.25)
    long_projected_target = recent_resistance + (recent_range * 0.35)
    short_projected_target = recent_support - (recent_range * 0.35)
    long_reward = max(long_projected_target - close_price, 0.0)
    short_reward = max(close_price - short_projected_target, 0.0)
    long_rr = long_reward / long_risk if long_risk > 0 else 0.0
    short_rr = short_reward / short_risk if short_risk > 0 else 0.0
    if long_rr >= 2.5:
        long_score += params["rr_weight"]
    elif long_rr >= 2.0:
        long_score += 10
    if short_rr >= 2.5:
        short_score += params["rr_weight"]
    elif short_rr >= 2.0:
        short_score += 10

    allow_long = market_bias == "bullish"
    allow_short = market_bias == "bearish"
    long_valid = all([trend_up, rsi14 > params["rsi_long_min"], rel_vol >= params["rel_vol_min"], confirmed_breakout, long_rr >= 2.0, allow_long, long_score >= params["score_threshold"]])
    short_valid = all([trend_down, rsi14 < params["rsi_short_max"], rel_vol >= params["rel_vol_min"], confirmed_breakdown, short_rr >= 2.0, allow_short, short_score >= params["score_threshold"]])

    if long_valid and (not short_valid or long_score >= short_score):
        return {
            "signal": "LONG SETUP",
            "entry": round(close_price, 4),
            "stop_loss": round(long_stop_base, 4),
            "take_profit_1": round(close_price + max((long_risk * params["stop_multiplier"]) * params["tp1_multiplier"], recent_range * 0.2), 4),
            "take_profit_2": round(close_price + max((long_risk * params["stop_multiplier"]) * params["tp2_multiplier"], recent_range * 0.35), 4),
            "score": int(long_score),
            "reason": build_trade_reason("LONG SETUP", trend_up, True, rel_vol >= params["rel_vol_min"], confirmed_breakout, long_rr >= 2.0),
            "timeframe": "1d",
            "timestamp": str(work_df.index[-1]),
        }
    if short_valid:
        return {
            "signal": "SHORT SETUP",
            "entry": round(close_price, 4),
            "stop_loss": round(short_stop_base, 4),
            "take_profit_1": round(close_price - max((short_risk * params["stop_multiplier"]) * params["tp1_multiplier"], recent_range * 0.2), 4),
            "take_profit_2": round(close_price - max((short_risk * params["stop_multiplier"]) * params["tp2_multiplier"], recent_range * 0.35), 4),
            "score": int(short_score),
            "reason": build_trade_reason("SHORT SETUP", trend_down, True, rel_vol >= params["rel_vol_min"], confirmed_breakdown, short_rr >= 2.0),
            "timeframe": "1d",
            "timestamp": str(work_df.index[-1]),
        }
    return None


def create_paper_performance_curve(closed_trades_df: pd.DataFrame):
    if closed_trades_df is None or closed_trades_df.empty:
        trades = pd.DataFrame({"trade_num": [0], "pnl": [0.0]})
        trades["equity"] = STARTING_EQUITY
        trades["peak"] = STARTING_EQUITY
        trades["drawdown"] = 0.0
        return trades
    trades = closed_trades_df.copy().reset_index(drop=True)
    trades["trade_num"] = trades.index + 1
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["equity"] = STARTING_EQUITY + trades["pnl"].cumsum()
    trades["peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] - trades["peak"]
    return trades


def summarize_backtest_results(backtest_trades: pd.DataFrame):
    if backtest_trades is None or backtest_trades.empty:
        return {"total_pnl": 0.0, "win_rate": 0.0, "average_win": 0.0, "average_loss": 0.0, "max_drawdown": 0.0, "num_trades": 0, "learning_score": 0, "learning_penalty": 0}
    trades = backtest_trades.copy()
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["score"] = pd.to_numeric(trades.get("score", 0), errors="coerce").fillna(0.0)
    trades["result"] = trades.get("result", "").replace("", pd.NA)
    trades["result"] = trades["result"].fillna(trades["pnl"].apply(lambda pnl: "win" if pnl > 0 else "loss"))
    trades["reason"] = trades.get("reason", "").replace("", "Unspecified").fillna("Unspecified")
    trades["score_range"] = pd.cut(trades["score"], bins=[-0.1, 50, 70, float("inf")], labels=["0-50", "50-70", "70+"], include_lowest=True).astype(str)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    curve = create_paper_performance_curve(trades)
    reason_win_rates = trades.groupby("reason", dropna=False)["result"].apply(lambda s: round((s == "win").mean() * 100, 2)).to_dict()
    score_win_rates = trades.groupby("score_range", dropna=False)["result"].apply(lambda s: round((s == "win").mean() * 100, 2)).to_dict()
    pnl_by_symbol = trades.groupby("symbol", dropna=False)["pnl"].sum().round(2).to_dict()
    poor_reasons = sum(1 for value in reason_win_rates.values() if float(value) < 40.0)
    poor_score_ranges = sum(1 for value in score_win_rates.values() if value != "nan" and float(value) < 40.0)
    poor_symbols = sum(1 for value in pnl_by_symbol.values() if float(value) < 0.0)
    strong_reasons = sum(1 for value in reason_win_rates.values() if float(value) >= 55.0)
    strong_score_ranges = sum(1 for value in score_win_rates.values() if value != "nan" and float(value) >= 55.0)
    strong_symbols = sum(1 for value in pnl_by_symbol.values() if float(value) > 0.0)
    return {
        "total_pnl": round(float(trades["pnl"].sum()), 2),
        "win_rate": round(float((trades["pnl"] > 0).mean() * 100), 2),
        "average_win": round(float(wins["pnl"].mean()), 2) if not wins.empty else 0.0,
        "average_loss": round(float(losses["pnl"].mean()), 2) if not losses.empty else 0.0,
        "max_drawdown": round(float(curve["drawdown"].min()), 2) if not curve.empty else 0.0,
        "num_trades": int(len(trades)),
        "learning_penalty": poor_reasons + poor_score_ranges + poor_symbols,
        "learning_reward": strong_reasons + strong_score_ranges + strong_symbols,
        "learning_score": (strong_reasons + strong_score_ranges + strong_symbols) - (poor_reasons + poor_score_ranges + poor_symbols),
    }


def compare_strategy_results(champion_summary, challenger_summary):
    drawdown_limit = abs(float(champion_summary.get("max_drawdown", 0.0))) * (1 + MAX_DRAWDOWN_DEGRADATION)
    checks = [
        challenger_summary.get("num_trades", 0) >= MIN_PROMOTION_TRADES,
        challenger_summary.get("total_pnl", 0.0) > champion_summary.get("total_pnl", 0.0),
        challenger_summary.get("win_rate", 0.0) >= champion_summary.get("win_rate", 0.0),
        abs(float(challenger_summary.get("max_drawdown", 0.0))) <= max(drawdown_limit, 1.0),
        challenger_summary.get("learning_score", 0) >= champion_summary.get("learning_score", 0),
    ]
    return all(checks)


def generate_strategy_candidates(base_params):
    base = sanitize_strategy_parameters(base_params)
    candidate_specs = [
        {"score_threshold": 75},
        {"rsi_long_min": 55, "rsi_short_max": 45},
        {"rsi_long_min": 60, "rsi_short_max": 40},
        {"rel_vol_min": 1.5},
        {"rel_vol_min": 1.8},
        {"stop_multiplier": 1.1},
        {"stop_multiplier": 1.2},
        {"tp1_multiplier": 1.5, "tp2_multiplier": 2.3},
        {"tp1_multiplier": 1.7, "tp2_multiplier": 2.6},
        {"ema_short_len": 12, "ema_long_len": 34},
        {"ema_short_len": 10, "ema_long_len": 60},
    ]
    candidates = []
    seen = set()
    for spec in candidate_specs:
        params = sanitize_strategy_parameters({**base, **spec})
        key = json.dumps(params, sort_keys=True)
        if key == json.dumps(base, sort_keys=True) or key in seen:
            continue
        seen.add(key)
        candidates.append(params)
    return candidates


def run_strategy_backtest(symbols, params) -> pd.DataFrame:
    results = []
    market_history = fetch_history("SPY", period="3y", interval="1d")
    for symbol in symbols:
        history = fetch_history(symbol, period="3y", interval="1d")
        if history.empty or len(history) < 80:
            continue
        open_trade = None
        for idx in range(60, len(history)):
            current_bar = history.iloc[idx]
            current_time = history.index[idx]
            high_price = float(current_bar["High"])
            low_price = float(current_bar["Low"])
            if open_trade is not None:
                if open_trade["signal"] == "LONG SETUP":
                    if low_price <= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        status = "CLOSED LOSS"
                    elif high_price >= open_trade["take_profit_2"]:
                        exit_price = open_trade["take_profit_2"]
                        status = "CLOSED WIN"
                    else:
                        continue
                    pnl = round((exit_price - open_trade["entry"]) * open_trade["shares"], 2)
                else:
                    if high_price >= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        status = "CLOSED LOSS"
                    elif low_price <= open_trade["take_profit_2"]:
                        exit_price = open_trade["take_profit_2"]
                        status = "CLOSED WIN"
                    else:
                        continue
                    pnl = round((open_trade["entry"] - exit_price) * open_trade["shares"], 2)
                results.append({
                    "symbol": symbol,
                    "signal": open_trade["signal"],
                    "timestamp": open_trade["timestamp"],
                    "timeframe": open_trade["timeframe"],
                    "entry": open_trade["entry"],
                    "stop_loss": open_trade["stop_loss"],
                    "take_profit": open_trade["take_profit_2"],
                    "take_profit_1": open_trade["take_profit_1"],
                    "take_profit_2": open_trade["take_profit_2"],
                    "pnl": pnl,
                    "result": "win" if status == "CLOSED WIN" else "loss",
                    "status": status,
                    "score": open_trade["score"],
                    "reason": open_trade["reason"],
                    "close_time": current_time,
                })
                open_trade = None
                continue
            market_bias = get_market_bias(market_history.loc[:current_time])
            snapshot = build_signal_snapshot(history.iloc[: idx + 1], symbol, market_bias=market_bias, strategy_params=params)
            if not snapshot:
                continue
            open_trade = {
                "signal": snapshot["signal"],
                "timestamp": snapshot["timestamp"],
                "timeframe": snapshot["timeframe"],
                "entry": snapshot["entry"],
                "stop_loss": snapshot["stop_loss"],
                "take_profit_1": snapshot["take_profit_1"],
                "take_profit_2": snapshot["take_profit_2"],
                "score": snapshot["score"],
                "reason": snapshot["reason"],
                "shares": 100,
            }
    return pd.DataFrame(results)


def run_research_iteration():
    registry = load_strategy_registry()
    registry["research_worker_status"] = "running"
    registry["research_worker_last_seen"] = now_str()
    champion = registry["champion"]
    champion_params = sanitize_strategy_parameters(champion.get("parameters", {}))
    champion_summary = champion.get("results_summary", {})
    if not champion_summary:
        champion_trades = run_strategy_backtest(BACKTEST_SYMBOLS, champion_params)
        champion_summary = summarize_backtest_results(champion_trades)
        registry["champion"]["results_summary"] = champion_summary

    existing_keys = {json.dumps(exp.get("parameters", {}), sort_keys=True) for exp in registry.get("experiments", [])}
    next_candidate = None
    for params in generate_strategy_candidates(champion_params):
        key = json.dumps(params, sort_keys=True)
        if key not in existing_keys:
            next_candidate = params
            break

    registry["last_research_run"] = now_str()
    if next_candidate is None:
        registry["last_challenger_result"] = "No new approved challenger variation available"
        save_strategy_registry(registry)
        return registry

    challenger_trades = run_strategy_backtest(BACKTEST_SYMBOLS, next_candidate)
    challenger_summary = summarize_backtest_results(challenger_trades)
    eligible = compare_strategy_results(champion_summary, challenger_summary)
    experiment_version = len(registry.get("experiments", [])) + 1
    promotion_status = "Promotable" if eligible else "Rejected: Promotion gates not met"
    experiment_record = {
        "id": f"challenger-v{experiment_version}",
        "version": experiment_version,
        "status": "challenger",
        "parameters": next_candidate,
        "results_summary": challenger_summary,
        "promotion_status": promotion_status,
        "paper_probation_passed": eligible,
        "created_at": now_str(),
        "promotion_date": "",
        "promotion_checks": [],
        "last_tested_at": now_str(),
        "testing_status": "scheduled" if not eligible else "promotable",
        "latest_result_status": "Promotable" if eligible else "Rejected",
    }
    registry.setdefault("experiments", []).append(experiment_record)
    registry["challenger"] = experiment_record
    registry["last_challenger_result"] = promotion_status
    if eligible:
        registry["last_promotion_at"] = registry.get("last_promotion_at", "")
    else:
        registry["last_rejection_reason"] = "Promotion gates not met"
    save_strategy_registry(registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description="Continuous background strategy research worker")
    parser.add_argument("--once", action="store_true", help="Run one research iteration and exit")
    parser.add_argument("--sleep-seconds", type=int, default=RESEARCH_LOOP_MINUTES * 60)
    args = parser.parse_args()

    if args.once:
        run_research_iteration()
        return

    while True:
        try:
            run_research_iteration()
        except Exception:
            registry = load_strategy_registry()
            registry["research_worker_status"] = "error"
            registry["research_worker_last_seen"] = now_str()
            save_strategy_registry(registry)
        time.sleep(max(args.sleep_seconds, 60))


if __name__ == "__main__":
    main()
