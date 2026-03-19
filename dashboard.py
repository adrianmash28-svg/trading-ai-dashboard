import os
import smtplib
import time
import json
import hashlib
from datetime import datetime, timedelta
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="Mash Terminal", layout="wide")


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
EMAIL_SENDER = get_secret("EMAIL_SENDER", "")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD", "")
POLYGON_API_KEY = get_secret("POLYGON_API_KEY", "")
VERIZON_SMS_GATEWAY = "3109911161@vtext.com"

PAPER_TRADES_FILE = "paper_trades.csv"
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "QQQ", "SPY", "IWM", "DIA", "BTC-USD", "ETH-USD"]
BACKTEST_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "IWM", "DIA"]
STRATEGY_REGISTRY_FILE = "strategy_registry.json"
ALGO_UPDATE_STATE_FILE = "algo_update_state.json"
STARTING_EQUITY = 10000.0
RISK_PER_TRADE = 0.01
MAX_SIMULTANEOUS_TRADES = 3
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
APP_TIMEZONE = ZoneInfo("America/Los_Angeles")
RESEARCH_LOOP_HOURS = 4
MAX_ONLINE_SCORE_ADJUSTMENT = 6
NEWS_LOOKBACK_HOURS = 48


def get_openai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


client = get_openai_client()


def get_account_balance_from_trades(closed_trades_df: pd.DataFrame) -> float:
    if closed_trades_df is None or closed_trades_df.empty:
        return STARTING_EQUITY
    realized_pnl = pd.to_numeric(closed_trades_df["pnl"], errors="coerce").fillna(0.0).sum()
    return max(STARTING_EQUITY + float(realized_pnl), STARTING_EQUITY * 0.25)


def get_risk_fraction(score: float) -> float:
    score_value = float(pd.to_numeric(score, errors="coerce") or 0.0)
    if score_value >= 90:
        return 0.02
    if score_value >= 80:
        return 0.015
    return RISK_PER_TRADE


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


def default_strategy_registry():
    return {
        "champion": {
            "id": "champion-v1",
            "version": 1,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion",
            "paper_probation_passed": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_tested_at": "",
            "testing_status": "active",
            "latest_result_status": "Awaiting test",
        },
        "previous_champion": None,
        "challenger": None,
        "experiments": [],
        "experiment_index": 0,
        "last_research_run": "",
        "last_challenger_result": "",
    }


def ensure_strategy_registry():
    if not os.path.exists(STRATEGY_REGISTRY_FILE):
        with open(STRATEGY_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(default_strategy_registry(), f, indent=2)
        return

    try:
        with open(STRATEGY_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "champion" not in data:
            raise ValueError("Missing champion")
    except Exception:
        with open(STRATEGY_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(default_strategy_registry(), f, indent=2)


def load_strategy_registry():
    ensure_strategy_registry()
    with open(STRATEGY_REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    if registry["champion"].get("version", 1) < 2:
        registry["previous_champion"] = registry.get("champion")
        registry["champion"] = {
            "id": "champion-v2",
            "version": 2,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion tightened for higher-quality setups",
            "paper_probation_passed": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_strategy_registry(registry)
    if registry["champion"].get("version", 2) < 3:
        registry["previous_champion"] = registry.get("champion")
        registry["champion"] = {
            "id": "champion-v3",
            "version": 3,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion slightly loosened to surface more valid trades",
            "paper_probation_passed": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_strategy_registry(registry)
    if registry["champion"].get("version", 3) < 4:
        registry["previous_champion"] = registry.get("champion")
        registry["champion"] = {
            "id": "champion-v4",
            "version": 4,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion upgraded with stronger entry confirmation and trade quality filters",
            "paper_probation_passed": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_strategy_registry(registry)
    if registry["champion"].get("version", 4) < 5:
        registry["previous_champion"] = registry.get("champion")
        registry["champion"] = {
            "id": "champion-v5",
            "version": 5,
            "status": "champion",
            "parameters": default_strategy_parameters(),
            "results_summary": {},
            "promotion_status": "Live champion slightly loosened with debug-ready thresholds to surface valid trades",
            "paper_probation_passed": True,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_strategy_registry(registry)
    registry["champion"]["parameters"] = sanitize_strategy_parameters(registry["champion"].get("parameters", {}))
    registry["champion"] = normalize_strategy_record(registry["champion"], "active")
    if registry.get("challenger"):
        registry["challenger"]["parameters"] = sanitize_strategy_parameters(registry["challenger"].get("parameters", {}))
        registry["challenger"] = normalize_strategy_record(registry["challenger"], "scheduled")
    registry["experiments"] = [
        normalize_strategy_record(exp, "historical")
        for exp in registry.get("experiments", [])
    ]
    return registry


def save_strategy_registry(registry):
    with open(STRATEGY_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def default_algo_update_state():
    return {
        "last_sent_at": "",
        "last_schedule_slot": "",
        "last_strategy_signature": "",
        "last_strategy_signature_hash": "",
        "last_message": "",
        "last_pnl": 0.0,
        "last_win_rate": 0.0,
    }


def ensure_algo_update_state():
    if not os.path.exists(ALGO_UPDATE_STATE_FILE):
        with open(ALGO_UPDATE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_algo_update_state(), f, indent=2)
        return

    try:
        with open(ALGO_UPDATE_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "last_schedule_slot" not in data:
            raise ValueError("Missing algo update fields")
    except Exception:
        with open(ALGO_UPDATE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_algo_update_state(), f, indent=2)


def load_algo_update_state():
    ensure_algo_update_state()
    with open(ALGO_UPDATE_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_algo_update_state(state):
    with open(ALGO_UPDATE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def strategy_to_label(strategy):
    return f"{strategy['id']} (v{strategy['version']})"


def build_strategy_signature(strategy):
    signature_payload = {
        "strategy_id": strategy.get("id", ""),
        "version": strategy.get("version", ""),
        "parameters": sanitize_strategy_parameters(strategy.get("parameters", {})),
        "filters": {
            "market_bias_filter": True,
            "partial_take_profit": True,
            "breakeven_after_tp1": True,
            "risk_model": "1pct_account_risk",
        },
    }
    signature_text = json.dumps(signature_payload, sort_keys=True)
    signature_hash = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:12]
    return signature_text, signature_hash


def scheduled_algo_hours_for_day(dt_local: datetime):
    if dt_local.weekday() <= 4:
        return [8, 12, 16]
    return [12, 16, 20]


def get_latest_schedule_slot(dt_local: datetime):
    hours = scheduled_algo_hours_for_day(dt_local)
    slots = [
        dt_local.replace(hour=hour, minute=0, second=0, microsecond=0)
        for hour in hours
    ]
    eligible = [slot for slot in slots if slot <= dt_local]
    if eligible:
        return eligible[-1]

    previous_day = dt_local - timedelta(days=1)
    prev_hours = scheduled_algo_hours_for_day(previous_day)
    return previous_day.replace(hour=prev_hours[-1], minute=0, second=0, microsecond=0)


def get_next_schedule_slot(dt_local: datetime):
    for day_offset in range(0, 8):
        candidate_day = dt_local + timedelta(days=day_offset)
        hours = scheduled_algo_hours_for_day(candidate_day)
        slots = [
            candidate_day.replace(hour=hour, minute=0, second=0, microsecond=0)
            for hour in hours
        ]
        future_slots = [slot for slot in slots if slot > dt_local]
        if future_slots:
            return future_slots[0]
    return None


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
        {"rsi_long_min": 60, "rsi_short_max": 40, "rel_vol_min": 1.8},
        {"stop_multiplier": 1.1, "tp1_multiplier": 1.7, "tp2_multiplier": 2.6},
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


def send_sms_alert(message: str):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD]):
        return False, "Missing email SMS credentials"
    try:
        email_message = EmailMessage()
        email_message["From"] = EMAIL_SENDER
        email_message["To"] = VERIZON_SMS_GATEWAY
        email_message["Subject"] = "MashGPT Alert"
        email_message.set_content(str(message).strip()[:160])

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(email_message)
        return True, ""
    except Exception as e:
        return False, str(e).strip()

def get_polygon_last_trade(symbol: str):
    if not POLYGON_API_KEY:
        return None

    try:
        url = f"https://api.polygon.io/v2/last/trade/{symbol}"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=10)
        data = r.json()

        result = data.get("results")
        if not result:
            return None

        return {
            "price": result.get("p"),
            "size": result.get("s"),
            "timestamp": result.get("t"),
        }
    except Exception:
        return None


def get_polygon_prev_day(symbol: str):
    if not POLYGON_API_KEY:
        return None

    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        r = requests.get(
            url,
            params={"adjusted": "true", "apiKey": POLYGON_API_KEY},
            timeout=10,
        )
        data = r.json()

        results = data.get("results")
        if not results:
            return None

        row = results[0]
        return {
            "open": row.get("o"),
            "high": row.get("h"),
            "low": row.get("l"),
            "close": row.get("c"),
            "volume": row.get("v"),
        }
    except Exception:
        return None


def get_polygon_market_status():
    if not POLYGON_API_KEY:
        return None

    try:
        url = "https://api.polygon.io/v1/marketstatus/now"
        r = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=10)
        return r.json()
    except Exception:
        return None


def ensure_paper_trades_file():
    cols = [
        "symbol",
        "time",
        "timestamp",
        "timeframe",
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "shares",
        "risk_pct",
        "account_balance",
        "base_score",
        "score",
        "reason",
        "signal",
        "sentiment_score",
        "recent_news",
        "macro_regime",
        "event_caution",
        "online_score_adjustment",
        "news_affected",
        "status",
        "result",
        "pnl",
        "exit_price",
        "exit_reason",
        "close_time",
    ]
    if not os.path.exists(PAPER_TRADES_FILE):
        pd.DataFrame(columns=cols).to_csv(PAPER_TRADES_FILE, index=False)
        return

    try:
        df = pd.read_csv(PAPER_TRADES_FILE)
        changed = False
        for col in cols:
            if col not in df.columns:
                df[col] = ""
                changed = True
        if changed:
            df = df[cols]
            df.to_csv(PAPER_TRADES_FILE, index=False)
    except Exception:
        pd.DataFrame(columns=cols).to_csv(PAPER_TRADES_FILE, index=False)


def load_paper_trades():
    ensure_paper_trades_file()
    return pd.read_csv(PAPER_TRADES_FILE)


def save_paper_trades(df: pd.DataFrame):
    df.to_csv(PAPER_TRADES_FILE, index=False)


@st.cache_data(ttl=120)
def fetch_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

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
    if not parts:
        parts.append("Scanner")
    return " + ".join(parts)


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


@st.cache_data(ttl=900)
def get_polygon_news_context(symbol: str):
    fallback = {
        "sentiment_score": 0.0,
        "recent_news": "No",
        "event_caution": False,
        "event_caution_label": "None",
        "news_adjustment": 0,
        "news_affected": False,
    }
    if not POLYGON_API_KEY:
        return fallback

    normalized_symbol = str(symbol).upper()
    if "-" in normalized_symbol or normalized_symbol.startswith("^"):
        return fallback

    try:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=NEWS_LOOKBACK_HOURS)
        response = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker": normalized_symbol,
                "limit": 10,
                "order": "desc",
                "sort": "published_utc",
                "apiKey": POLYGON_API_KEY,
            },
            timeout=10,
        )
        payload = response.json() if response.ok else {}
        news_items = payload.get("results", []) or []
    except Exception:
        return fallback

    if not news_items:
        return fallback

    positive_terms = ["beat", "surge", "growth", "upside", "bullish", "strong", "expands", "record", "gain"]
    negative_terms = ["miss", "drop", "cut", "lawsuit", "downgrade", "weak", "bearish", "loss", "fall"]
    caution_terms = ["earnings", "fomc", "fed", "cpi", "inflation", "jobs report", "guidance", "sec", "investigation"]

    recent_items = []
    for item in news_items:
        published = pd.to_datetime(item.get("published_utc"), utc=True, errors="coerce")
        if pd.isna(published) or published < cutoff:
            continue
        recent_items.append(item)

    if not recent_items:
        return fallback

    sentiment_points = 0
    caution_hits = 0
    for item in recent_items[:6]:
        insights = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("article_url", "")),
                " ".join(str(keyword) for keyword in item.get("keywords", []) or []),
            ]
        ).lower()
        sentiment_points += sum(term in insights for term in positive_terms)
        sentiment_points -= sum(term in insights for term in negative_terms)
        caution_hits += sum(term in insights for term in caution_terms)

    sentiment_score = max(min(sentiment_points / 6.0, 1.0), -1.0)
    event_caution = caution_hits > 0
    news_adjustment = 0
    if sentiment_score >= 0.35:
        news_adjustment = 3
    elif sentiment_score <= -0.35:
        news_adjustment = -3
    if event_caution:
        news_adjustment -= 2 if news_adjustment >= 0 else 1
    news_adjustment = int(max(min(news_adjustment, 10), -10))

    return {
        "sentiment_score": round(sentiment_score, 2),
        "recent_news": "Yes",
        "event_caution": event_caution,
        "event_caution_label": "High" if event_caution else "None",
        "news_adjustment": news_adjustment,
        "news_affected": bool(news_adjustment or event_caution),
    }


@st.cache_data(ttl=900)
def get_macro_regime_context():
    try:
        spy = fetch_history("SPY", period="6mo", interval="1d")
        qqq = fetch_history("QQQ", period="6mo", interval="1d")
        iwm = fetch_history("IWM", period="6mo", interval="1d")
        vix = fetch_history("^VIX", period="6mo", interval="1d")
    except Exception:
        return {"macro_regime": "neutral", "macro_adjustment_long": 0, "macro_adjustment_short": 0}

    if spy.empty or qqq.empty or iwm.empty:
        return {"macro_regime": "neutral", "macro_adjustment_long": 0, "macro_adjustment_short": 0}

    def above_ema50(df: pd.DataFrame) -> bool:
        if df.empty or len(df) < 60:
            return False
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()
        return float(df["Close"].iloc[-1]) > float(ema50.iloc[-1])

    breadth_score = sum([above_ema50(spy), above_ema50(qqq), above_ema50(iwm)])
    vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else 20.0

    if breadth_score >= 2 and vix_last < 20:
        regime = "risk_on"
        long_adj, short_adj = 2, -2
    elif breadth_score <= 1 and vix_last > 24:
        regime = "risk_off"
        long_adj, short_adj = -2, 2
    else:
        regime = "neutral"
        long_adj, short_adj = 0, 0

    return {
        "macro_regime": regime,
        "macro_adjustment_long": long_adj,
        "macro_adjustment_short": short_adj,
    }


def build_signal_snapshot(
    df: pd.DataFrame,
    symbol: str,
    market_bias: str = "neutral",
    account_equity: float = STARTING_EQUITY,
    strategy_params=None,
):
    if df.empty or len(df) < 60:
        return None

    params = sanitize_strategy_parameters(strategy_params)
    news_context = get_polygon_news_context(symbol)
    macro_context = get_macro_regime_context()
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
    elif rsi14 > 72:
        long_score += 4
    if 28 <= rsi14 <= params["rsi_short_max"] and rsi14 <= prev_rsi:
        short_score += params["momentum_weight"]
    elif rsi14 < 28:
        short_score += 4

    if rel_vol >= 1.8:
        long_score += params["volume_weight"] + 8
        short_score += params["volume_weight"] + 8
    elif rel_vol >= params["rel_vol_min"]:
        long_score += params["volume_weight"]
        short_score += params["volume_weight"]

    breakout_level = recent_resistance
    breakdown_level = recent_support
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
    long_reason = build_trade_reason(
        "LONG SETUP",
        trend_up,
        rsi14 > params["rsi_long_min"],
        rel_vol >= params["rel_vol_min"],
        confirmed_breakout,
        long_rr >= 2.0,
    )
    short_reason = build_trade_reason(
        "SHORT SETUP",
        trend_down,
        rsi14 < params["rsi_short_max"],
        rel_vol >= params["rel_vol_min"],
        confirmed_breakdown,
        short_rr >= 2.0,
    )
    timeframe = "15m"
    snapshot_timestamp = str(work_df.index[-1])
    preferred_direction = "LONG" if long_score >= short_score else "SHORT"
    selected_score = long_score if preferred_direction == "LONG" else short_score
    selected_trend_pass = trend_up if preferred_direction == "LONG" else trend_down
    selected_rsi_pass = rsi14 > params["rsi_long_min"] if preferred_direction == "LONG" else rsi14 < params["rsi_short_max"]
    selected_volume_pass = rel_vol >= params["rel_vol_min"]
    selected_breakout_pass = confirmed_breakout if preferred_direction == "LONG" else confirmed_breakdown
    selected_rr = long_rr if preferred_direction == "LONG" else short_rr
    selected_rr_pass = selected_rr >= 2.0
    selected_score_pass = selected_score >= params["score_threshold"]
    selected_market_pass = allow_long if preferred_direction == "LONG" else allow_short

    rejection_reason = ""
    if not selected_trend_pass:
        rejection_reason = "failed trend filter"
    elif not selected_rsi_pass:
        rejection_reason = "failed RSI filter"
    elif not selected_volume_pass:
        rejection_reason = "failed volume filter"
    elif not selected_breakout_pass:
        rejection_reason = "failed breakout filter"
    elif not selected_rr_pass:
        rejection_reason = "failed reward/risk filter"
    elif not selected_score_pass:
        rejection_reason = "failed score filter"
    elif not selected_market_pass:
        rejection_reason = "failed market regime filter"

    if long_score >= short_score and long_score >= params["score_threshold"] and trend_up and rel_vol >= params["rel_vol_min"] and rsi14 > params["rsi_long_min"] and long_rr >= 2.0 and confirmed_breakout and allow_long:
        signal = "LONG SETUP"
        base_score = long_score
        risk = long_risk * params["stop_multiplier"]
        stop_loss = round(long_stop_base, 4)
        tp1 = round(close_price + max(risk * params["tp1_multiplier"], recent_range * 0.2), 4)
        tp2 = round(close_price + max(risk * params["tp2_multiplier"], recent_range * 0.35), 4)
        reason = long_reason
    elif short_score > long_score and short_score >= params["score_threshold"] and trend_down and rel_vol >= params["rel_vol_min"] and rsi14 < params["rsi_short_max"] and short_rr >= 2.0 and confirmed_breakdown and allow_short:
        signal = "SHORT SETUP"
        base_score = short_score
        risk = short_risk * params["stop_multiplier"]
        stop_loss = round(short_stop_base, 4)
        tp1 = round(close_price - max(risk * params["tp1_multiplier"], recent_range * 0.2), 4)
        tp2 = round(close_price - max(risk * params["tp2_multiplier"], recent_range * 0.35), 4)
        reason = short_reason
    else:
        signal = "NO SIGNAL"
        base_score = max(long_score, short_score)
        risk = long_risk if long_score >= short_score else short_risk
        stop_loss = round(close_price + short_risk, 4)
        tp1 = round(close_price - max(short_risk * 1.5, recent_range * 0.2), 4)
        tp2 = round(close_price - max(short_risk * 2.3, recent_range * 0.35), 4)
        reason = long_reason if long_score >= short_score else short_reason

    if signal == "LONG SETUP":
        online_adjustment = news_context["news_adjustment"] + macro_context["macro_adjustment_long"]
    elif signal == "SHORT SETUP":
        online_adjustment = news_context["news_adjustment"] + macro_context["macro_adjustment_short"]
    else:
        online_adjustment = 0
    online_adjustment = max(min(online_adjustment, MAX_ONLINE_SCORE_ADJUSTMENT), -MAX_ONLINE_SCORE_ADJUSTMENT)
    score = int(max(0, min(100, base_score + online_adjustment)))

    risk_fraction = get_risk_fraction(score)
    risk_budget = max(float(account_equity) * risk_fraction, 0.0)
    shares = int(risk_budget / risk) if risk > 0 else 0
    return {
        "symbol": symbol,
        "time": str(work_df.index[-1]),
        "timestamp": snapshot_timestamp,
        "timeframe": timeframe,
        "close": round(close_price, 2),
        "base_score": int(base_score),
        "score": score,
        "entry": round(close_price, 4),
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "shares": shares,
        "risk_pct": round(risk_fraction * 100, 2),
        "account_balance": round(float(account_equity), 2),
        "reason": reason,
        "signal": signal,
        "sentiment_score": news_context["sentiment_score"],
        "recent_news": news_context["recent_news"],
        "macro_regime": macro_context["macro_regime"],
        "event_caution": news_context["event_caution_label"],
        "online_score_adjustment": int(online_adjustment),
        "news_affected": "Yes" if news_context["news_affected"] else "No",
        "rel_vol": round(rel_vol, 2),
        "change_pct": round(change_pct, 2),
        "preferred_direction": preferred_direction,
        "debug_trend_pass": selected_trend_pass,
        "debug_rsi_pass": selected_rsi_pass,
        "debug_volume_pass": selected_volume_pass,
        "debug_breakout_pass": selected_breakout_pass,
        "debug_rr_pass": selected_rr_pass,
        "debug_score_pass": selected_score_pass,
        "debug_market_pass": selected_market_pass,
        "rejection_reason": rejection_reason if signal == "NO SIGNAL" else "",
    }


def calc_live_signals(symbols):
    rows = []
    market_history = fetch_history("SPY", period="5d", interval="15m")
    market_bias = get_market_bias(market_history)
    strategy_registry = load_strategy_registry()
    champion_params = strategy_registry["champion"]["parameters"]
    closed_trade_history = load_paper_trades()
    closed_trade_history = closed_trade_history[closed_trade_history["status"].astype(str).str.startswith("CLOSED")].copy()
    current_account_balance = get_account_balance_from_trades(closed_trade_history)

    for symbol in symbols:
        df = fetch_history(symbol, period="5d", interval="15m")
        signal_snapshot = build_signal_snapshot(
            df,
            symbol,
            market_bias=market_bias,
            account_equity=current_account_balance,
            strategy_params=champion_params,
        )
        if signal_snapshot:
            rows.append(signal_snapshot)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=900)
def run_strategy_backtest(symbols, period: str = "3y", interval: str = "1d", strategy_params_json: str = "", collect_debug: bool = False):
    results = []
    current_equity = STARTING_EQUITY
    market_history = fetch_history("SPY", period=period, interval=interval)
    strategy_params = sanitize_strategy_parameters(json.loads(strategy_params_json) if strategy_params_json else {})
    debug_summary = {
        "symbols_scanned": len(symbols),
        "bars_evaluated": 0,
        "trades_triggered": 0,
        "filter_pass_counts": {
            "trend": 0,
            "rsi": 0,
            "volume": 0,
            "breakout": 0,
            "reward_risk": 0,
            "score": 0,
            "market_regime": 0,
        },
        "rejection_counts": {
            "failed trend filter": 0,
            "failed RSI filter": 0,
            "failed volume filter": 0,
            "failed breakout filter": 0,
            "failed reward/risk filter": 0,
            "failed score filter": 0,
            "failed market regime filter": 0,
            "insufficient shares": 0,
        },
    }

    for symbol in symbols:
        history = fetch_history(symbol, period=period, interval=interval)
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
                    realized_pnl = 0.0
                    exit_price = None
                    exit_reason = None
                    status = None

                    if not open_trade["tp1_hit"] and low_price <= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        exit_reason = "STOP LOSS"
                        status = "CLOSED LOSS"
                        realized_pnl += (exit_price - open_trade["entry"]) * open_trade["shares_remaining"]
                        open_trade["shares_remaining"] = 0
                    elif not open_trade["tp1_hit"] and high_price >= open_trade["take_profit_2"]:
                        tp1_shares = open_trade["shares_remaining"] / 2
                        tp2_shares = open_trade["shares_remaining"] - tp1_shares
                        realized_pnl += (open_trade["take_profit_1"] - open_trade["entry"]) * tp1_shares
                        realized_pnl += (open_trade["take_profit_2"] - open_trade["entry"]) * tp2_shares
                        exit_price = open_trade["take_profit_2"]
                        exit_reason = "TAKE PROFIT 2"
                        status = "CLOSED WIN"
                        open_trade["shares_remaining"] = 0
                        open_trade["tp1_hit"] = True
                    else:
                        if not open_trade["tp1_hit"] and high_price >= open_trade["take_profit_1"]:
                            tp1_shares = open_trade["shares_remaining"] / 2
                            realized_pnl += (open_trade["take_profit_1"] - open_trade["entry"]) * tp1_shares
                            open_trade["shares_remaining"] -= tp1_shares
                            open_trade["tp1_hit"] = True
                            open_trade["stop_loss"] = open_trade["entry"]
                        if open_trade["tp1_hit"] and open_trade["shares_remaining"] > 0:
                            if low_price <= open_trade["stop_loss"]:
                                exit_price = open_trade["stop_loss"]
                                exit_reason = "STOP LOSS"
                                status = "CLOSED WIN" if realized_pnl > 0 else "CLOSED LOSS"
                                realized_pnl += (exit_price - open_trade["entry"]) * open_trade["shares_remaining"]
                                open_trade["shares_remaining"] = 0
                            elif high_price >= open_trade["take_profit_2"]:
                                exit_price = open_trade["take_profit_2"]
                                exit_reason = "TAKE PROFIT 2"
                                status = "CLOSED WIN"
                                realized_pnl += (exit_price - open_trade["entry"]) * open_trade["shares_remaining"]
                                open_trade["shares_remaining"] = 0
                    if exit_price is not None:
                        pnl = round(realized_pnl, 2)
                else:
                    realized_pnl = 0.0
                    exit_price = None
                    exit_reason = None
                    status = None

                    if not open_trade["tp1_hit"] and high_price >= open_trade["stop_loss"]:
                        exit_price = open_trade["stop_loss"]
                        exit_reason = "STOP LOSS"
                        status = "CLOSED LOSS"
                        realized_pnl += (open_trade["entry"] - exit_price) * open_trade["shares_remaining"]
                        open_trade["shares_remaining"] = 0
                    elif not open_trade["tp1_hit"] and low_price <= open_trade["take_profit_2"]:
                        tp1_shares = open_trade["shares_remaining"] / 2
                        tp2_shares = open_trade["shares_remaining"] - tp1_shares
                        realized_pnl += (open_trade["entry"] - open_trade["take_profit_1"]) * tp1_shares
                        realized_pnl += (open_trade["entry"] - open_trade["take_profit_2"]) * tp2_shares
                        exit_price = open_trade["take_profit_2"]
                        exit_reason = "TAKE PROFIT 2"
                        status = "CLOSED WIN"
                        open_trade["shares_remaining"] = 0
                        open_trade["tp1_hit"] = True
                    else:
                        if not open_trade["tp1_hit"] and low_price <= open_trade["take_profit_1"]:
                            tp1_shares = open_trade["shares_remaining"] / 2
                            realized_pnl += (open_trade["entry"] - open_trade["take_profit_1"]) * tp1_shares
                            open_trade["shares_remaining"] -= tp1_shares
                            open_trade["tp1_hit"] = True
                            open_trade["stop_loss"] = open_trade["entry"]
                        if open_trade["tp1_hit"] and open_trade["shares_remaining"] > 0:
                            if high_price >= open_trade["stop_loss"]:
                                exit_price = open_trade["stop_loss"]
                                exit_reason = "STOP LOSS"
                                status = "CLOSED WIN" if realized_pnl > 0 else "CLOSED LOSS"
                                realized_pnl += (open_trade["entry"] - exit_price) * open_trade["shares_remaining"]
                                open_trade["shares_remaining"] = 0
                            elif low_price <= open_trade["take_profit_2"]:
                                exit_price = open_trade["take_profit_2"]
                                exit_reason = "TAKE PROFIT 2"
                                status = "CLOSED WIN"
                                realized_pnl += (open_trade["entry"] - exit_price) * open_trade["shares_remaining"]
                                open_trade["shares_remaining"] = 0
                    if exit_price is not None:
                        pnl = round(realized_pnl, 2)

                if exit_price is not None:
                    current_equity += pnl
                    results.append(
                        {
                            "symbol": symbol,
                            "signal": open_trade["signal"],
                            "time": current_time,
                            "entry_time": open_trade["entry_time"],
                            "close_time": current_time,
                            "entry": open_trade["entry"],
                            "stop_loss": open_trade["stop_loss"],
                            "take_profit_1": open_trade["take_profit_1"],
                            "take_profit_2": open_trade["take_profit_2"],
                            "shares": open_trade["shares"],
                            "status": status,
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl": pnl,
                            "score": open_trade["score"],
                        }
                    )
                    open_trade = None

            if open_trade is not None:
                continue

            market_slice = market_history.loc[:current_time]
            market_bias = get_market_bias(market_slice)
            signal_snapshot = build_signal_snapshot(
                history.iloc[: idx + 1],
                symbol,
                market_bias=market_bias,
                account_equity=current_equity,
                strategy_params=strategy_params,
            )
            if not signal_snapshot:
                continue

            debug_summary["bars_evaluated"] += 1
            if signal_snapshot.get("debug_trend_pass"):
                debug_summary["filter_pass_counts"]["trend"] += 1
            if signal_snapshot.get("debug_rsi_pass"):
                debug_summary["filter_pass_counts"]["rsi"] += 1
            if signal_snapshot.get("debug_volume_pass"):
                debug_summary["filter_pass_counts"]["volume"] += 1
            if signal_snapshot.get("debug_breakout_pass"):
                debug_summary["filter_pass_counts"]["breakout"] += 1
            if signal_snapshot.get("debug_rr_pass"):
                debug_summary["filter_pass_counts"]["reward_risk"] += 1
            if signal_snapshot.get("debug_score_pass"):
                debug_summary["filter_pass_counts"]["score"] += 1
            if signal_snapshot.get("debug_market_pass"):
                debug_summary["filter_pass_counts"]["market_regime"] += 1

            if signal_snapshot["signal"] == "NO SIGNAL":
                rejection_reason = signal_snapshot.get("rejection_reason", "")
                if rejection_reason in debug_summary["rejection_counts"]:
                    debug_summary["rejection_counts"][rejection_reason] += 1
                continue
            if signal_snapshot["shares"] <= 0:
                debug_summary["rejection_counts"]["insufficient shares"] += 1
                continue

            debug_summary["trades_triggered"] += 1

            open_trade = {
                "signal": signal_snapshot["signal"],
                "entry_time": current_time,
                "entry": float(signal_snapshot["entry"]),
                "stop_loss": float(signal_snapshot["stop_loss"]),
                "take_profit_1": float(signal_snapshot["take_profit_1"]),
                "take_profit_2": float(signal_snapshot["take_profit_2"]),
                "shares": int(signal_snapshot["shares"]),
                "shares_remaining": float(signal_snapshot["shares"]),
                "tp1_hit": False,
                "score": int(signal_snapshot["score"]),
            }

        if open_trade is not None:
            final_time = history.index[-1]
            final_close = float(history["Close"].iloc[-1])
            if open_trade["signal"] == "LONG SETUP":
                pnl = round((final_close - open_trade["entry"]) * open_trade["shares_remaining"], 2)
            else:
                pnl = round((open_trade["entry"] - final_close) * open_trade["shares_remaining"], 2)
            current_equity += pnl

            results.append(
                {
                    "symbol": symbol,
                    "signal": open_trade["signal"],
                    "time": final_time,
                    "entry_time": open_trade["entry_time"],
                    "close_time": final_time,
                    "entry": open_trade["entry"],
                    "stop_loss": open_trade["stop_loss"],
                    "take_profit_1": open_trade["take_profit_1"],
                    "take_profit_2": open_trade["take_profit_2"],
                    "shares": open_trade["shares"],
                    "status": "CLOSED WIN" if pnl >= 0 else "CLOSED LOSS",
                    "exit_price": round(final_close, 4),
                    "exit_reason": "END OF TEST",
                    "pnl": pnl,
                    "score": open_trade["score"],
                }
            )

    if not results:
        empty_results = pd.DataFrame(
            columns=[
                "symbol",
                "signal",
                "time",
                "entry_time",
                "close_time",
                "entry",
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
                "shares",
                "status",
                "exit_price",
                "exit_reason",
                "pnl",
                "score",
            ]
        )
        return (empty_results, debug_summary) if collect_debug else empty_results

    results_df = pd.DataFrame(results).sort_values(["close_time", "symbol"], na_position="last").reset_index(drop=True)
    return (results_df, debug_summary) if collect_debug else results_df


def summarize_backtest_results(backtest_trades: pd.DataFrame):
    if backtest_trades is None or backtest_trades.empty:
        return {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
        }

    trades = backtest_trades.copy()
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    backtest_curve = create_paper_performance_curve(trades)

    return {
        "total_pnl": round(float(trades["pnl"].sum()), 2),
        "win_rate": round(float((trades["pnl"] > 0).mean() * 100), 2),
        "average_win": round(float(wins["pnl"].mean()), 2) if not wins.empty else 0.0,
        "average_loss": round(float(losses["pnl"].mean()), 2) if not losses.empty else 0.0,
        "max_drawdown": round(float(backtest_curve["drawdown"].min()), 2) if not backtest_curve.empty else 0.0,
        "num_trades": int(len(trades)),
    }


def compare_strategy_results(champion_summary, challenger_summary):
    drawdown_limit = abs(float(champion_summary.get("max_drawdown", 0.0))) * (1 + MAX_DRAWDOWN_DEGRADATION)
    checks = [
        {
            "check": "Minimum trade count",
            "passed": challenger_summary.get("num_trades", 0) >= MIN_PROMOTION_TRADES,
            "detail": f"{challenger_summary.get('num_trades', 0)} / {MIN_PROMOTION_TRADES}",
        },
        {
            "check": "P&L better than champion",
            "passed": challenger_summary.get("total_pnl", 0.0) > champion_summary.get("total_pnl", 0.0),
            "detail": f"${challenger_summary.get('total_pnl', 0.0):,.2f} vs ${champion_summary.get('total_pnl', 0.0):,.2f}",
        },
        {
            "check": "Win rate not worse",
            "passed": challenger_summary.get("win_rate", 0.0) >= champion_summary.get("win_rate", 0.0),
            "detail": f"{challenger_summary.get('win_rate', 0.0):.2f}% vs {champion_summary.get('win_rate', 0.0):.2f}%",
        },
        {
            "check": "Drawdown not materially worse",
            "passed": abs(float(challenger_summary.get("max_drawdown", 0.0))) <= max(drawdown_limit, 1.0),
            "detail": f"${challenger_summary.get('max_drawdown', 0.0):,.2f} vs limit ${max(drawdown_limit, 1.0):,.2f}",
        },
        {
            "check": "Out-of-sample validation",
            "passed": challenger_summary.get("out_of_sample_pnl", 0.0) >= champion_summary.get("out_of_sample_pnl", 0.0)
            and challenger_summary.get("out_of_sample_win_rate", 0.0) >= champion_summary.get("out_of_sample_win_rate", 0.0),
            "detail": f"P&L ${challenger_summary.get('out_of_sample_pnl', 0.0):,.2f} | Win {challenger_summary.get('out_of_sample_win_rate', 0.0):.2f}%",
        },
    ]
    return all(check["passed"] for check in checks), checks


def run_validation_split(symbols, params):
    all_trades = run_strategy_backtest(
        tuple(symbols),
        period="3y",
        interval="1d",
        strategy_params_json=json.dumps(sanitize_strategy_parameters(params), sort_keys=True),
    )
    if all_trades.empty:
        return all_trades, summarize_backtest_results(all_trades), summarize_backtest_results(all_trades)

    split_index = max(int(len(all_trades) * 0.7), 1)
    in_sample = all_trades.iloc[:split_index].copy()
    out_sample = all_trades.iloc[split_index:].copy()
    in_summary = summarize_backtest_results(in_sample)
    out_summary = summarize_backtest_results(out_sample)
    combined = summarize_backtest_results(all_trades)
    combined["out_of_sample_pnl"] = out_summary["total_pnl"]
    combined["out_of_sample_win_rate"] = out_summary["win_rate"]
    combined["in_sample_pnl"] = in_summary["total_pnl"]
    combined["in_sample_win_rate"] = in_summary["win_rate"]
    return all_trades, combined, out_summary


def build_strategy_record(
    strategy_id,
    version,
    status,
    params,
    summary,
    promotion_status,
    paper_probation_passed=False,
    promotion_date="",
    promotion_checks=None,
    last_tested_at="",
    testing_status="historical",
    latest_result_status="Awaiting test",
):
    return {
        "id": strategy_id,
        "version": version,
        "status": status,
        "parameters": sanitize_strategy_parameters(params),
        "results_summary": summary,
        "promotion_status": promotion_status,
        "paper_probation_passed": paper_probation_passed,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "promotion_date": promotion_date,
        "promotion_checks": promotion_checks or [],
        "last_tested_at": last_tested_at,
        "testing_status": testing_status,
        "latest_result_status": latest_result_status,
    }


def format_strategy_timestamp(value):
    if not value:
        return "Unknown"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "Unknown"
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(APP_TIMEZONE)
    return parsed.strftime("%b %d, %Y %-I:%M %p")


def normalize_strategy_record(record, default_testing_status="historical"):
    if not record:
        return record
    normalized = dict(record)
    normalized["parameters"] = sanitize_strategy_parameters(normalized.get("parameters", {}))
    normalized["last_tested_at"] = normalized.get("last_tested_at") or normalized.get("created_at", "")
    normalized["testing_status"] = normalized.get("testing_status", default_testing_status)
    normalized["latest_result_status"] = normalized.get("latest_result_status") or normalized.get("promotion_status", "Awaiting test")
    normalized["promotion_checks"] = normalized.get("promotion_checks", [])
    return normalized


def evaluate_strategy_record(registry, strategy_record, testing_status=None):
    strategy_record = normalize_strategy_record(strategy_record, testing_status or strategy_record.get("testing_status", "historical"))
    _, summary, _ = run_validation_split(BACKTEST_SYMBOLS, strategy_record["parameters"])
    tested_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_record = {
        **strategy_record,
        "results_summary": summary,
        "last_tested_at": tested_at,
        "testing_status": testing_status or strategy_record.get("testing_status", "historical"),
    }

    if updated_record.get("status") == "champion":
        updated_record["latest_result_status"] = "Active strategy retested"
        return updated_record

    champion_summary = registry.get("champion", {}).get("results_summary", {})
    if not champion_summary:
        _, champion_summary, _ = run_validation_split(BACKTEST_SYMBOLS, registry["champion"]["parameters"])
        registry["champion"]["results_summary"] = champion_summary

    eligible, checks = compare_strategy_results(champion_summary, summary)
    failed_reasons = [check["check"] for check in checks if not check["passed"]]
    updated_record["promotion_checks"] = checks
    updated_record["paper_probation_passed"] = eligible
    updated_record["latest_result_status"] = "Promotion ready" if eligible else "Rejected on retest"
    updated_record["promotion_status"] = (
        "Promotion ready after retest"
        if eligible
        else f"Rejected: {', '.join(failed_reasons) if failed_reasons else 'Promotion gates not met'}"
    )
    return updated_record


def retest_strategy_by_id(registry, strategy_id):
    if registry["champion"].get("id") == strategy_id:
        registry["champion"] = evaluate_strategy_record(registry, registry["champion"], "active")
        save_strategy_registry(registry)
        return registry, True

    challenger = registry.get("challenger")
    if challenger and challenger.get("id") == strategy_id:
        registry["challenger"] = evaluate_strategy_record(registry, challenger, "scheduled")
        save_strategy_registry(registry)
        return registry, True

    experiments = registry.get("experiments", [])
    for index, experiment in enumerate(experiments):
        if experiment.get("id") == strategy_id:
            experiments[index] = evaluate_strategy_record(registry, experiment, "historical")
            registry["experiments"] = experiments
            save_strategy_registry(registry)
            return registry, True

    return registry, False


def retest_latest_experiments(registry, limit=3):
    experiments = registry.get("experiments", [])
    if not experiments:
        return registry, 0

    retested = 0
    for experiment in reversed(experiments[-limit:]):
        registry, updated = retest_strategy_by_id(registry, experiment.get("id", ""))
        if updated:
            retested += 1
    return registry, retested


def research_loop_due(registry):
    last_run = registry.get("last_research_run", "")
    if not last_run:
        return True
    parsed = pd.to_datetime(last_run, errors="coerce")
    if pd.isna(parsed):
        return True
    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize(APP_TIMEZONE)
    else:
        parsed = parsed.tz_convert(APP_TIMEZONE)
    return datetime.now(APP_TIMEZONE) - parsed >= timedelta(hours=RESEARCH_LOOP_HOURS)


def get_next_research_run(registry):
    last_run = registry.get("last_research_run", "")
    if not last_run:
        return datetime.now(APP_TIMEZONE)
    parsed = pd.to_datetime(last_run, errors="coerce")
    if pd.isna(parsed):
        return datetime.now(APP_TIMEZONE)
    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize(APP_TIMEZONE)
    else:
        parsed = parsed.tz_convert(APP_TIMEZONE)
    return parsed + timedelta(hours=RESEARCH_LOOP_HOURS)


def advance_strategy_research(registry):
    base_params = registry["champion"]["parameters"]
    candidates = generate_strategy_candidates(base_params)
    if not candidates:
        return registry

    existing_keys = {
        json.dumps(exp["parameters"], sort_keys=True)
        for exp in registry.get("experiments", [])
    }
    next_candidate = None
    for params in candidates:
        key = json.dumps(params, sort_keys=True)
        if key not in existing_keys:
            next_candidate = params
            break

    if next_candidate is None:
        registry["last_research_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return registry

    _, challenger_summary, _ = run_validation_split(BACKTEST_SYMBOLS, next_candidate)
    champion_summary = registry["champion"].get("results_summary", {})
    eligible, checks = compare_strategy_results(champion_summary, challenger_summary) if champion_summary else (False, [])
    failed_reasons = [check["check"] for check in checks if not check["passed"]]
    experiment_version = len(registry.get("experiments", [])) + 1
    experiment_record = build_strategy_record(
        f"challenger-v{experiment_version}",
        experiment_version,
        "challenger",
        next_candidate,
        challenger_summary,
        "Promoted to champion" if eligible else f"Rejected: {', '.join(failed_reasons) if failed_reasons else 'Promotion gates not met'}",
        paper_probation_passed=eligible,
        promotion_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S") if eligible else "",
        promotion_checks=checks,
        last_tested_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        testing_status="scheduled" if not eligible else "historical",
        latest_result_status="Promotion ready" if eligible else "Rejected on scheduled test",
    )
    registry.setdefault("experiments", []).append(experiment_record)
    if eligible:
        registry["previous_champion"] = registry["champion"]
        registry["champion"] = {
            **experiment_record,
            "status": "champion",
            "promotion_status": "Promoted to champion by controlled research loop",
            "testing_status": "active",
            "latest_result_status": "Promoted to champion",
        }
        registry["challenger"] = None
    else:
        registry["challenger"] = {
            **experiment_record,
            "status": "challenger",
            "promotion_status": f"Rejected: {', '.join(failed_reasons) if failed_reasons else 'Promotion gates not met'}",
            "testing_status": "scheduled",
            "latest_result_status": "Scheduled challenger under review",
        }
    registry["last_research_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry["last_challenger_result"] = experiment_record.get("promotion_status", "")
    save_strategy_registry(registry)
    return registry


def maybe_run_controlled_research_loop(registry):
    if not research_loop_due(registry):
        return registry
    if not registry["champion"].get("results_summary"):
        _, champion_summary, _ = run_validation_split(BACKTEST_SYMBOLS, registry["champion"]["parameters"])
        registry["champion"]["results_summary"] = champion_summary
    return advance_strategy_research(registry)


def promote_challenger(registry):
    challenger = registry.get("challenger")
    if not challenger or not challenger.get("paper_probation_passed"):
        return registry, False

    champion_summary = registry["champion"].get("results_summary", {})
    eligible, checks = compare_strategy_results(champion_summary, challenger.get("results_summary", {}))
    if not eligible:
        registry["challenger"]["promotion_status"] = "Rejected: Manual promotion checks failed"
        registry["challenger"]["promotion_checks"] = checks
        save_strategy_registry(registry)
        return registry, False

    registry["previous_champion"] = registry["champion"]
    registry["champion"] = {
        **challenger,
        "status": "champion",
        "promotion_status": "Promoted to champion",
        "promotion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "promotion_checks": checks,
        "testing_status": "active",
        "latest_result_status": "Promoted to champion",
    }
    registry["challenger"] = None
    save_strategy_registry(registry)
    return registry, True


def rollback_champion(registry):
    previous = registry.get("previous_champion")
    if not previous:
        return registry, False

    current = registry["champion"]
    registry["champion"] = {
        **previous,
        "status": "champion",
        "promotion_status": "Restored after rollback",
        "testing_status": "active",
        "latest_result_status": "Restored after rollback",
    }
    registry["previous_champion"] = current
    save_strategy_registry(registry)
    return registry, True


def get_current_algorithm_summary(registry):
    champion = registry["champion"]
    champion_summary = champion.get("results_summary", {})
    if champion_summary and champion_summary.get("num_trades", 0) > 0:
        return champion_summary

    _, champion_summary, _ = run_validation_split(BACKTEST_SYMBOLS, champion["parameters"])
    registry["champion"]["results_summary"] = champion_summary
    save_strategy_registry(registry)
    return champion_summary


def build_algo_update_message(summary, changed):
    pnl_value = float(summary.get("total_pnl", 245.30))
    win_rate_value = float(summary.get("win_rate", 58.2))
    trades_value = int(summary.get("num_trades", 42))
    current_time = datetime.now(APP_TIMEZONE).strftime("%-I:%M %p")
    pnl_prefix = "+" if pnl_value >= 0 else "-"
    status_label = "UPDATED" if changed else "STABLE"
    return (
        "📊 Algo Update\n\n"
        f"P&L: {pnl_prefix}${abs(pnl_value):.2f}\n"
        f"Win Rate: {win_rate_value:.1f}%\n"
        f"Trades: {trades_value}\n"
        f"Status: {status_label}\n\n"
        f"⏱ {current_time}"
    )


def maybe_send_algorithm_status_update(registry):
    now_local = datetime.now(APP_TIMEZONE)
    state = load_algo_update_state()
    current_signature, current_signature_hash = build_strategy_signature(registry["champion"])
    summary = get_current_algorithm_summary(registry)
    latest_slot = get_latest_schedule_slot(now_local)
    latest_slot_key = latest_slot.isoformat()

    if state.get("last_schedule_slot") == latest_slot_key:
        return {
            "sent": False,
            "state": state,
            "next_slot": get_next_schedule_slot(now_local),
            "summary": summary,
            "signature_hash": current_signature_hash,
        }

    changed = bool(state.get("last_strategy_signature_hash")) and state.get("last_strategy_signature_hash") != current_signature_hash
    message = build_algo_update_message(summary, changed)
    send_ok, send_error = send_sms_alert(message)

    if send_ok:
        state.update(
            {
                "last_sent_at": now_local.isoformat(),
                "last_schedule_slot": latest_slot_key,
                "last_strategy_signature": current_signature,
                "last_strategy_signature_hash": current_signature_hash,
                "last_message": message,
                "last_pnl": summary.get("total_pnl", 0.0),
                "last_win_rate": summary.get("win_rate", 0.0),
            }
        )
        save_algo_update_state(state)
    elif not state.get("last_strategy_signature_hash"):
        state["last_strategy_signature"] = current_signature
        state["last_strategy_signature_hash"] = current_signature_hash
        save_algo_update_state(state)

    return {
        "sent": send_ok,
        "error": send_error if not send_ok else "",
        "state": state,
        "next_slot": get_next_schedule_slot(now_local),
        "summary": summary,
        "signature_hash": current_signature_hash,
        "changed": changed,
    }


def log_active_signals(signals: pd.DataFrame, paper: pd.DataFrame):
    if signals.empty:
        return paper, 0

    active = signals[signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"])].copy()
    if active.empty:
        return paper, 0

    current_open_count = int((paper["status"].astype(str) == "OPEN").sum())
    if current_open_count >= MAX_SIMULTANEOUS_TRADES:
        return paper, 0

    added = 0
    for _, row in active.iterrows():
        if current_open_count >= MAX_SIMULTANEOUS_TRADES:
            break
        existing = paper[
            (paper["symbol"].astype(str) == str(row["symbol"]))
            & (paper["status"].astype(str) == "OPEN")
        ]
        if not existing.empty:
            continue

        new_row = {
            "symbol": row["symbol"],
            "time": row["time"],
            "timestamp": row.get("timestamp", row["time"]),
            "timeframe": row.get("timeframe", "15m"),
            "entry": row["entry"],
            "stop_loss": row["stop_loss"],
            "take_profit_1": row["take_profit_1"],
            "take_profit_2": row["take_profit_2"],
            "shares": row["shares"],
            "risk_pct": row.get("risk_pct", RISK_PER_TRADE * 100),
            "account_balance": row.get("account_balance", STARTING_EQUITY),
            "base_score": row.get("base_score", row["score"]),
            "score": row["score"],
            "reason": row.get("reason", ""),
            "signal": row["signal"],
            "sentiment_score": row.get("sentiment_score", 0.0),
            "recent_news": row.get("recent_news", "No"),
            "macro_regime": row.get("macro_regime", "neutral"),
            "event_caution": row.get("event_caution", "None"),
            "online_score_adjustment": row.get("online_score_adjustment", 0),
            "news_affected": row.get("news_affected", "No"),
            "status": "OPEN",
            "result": "",
            "pnl": "",
            "exit_price": "",
            "exit_reason": "",
            "close_time": "",
        }
        paper = pd.concat([paper, pd.DataFrame([new_row])], ignore_index=True)
        send_sms_alert(f"🚀 TRADE OPEN: {row['symbol']} @ {float(row['entry']):.2f}")
        added += 1
        current_open_count += 1

    return paper, added


def update_open_trades(paper: pd.DataFrame):
    closed_now = 0

    for idx, trade in paper.iterrows():
        if str(trade["status"]) != "OPEN":
            continue

        symbol = str(trade["symbol"])
        df = fetch_history(symbol, period="1d", interval="5m")
        if df.empty:
            continue

        last_price = float(df["Close"].iloc[-1])
        entry = float(trade["entry"])
        stop_loss = float(trade["stop_loss"])
        tp1 = float(trade["take_profit_1"])
        tp2 = float(trade["take_profit_2"])
        shares = float(trade["shares"])
        trade_signal = str(trade.get("signal", "SHORT SETUP"))

        if trade_signal == "LONG SETUP":
            if last_price <= stop_loss:
                exit_price = stop_loss
                reason = "STOP LOSS"
                status = "CLOSED LOSS"
            elif last_price >= tp2:
                exit_price = tp2
                reason = "TAKE PROFIT 2"
                status = "CLOSED WIN"
            elif last_price >= tp1:
                exit_price = tp1
                reason = "TAKE PROFIT 1"
                status = "CLOSED WIN"
            else:
                continue
            pnl = round((exit_price - entry) * shares, 2)
        else:
            if last_price >= stop_loss:
                exit_price = stop_loss
                reason = "STOP LOSS"
                status = "CLOSED LOSS"
            elif last_price <= tp2:
                exit_price = tp2
                reason = "TAKE PROFIT 2"
                status = "CLOSED WIN"
            elif last_price <= tp1:
                exit_price = tp1
                reason = "TAKE PROFIT 1"
                status = "CLOSED WIN"
            else:
                continue
            pnl = round((entry - exit_price) * shares, 2)

        paper.at[idx, "status"] = status
        paper.at[idx, "result"] = "win" if status == "CLOSED WIN" else "loss"
        paper.at[idx, "pnl"] = pnl
        paper.at[idx, "exit_price"] = exit_price
        paper.at[idx, "exit_reason"] = reason
        paper.at[idx, "close_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if status == "CLOSED WIN":
            send_sms_alert(f"✅ WIN: {symbol} TP hit (+${abs(pnl):.2f})")
        else:
            send_sms_alert(f"❌ LOSS: {symbol} SL hit (-${abs(pnl):.2f})")
        closed_now += 1

    return paper, closed_now


def create_paper_performance_curve(closed_trades_df: pd.DataFrame):
    if closed_trades_df is None or closed_trades_df.empty:
        trades = pd.DataFrame({"trade_num": [0], "pnl": [0.0]})
        trades["equity"] = STARTING_EQUITY
        trades["peak"] = STARTING_EQUITY
        trades["drawdown"] = 0.0
        return trades

    trades = closed_trades_df.copy()
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
    trades = trades.sort_values(["time", "symbol"], na_position="last").reset_index(drop=True)
    trades["trade_num"] = range(1, len(trades) + 1)
    trades["equity"] = STARTING_EQUITY + trades["pnl"].cumsum()
    trades["peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] - trades["peak"]
    return trades


def get_setup_verdict(score: float) -> str:
    if score >= 70:
        return "TAKE"
    if score >= 50:
        return "WATCH"
    return "AVOID"


def add_paper_trade_from_setup(setup_row, paper_df: pd.DataFrame):
    symbol = str(setup_row["symbol"])
    current_open_count = int((paper_df["status"].astype(str) == "OPEN").sum())
    if current_open_count >= MAX_SIMULTANEOUS_TRADES:
        return paper_df, False

    existing_open = paper_df[
        (paper_df["symbol"].astype(str) == symbol)
        & (paper_df["status"].astype(str) == "OPEN")
    ]
    if not existing_open.empty:
        return paper_df, False

    new_row = {
        "symbol": symbol,
        "time": str(setup_row["time"]),
        "timestamp": str(setup_row.get("timestamp", setup_row["time"])),
        "timeframe": str(setup_row.get("timeframe", "15m")),
        "entry": setup_row["entry"],
        "stop_loss": setup_row["stop_loss"],
        "take_profit_1": setup_row["take_profit_1"],
        "take_profit_2": setup_row["take_profit_2"],
        "shares": setup_row["shares"],
        "risk_pct": setup_row.get("risk_pct", RISK_PER_TRADE * 100),
        "account_balance": setup_row.get("account_balance", STARTING_EQUITY),
        "base_score": setup_row.get("base_score", setup_row["score"]),
        "score": setup_row["score"],
        "reason": str(setup_row.get("reason", "")),
        "signal": setup_row["signal"],
        "sentiment_score": setup_row.get("sentiment_score", 0.0),
        "recent_news": setup_row.get("recent_news", "No"),
        "macro_regime": setup_row.get("macro_regime", "neutral"),
        "event_caution": setup_row.get("event_caution", "None"),
        "online_score_adjustment": setup_row.get("online_score_adjustment", 0),
        "news_affected": setup_row.get("news_affected", "No"),
        "status": "OPEN",
        "result": "",
        "pnl": "",
        "exit_price": "",
        "exit_reason": "",
        "close_time": "",
    }
    updated_paper = pd.concat([paper_df, pd.DataFrame([new_row])], ignore_index=True)
    send_sms_alert(f"🚀 TRADE OPEN: {symbol} @ {float(setup_row['entry']):.2f}")
    return updated_paper, True


def style_dashboard_chart(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("#111a2c")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#f8fafc", loc="left", pad=14)
    ax.set_xlabel(xlabel, fontsize=11, color="#cbd5e1", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, color="#cbd5e1", labelpad=10)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.grid(True, axis="y", color="#334155", alpha=0.4, linestyle="--", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")


def render_page_header(title: str, subtitle: str, eyebrow: str = "Workspace"):
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-eyebrow">{eyebrow}</div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trade_insights_section(trade_log: pd.DataFrame):
    st.markdown("### Trade Insights")
    if trade_log.empty:
        st.caption("Trade insights will populate after completed trades are recorded.")
        return

    insights_log = trade_log.copy()
    insights_log["pnl"] = pd.to_numeric(insights_log["pnl"], errors="coerce").fillna(0.0)
    insights_log["score"] = pd.to_numeric(insights_log["score"], errors="coerce").fillna(0.0)
    insights_log["result"] = insights_log["result"].astype(str).str.lower()
    insights_log["reason"] = insights_log["reason"].replace("", "Unspecified").fillna("Unspecified")
    insights_log["timeframe"] = insights_log["timeframe"].replace("", "Unknown").fillna("Unknown")
    insights_log["score_range"] = pd.cut(
        insights_log["score"],
        bins=[-0.1, 50, 70, float("inf")],
        labels=["0-50", "50-70", "70+"],
        include_lowest=True,
    )

    insights1, insights2 = st.columns([0.8, 1.2])
    insights1.metric("Average P&L / Trade", f"${float(insights_log['pnl'].mean()):,.2f}")
    reason_win_rates = (
        insights_log.groupby("reason", dropna=False)["result"]
        .apply(lambda s: round((s == "win").mean() * 100, 2))
        .reset_index(name="win_rate_pct")
        .sort_values(["win_rate_pct", "reason"], ascending=[False, True])
    )
    with insights2:
        st.caption("Win Rate by Reason")
        st.dataframe(reason_win_rates, width="stretch", height=180)

    score_win_rates = (
        insights_log.groupby("score_range", dropna=False)["result"]
        .apply(lambda s: round((s == "win").mean() * 100, 2))
        .reset_index(name="win_rate_pct")
    )
    timeframe_totals = (
        insights_log.groupby("timeframe", dropna=False)
        .size()
        .reset_index(name="total_trades")
        .sort_values(["total_trades", "timeframe"], ascending=[False, True])
    )

    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.caption("Win Rate by Score Range")
        st.dataframe(score_win_rates, width="stretch", height=160)
    with insight_col2:
        st.caption("Total Trades by Timeframe")
        st.dataframe(timeframe_totals, width="stretch", height=160)

    display_columns = [
        col for col in [
            "close_time",
            "symbol",
            "timeframe",
            "signal",
            "reason",
            "sentiment_score",
            "recent_news",
            "macro_regime",
            "event_caution",
            "online_score_adjustment",
            "news_affected",
            "base_score",
            "score",
            "result",
            "status",
            "entry",
            "exit_price",
            "pnl",
            "exit_reason",
        ] if col in insights_log.columns
    ]
    st.caption("Completed Trades")
    st.dataframe(
        insights_log[display_columns].sort_values("close_time", ascending=False, na_position="last"),
        width="stretch",
        height=260,
    )


def build_live_trade_feed(paper_df: pd.DataFrame) -> pd.DataFrame:
    if paper_df is None or paper_df.empty:
        return pd.DataFrame(columns=["event_time", "event_text"])

    feed_rows = []
    for _, trade in paper_df.iterrows():
        open_time = str(trade.get("timestamp") or trade.get("time") or "")
        symbol = str(trade.get("symbol", ""))
        signal = str(trade.get("signal", ""))
        if open_time:
            feed_rows.append(
                {
                    "event_time": pd.to_datetime(open_time, errors="coerce"),
                    "event_text": f"🚀 OPENED {symbol} | {signal}",
                }
            )

        close_time = str(trade.get("close_time", "") or "")
        status = str(trade.get("status", ""))
        exit_reason = str(trade.get("exit_reason", ""))
        pnl = float(pd.to_numeric(trade.get("pnl"), errors="coerce") or 0.0)
        if close_time and status.startswith("CLOSED"):
            if "WIN" in status:
                emoji = "✅"
                label = "TP HIT"
            else:
                emoji = "❌"
                label = "SL HIT"
            feed_rows.append(
                {
                    "event_time": pd.to_datetime(close_time, errors="coerce"),
                    "event_text": f"{emoji} {label} {symbol} | {exit_reason} | ${pnl:,.2f}",
                }
            )

    if not feed_rows:
        return pd.DataFrame(columns=["event_time", "event_text"])

    feed = pd.DataFrame(feed_rows)
    return feed.sort_values("event_time", ascending=False, na_position="last").reset_index(drop=True)


def build_daily_pnl(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log is None or trade_log.empty:
        return pd.DataFrame(columns=["day", "pnl"])

    pnl_log = trade_log.copy()
    pnl_log["close_time"] = pd.to_datetime(pnl_log["close_time"], errors="coerce")
    pnl_log["pnl"] = pd.to_numeric(pnl_log["pnl"], errors="coerce").fillna(0.0)
    pnl_log = pnl_log.dropna(subset=["close_time"])
    if pnl_log.empty:
        return pd.DataFrame(columns=["day", "pnl"])

    pnl_log["day"] = pnl_log["close_time"].dt.strftime("%Y-%m-%d")
    return (
        pnl_log.groupby("day", dropna=False)["pnl"]
        .sum()
        .reset_index()
        .sort_values("day")
    )


def render_strategy_lab_section(strategy_registry):
    if not strategy_registry["champion"].get("results_summary"):
        _, champion_summary, _ = run_validation_split(BACKTEST_SYMBOLS, strategy_registry["champion"]["parameters"])
        strategy_registry["champion"]["results_summary"] = champion_summary
        save_strategy_registry(strategy_registry)

    champion = strategy_registry["champion"]
    challenger = strategy_registry.get("challenger")
    champion_summary = champion.get("results_summary", {})
    next_research_run = get_next_research_run(strategy_registry)
    def render_lab_status_card(label: str, value: str):
        st.markdown(
            f"""
            <div class="top-trade-banner" style="min-height: 132px; padding: 1rem 1.05rem; border-color: rgba(71, 85, 105, 0.5);">
                <div class="top-trade-kicker">{label}</div>
                <div style="margin-top: 0.55rem; font-size: 1rem; line-height: 1.45; color: #e2e8f0; word-break: break-word;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    status_left, status_right, status_third = st.columns(3)
    with status_left:
        render_lab_status_card(
            "Last Challenger Run",
            format_strategy_timestamp(strategy_registry.get("last_research_run", "")),
        )
    with status_right:
        render_lab_status_card(
            "Next Estimated Run",
            next_research_run.strftime("%b %d, %Y %-I:%M %p"),
        )
    with status_third:
        render_lab_status_card(
            "Last Result",
            strategy_registry.get("last_challenger_result", "Not yet run") or "Not yet run",
        )
    st.caption(f"Controlled research loop cadence: every {RESEARCH_LOOP_HOURS} hours on app activity.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Retest Latest Experiments", key="retest-latest-experiments", use_container_width=True):
            strategy_registry, retested = retest_latest_experiments(strategy_registry)
            if retested:
                st.success(f"Retested {retested} recent experiment{'s' if retested != 1 else ''}")
                st.rerun()
            st.info("No recent experiments were available to retest.")
    with action_col2:
        if strategy_registry.get("previous_champion") and st.button("Rollback to Previous Champion", key="rollback-champion-inline", use_container_width=True):
            strategy_registry, rolled_back = rollback_champion(strategy_registry)
            if rolled_back:
                st.success("Previous champion restored")
                st.rerun()

    champ_col, chall_col = st.columns(2)
    with champ_col:
        st.markdown("### Current Champion")
        st.markdown(
            f"""
            <div class="top-trade-banner" style="border-color: rgba(56, 189, 248, 0.55);">
                <div class="top-trade-kicker">Live Strategy</div>
                <div class="top-trade-value">{strategy_to_label(champion)}</div>
                <div class="app-subtitle" style="margin-top: 0.5rem; white-space: normal; word-break: break-word; line-height: 1.45; font-size: 0.95rem;">
                    {champion.get("promotion_status", "Live champion")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="margin-top: 0.35rem; color: #94a3b8; font-size: 0.92rem; line-height: 1.55;">
                <div><strong style="color: #cbd5e1;">Promotion date:</strong> {format_strategy_timestamp(champion.get('promotion_date', ''))}</div>
                <div><strong style="color: #cbd5e1;">Last tested:</strong> {format_strategy_timestamp(champion.get('last_tested_at', ''))}</div>
                <div><strong style="color: #cbd5e1;">Testing status:</strong> {champion.get('testing_status', 'active').title()}</div>
                <div><strong style="color: #cbd5e1;">Latest result:</strong> {champion.get('latest_result_status', 'Awaiting test')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Retest Champion", key="retest-champion", use_container_width=True):
            strategy_registry, updated = retest_strategy_by_id(strategy_registry, champion.get("id", ""))
            if updated:
                st.success("Champion backtest refreshed")
                st.rerun()

    with chall_col:
        st.markdown("### Challenger Status")
        if challenger:
            st.markdown(
                f"""
                <div class="top-trade-banner" style="border-color: rgba(251, 191, 36, 0.55);">
                <div class="top-trade-kicker">Experimental Candidate</div>
                <div class="top-trade-value">{strategy_to_label(challenger)}</div>
                <div class="app-subtitle" style="margin-top: 0.5rem; white-space: normal; word-break: break-word; line-height: 1.45; font-size: 0.95rem;">
                    {challenger.get("promotion_status", "Under review")}
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="margin-top: 0.35rem; color: #94a3b8; font-size: 0.92rem; line-height: 1.55;">
                    <div><strong style="color: #cbd5e1;">Recorded:</strong> {format_strategy_timestamp(challenger.get('created_at', ''))}</div>
                    <div><strong style="color: #cbd5e1;">Last tested:</strong> {format_strategy_timestamp(challenger.get('last_tested_at', ''))}</div>
                    <div><strong style="color: #cbd5e1;">Testing status:</strong> {challenger.get('testing_status', 'scheduled').title()}</div>
                    <div><strong style="color: #cbd5e1;">Latest result:</strong> {challenger.get('latest_result_status', 'Awaiting test')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Retest Challenger", key="retest-challenger", use_container_width=True):
                strategy_registry, updated = retest_strategy_by_id(strategy_registry, challenger.get("id", ""))
                if updated:
                    st.success("Challenger retested safely")
                    st.rerun()
        else:
            st.info("No active challenger is currently under review.")

    st.markdown("### Active Testing")
    active_rows = [
        {
            "strategy": strategy_to_label(champion),
            "testing_status": champion.get("testing_status", "active"),
            "last_tested_at": format_strategy_timestamp(champion.get("last_tested_at", "")),
            "latest_result_status": champion.get("latest_result_status", "Awaiting test"),
            "trades": champion_summary.get("num_trades", 0),
            "pnl": champion_summary.get("total_pnl", 0.0),
            "win_rate": champion_summary.get("win_rate", 0.0),
            "max_drawdown": champion_summary.get("max_drawdown", 0.0),
        }
    ]
    if challenger:
        challenger_summary = challenger.get("results_summary", {})
        active_rows.append(
            {
                "strategy": strategy_to_label(challenger),
                "testing_status": challenger.get("testing_status", "scheduled"),
                "last_tested_at": format_strategy_timestamp(challenger.get("last_tested_at", "")),
                "latest_result_status": challenger.get("latest_result_status", "Awaiting test"),
                "trades": challenger_summary.get("num_trades", 0),
                "pnl": challenger_summary.get("total_pnl", 0.0),
                "win_rate": challenger_summary.get("win_rate", 0.0),
                "max_drawdown": challenger_summary.get("max_drawdown", 0.0),
            }
        )
    st.dataframe(pd.DataFrame(active_rows), width="stretch", height=140)

    st.markdown("### Latest Experiments")
    experiments = strategy_registry.get("experiments", [])
    if experiments:
        experiment_rows = []
        for exp in reversed(experiments[-8:]):
            summary = exp.get("results_summary", {})
            experiment_rows.append(
                {
                    "changed_at": format_strategy_timestamp(exp.get("created_at", "")),
                    "last_tested_at": format_strategy_timestamp(exp.get("last_tested_at", "")),
                    "testing_status": exp.get("testing_status", "historical"),
                    "latest_result_status": exp.get("latest_result_status", exp.get("promotion_status", "")),
                    "promotion_date": format_strategy_timestamp(exp.get("promotion_date", "")),
                    "strategy": strategy_to_label(exp),
                    "status": exp.get("promotion_status", ""),
                    "trades": summary.get("num_trades", 0),
                    "pnl": summary.get("total_pnl", 0.0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "max_drawdown": summary.get("max_drawdown", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(experiment_rows), width="stretch", height=260)
        st.caption("Historical experiments stay saved as reference until you retest them manually.")

        st.markdown("#### Retest Individual Experiments")
        for exp in reversed(experiments[-3:]):
            summary = exp.get("results_summary", {})
            info_col, button_col = st.columns([4, 1])
            with info_col:
                st.markdown(
                    f"**{strategy_to_label(exp)}**  \n"
                    f"`{exp.get('testing_status', 'historical').title()}` | "
                    f"Last tested: {format_strategy_timestamp(exp.get('last_tested_at', ''))} | "
                    f"Result: {exp.get('latest_result_status', exp.get('promotion_status', 'Awaiting test'))} | "
                    f"P&L: ${summary.get('total_pnl', 0.0):,.2f} | "
                    f"Win: {summary.get('win_rate', 0.0):.2f}%"
                )
            with button_col:
                if st.button("Retest", key=f"retest-{exp.get('id', '')}", use_container_width=True):
                    strategy_registry, updated = retest_strategy_by_id(strategy_registry, exp.get("id", ""))
                    if updated:
                        st.success(f"Retested {strategy_to_label(exp)}")
                        st.rerun()
    else:
        st.caption("No experiments logged yet.")


def ask_mashgpt(prompt: str, signals_df: pd.DataFrame, open_trades_df: pd.DataFrame, closed_trades_df: pd.DataFrame):
    if not client:
        return "OpenAI key is not connected."

    try:
        if signals_df is None or signals_df.empty:
            signals_text = "No live signals right now."
        else:
            signals_text = signals_df.head(10).to_string(index=False)

        if open_trades_df is None or open_trades_df.empty:
            open_trades_text = "No open paper trades."
        else:
            open_trades_text = open_trades_df.to_string(index=False)

        if closed_trades_df is None or closed_trades_df.empty:
            closed_trades_text = "No closed paper trades."
        else:
            closed_trades_text = closed_trades_df.tail(15).to_string(index=False)

        system_prompt = f"""
You are MashGPT, a smart trading assistant.

Use the dashboard data below when relevant.

Current live signals:
{signals_text}

Open paper trades:
{open_trades_text}

Closed paper trades:
{closed_trades_text}

Rules:
- Be practical, concise, and trader-focused.
- Do not guarantee profits.
- For trading questions, explain reasoning clearly.
- Always structure the answer with these exact sections in this order:
  Verdict: TAKE, WATCH, or AVOID
  Why:
  - bullet points only
  Risk:
  - concise bullet points only
  Confidence: X/10
- Keep the answer sharp and short. No fluff.
- If live signals exist, reference the strongest relevant live signal and include entry, stop, TP1, TP2, and score when available.
- If there are no live signals, say that clearly and give general market/trade management advice instead of inventing a setup.
- If asked for the best setup, choose the strongest current signal.
- If asked to rate a setup, give a confidence score from 1 to 10.
- For "should I take this trade", give pros and risks, not certainty.
- For "why did this trigger", explain the likely signal factors.
"""

        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return "I couldn't generate a response."
    except Exception as e:
        return f"Error: {e}"


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 24%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.08), transparent 22%),
            linear-gradient(180deg, #0a0f1c 0%, #0d1320 100%);
        color: #e8eefc;
    }
    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        margin-top: 0.35rem;
        margin-bottom: 0.7rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border-right: 1px solid #1f2a44;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] .stRadio [role="radio"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(51, 65, 85, 0.9);
        border-radius: 14px;
        padding: 0.55rem 0.7rem;
        transition: all 0.18s ease;
    }
    [data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.96) 0%, rgba(15, 23, 42, 1) 100%);
        border-color: rgba(56, 189, 248, 0.65);
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.18);
    }
    [data-testid="stSidebar"] .stRadio [role="radio"] label,
    [data-testid="stSidebar"] .stRadio [role="radio"] div {
        color: #e2e8f0;
    }
    .sidebar-shell {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(15, 23, 42, 0.55) 100%);
        border: 1px solid rgba(51, 65, 85, 0.65);
        border-radius: 18px;
        padding: 0.9rem 1rem 1rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
    }
    .sidebar-kicker {
        color: #38bdf8;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .sidebar-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sidebar-copy {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .sidebar-section-label {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.2rem 0 0.45rem 0;
    }
    .sidebar-info-card {
        background: rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(51, 65, 85, 0.75);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        margin-top: 0.4rem;
    }
    .sidebar-info-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.45rem;
    }
    .sidebar-info-row:last-child {
        margin-bottom: 0;
    }
    .sidebar-info-label {
        color: #94a3b8;
        font-size: 0.84rem;
    }
    .sidebar-info-value {
        color: #f8fafc;
        font-size: 0.88rem;
        font-weight: 700;
        text-align: right;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(17, 26, 44, 0.96) 0%, rgba(13, 19, 32, 0.98) 100%);
        border: 1px solid rgba(51, 65, 85, 0.82);
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(51, 65, 85, 0.75);
        border-radius: 18px;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(51, 65, 85, 0.72);
        background: rgba(15, 23, 42, 0.78);
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(51, 65, 85, 0.74);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.55);
        overflow: hidden;
    }
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.64);
        border: 1px solid rgba(51, 65, 85, 0.74);
        border-radius: 18px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.14);
    }
    .stTextInput input, .stChatInput input, .stTextArea textarea {
        background: rgba(15, 23, 42, 0.88);
        color: #f8fafc;
        border: 1px solid rgba(51, 65, 85, 0.88);
        border-radius: 14px;
    }
    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.32);
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 1) 100%);
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: 0.02em;
        min-height: 2.6rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.14);
    }
    .stButton > button:hover {
        border-color: rgba(56, 189, 248, 0.55);
        color: #ffffff;
    }
    .app-hero {
        background:
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 30%),
            linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 26, 44, 0.98) 100%);
        border: 1px solid rgba(51, 65, 85, 0.76);
        border-radius: 24px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.18);
    }
    .app-kicker {
        color: #38bdf8;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .app-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .page-hero {
        margin: 0.1rem 0 1rem 0;
    }
    .page-eyebrow {
        color: #38bdf8;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }
    .page-title {
        color: #f8fafc;
        font-size: 1.42rem;
        font-weight: 780;
        line-height: 1.08;
        letter-spacing: -0.024em;
        margin-bottom: 0.28rem;
    }
    .page-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 920px;
    }
    .top-trade-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid #f59e0b;
        border-radius: 20px;
        padding: 18px 22px;
        margin: 0.25rem 0 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    }
    .top-trade-kicker {
        color: #fbbf24;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .top-trade-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.9rem;
    }
    .top-trade-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
    }
    .top-trade-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .top-trade-value {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .setup-card {
        border-radius: 18px;
        padding: 16px 18px;
        margin: 0.75rem 0;
        border: 1px solid rgba(71, 85, 105, 0.45);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    .setup-card.take {
        background: linear-gradient(135deg, rgba(20, 83, 45, 0.32) 0%, rgba(15, 23, 42, 0.96) 100%);
        border-color: rgba(34, 197, 94, 0.85);
    }
    .setup-card.watch {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.28) 0%, rgba(15, 23, 42, 0.96) 100%);
        border-color: rgba(251, 191, 36, 0.72);
    }
    .setup-card.avoid {
        background: linear-gradient(135deg, rgba(51, 65, 85, 0.38) 0%, rgba(15, 23, 42, 0.94) 100%);
        border-color: rgba(100, 116, 139, 0.55);
        opacity: 0.92;
    }
    .setup-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.85rem;
    }
    .setup-symbol {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .setup-signal {
        color: #94a3b8;
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.18rem;
    }
    .setup-badge {
        border-radius: 999px;
        padding: 0.32rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: inline-block;
    }
    .setup-badge.take {
        background: rgba(34, 197, 94, 0.18);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.45);
    }
    .setup-badge.watch {
        background: rgba(251, 191, 36, 0.18);
        color: #fde68a;
        border: 1px solid rgba(251, 191, 36, 0.42);
    }
    .setup-badge.avoid {
        background: rgba(148, 163, 184, 0.14);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.34);
    }
    .block-container {
        padding-top: 1.45rem;
        padding-bottom: 2.25rem;
        max-width: 1480px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


symbols = DEFAULT_SYMBOLS
paper = load_paper_trades()
strategy_registry = load_strategy_registry()
strategy_registry = maybe_run_controlled_research_loop(strategy_registry)
algo_update_info = maybe_send_algorithm_status_update(strategy_registry)
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "Auto"

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-shell">
            <div class="sidebar-kicker">Mash Trading</div>
            <div class="sidebar-title">Terminal</div>
            <div class="sidebar-copy">Live signals, paper execution, and market monitoring in one workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_mode = st.segmented_control(
        "Trading Mode",
        options=["Manual", "Auto"],
        default=st.session_state.trading_mode,
        key="trading_mode_selector",
    )
    if selected_mode is not None:
        st.session_state.trading_mode = selected_mode

trading_mode = st.session_state.trading_mode
signals = calc_live_signals(symbols)
paper, new_logged = log_active_signals(signals, paper) if trading_mode == "Auto" else (paper, 0)
paper, newly_closed = update_open_trades(paper)
save_paper_trades(paper)

open_trades = paper[paper["status"].astype(str) == "OPEN"].copy()
closed_trades = paper[paper["status"].astype(str).str.startswith("CLOSED")].copy()
trade_log = closed_trades.copy()
if not trade_log.empty and "result" in trade_log.columns:
    missing_result = trade_log["result"].astype(str).str.strip() == ""
    trade_log.loc[missing_result, "result"] = trade_log.loc[missing_result, "status"].astype(str).apply(
        lambda status: "win" if "WIN" in status else "loss"
    )
paper_pnl = pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0).sum()
current_account_balance = get_account_balance_from_trades(closed_trades)
st.session_state.open_trades = open_trades.to_dict("records")
st.session_state.trade_log = trade_log.to_dict("records")
st.session_state.account_balance = current_account_balance

performance = create_paper_performance_curve(closed_trades)
win_rate = round((performance["pnl"] > 0).mean() * 100, 2) if not closed_trades.empty else 0.0
total_pnl = round(float(performance["pnl"].sum()), 2)


page = st.sidebar.radio(
    "Workspace",
    ["Home", "Money View", "Performance", "Strategy Lab", "Trade Insights", "Market", "MashGPT"],
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">System</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div class="sidebar-info-card">
        <div class="sidebar-info-row">
            <div class="sidebar-info-label">Trading Mode</div>
            <div class="sidebar-info-value">{trading_mode}</div>
        </div>
        <div class="sidebar-info-row">
            <div class="sidebar-info-label">Watchlist</div>
            <div class="sidebar-info-value">{', '.join(symbols)}</div>
        </div>
        <div class="sidebar-info-row">
            <div class="sidebar-info-label">Active Opps</div>
            <div class="sidebar-info-value">{int(signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"]).sum()) if not signals.empty else 0}</div>
        </div>
        <div class="sidebar-info-row">
            <div class="sidebar-info-label">Phone Alerts</div>
            <div class="sidebar-info-value">{'On' if all([EMAIL_SENDER, EMAIL_PASSWORD]) else 'Off'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if st.sidebar.button("Send Test SMS", use_container_width=True):
    sms_ok, sms_error = send_sms_alert("✅ MashGPT test SMS is working.")
    if sms_ok:
        st.sidebar.success("Test SMS sent")
    else:
        st.sidebar.error(f"Test SMS could not be sent{f': {sms_error}' if sms_error else ''}")
    st.sidebar.caption(f"EMAIL_SENDER set: {'Yes' if bool(EMAIL_SENDER) else 'No'}")
    st.sidebar.caption(f"EMAIL_PASSWORD set: {'Yes' if bool(EMAIL_PASSWORD) else 'No'}")
    st.sidebar.caption(f"Gateway address: {VERIZON_SMS_GATEWAY}")


if page == "Home":
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-kicker">Mash Terminal</div>
            <div class="app-title">A sharper command layer for realtime trading signals, paper execution, and market intelligence.</div>
            <div class="app-subtitle">Track the best setup, monitor paper performance, and move from signal to decision inside one polished trading workspace. Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_page_header(
        "Profit Command Center",
        "Stay focused on profit, trade status, and the next action that matters.",
        "Home",
    )
    active_setup_count = int(signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"]).sum()) if not signals.empty else 0
    if len(open_trades) > 0:
        current_status = "IN TRADE"
    elif active_setup_count == 0:
        current_status = "NO SETUPS"
    else:
        current_status = "RUNNING"

    st.markdown(
        f"""
        <div class="top-trade-banner" style="border-color: rgba(34, 197, 94, 0.42);">
            <div class="top-trade-kicker">Profit Command Center</div>
            <div style="font-size: 2.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.7rem;">
                ${round(float(paper_pnl), 2):,.2f}
            </div>
            <div class="top-trade-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
                <div class="top-trade-item">
                    <div class="top-trade-label">Win Rate</div>
                    <div class="top-trade-value">{win_rate:.2f}%</div>
                </div>
                <div class="top-trade-item">
                    <div class="top-trade-label">Current Status</div>
                    <div class="top-trade-value">{current_status}</div>
                </div>
                <div class="top-trade-item">
                    <div class="top-trade-label">Mode</div>
                    <div class="top-trade-value">{trading_mode.upper()}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_left, summary_right = st.columns([1.35, 0.65])
    with summary_left:
        st.markdown("### Top Setup")
        if signals.empty:
            st.caption("No ranked setup is active at the moment.")
        else:
            best_signal = signals.sort_values("score", ascending=False).iloc[0]
            best_score = int(best_signal["score"])
            best_verdict = get_setup_verdict(best_score)
            st.markdown(
                f"""
                <div class="top-trade-banner">
                    <div class="top-trade-kicker">{best_verdict}</div>
                    <div class="top-trade-grid">
                        <div class="top-trade-item">
                            <div class="top-trade-label">Symbol</div>
                            <div class="top-trade-value">{best_signal["symbol"]}</div>
                        </div>
                        <div class="top-trade-item">
                            <div class="top-trade-label">Signal</div>
                            <div class="top-trade-value">{best_signal["signal"]}</div>
                        </div>
                        <div class="top-trade-item">
                            <div class="top-trade-label">Score</div>
                            <div class="top-trade-value">{best_score}</div>
                        </div>
                        <div class="top-trade-item">
                            <div class="top-trade-label">Entry</div>
                            <div class="top-trade-value">{float(best_signal["entry"]):.4f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with summary_right:
        st.markdown("### Live Trade Feed")
        feed = build_live_trade_feed(paper)
        if feed.empty:
            st.caption("No trade events recorded yet.")
        else:
            for _, event in feed.head(6).iterrows():
                event_time = event["event_time"]
                event_label = event_time.strftime("%b %d %I:%M %p") if pd.notna(event_time) else "Recent"
                st.caption(f"{event_label}  {event['event_text']}")

    st.markdown("### Active Opportunities")
    if signals.empty:
        st.caption("No active opportunities right now.")
    else:
        active_opportunities = signals.sort_values("score", ascending=False).head(2).reset_index(drop=True)
        for _, setup in active_opportunities.iterrows():
            score = int(setup["score"])
            setup_verdict = get_setup_verdict(score)
            verdict_class = setup_verdict.lower()
            st.markdown(
                f"""
                <div class="setup-card {verdict_class}">
                    <div class="setup-card-header">
                        <div>
                            <div class="setup-symbol">{setup["symbol"]}</div>
                            <div class="setup-signal">{setup["signal"]}</div>
                        </div>
                        <div class="setup-badge {verdict_class}">{setup_verdict}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            top_metrics = st.columns(4)
            top_metrics[0].metric("Score", score)
            top_metrics[1].metric("Entry", f'{float(setup["entry"]):.4f}')
            top_metrics[2].metric("Stop", f'{float(setup["stop_loss"]):.4f}')
            top_metrics[3].metric("Rel Vol", f'{float(setup["rel_vol"]):.2f}x')

            bottom_metrics = st.columns(4)
            bottom_metrics[0].metric("TP1", f'{float(setup["take_profit_1"]):.4f}')
            bottom_metrics[1].metric("TP2", f'{float(setup["take_profit_2"]):.4f}')
            bottom_metrics[2].metric("Change %", f'{float(setup["change_pct"]):.2f}%')
            bottom_metrics[3].metric("Signal", str(setup["signal"]))
            st.caption(
                f"Context: sentiment {float(setup.get('sentiment_score', 0.0)):+.2f} | "
                f"recent news {str(setup.get('recent_news', 'No'))} | "
                f"macro {str(setup.get('macro_regime', 'neutral'))} | "
                f"caution {str(setup.get('event_caution', 'None'))} | "
                f"online adj {int(pd.to_numeric(setup.get('online_score_adjustment', 0), errors='coerce') or 0):+d}"
            )

            open_symbols = {str(trade.get("symbol", "")) for trade in st.session_state.open_trades}
            if str(setup["symbol"]) in open_symbols:
                st.caption("Already open in Paper Trades")
            elif st.button("Take Trade", key=f'command-center-take-{setup["symbol"]}', use_container_width=True):
                paper, added = add_paper_trade_from_setup(setup, paper)
                if added:
                    st.session_state.open_trades.append(
                        {
                            "symbol": str(setup["symbol"]),
                            "time": str(setup["time"]),
                            "timestamp": str(setup.get("timestamp", setup["time"])),
                            "timeframe": str(setup.get("timeframe", "15m")),
                            "entry": setup["entry"],
                            "stop_loss": setup["stop_loss"],
                            "take_profit_1": setup["take_profit_1"],
                            "take_profit_2": setup["take_profit_2"],
                            "shares": setup["shares"],
                            "risk_pct": setup.get("risk_pct", RISK_PER_TRADE * 100),
                            "account_balance": setup.get("account_balance", current_account_balance),
                            "base_score": setup.get("base_score", setup["score"]),
                            "score": setup["score"],
                            "reason": str(setup.get("reason", "")),
                            "signal": str(setup["signal"]),
                            "sentiment_score": setup.get("sentiment_score", 0.0),
                            "recent_news": setup.get("recent_news", "No"),
                            "macro_regime": setup.get("macro_regime", "neutral"),
                            "event_caution": setup.get("event_caution", "None"),
                            "online_score_adjustment": setup.get("online_score_adjustment", 0),
                            "news_affected": setup.get("news_affected", "No"),
                            "status": "OPEN",
                            "result": "",
                        }
                    )
                    save_paper_trades(paper)
                    st.success("Trade added to Paper Trades")
                else:
                    st.info("Trade already open")

    st.markdown("### Open Trades")
    if open_trades.empty:
        st.caption("No open paper trades.")
    else:
        open_summary = open_trades[["symbol", "signal", "entry", "stop_loss", "take_profit_1", "take_profit_2", "sentiment_score", "recent_news", "macro_regime", "event_caution", "online_score_adjustment"]].copy()
        st.dataframe(open_summary, width="stretch", height=220)

    st.markdown("### Algo Status")
    algo_state = algo_update_info.get("state", {})
    next_slot = algo_update_info.get("next_slot")
    st.caption(f"Last update: {algo_state.get('last_sent_at', '') or 'Not yet sent'}")
    st.caption(f"Signature: {algo_update_info.get('signature_hash', 'n/a')}")
    st.caption(
        "Next update: "
        + (next_slot.strftime("%Y-%m-%d %I:%M %p %Z") if next_slot else "Unavailable")
    )
    if algo_state.get("last_message"):
        st.info(algo_state["last_message"])
    if st.button("Send Sample Algo Update", key="send-sample-algo-update", use_container_width=True):
        sample_message = build_algo_update_message(
            algo_update_info.get("summary", {}),
            algo_update_info.get("changed", False),
        )
        sample_ok, sample_error = send_sms_alert(sample_message)
        if sample_ok:
            st.success("Sample algo update sent")
        else:
            st.error(f"Sample algo update could not be sent{f': {sample_error}' if sample_error else ''}")

elif page == "Money View":
    render_page_header(
        "Money View",
        "Focus on equity growth, daily profit flow, and the trades that are moving the account.",
        "Profit",
    )
    daily_pnl = build_daily_pnl(trade_log)
    best_trade = float(pd.to_numeric(trade_log["pnl"], errors="coerce").max()) if not trade_log.empty else 0.0
    worst_trade = float(pd.to_numeric(trade_log["pnl"], errors="coerce").min()) if not trade_log.empty else 0.0

    mv1, mv2, mv3, mv4 = st.columns(4)
    mv1.metric("Open Trades", int(len(open_trades)))
    mv2.metric("Best Trade", f"${best_trade:,.2f}")
    mv3.metric("Worst Trade", f"${worst_trade:,.2f}")
    mv4.metric("Realized P&L", f"${float(paper_pnl):,.2f}")

    money_left, money_right = st.columns(2)
    with money_left:
        fig_money, ax_money = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
        ax_money.plot(performance["trade_num"], performance["equity"], linewidth=2.8, color="#22c55e")
        ax_money.fill_between(
            performance["trade_num"],
            performance["equity"],
            performance["equity"].min(),
            color="#22c55e",
            alpha=0.12,
        )
        style_dashboard_chart(ax_money, "Equity Curve", "Trade Number", "Account Value")
        fig_money.tight_layout(pad=1.2)
        st.pyplot(fig_money)

    with money_right:
        if daily_pnl.empty:
            st.caption("Daily P&L will appear after trades close on multiple dates.")
        else:
            fig_daily, ax_daily = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
            daily_colors = ["#22c55e" if pnl >= 0 else "#ef4444" for pnl in daily_pnl["pnl"]]
            ax_daily.bar(daily_pnl["day"], daily_pnl["pnl"], color=daily_colors, width=0.6)
            style_dashboard_chart(ax_daily, "Daily P&L", "Day", "P&L ($)")
            plt.setp(ax_daily.get_xticklabels(), rotation=30, ha="right")
            fig_daily.tight_layout(pad=1.2)
            st.pyplot(fig_daily)

elif page == "Dashboard":
    render_page_header(
        "Performance Dashboard",
        "Monitor paper equity, drawdown, and current system activity from one clean trading overview.",
        "Overview",
    )
    if not signals.empty:
        best_signal = signals.sort_values("score", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="top-trade-banner">
                <div class="top-trade-kicker">Top Trade Banner</div>
                <div class="top-trade-grid">
                    <div class="top-trade-item">
                        <div class="top-trade-label">Symbol</div>
                        <div class="top-trade-value">{best_signal["symbol"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Score</div>
                        <div class="top-trade-value">{int(best_signal["score"])}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Entry</div>
                        <div class="top-trade-value">{best_signal["entry"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Stop</div>
                        <div class="top-trade-value">{best_signal["stop_loss"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP1</div>
                        <div class="top-trade-value">{best_signal["take_profit_1"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP2</div>
                        <div class="top-trade-value">{best_signal["take_profit_2"]}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Equity Curve")
        fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
        ax.plot(performance["trade_num"], performance["equity"], linewidth=2.8, color="#38bdf8")
        ax.fill_between(
            performance["trade_num"],
            performance["equity"],
            performance["equity"].min(),
            color="#38bdf8",
            alpha=0.12,
        )
        style_dashboard_chart(ax, "Paper Equity Curve", "Trade Number", "Account Value")
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)

    with c2:
        st.subheader("Drawdown")
        fig2, ax2 = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
        ax2.plot(performance["trade_num"], performance["drawdown"], linewidth=2.8, color="#f97316")
        ax2.fill_between(
            performance["trade_num"],
            performance["drawdown"],
            0,
            color="#f97316",
            alpha=0.14,
        )
        style_dashboard_chart(ax2, "Drawdown Profile", "Trade Number", "Drawdown ($)")
        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2)

    st.subheader("System Activity")
    a1, a2 = st.columns(2)
    a1.metric("New Logged This Refresh", new_logged)
    a2.metric("Closed This Refresh", newly_closed)

elif page == "Performance":
    render_page_header(
        "Performance",
        "Review the current strategy with a concise set of metrics and charts, then open Strategy Lab when you need deeper research controls.",
        "Analytics",
    )

    champion_params = strategy_registry["champion"]["parameters"]
    st.caption(
        f"Backtest universe: {', '.join(BACKTEST_SYMBOLS)} | Lookback: ~3 years of daily candles per symbol"
    )
    with st.spinner("Running historical backtest..."):
        backtest_trades = run_strategy_backtest(
            tuple(BACKTEST_SYMBOLS),
            period="3y",
            interval="1d",
            strategy_params_json=json.dumps(champion_params, sort_keys=True),
        )
    backtest_summary = summarize_backtest_results(backtest_trades)
    backtest_curve = create_paper_performance_curve(backtest_trades)
    closed_trade_pnl = pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0.0) if not closed_trades.empty else pd.Series(dtype=float)
    closed_trade_wins = closed_trade_pnl[closed_trade_pnl > 0]
    closed_trade_losses = closed_trade_pnl[closed_trade_pnl < 0]
    backtest_summary["out_of_sample_pnl"] = backtest_summary["total_pnl"]
    backtest_summary["out_of_sample_win_rate"] = backtest_summary["win_rate"]
    strategy_registry["champion"]["results_summary"] = backtest_summary
    save_strategy_registry(strategy_registry)

    if backtest_trades.empty:
        st.markdown(
            """
            <div class="top-trade-banner" style="border-color: rgba(71, 85, 105, 0.7);">
                <div class="top-trade-kicker" style="color: #cbd5e1;">No Backtest Trades Yet</div>
                <div class="app-subtitle">The historical simulation did not generate enough completed setups for the selected watchlist and timeframe.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total P&L", f"${backtest_summary['total_pnl']:,.2f}")
        p2.metric("Win Rate", f"{backtest_summary['win_rate']:.2f}%")
        p3.metric("Trades", backtest_summary["num_trades"])
        p4.metric("Max Drawdown", f"${backtest_summary['max_drawdown']:,.2f}")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_perf, ax_perf = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
            ax_perf.plot(backtest_curve["trade_num"], backtest_curve["equity"], linewidth=2.8, color="#38bdf8")
            ax_perf.fill_between(
                backtest_curve["trade_num"],
                backtest_curve["equity"],
                backtest_curve["equity"].min(),
                color="#38bdf8",
                alpha=0.12,
            )
            style_dashboard_chart(ax_perf, "Backtest Equity Curve", "Trade Number", "Account Value")
            fig_perf.tight_layout(pad=1.2)
            st.pyplot(fig_perf)

        with chart_col2:
            fig_dd, ax_dd = plt.subplots(figsize=(10.5, 4.8), facecolor="#0f172a")
            ax_dd.plot(backtest_curve["trade_num"], backtest_curve["drawdown"], linewidth=2.8, color="#f97316")
            ax_dd.fill_between(
                backtest_curve["trade_num"],
                backtest_curve["drawdown"],
                0,
                color="#f97316",
                alpha=0.14,
            )
            style_dashboard_chart(ax_dd, "Backtest Drawdown", "Trade Number", "Drawdown ($)")
            fig_dd.tight_layout(pad=1.2)
            st.pyplot(fig_dd)

    st.markdown("### Closed Trade Analytics")
    analytics1, analytics2, analytics3, analytics4 = st.columns(4)
    analytics1.metric("Closed Trades", int(len(closed_trades)))
    analytics2.metric("Average Win", f"${float(closed_trade_wins.mean()) if not closed_trade_wins.empty else 0.0:,.2f}")
    analytics3.metric("Average Loss", f"${float(closed_trade_losses.mean()) if not closed_trade_losses.empty else 0.0:,.2f}")
    analytics4.metric("Realized P&L", f"${float(closed_trade_pnl.sum()) if not closed_trade_pnl.empty else 0.0:,.2f}")

    if closed_trades.empty:
        st.caption("Closed trade analytics will populate once paper trades complete.")
    else:
        closed_trade_table = closed_trades.copy()
        for col in ["entry", "exit_price", "pnl"]:
            if col in closed_trade_table.columns:
                closed_trade_table[col] = pd.to_numeric(closed_trade_table[col], errors="coerce")
        display_columns = [
            col for col in [
                "close_time",
                "symbol",
                "timeframe",
                "signal",
                "reason",
                "score",
                "result",
                "status",
                "entry",
                "exit_price",
                "pnl",
                "exit_reason",
            ] if col in closed_trade_table.columns
        ]
        st.dataframe(
            closed_trade_table[display_columns].sort_index(ascending=False),
            width="stretch",
            height=240,
        )

    st.caption("Open Trade Insights for the setup-quality breakdown and completed-trade metadata view.")

    with st.expander("Strategy Lab", expanded=False):
        render_strategy_lab_section(strategy_registry)

elif page == "Trade Insights":
    render_page_header(
        "Trade Insights",
        "Understand which setups actually work by reviewing completed trades, quality signals, and outcome patterns.",
        "Analytics",
    )
    render_trade_insights_section(trade_log)

elif page == "Strategy Lab":
    render_page_header(
        "Strategy Lab",
        "One place to understand the live algorithm, backtest behavior, strategy settings, and recent research progress.",
        "Research",
    )

    champion_params = strategy_registry["champion"]["parameters"]
    algo_state = algo_update_info.get("state", {})
    algo_changed = bool(algo_update_info.get("changed", False))
    live_status = "UPDATED" if algo_changed else "STABLE"

    st.markdown("### Live Algorithm Status")
    live1, live2, live3, live4, live5 = st.columns(5)
    live1.metric("Current P&L", f"${float(paper_pnl):,.2f}")
    live2.metric("Current Win Rate", f"{win_rate:.2f}%")
    live3.metric("Trades", int(len(closed_trades)))
    live4.metric("Status", live_status)
    live5.metric("Changed Recently", "Yes" if algo_changed else "No")
    if algo_state.get("last_message"):
        st.caption(f"Last algo update: {algo_state.get('last_sent_at', '') or 'Not yet sent'}")
        st.info(algo_state["last_message"])

    st.markdown("### Backtest Results")
    st.caption(
        f"Backtest universe: {', '.join(BACKTEST_SYMBOLS)} | Lookback: ~3 years of daily candles per symbol"
    )
    with st.spinner("Running strategy lab backtest..."):
        lab_backtest_trades, backtest_debug = run_strategy_backtest(
            tuple(BACKTEST_SYMBOLS),
            period="3y",
            interval="1d",
            strategy_params_json=json.dumps(champion_params, sort_keys=True),
            collect_debug=True,
        )
    lab_backtest_summary = summarize_backtest_results(lab_backtest_trades)
    lab_backtest_curve = create_paper_performance_curve(lab_backtest_trades)
    strategy_registry["champion"]["results_summary"] = lab_backtest_summary
    save_strategy_registry(strategy_registry)

    bt1, bt2, bt3, bt4 = st.columns(4)
    bt1.metric("Backtest P&L", f"${lab_backtest_summary.get('total_pnl', 0.0):,.2f}")
    bt2.metric("Win Rate", f"{lab_backtest_summary.get('win_rate', 0.0):.2f}%")
    bt3.metric("Max Drawdown", f"${lab_backtest_summary.get('max_drawdown', 0.0):,.2f}")
    bt4.metric("Trade Count", int(lab_backtest_summary.get("num_trades", 0)))

    fig_lab, ax_lab = plt.subplots(figsize=(11.0, 4.8), facecolor="#0f172a")
    ax_lab.plot(lab_backtest_curve["trade_num"], lab_backtest_curve["equity"], linewidth=2.8, color="#38bdf8")
    ax_lab.fill_between(
        lab_backtest_curve["trade_num"],
        lab_backtest_curve["equity"],
        lab_backtest_curve["equity"].min(),
        color="#38bdf8",
        alpha=0.12,
    )
    style_dashboard_chart(ax_lab, "Backtest Equity Curve", "Trade Number", "Account Value")
    fig_lab.tight_layout(pad=1.2)
    st.pyplot(fig_lab)

    st.markdown("#### Backtest Filter Debug")
    debug1, debug2, debug3 = st.columns(3)
    debug1.metric("Symbols Scanned", int(backtest_debug.get("symbols_scanned", 0)))
    debug2.metric("Bars Evaluated", int(backtest_debug.get("bars_evaluated", 0)))
    debug3.metric("Trades Triggered", int(backtest_debug.get("trades_triggered", 0)))

    pass_counts = backtest_debug.get("filter_pass_counts", {})
    rejection_counts = backtest_debug.get("rejection_counts", {})
    pass_rows = pd.DataFrame(
        [
            {"filter_stage": "Trend", "passed_count": int(pass_counts.get("trend", 0))},
            {"filter_stage": "RSI", "passed_count": int(pass_counts.get("rsi", 0))},
            {"filter_stage": "Volume", "passed_count": int(pass_counts.get("volume", 0))},
            {"filter_stage": "Breakout", "passed_count": int(pass_counts.get("breakout", 0))},
            {"filter_stage": "Reward / Risk", "passed_count": int(pass_counts.get("reward_risk", 0))},
            {"filter_stage": "Score", "passed_count": int(pass_counts.get("score", 0))},
            {"filter_stage": "Market Regime", "passed_count": int(pass_counts.get("market_regime", 0))},
        ]
    )
    rejection_rows = pd.DataFrame(
        [
            {"rejection_reason": reason, "count": int(count)}
            for reason, count in rejection_counts.items()
            if int(count) > 0
        ]
    ).sort_values("count", ascending=False)

    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.dataframe(pass_rows, width="stretch", height=280)
    with debug_col2:
        if rejection_rows.empty:
            st.success("No rejection reasons were recorded in this backtest run.")
        else:
            st.dataframe(rejection_rows, width="stretch", height=280)
            top_rejection = rejection_rows.iloc[0]
            st.caption(
                f"Most common blocker: {top_rejection['rejection_reason']} ({int(top_rejection['count'])} occurrences)"
            )

    st.markdown("### Current Strategy Settings")
    settings_rows = pd.DataFrame(
        [
            {"setting": "Score Threshold", "value": champion_params.get("score_threshold", "")},
            {"setting": "RSI Long Minimum", "value": champion_params.get("rsi_long_min", "")},
            {"setting": "RSI Short Maximum", "value": champion_params.get("rsi_short_max", "")},
            {"setting": "Relative Volume Minimum", "value": champion_params.get("rel_vol_min", "")},
            {"setting": "Trend Filter", "value": f"EMA {champion_params.get('ema_long_len', 50)} with distance confirmation"},
            {"setting": "EMA Short / Long", "value": f"{champion_params.get('ema_short_len', '')} / {champion_params.get('ema_long_len', '')}"},
            {"setting": "Risk / Reward Threshold", "value": ">= 2.0"},
            {"setting": "Stop Model", "value": "Recent swing structure"},
            {"setting": "Online Context Layer", "value": "Enabled, capped confidence adjustment only"},
            {"setting": "Current Macro Regime", "value": get_macro_regime_context().get("macro_regime", "neutral")},
        ]
    )
    st.dataframe(settings_rows, width="stretch", height=280)

    challenger = strategy_registry.get("challenger")
    if challenger:
        st.markdown("### Champion vs Challenger")
        champion_summary = strategy_registry["champion"].get("results_summary", {})
        challenger_summary = challenger.get("results_summary", {})
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.markdown(f"**Champion:** `{strategy_to_label(strategy_registry['champion'])}`")
            st.caption(strategy_registry["champion"].get("promotion_status", "Live champion"))
            st.metric("Champion P&L", f"${champion_summary.get('total_pnl', 0.0):,.2f}")
            st.metric("Champion Win Rate", f"{champion_summary.get('win_rate', 0.0):.2f}%")
        with cmp2:
            st.markdown(f"**Challenger:** `{strategy_to_label(challenger)}`")
            st.caption(challenger.get("promotion_status", "Under review"))
            st.metric("Challenger P&L", f"${challenger_summary.get('total_pnl', 0.0):,.2f}")
            st.metric("Challenger Win Rate", f"{challenger_summary.get('win_rate', 0.0):.2f}%")

        eligible, checks = compare_strategy_results(champion_summary, challenger_summary)
        checks_df = pd.DataFrame(
            [
                {
                    "check": check.get("check", ""),
                    "passed": "Yes" if check.get("passed") else "No",
                    "detail": check.get("detail", ""),
                }
                for check in checks
            ]
        )
        st.dataframe(checks_df, width="stretch", height=210)
        st.caption("Promotion ready" if eligible else "Challenger has not passed promotion gates yet.")

    st.markdown("### Latest Experiments")
    render_strategy_lab_section(strategy_registry)

elif page == "About":
    render_page_header(
        "About Mash Terminal",
        "A focused trading workspace built for research, signal review, and simulated execution.",
        "Product Overview",
    )

    st.markdown(
        """
        Mash Terminal is designed to bring market scanning, setup review, paper execution, and trader-facing intelligence into one streamlined environment. Instead of splitting the workflow across disconnected tools, the platform combines live signal discovery, trade tracking, analytics, and AI-assisted commentary inside a single interface built to feel more like a professional trading product than a collection of utilities.

        At its core, the goal of Mash Terminal is simple: help users move from observation to decision with greater clarity. The platform continuously watches a defined symbol list, evaluates market structure, momentum, relative volume, and reward-to-risk characteristics, and surfaces the strongest long or short ideas in a format that is easier to scan quickly. Rather than forcing users to interpret raw outputs alone, the app organizes those signals into ranked views, top-setup callouts, and cleaner dashboards that make quality differences more obvious.

        The scanning layer is paired with a simulated execution workflow through paper trading. When a trade is opened, the app records the setup, tracks price against its stop and profit targets, and moves that trade through its lifecycle as market conditions change. Closed trades feed directly into the performance views, allowing the platform to function not only as a scanner, but also as a lightweight trade journal for reviewing outcomes, equity growth, and win-loss behavior over time.

        MashGPT extends that workflow by adding an intelligence layer on top of the platform data. It can interpret the strongest current setup, explain why a signal may or may not be compelling, frame trade risk, and respond to broader market questions using the same context the rest of the application is displaying. The goal is not to replace trader judgment, but to make the surrounding analysis faster, more structured, and easier to act on.

        The Live Market view supports that decision process by giving users a cleaner way to monitor a symbol in real time. It combines a live chart, quick symbol switching, and current market context so the active name can be watched more closely once a setup becomes interesting. In practice, this makes the app useful both for discovering opportunities and for staying focused on the few symbols that matter most in the moment.

        Mash Terminal also supports two operating styles. In Manual mode, the user stays in control of which setups become paper trades, typically through the explicit trade actions on the setup views. In Auto mode, the platform can log qualifying signals into paper trades automatically using the same underlying rules and tracking system. This allows the product to serve both discretionary review workflows and more systematic simulated monitoring.

        The platform is intended for research, analysis, and simulated trading. It is built to help users evaluate setups, understand trade structure, and review performance in a more disciplined way, without presenting itself as a guarantee of outcomes or a substitute for independent judgment. In that sense, Mash Terminal is best understood as a decision-support product: a professional-feeling workspace for scanning markets, organizing information, and practicing execution with greater structure.
        """
    )

elif page == "Trades":
    render_page_header(
        "Trades",
        "One place to review setups, open positions, and closed-trade performance without jumping across redundant pages.",
        "Execution Hub",
    )

    performance_trades = closed_trades.copy()
    performance_trades["pnl"] = pd.to_numeric(performance_trades["pnl"], errors="coerce").fillna(0.0)
    wins = performance_trades[performance_trades["pnl"] > 0]
    losses = performance_trades[performance_trades["pnl"] < 0]
    total_closed_trades = int(len(performance_trades))
    average_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
    average_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Live Setups", int(signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"]).sum()) if not signals.empty else 0)
    t2.metric("Open Trades", int(len(open_trades)))
    t3.metric("Total P&L", f"${total_pnl:,.2f}")
    t4.metric("Win Rate", f"{win_rate:.2f}%")
    t5.metric("Average Loss", f"${average_loss:,.2f}")

    if signals.empty:
        st.markdown(
            """
            <div class="top-trade-banner" style="border-color: rgba(71, 85, 105, 0.7);">
                <div class="top-trade-kicker" style="color: #cbd5e1;">Scanner Standing By</div>
                <div class="app-subtitle">No live setups are active right now. Open trades and journal performance remain available below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        sorted_signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        best_setup = sorted_signals.iloc[0]
        best_score = int(best_setup["score"])
        verdict = get_setup_verdict(best_score)

        st.markdown(
            f"""
            <div class="top-trade-banner">
                <div class="top-trade-kicker">Top Setup</div>
                <div class="top-trade-grid">
                    <div class="top-trade-item">
                        <div class="top-trade-label">Symbol</div>
                        <div class="top-trade-value">{best_setup["symbol"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Signal</div>
                        <div class="top-trade-value">{best_setup["signal"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Score</div>
                        <div class="top-trade-value">{best_score}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Entry</div>
                        <div class="top-trade-value">{best_setup["entry"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Stop</div>
                        <div class="top-trade-value">{best_setup["stop_loss"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Verdict</div>
                        <div class="top-trade-value">{verdict}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Ranked Setups")
        for _, setup in sorted_signals.iterrows():
            score = int(setup["score"])
            setup_verdict = get_setup_verdict(score)
            verdict_class = setup_verdict.lower()

            st.markdown(
                f"""
                <div class="setup-card {verdict_class}">
                    <div class="setup-card-header">
                        <div>
                            <div class="setup-symbol">{setup["symbol"]}</div>
                            <div class="setup-signal">{setup["signal"]}</div>
                        </div>
                        <div class="setup-badge {verdict_class}">{setup_verdict}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            top_metrics = st.columns(4)
            top_metrics[0].metric("Score", score)
            top_metrics[1].metric("Entry", f'{float(setup["entry"]):.4f}')
            top_metrics[2].metric("Stop", f'{float(setup["stop_loss"]):.4f}')
            top_metrics[3].metric("Rel Vol", f'{float(setup["rel_vol"]):.2f}x')

            bottom_metrics = st.columns(4)
            bottom_metrics[0].metric("TP1", f'{float(setup["take_profit_1"]):.4f}')
            bottom_metrics[1].metric("TP2", f'{float(setup["take_profit_2"]):.4f}')
            bottom_metrics[2].metric("Change %", f'{float(setup["change_pct"]):.2f}%')
            bottom_metrics[3].metric("Shares", int(setup["shares"]))

            open_symbols = {str(trade.get("symbol", "")) for trade in st.session_state.open_trades}
            if str(setup["symbol"]) in open_symbols:
                st.caption("Already open in Paper Trades")
            elif st.button("Take Trade", key=f'trades-take-{setup["symbol"]}', use_container_width=True):
                paper, added = add_paper_trade_from_setup(setup, paper)
                if added:
                    session_trade = {
                        "symbol": str(setup["symbol"]),
                        "time": str(setup["time"]),
                        "timestamp": str(setup.get("timestamp", setup["time"])),
                        "timeframe": str(setup.get("timeframe", "15m")),
                        "entry": setup["entry"],
                        "stop_loss": setup["stop_loss"],
                        "take_profit_1": setup["take_profit_1"],
                        "take_profit_2": setup["take_profit_2"],
                        "shares": setup["shares"],
                        "risk_pct": setup.get("risk_pct", RISK_PER_TRADE * 100),
                        "account_balance": setup.get("account_balance", current_account_balance),
                        "base_score": setup.get("base_score", setup["score"]),
                        "score": setup["score"],
                        "reason": str(setup.get("reason", "")),
                        "signal": str(setup["signal"]),
                        "sentiment_score": setup.get("sentiment_score", 0.0),
                        "recent_news": setup.get("recent_news", "No"),
                        "macro_regime": setup.get("macro_regime", "neutral"),
                        "event_caution": setup.get("event_caution", "None"),
                        "online_score_adjustment": setup.get("online_score_adjustment", 0),
                        "news_affected": setup.get("news_affected", "No"),
                        "status": "OPEN",
                        "result": "",
                    }
                    st.session_state.open_trades.append(session_trade)
                    save_paper_trades(paper)
                    st.success("Trade added to Paper Trades")
                else:
                    st.info("Trade already open")

            st.markdown("")

    st.markdown("### Portfolio Snapshot")
    portfolio_left, portfolio_right = st.columns(2)

    with portfolio_left:
        st.markdown("#### Open Paper Trades")
        if open_trades.empty:
            st.caption("No open paper trades.")
        else:
            open_columns = ["symbol", "signal", "entry", "stop_loss", "take_profit_1", "take_profit_2", "shares"]
            available_open_columns = [col for col in open_columns if col in open_trades.columns]
            st.dataframe(open_trades[available_open_columns].copy(), width="stretch", height=260)

    with portfolio_right:
        st.markdown("#### Journal Summary")
        j1, j2, j3 = st.columns(3)
        j1.metric("Closed Trades", total_closed_trades)
        j2.metric("Average Win", f"${average_win:,.2f}")
        j3.metric("Average Loss", f"${average_loss:,.2f}")

    if performance_trades.empty:
        st.caption("Performance charts and journal entries will appear once trades close.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_perf, ax_perf = plt.subplots(figsize=(10.5, 4.6), facecolor="#0f172a")
            ax_perf.plot(performance["trade_num"], performance["equity"], linewidth=2.8, color="#38bdf8")
            ax_perf.fill_between(
                performance["trade_num"],
                performance["equity"],
                performance["equity"].min(),
                color="#38bdf8",
                alpha=0.12,
            )
            style_dashboard_chart(ax_perf, "Equity Curve", "Trade Number", "Account Value")
            fig_perf.tight_layout(pad=1.2)
            st.pyplot(fig_perf)

        with chart_col2:
            fig_pnl, ax_pnl = plt.subplots(figsize=(10.5, 4.6), facecolor="#0f172a")
            pnl_colors = ["#22c55e" if pnl >= 0 else "#ef4444" for pnl in performance["pnl"]]
            ax_pnl.bar(performance["trade_num"], performance["pnl"], color=pnl_colors, width=0.65)
            style_dashboard_chart(ax_pnl, "P&L Per Trade", "Trade Number", "P&L ($)")
            ax_pnl.axhline(0, color="#64748b", linewidth=1.0, alpha=0.8)
            fig_pnl.tight_layout(pad=1.2)
            st.pyplot(fig_pnl)

        st.markdown("### Closed Trade Journal")
        journal_columns = [
            "symbol",
            "signal",
            "time",
            "close_time",
            "entry",
            "exit_price",
            "shares",
            "status",
            "exit_reason",
            "pnl",
        ]
        available_journal_columns = [col for col in journal_columns if col in performance_trades.columns]
        st.dataframe(performance_trades[available_journal_columns].copy(), width="stretch", height=320)

elif page == "Live Signals":
    render_page_header(
        "Live Signals",
        "Scan the current signal feed and review what the algorithm is flagging right now.",
        "Market Feed",
    )
    if signals.empty:
        st.markdown(
            """
            <div class="top-trade-banner" style="border-color: rgba(71, 85, 105, 0.7);">
                <div class="top-trade-kicker" style="color: #cbd5e1;">Scanner Standing By</div>
                <div class="app-subtitle">No live signals are active right now. The scanner is still monitoring the watchlist for stronger long or short setups.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        live_signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        top_signal = live_signals.iloc[0]
        top_score = int(top_signal["score"])
        top_verdict = get_setup_verdict(top_score)

        st.markdown(
            f"""
            <div class="top-trade-banner">
                <div class="top-trade-kicker">Top Signal</div>
                <div class="top-trade-grid">
                    <div class="top-trade-item">
                        <div class="top-trade-label">Symbol</div>
                        <div class="top-trade-value">{top_signal["symbol"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Signal</div>
                        <div class="top-trade-value">{top_signal["signal"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Score</div>
                        <div class="top-trade-value">{top_score}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Entry</div>
                        <div class="top-trade-value">{float(top_signal["entry"]):.4f}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Stop</div>
                        <div class="top-trade-value">{float(top_signal["stop_loss"]):.4f}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Verdict</div>
                        <div class="top-trade-value">{top_verdict}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Signal Scanner")
        for _, signal_row in live_signals.iterrows():
            score = int(signal_row["score"])
            signal_verdict = get_setup_verdict(score)
            verdict_class = signal_verdict.lower()

            st.markdown(
                f"""
                <div class="setup-card {verdict_class}">
                    <div class="setup-card-header">
                        <div>
                            <div class="setup-symbol">{signal_row["symbol"]}</div>
                            <div class="setup-signal">{signal_row["signal"]}</div>
                        </div>
                        <div class="setup-badge {verdict_class}">{signal_verdict}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            signal_top_metrics = st.columns(5)
            signal_top_metrics[0].metric("Score", score)
            signal_top_metrics[1].metric("Entry", f'{float(signal_row["entry"]):.4f}')
            signal_top_metrics[2].metric("Stop", f'{float(signal_row["stop_loss"]):.4f}')
            signal_top_metrics[3].metric("TP1", f'{float(signal_row["take_profit_1"]):.4f}')
            signal_top_metrics[4].metric("TP2", f'{float(signal_row["take_profit_2"]):.4f}')

            signal_bottom_metrics = st.columns(4)
            signal_bottom_metrics[0].metric("Shares", int(signal_row["shares"]))
            signal_bottom_metrics[1].metric("Rel Vol", f'{float(signal_row["rel_vol"]):.2f}x')
            signal_bottom_metrics[2].metric("Change %", f'{float(signal_row["change_pct"]):.2f}%')
            signal_bottom_metrics[3].metric("Signal Type", str(signal_row["signal"]))

            st.markdown("")

elif page == "Setups":
    render_page_header(
        "Setups",
        "Ranked trade opportunities with cleaner scoring, verdicts, and fast paper-trade execution.",
        "Opportunity Board",
    )
    if signals.empty:
        st.info("No setups available right now.")
    else:
        sorted_signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
        best_setup = sorted_signals.iloc[0]
        best_score = int(best_setup["score"])
        verdict = get_setup_verdict(best_score)

        st.markdown(
            f"""
            <div class="top-trade-banner">
                <div class="top-trade-kicker">Top Setup</div>
                <div class="top-trade-grid">
                    <div class="top-trade-item">
                        <div class="top-trade-label">Symbol</div>
                        <div class="top-trade-value">{best_setup["symbol"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Score</div>
                        <div class="top-trade-value">{best_score}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Entry</div>
                        <div class="top-trade-value">{best_setup["entry"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Stop</div>
                        <div class="top-trade-value">{best_setup["stop_loss"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP1</div>
                        <div class="top-trade-value">{best_setup["take_profit_1"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP2</div>
                        <div class="top-trade-value">{best_setup["take_profit_2"]}</div>
                    </div>
                </div>
                <div class="top-trade-kicker" style="margin-top: 0.9rem; margin-bottom: 0;">Verdict: {verdict} | Signal: {best_setup["signal"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Score Overview")
        fig3, ax3 = plt.subplots(figsize=(10.5, 3.6), facecolor="#0f172a")
        bar_colors = ["#22c55e" if score >= 70 else "#f59e0b" if score >= 50 else "#64748b" for score in sorted_signals["score"]]
        ax3.bar(sorted_signals["symbol"], sorted_signals["score"], color=bar_colors, width=0.6)
        style_dashboard_chart(ax3, "Setup Score Overview", "Symbol", "Score")
        ax3.set_ylim(0, max(100, float(sorted_signals["score"].max()) + 10))
        plt.setp(ax3.get_xticklabels(), rotation=0, ha="center")
        fig3.tight_layout(pad=1.1)
        st.pyplot(fig3)

        st.markdown("### Ranked Setups")
        for _, setup in sorted_signals.iterrows():
            score = int(setup["score"])
            setup_verdict = get_setup_verdict(score)
            verdict_class = setup_verdict.lower()

            st.markdown(
                f"""
                <div class="setup-card {verdict_class}">
                    <div class="setup-card-header">
                        <div>
                            <div class="setup-symbol">{setup["symbol"]}</div>
                            <div class="setup-signal">{setup["signal"]}</div>
                        </div>
                        <div class="setup-badge {verdict_class}">{setup_verdict}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            top_metrics = st.columns(4)
            top_metrics[0].metric("Score", score)
            top_metrics[1].metric("Entry", f'{float(setup["entry"]):.4f}')
            top_metrics[2].metric("Stop", f'{float(setup["stop_loss"]):.4f}')
            top_metrics[3].metric("Rel Vol", f'{float(setup["rel_vol"]):.2f}x')

            bottom_metrics = st.columns(4)
            bottom_metrics[0].metric("TP1", f'{float(setup["take_profit_1"]):.4f}')
            bottom_metrics[1].metric("TP2", f'{float(setup["take_profit_2"]):.4f}')
            bottom_metrics[2].metric("Change %", f'{float(setup["change_pct"]):.2f}%')
            bottom_metrics[3].metric("Signal", str(setup["signal"]))

            open_symbols = {str(trade.get("symbol", "")) for trade in st.session_state.open_trades}
            if str(setup["symbol"]) in open_symbols:
                st.caption("Already open in Paper Trades")
            elif st.button("Take Trade", key=f'take-trade-{setup["symbol"]}', use_container_width=True):
                paper, added = add_paper_trade_from_setup(setup, paper)
                if added:
                    session_trade = {
                        "symbol": str(setup["symbol"]),
                        "time": str(setup["time"]),
                        "timestamp": str(setup.get("timestamp", setup["time"])),
                        "timeframe": str(setup.get("timeframe", "15m")),
                        "entry": setup["entry"],
                        "stop_loss": setup["stop_loss"],
                        "take_profit_1": setup["take_profit_1"],
                        "take_profit_2": setup["take_profit_2"],
                        "shares": setup["shares"],
                        "risk_pct": setup.get("risk_pct", RISK_PER_TRADE * 100),
                        "account_balance": setup.get("account_balance", current_account_balance),
                        "base_score": setup.get("base_score", setup["score"]),
                        "score": setup["score"],
                        "reason": str(setup.get("reason", "")),
                        "signal": str(setup["signal"]),
                        "sentiment_score": setup.get("sentiment_score", 0.0),
                        "recent_news": setup.get("recent_news", "No"),
                        "macro_regime": setup.get("macro_regime", "neutral"),
                        "event_caution": setup.get("event_caution", "None"),
                        "online_score_adjustment": setup.get("online_score_adjustment", 0),
                        "news_affected": setup.get("news_affected", "No"),
                        "status": "OPEN",
                        "result": "",
                    }
                    st.session_state.open_trades.append(session_trade)
                    save_paper_trades(paper)
                    st.success("Trade added to Paper Trades")
                else:
                    st.info("Trade already open")

            st.markdown("")

elif page == "Paper Trades":
    render_page_header(
        "Paper Trades",
        "Review open exposure, closed outcomes, and overall paper performance in one place.",
        "Execution",
    )
    st.markdown("### Open Paper Trades")
    if open_trades.empty:
        st.write("No open paper trades.")
    else:
        st.dataframe(open_trades, width="stretch", height=260)

    st.markdown("### Closed Paper Trades")
    if closed_trades.empty:
        st.write("No closed paper trades.")
    else:
        st.dataframe(closed_trades, width="stretch", height=260)

    st.metric("Paper P&L", f"${round(float(paper_pnl), 2)}")

elif page == "MashGPT":
    render_page_header(
        "MashGPT",
        "Ask for trader-focused readouts on live setups, risk, and market context with faster feedback.",
        "Market Intelligence",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", "Ask me about the best setup, trade confidence, risk/reward, or any general question.")
        ]
    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    for role, content in st.session_state.chat_history:
        avatar = "🙂" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(content)

    prompt = st.chat_input("Ask MashGPT anything")

    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user", avatar="🙂"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            recent_times = st.session_state.response_times
            estimated_time = sum(recent_times) / len(recent_times) if recent_times else 4.0
            thinking_placeholder = st.empty()
            status_placeholder = st.empty()
            status_placeholder.caption(f"Estimated time remaining: {estimated_time:.1f}s")
            thinking_messages = [
                "Analyzing price action...",
                "Checking volume...",
                "Evaluating setup quality...",
                "Calculating risk/reward...",
            ]
            for message in thinking_messages:
                thinking_placeholder.caption(message)
                time.sleep(0.5)
            start_time = time.time()
            with st.spinner("MashGPT is analyzing the market..."):
                reply = ask_mashgpt(prompt, signals, open_trades, closed_trades)
            duration = time.time() - start_time
            st.session_state.response_times = (recent_times + [duration])[-5:]
            thinking_placeholder.empty()
            status_placeholder.caption(f"Done in {duration:.2f}s")
            st.write(reply)

        st.session_state.chat_history.append(("assistant", reply))

elif page == "Market":
    render_page_header(
        "Market",
        "Monitor the watchlist and stay focused on the chart plus the few market numbers that matter.",
        "Watchlist",
    )
    st_autorefresh(interval=5000, key="live_market_refresh")

    if "live_market_symbol" not in st.session_state:
        st.session_state["live_market_symbol"] = "NVDA"

    selected_symbol = st.session_state.get("live_market_symbol", "NVDA")
    # default timeframe (TradingView will handle UI anyway)
    timeframe = "60"

    st.session_state["live_market_symbol"] = selected_symbol
    quick1, quick2, quick3, quick4 = st.columns(4)
    if quick1.button("NVDA", use_container_width=True):
        st.session_state["live_market_symbol"] = "NVDA"
        st.rerun()
    if quick2.button("AAPL", use_container_width=True):
        st.session_state["live_market_symbol"] = "AAPL"
        st.rerun()
    if quick3.button("TSLA", use_container_width=True):
        st.session_state["live_market_symbol"] = "TSLA"
        st.rerun()
    if quick4.button("SPY", use_container_width=True):
        st.session_state["live_market_symbol"] = "SPY"
        st.rerun()

    selected_symbol = st.session_state["live_market_symbol"]

    st.markdown("### Watchlist")
    watchlist_symbols = ["NVDA", "AAPL", "TSLA", "SPY"]
    watchlist_snapshot = []
    if not signals.empty:
        ranked_signals = signals.sort_values("score", ascending=False).drop_duplicates(subset=["symbol"])
        for symbol_name in watchlist_symbols:
            matched = ranked_signals[ranked_signals["symbol"].astype(str) == symbol_name]
            if matched.empty:
                watchlist_snapshot.append(
                    {
                        "symbol": symbol_name,
                        "signal": "NO SIGNAL",
                        "score": 0,
                        "verdict": "AVOID",
                    }
                )
            else:
                row = matched.iloc[0]
                watchlist_snapshot.append(
                    {
                        "symbol": symbol_name,
                        "signal": str(row["signal"]),
                        "score": int(row["score"]),
                        "verdict": get_setup_verdict(float(row["score"])),
                    }
                )
    else:
        for symbol_name in watchlist_symbols:
            watchlist_snapshot.append(
                {"symbol": symbol_name, "signal": "NO SIGNAL", "score": 0, "verdict": "AVOID"}
            )

    watch_cols = st.columns(len(watchlist_snapshot))
    for idx, snapshot in enumerate(watchlist_snapshot):
        with watch_cols[idx]:
            st.markdown(
                f"""
                <div class="setup-card {snapshot["verdict"].lower()}">
                    <div class="setup-card-header">
                        <div>
                            <div class="setup-symbol">{snapshot["symbol"]}</div>
                            <div class="setup-signal">{snapshot["signal"]}</div>
                        </div>
                        <div class="setup-badge {snapshot["verdict"].lower()}">{snapshot["verdict"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Score: {snapshot['score']}")

    market_df = fetch_history(
        selected_symbol,
        period="6mo" if timeframe in ["D", "W"] else "5d",
        interval="1d" if timeframe in ["D", "W"] else "15m",
    )

    st.markdown("")
    polygon_trade = get_polygon_last_trade(selected_symbol)
    polygon_prev = get_polygon_prev_day(selected_symbol)
    market_status = get_polygon_market_status()

    if market_status:
        market_name = market_status.get("market", "unknown")
        st.caption(f"Market status: {market_name}")

    tradingview_html = f"""
    <div id="tradingview_chart" style="width:100%; height:720px;"></div>

    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 720,
        "symbol": "{selected_symbol}",
        "interval": "{timeframe}",
        "timezone": "America/Los_Angeles",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#0b1220",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "withdateranges": true,
        "container_id": "tradingview_chart"
      }});
    </script>
    """
    st.markdown("### Live Chart")
    components.html(tradingview_html, height=740)
    st.caption("Chart display by TradingView. Live stats use Polygon when available.")

    st.markdown("### Market Snapshot")

    if polygon_trade and polygon_prev:
        latest_close = float(polygon_trade["price"]) if polygon_trade.get("price") is not None else None
        prev_close = float(polygon_prev["close"]) if polygon_prev.get("close") is not None else None
        high_val = float(polygon_prev["high"]) if polygon_prev.get("high") is not None else None
        low_val = float(polygon_prev["low"]) if polygon_prev.get("low") is not None else None
        latest_volume = int(polygon_prev["volume"]) if polygon_prev.get("volume") is not None else None

        if latest_close is not None and prev_close not in (None, 0):
            change_pct = ((latest_close - prev_close) / prev_close) * 100
        else:
            change_pct = None

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Live Price", f"{latest_close:.2f}" if latest_close is not None else "N/A")
        s2.metric("Change %", f"{change_pct:.2f}%" if change_pct is not None else "N/A")
        s3.metric("Day Range", f"{low_val:.2f} - {high_val:.2f}" if low_val is not None and high_val is not None else "N/A")
        s4.metric("Volume", f"{latest_volume:,}" if latest_volume is not None else "N/A")

        st.caption("Live price powered by Polygon")

    elif not market_df.empty:
        latest_close = float(market_df["Close"].iloc[-1])
        first_close = float(market_df["Close"].iloc[0])
        change_pct = ((latest_close - first_close) / first_close) * 100 if first_close != 0 else 0.0
        latest_volume = int(market_df["Volume"].iloc[-1])
        high_val = float(market_df["High"].max())
        low_val = float(market_df["Low"].min())

        stats_left, stats_center, stats_right = st.columns([0.04, 0.92, 0.04])
        with stats_center:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Last", f"{latest_close:.2f}")
            s2.metric("Change %", f"{change_pct:.2f}%")
            s3.metric("Range", f"{low_val:.2f} - {high_val:.2f}")
            s4.metric("Volume", f"{latest_volume:,}")

        st.caption("Fallback data from yfinance")
    else:
        st.warning(f"No data found for {selected_symbol}")
