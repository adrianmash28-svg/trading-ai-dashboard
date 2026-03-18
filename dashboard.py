import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage

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


st.set_page_config(page_title="Mash Trading Dashboard", layout="wide")


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
POLYGON_API_KEY = get_secret("ypKE7G5kgwYcGEPApyKMWjpgp4JGpCTT", "")
VERIZON_SMS_GATEWAY = "3109911161@vtext.com"

PAPER_TRADES_FILE = "paper_trades.csv"
DEFAULT_SYMBOLS = ["META", "NVDA", "AAPL", "MSFT"]
STARTING_EQUITY = 10000.0


def get_openai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


client = get_openai_client()


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
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "shares",
        "score",
        "signal",
        "status",
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


def calc_live_signals(symbols):
    rows = []

    for symbol in symbols:
        df = fetch_history(symbol, period="5d", interval="15m")
        if df.empty or len(df) < 60:
            continue

        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["vol_avg20"] = df["Volume"].rolling(20).mean()
        df["rel_vol"] = df["Volume"] / df["vol_avg20"]
        delta = df["Close"].diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        df["rsi14"] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]
        prev = df.iloc[-2]
        recent_window = df.iloc[-21:-1].copy()
        if recent_window.empty:
            continue

        close_price = float(last["Close"])
        ema20 = float(last["ema20"]) if pd.notna(last["ema20"]) else close_price
        ema50 = float(last["ema50"]) if pd.notna(last["ema50"]) else close_price
        rel_vol = float(last["rel_vol"]) if pd.notna(last["rel_vol"]) else 0.0
        rsi14 = float(last["rsi14"]) if pd.notna(last["rsi14"]) else 50.0
        prev_rsi = float(prev["rsi14"]) if pd.notna(prev["rsi14"]) else rsi14
        change_pct = ((float(last["Close"]) - float(prev["Close"])) / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0.0
        recent_support = float(recent_window["Low"].min())
        recent_resistance = float(recent_window["High"].max())
        recent_range = max(recent_resistance - recent_support, close_price * 0.01)

        long_score = 0
        short_score = 0

        trend_up = ema20 > ema50 and close_price > ema20
        trend_down = ema20 < ema50 and close_price < ema20
        if trend_up:
            long_score += 30
        elif close_price > ema20:
            long_score += 10
        if trend_down:
            short_score += 30
        elif close_price < ema20:
            short_score += 10

        if 50 <= rsi14 <= 70 and rsi14 >= prev_rsi:
            long_score += 18
        elif rsi14 > 70:
            long_score += 8
        if 30 <= rsi14 <= 50 and rsi14 <= prev_rsi:
            short_score += 18
        elif rsi14 < 30:
            short_score += 8

        if rel_vol >= 1.8:
            long_score += 20
            short_score += 20
        elif rel_vol >= 1.35:
            long_score += 12
            short_score += 12

        breakout_level = recent_resistance * 0.997
        breakdown_level = recent_support * 1.003
        if close_price >= breakout_level:
            long_score += 18
        if close_price <= breakdown_level:
            short_score += 18

        if change_pct >= 0.6:
            long_score += 10
        elif change_pct >= 0.3:
            long_score += 5
        if change_pct <= -0.6:
            short_score += 10
        elif change_pct <= -0.3:
            short_score += 5

        long_risk = max(close_price - recent_support, close_price * 0.004, 0.25)
        short_risk = max(recent_resistance - close_price, close_price * 0.004, 0.25)
        long_projected_target = recent_resistance + (recent_range * 0.35)
        short_projected_target = recent_support - (recent_range * 0.35)
        long_reward = max(long_projected_target - close_price, 0.0)
        short_reward = max(close_price - short_projected_target, 0.0)
        long_rr = long_reward / long_risk if long_risk > 0 else 0.0
        short_rr = short_reward / short_risk if short_risk > 0 else 0.0

        if long_rr >= 2.2:
            long_score += 20
        elif long_rr >= 1.7:
            long_score += 10
        if short_rr >= 2.2:
            short_score += 20
        elif short_rr >= 1.7:
            short_score += 10

        if long_score >= short_score and long_score >= 60 and trend_up and rel_vol >= 1.1 and long_rr >= 1.7:
            signal = "LONG SETUP"
            score = long_score
            risk = long_risk
            stop_loss = round(close_price - risk, 4)
            tp1 = round(close_price + max(risk * 1.5, recent_range * 0.2), 4)
            tp2 = round(close_price + max(risk * 2.3, recent_range * 0.35), 4)
        elif short_score > long_score and short_score >= 60 and trend_down and rel_vol >= 1.1 and short_rr >= 1.7:
            signal = "SHORT SETUP"
            score = short_score
            risk = short_risk
            stop_loss = round(close_price + risk, 4)
            tp1 = round(close_price - max(risk * 1.5, recent_range * 0.2), 4)
            tp2 = round(close_price - max(risk * 2.3, recent_range * 0.35), 4)
        else:
            signal = "NO SIGNAL"
            score = max(long_score, short_score)
            risk = long_risk if long_score >= short_score else short_risk
            stop_loss = round(close_price + short_risk, 4)
            tp1 = round(close_price - max(short_risk * 1.5, recent_range * 0.2), 4)
            tp2 = round(close_price - max(short_risk * 2.3, recent_range * 0.35), 4)

        shares = int(100 / risk) if risk > 0 else 0

        rows.append(
            {
                "symbol": symbol,
                "time": str(df.index[-1]),
                "close": round(close_price, 2),
                "score": score,
                "entry": round(close_price, 4),
                "stop_loss": stop_loss,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "shares": shares,
                "signal": signal,
                "rel_vol": round(rel_vol, 2),
                "change_pct": round(change_pct, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def log_active_signals(signals: pd.DataFrame, paper: pd.DataFrame):
    if signals.empty:
        return paper, 0

    active = signals[signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"])].copy()
    if active.empty:
        return paper, 0

    added = 0
    for _, row in active.iterrows():
        existing = paper[
            (paper["symbol"].astype(str) == str(row["symbol"]))
            & (paper["status"].astype(str) == "OPEN")
        ]
        if not existing.empty:
            continue

        new_row = {
            "symbol": row["symbol"],
            "time": row["time"],
            "entry": row["entry"],
            "stop_loss": row["stop_loss"],
            "take_profit_1": row["take_profit_1"],
            "take_profit_2": row["take_profit_2"],
            "shares": row["shares"],
            "score": row["score"],
            "signal": row["signal"],
            "status": "OPEN",
            "pnl": "",
            "exit_price": "",
            "exit_reason": "",
            "close_time": "",
        }
        paper = pd.concat([paper, pd.DataFrame([new_row])], ignore_index=True)
        send_sms_alert(f"🚀 TRADE OPEN: {row['symbol']} @ {float(row['entry']):.2f}")
        added += 1

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
    existing_open = paper_df[
        (paper_df["symbol"].astype(str) == symbol)
        & (paper_df["status"].astype(str) == "OPEN")
    ]
    if not existing_open.empty:
        return paper_df, False

    new_row = {
        "symbol": symbol,
        "time": str(setup_row["time"]),
        "entry": setup_row["entry"],
        "stop_loss": setup_row["stop_loss"],
        "take_profit_1": setup_row["take_profit_1"],
        "take_profit_2": setup_row["take_profit_2"],
        "shares": setup_row["shares"],
        "score": setup_row["score"],
        "signal": setup_row["signal"],
        "status": "OPEN",
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
        padding-top: 0.85rem;
        padding-bottom: 2.25rem;
        max-width: 1480px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


symbols = DEFAULT_SYMBOLS
paper = load_paper_trades()
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "Manual"

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
paper_pnl = pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0).sum()
st.session_state.open_trades = open_trades.to_dict("records")

performance = create_paper_performance_curve(closed_trades)
win_rate = round((performance["pnl"] > 0).mean() * 100, 2) if not closed_trades.empty else 0.0
total_pnl = round(float(performance["pnl"].sum()), 2)


page = st.sidebar.radio(
    "Navigate",
    ["Command Center", "Dashboard", "Setups", "Live Signals", "Paper Trades", "MashGPT", "Live Market"],
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
            <div class="sidebar-info-label">Live Setups</div>
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


st.markdown(
    f"""
    <div class="app-hero">
        <div class="app-kicker">Mash Trading Platform</div>
        <div class="app-title">Realtime signals, execution, and market intelligence in one terminal.</div>
        <div class="app-subtitle">Command your watchlist, monitor paper performance, and act on long or short setups with a cleaner product-grade workspace. Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Paper Win Rate %", win_rate)
m2.metric("Paper P&L", f"${total_pnl}")
m3.metric("Open Paper Trades", int(len(open_trades)))
m4.metric("Live Setups", int(signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"]).sum()) if not signals.empty else 0)


if page == "Command Center":
    st.subheader("Command Center")

    if not signals.empty:
        top_setup = signals.sort_values("score", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="top-trade-banner">
                <div class="top-trade-kicker">Top Setup Right Now</div>
                <div class="top-trade-grid">
                    <div class="top-trade-item">
                        <div class="top-trade-label">Symbol</div>
                        <div class="top-trade-value">{top_setup["symbol"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Score</div>
                        <div class="top-trade-value">{int(top_setup["score"])}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Entry</div>
                        <div class="top-trade-value">{top_setup["entry"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">Stop</div>
                        <div class="top-trade-value">{top_setup["stop_loss"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP1</div>
                        <div class="top-trade-value">{top_setup["take_profit_1"]}</div>
                    </div>
                    <div class="top-trade-item">
                        <div class="top-trade-label">TP2</div>
                        <div class="top-trade-value">{top_setup["take_profit_2"]}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No live setup available right now.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Trading Mode", trading_mode)
    cc2.metric("Open Trades", int(len(open_trades)))
    cc3.metric("Live Setups", int(signals["signal"].astype(str).isin(["LONG SETUP", "SHORT SETUP"]).sum()) if not signals.empty else 0)
    cc4.metric("Paper P&L", f"${round(float(paper_pnl), 2)}")

    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown("### Open Paper Trades")
        if open_trades.empty:
            st.caption("No open paper trades.")
        else:
            open_summary = open_trades[["symbol", "entry", "stop_loss", "take_profit_1", "take_profit_2"]].copy()
            st.dataframe(open_summary, width="stretch", height=240)

        st.markdown("### Quick Prompt")
        quick_prompt = st.text_input(
            "Ask MashGPT from Command Center",
            key="command_center_prompt",
            placeholder="What is the best setup right now?",
            label_visibility="collapsed",
        )
        if st.button("Ask MashGPT", key="command-center-ask", use_container_width=True) and quick_prompt.strip():
            st.session_state.command_center_reply = ask_mashgpt(quick_prompt, signals, open_trades, closed_trades)
        if st.session_state.get("command_center_reply"):
            st.info(st.session_state["command_center_reply"])

    with right_col:
        st.markdown("### Recent Activity")
        recent_items = []
        if new_logged:
            recent_items.append(f"{new_logged} new trade(s) opened this refresh")
        if newly_closed:
            recent_items.append(f"{newly_closed} trade(s) closed this refresh")
        if not closed_trades.empty:
            last_closed = closed_trades.iloc[-1]
            recent_items.append(
                f"Last closed: {last_closed['symbol']} {last_closed['exit_reason']} ({last_closed['status']})"
            )
        if not signals.empty:
            recent_items.append(f"Active leader: {signals.iloc[0]['symbol']} score {int(signals.iloc[0]['score'])}")

        if recent_items:
            for item in recent_items[:4]:
                st.markdown(f"- {item}")
        else:
            st.caption("No recent activity yet.")

elif page == "Dashboard":
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

elif page == "Live Signals":
    st.subheader("Live Signals")
    if signals.empty:
        st.info("No live signals right now.")
    else:
        st.dataframe(signals, width="stretch", height=420)

elif page == "Setups":
    st.subheader("Setups")
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
                        "entry": setup["entry"],
                        "stop_loss": setup["stop_loss"],
                        "take_profit_1": setup["take_profit_1"],
                        "take_profit_2": setup["take_profit_2"],
                        "shares": setup["shares"],
                        "signal": str(setup["signal"]),
                        "status": "OPEN",
                    }
                    st.session_state.open_trades.append(session_trade)
                    save_paper_trades(paper)
                    st.success("Trade added to Paper Trades")
                else:
                    st.info("Trade already open")

            st.markdown("")

elif page == "Paper Trades":
    st.subheader("Open Paper Trades")
    if open_trades.empty:
        st.write("No open paper trades.")
    else:
        st.dataframe(open_trades, width="stretch", height=260)

    st.subheader("Closed Paper Trades")
    if closed_trades.empty:
        st.write("No closed paper trades.")
    else:
        st.dataframe(closed_trades, width="stretch", height=260)

    st.metric("Paper P&L", f"${round(float(paper_pnl), 2)}")

elif page == "MashGPT":
    st.subheader("MashGPT")

    if signals is not None and not signals.empty:
        best_signal = signals.sort_values("score", ascending=False).iloc[0]

        score_val = int(best_signal["score"])
        if score_val >= 70:
            confidence = "High Confidence"
        elif score_val >= 50:
            confidence = "Medium Confidence"
        else:
            confidence = "Low Confidence"

        st.markdown("### Top Trade Card")

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Setup", str(best_signal["symbol"]))
        c2.metric("Score", score_val)
        c3.metric("Confidence", confidence)

        c4, c5, c6, c7 = st.columns(4)
        c4.metric("Entry", best_signal["entry"])
        c5.metric("Stop", best_signal["stop_loss"])
        c6.metric("TP1", best_signal["take_profit_1"])
        c7.metric("TP2", best_signal["take_profit_2"])

        if score_val >= 70:
            verdict = "🔥 TAKE"
            reason = "Strong setup, good momentum and risk/reward."
        elif score_val >= 50:
            verdict = "⚠️ WATCH"
            reason = "Decent setup but not the strongest."
        else:
            verdict = "❌ AVOID"
            reason = "Weak setup, low probability."

        st.markdown(f"### {verdict}")
        st.write(reason)
        st.markdown("---")
    else:
        st.info("No live setup available right now.")

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

elif page == "Live Market":
    st.markdown("## Live Market")
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

    st.markdown("---")

    # ---------------- CHART ----------------
    st.markdown(f"### {selected_symbol} Chart")

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

    with st.container():
        components.html(tradingview_html, height=740)

    st.caption("Chart display by TradingView. Live stats use Polygon when available.")

    st.markdown("")
    st.markdown("")

    # ---------------- STATS ----------------
    st.markdown("### Stats")

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

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Live Price", f"{latest_close:.2f}" if latest_close is not None else "N/A")
        s2.metric("Change %", f"{change_pct:.2f}%" if change_pct is not None else "N/A")
        s3.metric("High", f"{high_val:.2f}" if high_val is not None else "N/A")
        s4.metric("Low", f"{low_val:.2f}" if low_val is not None else "N/A")
        s5.metric("Volume", f"{latest_volume:,}" if latest_volume is not None else "N/A")

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
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Last", f"{latest_close:.2f}")
            s2.metric("Change %", f"{change_pct:.2f}%")
            s3.metric("High", f"{high_val:.2f}")
            s4.metric("Low", f"{low_val:.2f}")
            s5.metric("Volume", f"{latest_volume:,}")

        st.caption("Fallback data from yfinance")
    else:
        st.warning(f"No data found for {selected_symbol}")

    st.markdown("")

    with st.expander("Show raw market data"):
        if not market_df.empty:
            st.dataframe(market_df.tail(100), width="stretch", height=220)
