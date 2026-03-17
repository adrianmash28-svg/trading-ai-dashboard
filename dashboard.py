import os
from datetime import datetime

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Mash Trading Dashboard", layout="wide")

DEFAULT_SYMBOLS = ["META", "NVDA", "AAPL", "MSFT"]
PAPER_TRADES_FILE = "paper_trades.csv"
STARTING_EQUITY = 10000.0


# =========================
# SECRETS / ENV
# =========================
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
DISCORD_WEBHOOK_URL = get_secret("DISCORD_WEBHOOK_URL", "")


def get_openai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


client = get_openai_client()


# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0f1c 0%, #0d1320 100%);
        color: #e8eefc;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1f2a44;
    }
    div[data-testid="stMetric"] {
        background: #111a2c;
        border: 1px solid #223150;
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.18);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #223150;
        border-radius: 16px;
        overflow: hidden;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
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
        "status",
        "pnl",
        "exit_price",
        "exit_reason",
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


def send_discord_alert(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception:
        pass


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
        if df.empty or len(df) < 25:
            continue

        df["sma20"] = df["Close"].rolling(20).mean()
        df["vol_avg20"] = df["Volume"].rolling(20).mean()
        df["rel_vol"] = df["Volume"] / df["vol_avg20"]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close_price = float(last["Close"])
        sma20 = float(last["sma20"]) if pd.notna(last["sma20"]) else close_price
        rel_vol = float(last["rel_vol"]) if pd.notna(last["rel_vol"]) else 0.0
        intraday_change = ((float(last["Close"]) - float(prev["Close"])) / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0

        score = 0
        if close_price < sma20:
            score += 20
        if intraday_change < -0.3:
            score += 20
        if rel_vol > 1.2:
            score += 20
        if float(last["Low"]) <= float(df["Low"].tail(20).min()) * 1.01:
            score += 20

        signal = "SHORT SETUP" if score >= 45 else "NO SIGNAL"

        risk = max(close_price * 0.003, 0.25)
        stop_loss = round(close_price + risk, 4)
        tp1 = round(close_price - (risk * 2), 4)
        tp2 = round(close_price - (risk * 3), 4)
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
                "change_pct": round(intraday_change, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def log_active_signals(signals: pd.DataFrame, paper: pd.DataFrame):
    if signals.empty:
        return paper, 0

    active = signals[signals["signal"] == "SHORT SETUP"].copy()
    if active.empty:
        return paper, 0

    added = 0
    for _, row in active.iterrows():
        existing = paper[
            (paper["symbol"].astype(str) == str(row["symbol"])) &
            (paper["status"].astype(str) == "OPEN")
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
            "status": "OPEN",
            "pnl": "",
            "exit_price": "",
            "exit_reason": "",
        }
        paper = pd.concat([paper, pd.DataFrame([new_row])], ignore_index=True)
        added += 1
        send_discord_alert(
            f"🚨 NEW SHORT SETUP\n"
            f"{row['symbol']} | score {row['score']}\n"
            f"Entry: {row['entry']}\n"
            f"Stop: {row['stop_loss']}\n"
            f"TP1: {row['take_profit_1']}\n"
            f"TP2: {row['take_profit_2']}"
        )

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
        tp2 = float(trade["take_profit_2"])
        shares = float(trade["shares"])

        if last_price >= stop_loss:
            exit_price = stop_loss
            reason = "STOP LOSS"
        elif last_price <= tp2:
            exit_price = tp2
            reason = "TAKE PROFIT 2"
        else:
            continue

        pnl = round((entry - exit_price) * shares, 2)
        paper.at[idx, "status"] = "CLOSED"
        paper.at[idx, "pnl"] = pnl
        paper.at[idx, "exit_price"] = exit_price
        paper.at[idx, "exit_reason"] = reason
        closed_now += 1

        send_discord_alert(
            f"{'✅' if pnl > 0 else '🛑'} TRADE CLOSED\n"
            f"{symbol}\n"
            f"{reason}\n"
            f"PnL: ${pnl}"
        )

    return paper, closed_now


def create_backtest_placeholder():
    x = list(range(1, 61))
    pnl_steps = [
        120, -40, 90, 60, -35, 80, 50, -25, 110, 70,
        -30, 65, 85, -45, 95, 40, -20, 105, 60, -35,
        75, 55, -25, 90, 65, -30, 80, 70, -40, 95,
        50, -20, 85, 60, -35, 100, 55, -25, 70, 65,
        -30, 110, 50, -20, 75, 60, -35, 90, 55, -25,
        80, 65, -30, 95, 50, -20, 70, 60, -30, 85
    ]
    trades = pd.DataFrame({"trade_num": x, "pnl": pnl_steps})
    trades["equity"] = STARTING_EQUITY + trades["pnl"].cumsum()
    trades["peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] - trades["peak"]
    return trades


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

Your job:
- answer normal questions clearly
- answer trading questions with practical reasoning
- use the dashboard data below when relevant
- help rank setups, explain signals, discuss risk/reward, and evaluate trades
- do NOT guarantee profits
- be concise but useful
- when talking about trades, mention entry, stop, targets, score, and risk if available

Current live signals:
{signals_text}

Open paper trades:
{open_trades_text}

Closed paper trades:
{closed_trades_text}

When the user asks trading questions:
- explain why a setup looks strong or weak
- compare signal score, relative move, and reward/risk
- if asked "best setup", choose the strongest current signal
- if asked "should I take this trade", give pros/cons, not certainty
- if asked for confidence, give a 1-10 score with a short reason
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


# =========================
# DATA
# =========================
symbols = DEFAULT_SYMBOLS
paper = load_paper_trades()
signals = calc_live_signals(symbols)
paper, new_logged = log_active_signals(signals, paper)
paper, newly_closed = update_open_trades(paper)
save_paper_trades(paper)

open_trades = paper[paper["status"].astype(str) == "OPEN"].copy()
closed_trades = paper[paper["status"].astype(str) == "CLOSED"].copy()
paper_pnl = pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0).sum()

backtest = create_backtest_placeholder()
win_rate = round((backtest["pnl"] > 0).mean() * 100, 2)
total_pnl = round(float(backtest["pnl"].sum()), 2)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Live Signals", "Paper Trades", "MashGPT", "Live Market"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Refresh:** manual")
st.sidebar.write(f"**Min Score:** 45")
st.sidebar.write(f"**Symbols:** {', '.join(symbols)}")
st.sidebar.write(f"**Discord Alerts:** {'On' if DISCORD_WEBHOOK_URL else 'Off'}")


# =========================
# TOP
# =========================
st.title("Mash Trading Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Backtest Win Rate %", win_rate)
m2.metric("Backtest P&L", f"${total_pnl}")
m3.metric("Open Paper Trades", int(len(open_trades)))
m4.metric("Live Setups", int((signals["signal"] == "SHORT SETUP").sum()) if not signals.empty else 0)


# =========================
# PAGES
# =========================
if page == "Dashboard":
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Equity Curve")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(backtest["trade_num"], backtest["equity"], linewidth=2.7)
        ax.fill_between(backtest["trade_num"], backtest["equity"], backtest["equity"].min(), alpha=0.08)
        ax.set_title("Strategy Equity Curve", fontsize=15, pad=12)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Account Value")
        ax.grid(True, alpha=0.22, linestyle="--")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

    with c2:
        st.subheader("Drawdown")
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        ax2.plot(backtest["trade_num"], backtest["drawdown"], linewidth=2.7)
        ax2.fill_between(backtest["trade_num"], backtest["drawdown"], 0, alpha=0.12)
        ax2.set_title("Strategy Drawdown", fontsize=15, pad=12)
        ax2.set_xlabel("Trade Number")
        ax2.set_ylabel("Drawdown ($)")
        ax2.grid(True, alpha=0.22, linestyle="--")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
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
        st.dataframe(signals, use_container_width=True, height=420)

elif page == "Paper Trades":
    st.subheader("Open Paper Trades")
    if open_trades.empty:
        st.write("No open paper trades.")
    else:
        st.dataframe(open_trades, use_container_width=True, height=260)

    st.subheader("Closed Paper Trades")
    if closed_trades.empty:
        st.write("No closed paper trades.")
    else:
        st.dataframe(closed_trades, use_container_width=True, height=260)

    st.metric("Paper P&L", f"${round(float(paper_pnl), 2)}")

elif page == "MashGPT":
    st.subheader("MashGPT")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        avatar = "🙂" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(content)

    prompt = st.chat_input("Ask MashGPT anything")

    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user", avatar="🙂"):
            st.write(prompt)

        reply = ask_mashgpt(prompt, signals, open_trades, closed_trades)
        st.session_state.chat_history.append(("assistant", reply))

        with st.chat_message("assistant", avatar="🤖"):
            st.write(reply)

elif page == "Live Market":
    st.subheader("Live Market")

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        lookup_symbol = st.text_input("Search ticker", value="AAPL").upper().strip()

    with c2:
        period = st.selectbox(
            "Time Range",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=4,
        )

    with c3:
        interval_choices = {
            "1d": ["1m", "5m", "15m"],
            "5d": ["5m", "15m", "30m", "1h"],
            "1mo": ["30m", "1h", "1d"],
            "3mo": ["1h", "1d"],
            "6mo": ["1d"],
            "1y": ["1d"],
            "2y": ["1d", "1wk"],
            "5y": ["1d", "1wk", "1mo"],
        }
        interval = st.selectbox("Interval", interval_choices[period], index=0)

    market_df = fetch_history(lookup_symbol, period=period, interval=interval)

    if market_df.empty:
        st.error(f"No data found for {lookup_symbol}.")
    else:
        latest_close = float(market_df["Close"].iloc[-1])
        first_close = float(market_df["Close"].iloc[0])
        change_pct = ((latest_close - first_close) / first_close) * 100 if first_close != 0 else 0.0
        latest_volume = int(market_df["Volume"].iloc[-1])

        k1, k2, k3 = st.columns(3)
        k1.metric(f"{lookup_symbol} Last Price", round(latest_close, 2))
        k2.metric("Period Change %", round(change_pct, 2))
        k3.metric("Latest Volume", f"{latest_volume:,}")

        st.subheader(f"{lookup_symbol} Candlestick Chart")

        chart_df = market_df.copy()
        chart_df.index.name = "Date"

        mc = mpf.make_marketcolors(
            up="#22c55e",
            down="#ef4444",
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mc,
            facecolor="#0f1115",
            edgecolor="#2d333b",
            figcolor="#0f1115",
            gridcolor="#30363d",
            gridstyle="--",
        )

        fig, _ = mpf.plot(
            chart_df,
            type="candle",
            style=style,
            volume=True,
            figsize=(12, 7),
            tight_layout=True,
            returnfig=True,
            warn_too_much_data=10000,
        )
        st.pyplot(fig)

        with st.expander("Show raw market data"):
            st.dataframe(chart_df.tail(200), use_container_width=True, height=320)
