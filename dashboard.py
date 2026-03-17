
import os
import json
from datetime import datetime

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

except Exception:
    OpenAI = None


# =========================
# FILES / CONSTANTS
# =========================
SETTINGS_FILE = "dashboard_settings.json"
TRADES_CSV = "tight_strategy_trades.csv"
SUMMARY_CSV = "tight_strategy_summary.csv"
PAPER_TRADES_CSV = "paper_trades.csv"
ALERT_STATE_JSON = "alert_state.json"

STARTING_EQUITY = 10000.0

DEFAULT_SETTINGS = {
    "bot_name": "MashGPT",
    "page_title": "Mash Trading Dashboard",
    "refresh_seconds": 60,
    "min_score": 45,
    "show_only_live_setups": False,
    "symbols": ["META", "NVDA", "AAPL", "MSFT"],
    "theme_mode": "gray",
    "chart_height": 5.2,
    "watchlist_size": 8,
    "discord_alerts_enabled": True,
}

MARKET_SYMBOL = "SPY"
LIVE_DATA_PERIOD = "5d"
LIVE_DATA_INTERVAL = "15m"

REL_VOL_MIN = 1.0
ATR_STOP_MULT = 1.0
TARGET1_R = 2.0
TARGET2_R = 3.0

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


# =========================
# SETTINGS
# =========================
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    merged = DEFAULT_SETTINGS.copy()
    merged.update(data)
    return merged


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


settings = load_settings()

st.set_page_config(
    page_title=settings["page_title"],
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(interval=int(settings["refresh_seconds"]) * 1000, key="live_refresh")


# =========================
# THEME
# =========================
def apply_theme_css(theme_mode: str):
    if theme_mode == "gray":
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #0f1115 0%, #12161d 100%);
                color: #e5e7eb;
            }
            [data-testid="stSidebar"] {
                background-color: #151922;
                border-right: 1px solid #2b313b;
            }
            div[data-testid="stMetric"] {
                background: #161b22;
                border: 1px solid #2d333b;
                border-radius: 18px;
                padding: 14px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.18);
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #2d333b;
                border-radius: 16px;
                overflow: hidden;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }
            .chat-wrap {
                border: 1px solid #2d333b;
                border-radius: 16px;
                padding: 12px;
                background: #121720;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: #0b1220;
            }
            [data-testid="stSidebar"] {
                background-color: #0f172a;
            }
            div[data-testid="stMetric"] {
                background: #111827;
                border: 1px solid #253041;
                border-radius: 18px;
                padding: 14px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.18);
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #253041;
                border-radius: 16px;
                overflow: hidden;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }
            .chat-wrap {
                border: 1px solid #253041;
                border-radius: 16px;
                padding: 12px;
                background: #0f172a;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


apply_theme_css(settings["theme_mode"])


# =========================
# OPENAI
# =========================
def get_openai_client():
    def get_openai_client():
        import os
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return None

        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None


client = get_openai_client()


client = get_openai_client()


# =========================
# DISCORD
# =========================
def load_alert_state():
    if not os.path.exists(ALERT_STATE_JSON):
        return {}
    try:
        with open(ALERT_STATE_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_alert_state(state):
    with open(ALERT_STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


def send_discord_message(content: str):
    if not settings.get("discord_alerts_enabled", True):
        return False
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


def send_new_signal_alert(row):
    content = (
        f"🚨 **NEW SHORT SETUP**\n"
        f"**Symbol:** {row['symbol']}\n"
        f"**Score:** {row['score']}\n"
        f"**Entry:** {row['entry']}\n"
        f"**Stop:** {row['stop_loss']}\n"
        f"**TP1:** {row['take_profit_1']}\n"
        f"**TP2:** {row['take_profit_2']}\n"
        f"**Shares:** {row['shares']}\n"
        f"**Time:** {row['time']}"
    )
    send_discord_message(content)


def send_close_alert(symbol, reason, pnl):
    emoji = "✅" if pnl > 0 else "🛑"
    content = (
        f"{emoji} **TRADE CLOSED**\n"
        f"**Symbol:** {symbol}\n"
        f"**Reason:** {reason}\n"
        f"**PnL:** ${round(float(pnl), 2)}"
    )
    send_discord_message(content)


def send_best_setup_alert(best_row):
    content = (
        f"⭐ **BEST SETUP RIGHT NOW**\n"
        f"**Symbol:** {best_row['symbol']}\n"
        f"**Score:** {best_row['score']}\n"
        f"**Entry:** {best_row['entry']}\n"
        f"**Stop:** {best_row['stop_loss']}\n"
        f"**TP1:** {best_row['take_profit_1']}\n"
        f"**TP2:** {best_row['take_profit_2']}"
    )
    send_discord_message(content)


# =========================
# FILE HELPERS
# =========================
def ensure_paper_trades_file():
    if not os.path.exists(PAPER_TRADES_CSV):
        pd.DataFrame(
            columns=[
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
        ).to_csv(PAPER_TRADES_CSV, index=False)


def load_paper_trades():
    ensure_paper_trades_file()
    return pd.read_csv(PAPER_TRADES_CSV)


def save_paper_trades(df):
    df.to_csv(PAPER_TRADES_CSV, index=False)


# =========================
# DATA HELPERS
# =========================
def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


@st.cache_data(ttl=45, show_spinner=False)
def download_data(symbol, period="5d", interval="15m"):
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

    df = flatten_yf_columns(df)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()

    df = df[needed].copy().dropna()
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    df["date"] = pd.to_datetime(df.index.date)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def download_market_lookup(symbol, period, interval):
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

    df = flatten_yf_columns(df)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()

    df = df[needed].copy().dropna()
    df.index = pd.to_datetime(df.index)
    return df


def prepare_indicators(df):
    out = df.copy()
    out["sma20"] = out["Close"].astype(float).rolling(20).mean()
    out["vol_avg20"] = out["Volume"].astype(float).rolling(20).mean()
    out["rel_vol"] = out["Volume"] / out["vol_avg20"]

    day_open = out.groupby("date")["Open"].transform("first")
    out["day_change_pct"] = ((out["Close"] - day_open) / day_open) * 100
    out["low20"] = out["Low"].rolling(20).min()

    prev_close = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close).abs()
    tr3 = (out["Low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr14"] = out["tr"].rolling(14).mean()
    return out


def build_market_day_map(spy_df):
    day_map = {}
    if spy_df.empty:
        return day_map

    for day, g in spy_df.groupby("date"):
        day_open = float(g["Open"].iloc[0])
        day_close = float(g["Close"].iloc[-1])
        change_pct = ((day_close - day_open) / day_open) * 100 if day_open != 0 else 0.0
        day_map[day] = change_pct
    return day_map


# =========================
# STRATEGY
# =========================
def short_score_row(row, market_day_change_pct):
    if market_day_change_pct >= -0.5:
        return 0

    score = 0
    if row["Close"] < row["sma20"]:
        score += 15
    if row["day_change_pct"] < -0.5:
        score += 15
    if row["rel_vol"] >= REL_VOL_MIN:
        score += 15
    if row["Close"] <= row["low20"] * 1.002:
        score += 30
    if row["day_change_pct"] < market_day_change_pct:
        score += 20
    return score


def scan_live_signals(symbols):
    spy_raw = download_data(MARKET_SYMBOL, period=LIVE_DATA_PERIOD, interval=LIVE_DATA_INTERVAL)
    if spy_raw.empty:
        fpd.DataFrame()

    spy = prepare_indicators(spy_raw)
    market_day_map = build_market_day_map(spy)
    today = max(spy["date"].unique())
    market_day_change_pct = market_day_map.get(today, 0.0)

    rows = []

    for symbol in symbols:
        raw = download_data(symbol, period=LIVE_DATA_PERIOD, interval=LIVE_DATA_INTERVAL)
        if raw.empty:
            continue

        df = prepare_indicators(raw)
        today_df = df[df["date"] == today].copy()
        if today_df.empty:
            continue

        row = today_df.iloc[-1]
        needed = ["sma20", "rel_vol", "low20", "atr14", "Close", "day_change_pct"]
        if any(pd.isna(row[c]) for c in needed):
            continue

        score = short_score_row(row, market_day_change_pct)
        close_price = float(row["Close"])
        atr = float(row["atr14"])

        stop_loss = round(close_price + ATR_STOP_MULT * atr, 4)
        risk_per_share = stop_loss - close_price

        shares = 0
        tp1 = close_price
        tp2 = close_price

        if risk_per_share > 0:
            shares = int(100.0 / risk_per_share)
            tp1 = round(close_price - TARGET1_R * risk_per_share, 4)
            tp2 = round(close_price - TARGET2_R * risk_per_share, 4)

        signal = "SHORT SETUP" if score >= int(settings["min_score"]) else "NO SIGNAL"

        rows.append(
            {
                "symbol": symbol,
                "time": str(today_df.index[-1]),
                "close": round(close_price, 2),
                "score": int(score),
                "entry": round(close_price, 4),
                "stop_loss": round(stop_loss, 4),
                "take_profit_1": round(tp1, 4),
                "take_profit_2": round(tp2, 4),
                "shares": int(shares),
                "signal": signal,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["score"], ascending=[False]).reset_index(drop=True)


def auto_log_active_signals(signals, paper):
    if signals.empty:
        return paper, 0

    active = signals[signals["signal"] == "SHORT SETUP"].copy()
    if active.empty:
        return paper, 0

    added = 0
    alert_state = load_alert_state()

    for _, row in active.iterrows():
        exists = pd.DataFrame()
        if not paper.empty:
            exists = paper[(paper["symbol"] == row["symbol"]) & (paper["status"] == "OPEN")]
        if not exists.empty:
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

        key = f"signal_{row['symbol']}_{row['time']}"
        if key not in alert_state:
            send_new_signal_alert(row)
            alert_state[key] = True

    if not active.empty:
        best = active.sort_values("score", ascending=False).iloc[0]
        best_key = f"best_{best['symbol']}_{best['time']}"
        if best_key not in alert_state:
            send_best_setup_alert(best)
            alert_state[best_key] = True

    save_alert_state(alert_state)
    return paper, added


def auto_update_open_trades(paper):
    if paper.empty:
        return paper, 0

    closed_now = 0
    alert_state = load_alert_state()

    for idx, trade in paper.iterrows():
        if str(trade.get("status", "")) != "OPEN":
            continue

        symbol = str(trade["symbol"])
        raw = download_data(symbol, period="1d", interval="5m")
        if raw.empty:
            continue

        last_price = float(raw["Close"].iloc[-1])
        entry = float(trade["entry"])
        stop_loss = float(trade["stop_loss"])
        tp2 = float(trade["take_profit_2"])
        shares = float(trade["shares"])

        exit_reason = None
        exit_price = None

        if last_price >= stop_loss:
            exit_reason = "STOP LOSS"
            exit_price = stop_loss
        elif last_price <= tp2:
            exit_reason = "TAKE PROFIT 2"
            exit_price = tp2

        if exit_reason is not None:
            pnl = round((entry - exit_price) * shares, 2)
            paper.at[idx, "status"] = "CLOSED"
            paper.at[idx, "pnl"] = pnl
            paper.at[idx, "exit_price"] = exit_price
            paper.at[idx, "exit_reason"] = exit_reason
            closed_now += 1

            key = f"closed_{symbol}_{trade['time']}"
            if key not in alert_state:
                send_close_alert(symbol, exit_reason, pnl)
                alert_state[key] = True

    save_alert_state(alert_state)
    return paper, closed_now


# =========================
# BOT
# =========================
def process_bot_command(prompt, settings):
    text = prompt.strip()
    lower = text.lower()

    if lower.startswith("change title to "):
        settings["page_title"] = text[len("change title to "):].strip()
        save_settings(settings)
        return f"{settings['bot_name']} changed the title to: {settings['page_title']}", True

    if lower.startswith("set min score to "):
        try:
            value = int(lower.replace("set min score to ", "").strip())
            settings["min_score"] = value
            save_settings(settings)
            return f"{settings['bot_name']} set min score to {value}.", True
        except Exception:
            return "I couldn't read that min score value.", False

    if lower.startswith("set refresh interval to "):
        try:
            value = int(lower.replace("set refresh interval to ", "").strip())
            settings["refresh_seconds"] = value
            save_settings(settings)
            return f"{settings['bot_name']} set refresh interval to {value} seconds.", True
        except Exception:
            return "I couldn't read that refresh interval.", False

    if lower.startswith("scan symbols "):
        raw = text[len("scan symbols "):].strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            return "I couldn't find any symbols in that command.", False
        settings["symbols"] = symbols
        save_settings(settings)
        return f"{settings['bot_name']} changed the symbol list to: {', '.join(symbols)}", True

    if lower == "show only live setups":
        settings["show_only_live_setups"] = True
        save_settings(settings)
        return f"{settings['bot_name']} turned on live-setup-only mode.", True

    if lower == "turn off live only mode":
        settings["show_only_live_setups"] = False
        save_settings(settings)
        return f"{settings['bot_name']} turned off live-setup-only mode.", True

    if lower == "make the site gray" or lower == "make site gray":
        settings["theme_mode"] = "gray"
        save_settings(settings)
        return "Done. The site theme is now gray.", True

    if lower == "make the site dark" or lower == "make site dark":
        settings["theme_mode"] = "dark"
        save_settings(settings)
        return "Done. The site theme is now dark.", True

    if lower.startswith("rename yourself to "):
        new_name = text[len("rename yourself to "):].strip()
        if new_name:
            settings["bot_name"] = new_name
            save_settings(settings)
            return f"My new name is {new_name}.", True

    if lower == "turn off discord alerts":
        settings["discord_alerts_enabled"] = False
        save_settings(settings)
        return "Discord alerts are now off.", True

    if lower == "turn on discord alerts":
        settings["discord_alerts_enabled"] = True
        save_settings(settings)
        return "Discord alerts are now on.", True

    return None, False


def local_dashboard_answer(question, signals, paper, summary, trades):
    q = question.lower().strip()

    open_trades = paper[paper["status"] == "OPEN"].copy() if not paper.empty else pd.DataFrame()
    closed_trades = paper[paper["status"] == "CLOSED"].copy() if not paper.empty else pd.DataFrame()

    if "open" in q and "trade" in q:
        if open_trades.empty:
            return "You have no open paper trades."
        return f"You have {len(open_trades)} open paper trade(s): {', '.join(open_trades['symbol'].astype(str).tolist())}."

    if "paper pnl" in q or "paper p&l" in q:
        if closed_trades.empty:
            return "Your paper P&L is $0 because you have no closed paper trades yet."
        pnl = pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0).sum()
        return f"Your paper P&L is ${round(float(pnl), 2)}."

    if "win rate" in q:
        wr = round(float(summary.loc[0, "win_rate_pct"]), 2)
        return f"Your backtest win rate is {wr}%."

    if "best symbol" in q:
        symbol_pnl = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        if symbol_pnl.empty:
            return "I couldn't find any trade history."
        return f"Your best symbol is {symbol_pnl.index[0]} with P&L of ${round(float(symbol_pnl.iloc[0]), 2)}."

    if "best setup" in q:
        if signals.empty:
            return "There are no live signals right now."
        best = signals.sort_values("score", ascending=False).iloc[0]
        return (
            f"Best setup right now is {best['symbol']} with score {best['score']}, "
            f"entry {best['entry']}, stop {best['stop_loss']}, TP1 {best['take_profit_1']}, TP2 {best['take_profit_2']}."
        )

    if "rate this setup" in q or "should i take this trade" in q:
        if signals.empty:
            return "There are no live signals to rate right now."
        best = signals.sort_values("score", ascending=False).iloc[0]
        score = int(best["score"])
        if score >= 60:
            rating = "8/10"
        elif score >= 45:
            rating = "6/10"
        else:
            rating = "4/10"
        return f"I’d rate the best current setup {rating}. The strongest symbol right now is {best['symbol']} with score {score}."

    return "I can answer dashboard questions, trading questions, and broader questions if your OpenAI API key is connected."


def ai_answer(question, signals, paper, summary, trades):
    if client is None:
        return local_dashboard_answer(question, signals, paper, summary, trades)

    open_trades = paper[paper["status"] == "OPEN"].copy() if not paper.empty else pd.DataFrame()
    closed_trades = paper[paper["status"] == "CLOSED"].copy() if not paper.empty else pd.DataFrame()

    context = f"""
Dashboard summary:
{summary.to_string(index=False)}

Live signals:
{signals.head(25).to_string(index=False) if not signals.empty else "No live signals"}

Open paper trades:
{open_trades.to_string(index=False) if not open_trades.empty else "No open paper trades"}

Closed paper trades:
{closed_trades.tail(20).to_string(index=False) if not closed_trades.empty else "No closed paper trades"}
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=f"""
You are MashGPT, a helpful trading dashboard assistant.

You can:
- answer questions about the trading dashboard using the provided context
- answer broader general questions
- answer current questions like weather or news using web search
- answer trading questions like: should I short today, which setup is best, rate this setup, explain this signal

Be practical, clear, and concise.

Context:
{context}

User question:
{question}
"""
        )
        text = getattr(response, "output_text", None)
        if text and text.strip():
            return text.strip()
        return "I couldn't generate a response right now."
    except Exception as e:
        return f"MashGPT error: {e}"


# =========================
# LOAD DATA
# =========================
try:
    trades = pd.read_csv(TRADES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
except FileNotFoundError:
    st.error("Could not find backtest CSV files.")
    st.stop()

paper = load_paper_trades()
signals = scan_live_signals(settings["symbols"])
paper, auto_logged = auto_log_active_signals(signals, paper)
paper, auto_closed = auto_update_open_trades(paper)
save_paper_trades(paper)

trades["equity"] = STARTING_EQUITY + trades["pnl"].cumsum()
trades["peak"] = trades["equity"].cummax()
trades["drawdown"] = trades["equity"] - trades["peak"]

open_trades = paper[paper["status"] == "OPEN"].copy() if not paper.empty else pd.DataFrame()
closed_trades = paper[paper["status"] == "CLOSED"].copy() if not paper.empty else pd.DataFrame()

paper_pnl = 0.0
if not closed_trades.empty:
    paper_pnl = float(pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0).sum())

live_count = 0 if signals.empty else int((signals["signal"] == "SHORT SETUP").sum())

# =========================
# SIDEBAR / NAV
# =========================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Live Signals", "Paper Trades", settings["bot_name"], "Live Market"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Refresh:** {settings['refresh_seconds']}s")
st.sidebar.write(f"**Min Score:** {settings['min_score']}")
st.sidebar.write(f"**Theme:** {settings['theme_mode']}")
st.sidebar.write(f"**Symbols:** {', '.join(settings['symbols'])}")
st.sidebar.write(f"**Discord Alerts:** {'On' if settings.get('discord_alerts_enabled', True) else 'Off'}")

# =========================
# TOP BAR
# =========================
st.title(settings["page_title"])
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Backtest Win Rate %", round(float(summary.loc[0, "win_rate_pct"]), 2))
m2.metric("Backtest P&L", round(float(summary.loc[0, "total_pnl"]), 2))
m3.metric("Open Paper Trades", int(len(open_trades)))
m4.metric("Live Setups", live_count)

a1, a2 = st.columns(2)
a1.metric("Auto-Logged This Refresh", auto_logged)
a2.metric("Auto-Closed This Refresh", auto_closed)

# =========================
# PAGES
# =========================
if page == "Dashboard":
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Equity Curve")
        fig, ax = plt.subplots(figsize=(10, float(settings["chart_height"])))
        ax.plot(trades.index, trades["equity"], linewidth=2.7)
        ax.fill_between(trades.index, trades["equity"], trades["equity"].min(), alpha=0.08)
        ax.set_title("Strategy Equity Curve", fontsize=16, pad=12)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Account Value")
        ax.grid(True, alpha=0.22, linestyle="--")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

    with c2:
        st.subheader("Drawdown")
        fig2, ax2 = plt.subplots(figsize=(10, float(settings["chart_height"])))
        ax2.plot(trades.index, trades["drawdown"], linewidth=2.7)
        ax2.fill_between(trades.index, trades["drawdown"], 0, alpha=0.12)
        ax2.set_title("Strategy Drawdown", fontsize=16, pad=12)
        ax2.set_xlabel("Trade Number")
        ax2.set_ylabel("Drawdown ($)")
        ax2.grid(True, alpha=0.22, linestyle="--")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        st.pyplot(fig2)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("P&L by Symbol")
        symbol_pnl = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        st.bar_chart(symbol_pnl)

    with c4:
        st.subheader("Win Rate by Symbol")
        win_rate = trades.assign(win=trades["pnl"] > 0).groupby("symbol")["win"].mean() * 100
        st.bar_chart(win_rate)

    st.subheader("Watchlist")
    if signals.empty:
        st.info("No watchlist data right now.")
    else:
        watch = signals.head(int(settings.get("watchlist_size", 8))).copy()
        st.dataframe(watch, width="stretch", height=280)

    st.subheader("Backtest Trades")
    st.dataframe(trades, width="stretch", height=340)

elif page == "Live Signals":
    st.subheader("Live Signals")
    if signals.empty:
        st.info("No live signal data available right now.")
    else:
        shown = signals.copy()
        if settings["show_only_live_setups"]:
            shown = shown[shown["signal"] == "SHORT SETUP"]
        st.dataframe(shown, width="stretch", height=460)

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

    st.metric("Paper P&L", round(paper_pnl, 2))

elif page == settings["bot_name"]:
    st.subheader(settings["bot_name"])
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    if "bot_messages" not in st.session_state:
        st.session_state.bot_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hi, I'm {settings['bot_name']}. I can answer trading questions, dashboard questions, "
                    f"general questions, and current questions if your API key is connected."
                )
            }
        ]

    for msg in st.session_state.bot_messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🙂"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    prompt = st.chat_input(f"Ask {settings['bot_name']} anything")
  if prompt:
    with st.chat_message("user", avatar="🙂"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        st.write("Thinking...")

        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are MashGPT, a smart trading assistant that helps with stocks, trading strategies, and general questions."},
                    {"role": "user", "content": prompt}
                ]
            )

            reply = response.choices[0].message.content
            st.write(reply)

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

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

    market_df = download_market_lookup(lookup_symbol, period, interval)

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
        s = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mc,
            facecolor="#0f1115",
            edgecolor="#2d333b",
            figcolor="#0f1115",
            gridcolor="#30363d",
            gridstyle="--",
            rc={"font.size": 10},
        )

        fig, _ = mpf.plot(
            chart_df,
            type="candle",
            style=s,
            volume=True,
            figsize=(12, 7),
            tight_layout=True,
            returnfig=True,
            warn_too_much_data=10000,
        )
        st.pyplot(fig)

        with st.expander("Show raw market data"):
            st.dataframe(chart_df.tail(200), width="stretch", height=320)
