import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import pandas as pd
import yfinance as yf


SYMBOLS = ["META", "NVDA", "AAPL", "AMZN"]
MARKET_SYMBOL = "SPY"

# Yahoo 15m data is limited to about 60 days
DATA_PERIOD = "60d"
DATA_INTERVAL = "15m"

STARTING_EQUITY = 10000.0
RISK_PER_TRADE = 100.0

REL_VOL_MIN = 1.0
MIN_SCORE = 45
TRAIN_SPLIT = 0.70

ENTRY_SLIPPAGE_BPS = 2
EXIT_SLIPPAGE_BPS = 2

ATR_STOP_MULT = 1.0
TARGET1_R = 2.0
TARGET2_R = 3.0

OUTPUT_TRADES_CSV = "tight_strategy_trades.csv"
OUTPUT_SUMMARY_CSV = "tight_strategy_summary.csv"


@dataclass
class Trade:
    symbol: str
    date: str
    entry_time: str
    exit_time: str
    side: str
    entry: float
    stop: float
    target1: float
    target2: float
    shares: int
    pnl: float
    r_multiple: float
    exit_reason: str


@dataclass
class Summary:
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_pnl: float
    avg_pnl: float
    avg_r: float
    best_trade: float
    worst_trade: float
    ending_equity: float
    max_losing_streak: int


def apply_short_entry_slippage(price: float) -> float:
    return round(price * (1 - ENTRY_SLIPPAGE_BPS / 10000), 4)


def apply_short_exit_slippage(price: float) -> float:
    return round(price * (1 + EXIT_SLIPPAGE_BPS / 10000), 4)


def summarize_trades(
    trades: List[Trade], starting_equity: float = STARTING_EQUITY
) -> Summary:
    if not trades:
        return Summary(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            avg_r=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            ending_equity=starting_equity,
            max_losing_streak=0,
        )

    pnls = [t.pnl for t in trades]
    rs = [t.r_multiple for t in trades]

    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)

    max_losing_streak = 0
    current_losing_streak = 0
    for p in pnls:
        if p <= 0:
            current_losing_streak += 1
            max_losing_streak = max(max_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0

    total_pnl = round(sum(pnls), 2)

    return Summary(
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate_pct=round((wins / len(trades)) * 100, 2),
        total_pnl=total_pnl,
        avg_pnl=round(total_pnl / len(trades), 2),
        avg_r=round(sum(rs) / len(rs), 3),
        best_trade=round(max(pnls), 2),
        worst_trade=round(min(pnls), 2),
        ending_equity=round(starting_equity + total_pnl, 2),
        max_losing_streak=max_losing_streak,
    )


def print_summary(title: str, summary: Summary) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total trades: {summary.total_trades}")
    print(f"Wins: {summary.wins}")
    print(f"Losses: {summary.losses}")
    print(f"Win rate: {summary.win_rate_pct}%")
    print(f"Total P&L: ${summary.total_pnl}")
    print(f"Average P&L/trade: ${summary.avg_pnl}")
    print(f"Average R: {summary.avg_r}")
    print(f"Best trade: ${summary.best_trade}")
    print(f"Worst trade: ${summary.worst_trade}")
    print(f"Ending equity: ${summary.ending_equity}")
    print(f"Max losing streak: {summary.max_losing_streak}")


def download_data(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    df = df[required].copy()
    df = df.dropna().copy()
    if df.empty:
        raise ValueError(f"No usable rows for {symbol}")

    df.index = pd.to_datetime(df.index)
    df["date"] = pd.to_datetime(df.index.date)
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sma20"] = out["Close"].rolling(20).mean()
    out["vol_avg20"] = out["Volume"].rolling(20).mean()
    out["rel_vol"] = out["Volume"] / out["vol_avg20"]

    day_open = out.groupby("date")["Open"].transform("first")
    out["day_change_pct"] = ((out["Close"] - day_open) / day_open) * 100

    out["low20"] = out["Low"].rolling(20).min()
    out["high20"] = out["High"].rolling(20).max()

    prev_close_shift = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close_shift).abs()
    tr3 = (out["Low"] - prev_close_shift).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr14"] = out["tr"].rolling(14).mean()

    return out


def build_market_day_map(spy_df: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    day_map: Dict[pd.Timestamp, float] = {}
    grouped = spy_df.groupby("date")
    for day, g in grouped:
        day_open = g["Open"].iloc[0]
        day_close = g["Close"].iloc[-1]
        change_pct = ((day_close - day_open) / day_open) * 100 if day_open != 0 else 0.0
        day_map[day] = change_pct
    return day_map


def short_score_row(row: pd.Series, market_day_change_pct: float) -> int:
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


def run_tight_short_strategy(
    stock_map: Dict[str, pd.DataFrame],
    market_day_map: Dict[pd.Timestamp, float],
) -> List[Trade]:
    trades: List[Trade] = []

    for symbol, df in stock_map.items():
        grouped = df.groupby("date")

        for day, day_df in grouped:
            if day not in market_day_map:
                continue

            market_day_change_pct = market_day_map[day]

            for i in range(20, len(day_df) - 1):
                row = day_df.iloc[i]

                if pd.isna(row["sma20"]) or pd.isna(row["rel_vol"]) or pd.isna(row["low20"]) or pd.isna(row["atr14"]):
                    continue

                score = short_score_row(row, market_day_change_pct)
                if score < MIN_SCORE:
                    continue

                atr = float(row["atr14"])
                if atr <= 0 or math.isnan(atr):
                    continue

                entry_price_raw = float(day_df.iloc[i + 1]["Open"])
                entry = apply_short_entry_slippage(entry_price_raw)

                stop = round(entry + ATR_STOP_MULT * atr, 4)
                risk_per_share = stop - entry
                if risk_per_share <= 0:
                    continue

                shares = int(RISK_PER_TRADE / risk_per_share)
                if shares <= 0:
                    continue

                target1 = round(entry - TARGET1_R * risk_per_share, 4)
                target2 = round(entry - TARGET2_R * risk_per_share, 4)

                remaining_shares = shares
                realized_pnl = 0.0
                got_target1 = False
                exit_reason = "EOD"
                exit_time = str(day_df.index[-1])

                future_df = day_df.iloc[i + 1 :]

                for ts_future, bar in future_df.iterrows():
                    bar_high = float(bar["High"])
                    bar_low = float(bar["Low"])
                    bar_close = float(bar["Close"])

                    if remaining_shares > 0 and bar_high >= stop:
                        exit_price = apply_short_exit_slippage(stop)
                        realized_pnl += (entry - exit_price) * remaining_shares
                        exit_reason = "STOP"
                        exit_time = str(ts_future)
                        remaining_shares = 0
                        break

                    if (not got_target1) and remaining_shares > 0 and bar_low <= target1:
                        exit_price = apply_short_exit_slippage(target1)
                        partial_shares = remaining_shares // 2
                        if partial_shares == 0:
                            partial_shares = remaining_shares
                        realized_pnl += (entry - exit_price) * partial_shares
                        remaining_shares -= partial_shares
                        got_target1 = True
                        exit_reason = "TARGET1"

                    if remaining_shares > 0 and bar_low <= target2:
                        exit_price = apply_short_exit_slippage(target2)
                        realized_pnl += (entry - exit_price) * remaining_shares
                        exit_reason = "TARGET2"
                        exit_time = str(ts_future)
                        remaining_shares = 0
                        break

                    exit_time = str(ts_future)

                    if ts_future == future_df.index[-1] and remaining_shares > 0:
                        exit_price = apply_short_exit_slippage(bar_close)
                        realized_pnl += (entry - exit_price) * remaining_shares
                        exit_reason = "EOD"
                        remaining_shares = 0
                        break

                total_risk = risk_per_share * shares
                r_multiple = realized_pnl / total_risk if total_risk > 0 else 0.0

                trades.append(
                    Trade(
                        symbol=symbol,
                        date=str(day.date()),
                        entry_time=str(day_df.index[i + 1]),
                        exit_time=exit_time,
                        side="SHORT",
                        entry=round(entry, 4),
                        stop=round(stop, 4),
                        target1=round(target1, 4),
                        target2=round(target2, 4),
                        shares=shares,
                        pnl=round(realized_pnl, 2),
                        r_multiple=round(r_multiple, 3),
                        exit_reason=exit_reason,
                    )
                )

                break

    trades.sort(key=lambda t: t.entry_time)
    return trades


def split_days(common_days: List[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    split_idx = max(1, int(len(common_days) * TRAIN_SPLIT))
    train_days = common_days[:split_idx]
    test_days = common_days[split_idx:]
    return train_days, test_days


def filter_df_by_days(df: pd.DataFrame, days: List[pd.Timestamp]) -> pd.DataFrame:
    day_set = set(days)
    return df[df["date"].isin(day_set)].copy()


def main() -> None:
    print("Downloading data...")

    raw_map: Dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS + [MARKET_SYMBOL]:
        print(f"Loading {symbol}...")
        raw_map[symbol] = download_data(symbol)

    prepared_map: Dict[str, pd.DataFrame] = {}
    for symbol, df in raw_map.items():
        prepared_map[symbol] = prepare_indicators(df)

    stock_map = {s: prepared_map[s] for s in SYMBOLS}
    spy_df = prepared_map[MARKET_SYMBOL]

    common_days = sorted(set(spy_df["date"].unique()))
    for symbol in SYMBOLS:
        common_days = sorted(set(common_days).intersection(set(stock_map[symbol]["date"].unique())))

    if len(common_days) < 10:
        raise ValueError("Not enough common trading days to run test")

    train_days, test_days = split_days(common_days)

    print(f"\nTotal common days: {len(common_days)}")
    print(f"In-sample days: {len(train_days)}")
    print(f"Out-of-sample days: {len(test_days)}")

    train_stock_map = {s: filter_df_by_days(stock_map[s], train_days) for s in SYMBOLS}
    test_stock_map = {s: filter_df_by_days(stock_map[s], test_days) for s in SYMBOLS}
    full_stock_map = {s: filter_df_by_days(stock_map[s], common_days) for s in SYMBOLS}

    train_spy = filter_df_by_days(spy_df, train_days)
    test_spy = filter_df_by_days(spy_df, test_days)
    full_spy = filter_df_by_days(spy_df, common_days)

    train_market_map = build_market_day_map(train_spy)
    test_market_map = build_market_day_map(test_spy)
    full_market_map = build_market_day_map(full_spy)

    train_trades = run_tight_short_strategy(train_stock_map, train_market_map)
    test_trades = run_tight_short_strategy(test_stock_map, test_market_map)
    full_trades = run_tight_short_strategy(full_stock_map, full_market_map)

    train_summary = summarize_trades(train_trades)
    test_summary = summarize_trades(test_trades)
    full_summary = summarize_trades(full_trades)

    print_summary("IN-SAMPLE SUMMARY", train_summary)
    print_summary("OUT-OF-SAMPLE SUMMARY", test_summary)
    print_summary("FULL PERIOD SUMMARY", full_summary)

    if full_trades:
        pd.DataFrame([asdict(t) for t in full_trades]).to_csv(OUTPUT_TRADES_CSV, index=False)
    pd.DataFrame([asdict(full_summary)]).to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"\nSaved trades to {OUTPUT_TRADES_CSV}")
    print(f"Saved summary to {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()