"""HFT Backtest — Main runner

Implements Shot, DepthShot, Averages, and Vector algorithms
using Binance aggTrades data from:
https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades

Usage:
  # Single file
  python main.py --file ./data/BTCUSDT-aggTrades-2024-01-15.zip --algo all

  # All files in directory for one symbol
  python main.py --dir ./data --symbol BTCUSDT --algo all

  # Download a week first, then backtest
  python download_data.py --symbol BTCUSDT --days 7 --start 2024-01-15
  python main.py --dir ./data --symbol BTCUSDT --algo all
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("hft")

from hft_types import (
    Side, TakeProfitConfig, StopLossConfig, AutoPriceDown,
    TradeResult, ExitReason, BacktestStats
)
from data import load_agg_trades_csv, build_synthetic_book, AggTrade
from algorithms.shot import ShotBacktest, ShotConfig
from algorithms.depth_shot import DepthShotBacktest, DepthShotConfig, Depth
import constants as C
from config import load_config, save_current_config
from slippage import SlippageModel, SlippageConfig
from algorithms.averages import AveragesBacktest, AveragesConfig, AveragesCondition
from algorithms.vector import VectorBacktest, VectorConfig, BorderRange, ShotDirection


def parse_side(s: str) -> Side:
    return Side.Sell if s.lower() in ("sell", "short") else Side.Buy


def collect_files(args) -> List[Path]:
    if args.file:
        return [Path(args.file)]

    if args.dir:
        d = Path(args.dir)
        files = [
            p for p in d.iterdir()
            if p.suffix.lower() in ('.zip', '.csv', '.gz')
        ]
        if args.symbol:
            files = [f for f in files if f.name.startswith(args.symbol)]
        files.sort()
        return files

    logger.error("Provide either --file or --dir")
    sys.exit(1)


def run_shot(trades: List[AggTrade], side: Side, slippage: Optional[SlippageModel] = None,
             maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
             position_sizer: Optional["PositionSizer"] = None) -> None:
    logger.info("--- Running Shot Algorithm ---")

    config = ShotConfig(
        side=side,
        distance_pct=C.SHOT_DISTANCE_PCT,
        buffer_pct=C.SHOT_BUFFER_PCT,
        follow_price_delay_secs=C.SHOT_FOLLOW_DELAY_SECS,
        replace_delay_secs=C.SHOT_REPLACE_DELAY_SECS,
        order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
        take_profit=TakeProfitConfig(
            enabled=True,
            percentage=C.SHOT_TP_PCT,
            auto_price_down=AutoPriceDown(
                timer_secs=C.SHOT_TP_AUTO_DOWN_TIMER,
                step_pct=C.SHOT_TP_AUTO_DOWN_STEP,
                limit_pct=C.SHOT_TP_AUTO_DOWN_LIMIT,
            ),
        ),
        stop_loss=StopLossConfig(
            enabled=True,
            percentage=C.SHOT_SL_PCT,
            spread_pct=C.SHOT_SL_SPREAD_PCT,
            delay_secs=C.SHOT_SL_DELAY_SECS,
            trailing=None,
            second_sl=None,
        ),
    )

    backtest = ShotBacktest(config, slippage=slippage, maker_fee_pct=maker_fee_pct,
                            taker_fee_pct=taker_fee_pct, position_sizer=position_sizer)
    backtest.run(trades)
    backtest.stats.print_summary("Shot")
    print_sample_trades(backtest.results, 5)


def run_depth_shot(trades: List[AggTrade], side: Side, slippage: Optional[SlippageModel] = None,
                   maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
                   position_sizer: Optional["PositionSizer"] = None) -> None:
    logger.info("--- Running Depth Shot Algorithm ---")

    config = DepthShotConfig(
        side=side,
        target_volume=C.DEPTH_TARGET_VOLUME,
        min_distance_pct=C.DEPTH_MIN_DISTANCE_PCT,
        max_distance_pct=C.DEPTH_MAX_DISTANCE_PCT,
        volume_buffer=C.DEPTH_VOLUME_BUFFER,
        min_buffer_pct=C.DEPTH_MIN_BUFFER_PCT,
        max_buffer_pct=C.DEPTH_MAX_BUFFER_PCT,
        stop_if_out_of_range=False,
        follow_price_delay_secs=C.SHOT_FOLLOW_DELAY_SECS,
        replace_delay_secs=C.SHOT_REPLACE_DELAY_SECS,
        order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
        take_profit_mode=Depth(percentage=C.DEPTH_TP_PERCENTAGE),
        auto_price_down=AutoPriceDown(
            timer_secs=C.DEPTH_AUTO_DOWN_TIMER,
            step_pct=C.DEPTH_AUTO_DOWN_STEP,
            limit_pct=C.DEPTH_AUTO_DOWN_LIMIT,
        ),
        stop_loss=StopLossConfig(
            enabled=True,
            percentage=C.DEPTH_SL_PCT,
            spread_pct=C.DEPTH_SL_SPREAD,
            delay_secs=C.DEPTH_SL_DELAY,
            trailing=None,
            second_sl=None,
        ),
    )

    tick_size = trades[0].price * C.DEPTH_TICK_SCALE if trades else 1.0

    backtest = DepthShotBacktest(config, slippage=slippage, maker_fee_pct=maker_fee_pct,
                                 taker_fee_pct=taker_fee_pct, position_sizer=position_sizer)
    backtest.run(trades, lambda window: build_synthetic_book(window, C.DEPTH_SYNTHETIC_LEVELS, tick_size))
    backtest.stats.print_summary("Depth Shot")
    print_sample_trades(backtest.results, 5)


def run_averages(trades: List[AggTrade], side: Side, slippage: Optional[SlippageModel] = None,
                 maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
                 position_sizer: Optional["PositionSizer"] = None) -> None:
    logger.info("--- Running Averages Algorithm ---")

    if side == Side.Buy:
        trigger_min, trigger_max = C.AVG_BUY_TRIGGER_MIN, C.AVG_BUY_TRIGGER_MAX
        order_distance_pct = C.AVG_ORDER_DISTANCE_BUY
    else:
        trigger_min, trigger_max = C.AVG_SELL_TRIGGER_MIN, C.AVG_SELL_TRIGGER_MAX
        order_distance_pct = C.AVG_ORDER_DISTANCE_SELL

    config = AveragesConfig(
        side=side,
        order_distance_pct=order_distance_pct,
        conditions=[
            AveragesCondition(
                long_period_secs=C.AVG_LONG_PERIOD_SECS,
                short_period_secs=C.AVG_SHORT_PERIOD_SECS,
                trigger_min_pct=trigger_min,
                trigger_max_pct=trigger_max,
            ),
        ],
        order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
        cancel_delay_secs=C.AVG_CANCEL_DELAY,
        do_not_trigger_if_active=True,
        restart_delay_secs=C.AVG_RESTART_DELAY,
        take_profit=TakeProfitConfig(
            enabled=True,
            percentage=C.AVG_TP_PCT,
            auto_price_down=None,
        ),
        stop_loss=StopLossConfig(
            enabled=True,
            percentage=C.AVG_SL_PCT,
            spread_pct=C.AVG_SL_SPREAD,
            delay_secs=0.0,
            trailing=None,
            second_sl=None,
        ),
        grid=None,
    )

    backtest = AveragesBacktest(config, slippage=slippage, maker_fee_pct=maker_fee_pct,
                                taker_fee_pct=taker_fee_pct, position_sizer=position_sizer)
    backtest.run(trades)
    backtest.stats.print_summary("Averages")
    print_sample_trades(backtest.results, 5)


def run_vector(trades: List[AggTrade], side: Side, slippage: Optional[SlippageModel] = None,
               maker_fee_pct: float = 0.02, taker_fee_pct: float = 0.05,
               position_sizer: Optional["PositionSizer"] = None) -> None:
    logger.info("--- Running Vector Algorithm ---")

    config = VectorConfig(
        side=side,
        frame_size_secs=C.VEC_FRAME_SIZE_SECS,
        time_frame_secs=C.VEC_TIME_FRAME_SECS,
        min_spread_size_pct=C.VEC_MIN_SPREAD_PCT,
        upper_border_range=BorderRange(enabled=False, min_pct=0.0, max_pct=100.0),
        lower_border_range=BorderRange(enabled=False, min_pct=0.0, max_pct=100.0),
        min_trades_per_frame=C.VEC_MIN_TRADES_PER_FRAME,
        min_quote_asset_volume=C.VEC_MIN_QUOTE_VOLUME,
        order_distance_pct=C.VEC_ORDER_DISTANCE_PCT,
        use_adaptive_order_distance=True,
        order_lifetime_secs=C.VEC_ORDER_LIFETIME_SECS,
        max_orders=C.VEC_MAX_ORDERS,
        order_frequency_secs=C.VEC_ORDER_FREQ_SECS,
        detect_shot=False,
        detect_shot_pullback_pct=C.VEC_PULLBACK_PCT,
        shot_direction=ShotDirection.Down,
        take_profit_spread_pct=C.VEC_TP_SPREAD_PCT,
        use_adaptive_take_profit=True,
        auto_price_down=None,
        stop_loss=StopLossConfig(
            enabled=True,
            percentage=C.VEC_SL_PCT,
            spread_pct=C.VEC_SL_SPREAD,
            delay_secs=0.0,
            trailing=None,
            second_sl=None,
        ),
        order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
    )

    backtest = VectorBacktest(config, slippage=slippage, maker_fee_pct=maker_fee_pct,
                              taker_fee_pct=taker_fee_pct, position_sizer=position_sizer)
    backtest.run(trades)
    backtest.stats.print_summary("Vector")
    print_sample_trades(backtest.results, 5)


def print_sample_trades(results: List[TradeResult], n: int) -> None:
    if not results:
        logger.info("  No trades executed.\n")
        return

    count = min(n, len(results))
    logger.info("  Sample trades (first %d):", count)
    for r in results[:count]:
        side_str = "BUY " if r.side == Side.Buy else "SELL"
        exit_str = "TP" if r.exit_reason == ExitReason.TakeProfit else "SL"
        logger.info("  %s entry=%.4f exit=%.4f pnl=%+.4f%% [%s]",
                     side_str, r.entry_price, r.exit_price, r.pnl_pct, exit_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hft_backtest",
        description="HFT Backtest: Shot, DepthShot, Averages, Vector algorithms on Binance aggTrades",
    )
    parser.add_argument("--file", help="Path to single aggTrades CSV or ZIP file")
    parser.add_argument("--dir", help="Directory containing multiple aggTrades ZIP files")
    parser.add_argument("--symbol", help="Filter by symbol when using --dir (e.g., BTCUSDT)")
    parser.add_argument("--algo", default="all",
                        help="Algorithm to run: shot | depth_shot | averages | vector | all")
    parser.add_argument("--side", default="buy", help="Trading side: buy | sell")
    parser.add_argument("--max-trades", type=int, default=0,
                        help="Maximum number of trades to process per file (0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only show WARNING and above")
    parser.add_argument("-j", "--parallel", action="store_true",
                        help="Run algorithms in parallel when --algo all")
    parser.add_argument("--config", help="JSON config file to override default constants")
    parser.add_argument("--save-config", metavar="PATH",
                        help="Save current config (after overrides) to JSON file and exit")
    parser.add_argument("--slippage-bps", type=float, default=0.0,
                        help="Fixed slippage in basis points (1 bp = 0.01%%)")
    parser.add_argument("--latency-ms", type=int, default=0,
                        help="Simulated order latency in milliseconds")
    parser.add_argument("--maker-fee", type=float, default=C.DEFAULT_MAKER_FEE_PCT,
                        help="Maker fee percentage (default: Binance maker 0.02%%)")
    parser.add_argument("--taker-fee", type=float, default=C.DEFAULT_TAKER_FEE_PCT,
                        help="Taker fee percentage (default: Binance taker 0.05%%)")
    parser.add_argument("--sizing-mode", type=str, default=C.POSITION_SIZING_MODE,
                        help="Position sizing: fixed | fractional | kelly")
    parser.add_argument("--sizing-pct", type=float, default=C.FRACTIONAL_SIZE_PCT,
                        help="Position size %% of equity (for fractional mode)")
    parser.add_argument("--initial-equity", type=float, default=C.INITIAL_EQUITY_USDT,
                        help="Starting equity in USDT")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_config(args.config)

    slippage = SlippageModel(SlippageConfig(
        fixed_bps=args.slippage_bps,
        latency_ms=args.latency_ms,
        enabled=(args.slippage_bps > 0 or args.latency_ms > 0),
    ))
    if slippage.config.enabled:
        logger.info("%s", slippage.describe())

    # Initialize position sizer
    from risk_management import PositionSizer, SizingConfig
    position_sizer = PositionSizer(SizingConfig(
        mode=args.sizing_mode,
        fixed_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
        fractional_pct=args.sizing_pct,
        kelly_cap=C.KELLY_CAP,
        kelly_lookback=C.KELLY_LOOKBACK_TRADES,
        initial_equity=args.initial_equity,
    ))
    logger.info("Position sizing: %s (initial equity: $%.2f)", args.sizing_mode, args.initial_equity)

    if args.save_config:
        save_current_config(args.save_config)
        return

    files = collect_files(args)

    if not files:
        logger.warning("No files found. Use --file or --dir with --symbol.")
        return

    logger.info("Files to process: %d", len(files))
    for f in files:
        logger.debug("  %s", f)

    all_trades: List[AggTrade] = []
    for file in files:
        logger.info("Loading %s ...", file.name)
        trades = load_agg_trades_csv(file)
        if args.max_trades > 0 and len(trades) > args.max_trades:
            trades = trades[:args.max_trades]
        logger.info("  %d trades loaded", len(trades))
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t.transact_time)

    logger.info("Total trades: %d", len(all_trades))

    if not all_trades:
        logger.warning("No trades to process.")
        return

    first_ts = all_trades[0].transact_time
    last_ts = all_trades[-1].transact_time
    first_price = all_trades[0].price
    last_price = all_trades[-1].price
    duration_days = (last_ts - first_ts) / 86_400_000.0

    logger.info("Period: %.1f days (%s → %s)", duration_days, first_ts, last_ts)
    logger.info("Price range: %.4f → %.4f", first_price, last_price)

    side = parse_side(args.side)
    algo = args.algo.lower()

    runners = []
    if algo in ("shot", "all"):
        runners.append(("Shot", run_shot, all_trades, side, slippage, args.maker_fee, args.taker_fee, position_sizer))
    if algo in ("depth_shot", "all"):
        runners.append(("DepthShot", run_depth_shot, all_trades, side, slippage, args.maker_fee, args.taker_fee, position_sizer))
    if algo in ("averages", "all"):
        runners.append(("Averages", run_averages, all_trades, side, slippage, args.maker_fee, args.taker_fee, position_sizer))
    if algo in ("vector", "all"):
        runners.append(("Vector", run_vector, all_trades, side, slippage, args.maker_fee, args.taker_fee, position_sizer))

    if args.parallel and len(runners) > 1:
        logger.info("Running %d algorithms in parallel ...", len(runners))

        def make_sizer():
            return PositionSizer(SizingConfig(
                mode=args.sizing_mode,
                fixed_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
                fractional_pct=args.sizing_pct,
                kelly_cap=C.KELLY_CAP,
                kelly_lookback=C.KELLY_LOOKBACK_TRADES,
                initial_equity=args.initial_equity,
            ))

        def make_slippage():
            return SlippageModel(SlippageConfig(
                fixed_bps=args.slippage_bps,
                latency_ms=args.latency_ms,
                enabled=(args.slippage_bps > 0 or args.latency_ms > 0),
            ))

        with ThreadPoolExecutor(max_workers=len(runners)) as pool:
            futures = {
                pool.submit(fn, trades, s, make_slippage(), mf, tf, make_sizer()): name
                for name, fn, trades, s, _slip, mf, tf, _ps in runners
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("%s raised: %s", name, exc)
    else:
        for name, fn, trades, s, slip, mf, tf, ps in runners:
            fn(trades, s, slip, mf, tf, ps)


if __name__ == "__main__":
    main()
