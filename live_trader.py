#!/usr/bin/env python3
"""Live Paper Trader — Binance WebSocket aggTrade stream.

Streams real-time trades from Binance Futures, feeds them into all 4
algorithms (Shot, DepthShot, Averages, Vector) plus the full 19-dimension
signal scorecard (Poincaré, Hecke, Frenet-Serret, p-adic, FEL, Simons SDEs…).

Usage:
    python live_trader.py --symbol BTCUSDT
    python live_trader.py --symbol ETHUSDT --equity 5000 --interval 10
    python live_trader.py --symbol BTCUSDT --no-signals   # faster, no scorecard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("live")

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data import AggTrade
from hft_types import Side, TakeProfitConfig, StopLossConfig, AutoPriceDown, BacktestStats
from slippage import SlippageModel, SlippageConfig
from risk_management import PositionSizer, SizingConfig
from signal_gate import SignalGate, GateConfig
from regime_detector import RegimeDetector
import constants as C

from algorithms.shot import ShotBacktest, ShotConfig
from algorithms.depth_shot import DepthShotBacktest, DepthShotConfig, Depth
from algorithms.averages import AveragesBacktest, AveragesConfig, AveragesCondition
from algorithms.vector import VectorBacktest, VectorConfig, BorderRange, ShotDirection
from data import build_synthetic_book


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket URL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _futures_ws_url(symbol: str) -> str:
    return f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"


def _spot_ws_url(symbol: str) -> str:
    return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@aggTrade"


def _parse_agg_trade(msg: dict) -> AggTrade:
    """Convert Binance aggTrade WebSocket message to AggTrade."""
    return AggTrade(
        agg_trade_id=int(msg["a"]),
        price=float(msg["p"]),
        quantity=float(msg["q"]),
        first_trade_id=int(msg["f"]),
        last_trade_id=int(msg["l"]),
        transact_time=int(msg["T"]),   # ms
        is_buyer_maker=bool(msg["m"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlgoBundle:
    """All four algorithms running in parallel on the same trade stream."""
    shot_buy:   ShotBacktest
    shot_sell:  ShotBacktest
    depth_buy:  DepthShotBacktest
    depth_sell: DepthShotBacktest
    avg_buy:    AveragesBacktest
    avg_sell:   AveragesBacktest
    vector_buy: VectorBacktest
    vector_sell: VectorBacktest
    # rolling trade buffer for DepthShot (needs synthetic book)
    _buffer: Deque[AggTrade] = field(default_factory=lambda: deque(maxlen=500))

    def on_trade(self, trade: AggTrade) -> None:
        self._buffer.append(trade)
        tick_size = trade.price * C.DEPTH_TICK_SCALE
        book = build_synthetic_book(list(self._buffer), C.DEPTH_SYNTHETIC_LEVELS, tick_size)

        self.shot_buy.on_trade(trade)
        self.shot_sell.on_trade(trade)
        self.avg_buy.on_trade(trade)
        self.avg_sell.on_trade(trade)
        self.vector_buy.on_trade(trade)
        self.vector_sell.on_trade(trade)
        self.depth_buy.on_trade(trade, book)
        self.depth_sell.on_trade(trade, book)

    def total_trades(self) -> int:
        return (
            self.shot_buy.stats.total_trades + self.shot_sell.stats.total_trades +
            self.depth_buy.stats.total_trades + self.depth_sell.stats.total_trades +
            self.avg_buy.stats.total_trades + self.avg_sell.stats.total_trades +
            self.vector_buy.stats.total_trades + self.vector_sell.stats.total_trades
        )

    def total_pnl(self) -> float:
        return (
            self.shot_buy.stats.total_pnl_pct + self.shot_sell.stats.total_pnl_pct +
            self.depth_buy.stats.total_pnl_pct + self.depth_sell.stats.total_pnl_pct +
            self.avg_buy.stats.total_pnl_pct + self.avg_sell.stats.total_pnl_pct +
            self.vector_buy.stats.total_pnl_pct + self.vector_sell.stats.total_pnl_pct
        )

    def rows(self) -> List[tuple]:
        """Return (name, stats) for display."""
        return [
            ("Shot  Buy",    self.shot_buy.stats),
            ("Shot  Sell",   self.shot_sell.stats),
            ("Depth Buy",    self.depth_buy.stats),
            ("Depth Sell",   self.depth_sell.stats),
            ("Avg   Buy",    self.avg_buy.stats),
            ("Avg   Sell",   self.avg_sell.stats),
            ("Vec   Buy",    self.vector_buy.stats),
            ("Vec   Sell",   self.vector_sell.stats),
        ]


def _build_algos(
    symbol: str,
    equity: float,
    signal_gate: Optional[SignalGate],
    slippage: SlippageModel,
) -> AlgoBundle:
    """Construct all 8 algorithm instances (4 algos × 2 sides)."""

    def sizer() -> PositionSizer:
        return PositionSizer(SizingConfig(
            mode="fractional",
            fixed_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
            fractional_pct=C.FRACTIONAL_SIZE_PCT,
            kelly_cap=C.KELLY_CAP,
            kelly_lookback=C.KELLY_LOOKBACK_TRADES,
            initial_equity=equity,
        ))

    def shot(side: Side) -> ShotBacktest:
        cfg = ShotConfig(
            side=side,
            distance_pct=C.SHOT_DISTANCE_PCT,
            buffer_pct=C.SHOT_BUFFER_PCT,
            follow_price_delay_secs=C.SHOT_FOLLOW_DELAY_SECS,
            replace_delay_secs=C.SHOT_REPLACE_DELAY_SECS,
            order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
            take_profit=TakeProfitConfig(
                enabled=True, percentage=C.SHOT_TP_PCT,
                auto_price_down=AutoPriceDown(
                    timer_secs=C.SHOT_TP_AUTO_DOWN_TIMER,
                    step_pct=C.SHOT_TP_AUTO_DOWN_STEP,
                    limit_pct=C.SHOT_TP_AUTO_DOWN_LIMIT,
                ),
            ),
            stop_loss=StopLossConfig(
                enabled=True, percentage=C.SHOT_SL_PCT,
                spread_pct=C.SHOT_SL_SPREAD_PCT, delay_secs=C.SHOT_SL_DELAY_SECS,
            ),
        )
        return ShotBacktest(cfg, slippage=slippage, position_sizer=sizer(),
                            signal_gate=signal_gate)

    def depth(side: Side) -> DepthShotBacktest:
        cfg = DepthShotConfig(
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
                enabled=True, percentage=C.DEPTH_SL_PCT,
                spread_pct=C.DEPTH_SL_SPREAD, delay_secs=C.DEPTH_SL_DELAY,
            ),
        )
        return DepthShotBacktest(cfg, slippage=slippage, position_sizer=sizer(),
                                 signal_gate=signal_gate)

    def avg(side: Side) -> AveragesBacktest:
        order_dist = C.AVG_ORDER_DISTANCE_BUY if side == Side.Buy else C.AVG_ORDER_DISTANCE_SELL
        t_min = C.AVG_BUY_TRIGGER_MIN if side == Side.Buy else C.AVG_SELL_TRIGGER_MIN
        t_max = C.AVG_BUY_TRIGGER_MAX if side == Side.Buy else C.AVG_SELL_TRIGGER_MAX
        cfg = AveragesConfig(
            side=side,
            order_distance_pct=order_dist,
            conditions=[AveragesCondition(
                long_period_secs=C.AVG_LONG_PERIOD_SECS,
                short_period_secs=C.AVG_SHORT_PERIOD_SECS,
                trigger_min_pct=t_min,
                trigger_max_pct=t_max,
            )],
            order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
            cancel_delay_secs=C.AVG_CANCEL_DELAY,
            do_not_trigger_if_active=True,
            restart_delay_secs=C.AVG_RESTART_DELAY,
            take_profit=TakeProfitConfig(enabled=True, percentage=C.AVG_TP_PCT),
            stop_loss=StopLossConfig(
                enabled=True, percentage=C.AVG_SL_PCT,
                spread_pct=C.AVG_SL_SPREAD, delay_secs=0.0,
            ),
        )
        return AveragesBacktest(cfg, slippage=slippage, position_sizer=sizer(),
                                signal_gate=signal_gate)

    def vec(side: Side) -> VectorBacktest:
        cfg = VectorConfig(
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
                enabled=True, percentage=C.VEC_SL_PCT,
                spread_pct=C.VEC_SL_SPREAD, delay_secs=0.0,
            ),
            order_size_usdt=C.DEFAULT_ORDER_SIZE_USDT,
        )
        return VectorBacktest(cfg, slippage=slippage, position_sizer=sizer(),
                              signal_gate=signal_gate)

    return AlgoBundle(
        shot_buy=shot(Side.Buy),   shot_sell=shot(Side.Sell),
        depth_buy=depth(Side.Buy), depth_sell=depth(Side.Sell),
        avg_buy=avg(Side.Buy),     avg_sell=avg(Side.Sell),
        vector_buy=vec(Side.Buy),  vector_sell=vec(Side.Sell),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Terminal dashboard
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_WHITE  = "\033[37m"
_DIM    = "\033[2m"


def _pnl_color(v: float) -> str:
    if v > 0:
        return _GREEN
    if v < 0:
        return _RED
    return _WHITE


def _clear() -> None:
    print("\033[2J\033[H", end="", flush=True)


def _render_dashboard(
    symbol: str,
    price: float,
    trade_count: int,
    uptime_secs: float,
    algos: AlgoBundle,
    gate: Optional[SignalGate],
    use_signals: bool,
) -> None:
    _clear()
    now = datetime.now().strftime("%H:%M:%S")

    # ── header ─────────────────────────────────────────────────────────────
    print(f"{_BOLD}{_CYAN}{'═'*72}{_RESET}")
    print(f"{_BOLD}{_CYAN}  HFT LIVE PAPER TRADER  {_RESET}"
          f"  {_BOLD}{symbol}{_RESET}  "
          f"price={_BOLD}{price:,.2f}{_RESET}  "
          f"trades={trade_count:,}  "
          f"uptime={uptime_secs:.0f}s  "
          f"time={now}")
    print(f"{_BOLD}{_CYAN}{'═'*72}{_RESET}")

    # ── signal scorecard ───────────────────────────────────────────────────
    if use_signals and gate is not None:
        sc = gate.scorecard
        regime = gate.regime.value
        if sc is not None:
            comp = sc.composite
            comp_color = _pnl_color(comp)
            print(f"\n{_BOLD}  19-DIM SCORECARD{_RESET}  "
                  f"composite={comp_color}{_BOLD}{comp:+.3f}{_RESET}  "
                  f"regime={_YELLOW}{regime}{_RESET}  "
                  f"({sc.compute_ms:.0f}ms)")
            print(f"  {'─'*68}")

            # Math / topology row
            dims_math = [
                ("topology",    sc.topology.score,    sc.topology.regime),
                ("torsion",     sc.torsion.score,     sc.torsion.signal),
                ("algebraic",   sc.algebraic.score * sc.algebraic.direction, sc.algebraic.zeta_signal),
                ("geometry",    sc.geometry.score,    sc.geometry.curvature_signal),
                ("polar",       sc.polar.score,       sc.polar.regime[:18]),
                ("spectral",    sc.spectral.score,    f"sig={sc.spectral.num_significant}"),
                ("fel",         sc.fel.score,         sc.fel.signal_strength),
                ("quaternion",  sc.quaternion.score,  sc.quaternion.regime),
                ("p-adic/NT",   sc.number_theory.score, f"rough={sc.number_theory.roughness:.3f}"),
                ("graph",       sc.graph.score,       f"dens={sc.graph.density:.3f}"),
            ]
            dims_finance = [
                ("hurst",       sc.hurst.score,       f"H={sc.hurst.H:.3f}"),
                ("volatility",  sc.volatility.score,  sc.volatility.regime),
                ("order_flow",  sc.order_flow.score,  f"OFI={sc.order_flow.ofi_ratio:+.3f}"),
                ("momentum",    sc.momentum.score,    f"RSI={sc.momentum.rsi:.1f}"),
                ("autocorr",    sc.autocorr.score,    f"lag1={sc.autocorr.lag1:+.3f}"),
                ("microstr",    sc.microstructure.score, f"illiq={sc.microstructure.amihud:.1e}"),
                ("vol_profile", sc.volume_profile.score, f"vwap Δ={sc.volume_profile.price_vs_vwap:+.4f}"),
                ("funding",     sc.funding.score,     sc.funding.signal),
                ("simons/SDE",  sc.simons.score,
                 f"{sc.simons.regime} hl={sc.simons.ou_half_life:.1f}b"),
            ]

            print(f"  {_DIM}Math/Topology:{_RESET}")
            for i in range(0, len(dims_math), 5):
                row = dims_math[i:i+5]
                parts = []
                for name, score, info in row:
                    c = _pnl_color(score)
                    parts.append(f"  {name:<12s}{c}{score:+.3f}{_RESET}({info})")
                print("".join(parts))

            print(f"  {_DIM}Classic Finance / SDEs:{_RESET}")
            for i in range(0, len(dims_finance), 4):
                row = dims_finance[i:i+4]
                parts = []
                for name, score, info in row:
                    c = _pnl_color(score)
                    parts.append(f"  {name:<13s}{c}{score:+.3f}{_RESET}({info})")
                print("".join(parts))
        else:
            print(f"\n  {_DIM}Scorecard: warming up… ({gate.config.min_prices} prices needed){_RESET}")
        print()

    # ── algorithm table ────────────────────────────────────────────────────
    print(f"  {_BOLD}{'Algorithm':<14} {'Trades':>6} {'Win%':>6} {'PnL%':>9} "
          f"{'MaxDD%':>8} {'Sharpe':>7} {'ProfFact':>9}{_RESET}")
    print(f"  {'─'*68}")

    for name, stats in algos.rows():
        wr = stats.win_rate()
        pnl = stats.total_pnl_pct
        dd = stats.max_drawdown_pct
        sharpe = stats.sharpe_ratio()
        pf = stats.profit_factor()
        pf_str = f"{pf:.2f}" if not (pf == float("inf") or pf != pf) else "∞"
        pnl_c = _pnl_color(pnl)
        print(f"  {name:<14} {stats.total_trades:>6}  {wr:>5.1f}%  "
              f"{pnl_c}{pnl:>+8.3f}%{_RESET}  {dd:>7.3f}%  "
              f"{sharpe:>6.2f}  {pf_str:>8}")

    # ── totals ─────────────────────────────────────────────────────────────
    total_t = algos.total_trades()
    total_p = algos.total_pnl()
    tp_c = _pnl_color(total_p)
    print(f"  {'─'*68}")
    print(f"  {_BOLD}{'TOTAL':<14} {total_t:>6}  {'':>6}  "
          f"{tp_c}{total_p:>+8.3f}%{_RESET}")
    print(f"\n{_DIM}  Press Ctrl+C to stop{_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Main async loop
# ─────────────────────────────────────────────────────────────────────────────

async def _run(
    symbol: str,
    equity: float,
    use_signals: bool,
    signal_interval: float,
    futures: bool,
    display_interval: float,
) -> None:
    import websockets

    url = _futures_ws_url(symbol) if futures else _spot_ws_url(symbol)
    logger.info("Connecting to %s …", url)

    # Build signal gate
    gate: Optional[SignalGate] = None
    if use_signals:
        gate = SignalGate(symbol, GateConfig(
            eval_interval_secs=signal_interval,
            window_size=500,
            min_prices=60,
            hard_block=False,
            regime_enabled=True,
            regime_interval_secs=30.0,
        ))

    slippage = SlippageModel(SlippageConfig(enabled=False))
    algos = _build_algos(symbol, equity, gate, slippage)

    trade_count = 0
    last_price = 0.0
    start_time = time.monotonic()
    last_display = 0.0

    async for ws in websockets.connect(url, ping_interval=20, ping_timeout=10):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("e") != "aggTrade":
                    continue

                trade = _parse_agg_trade(msg)
                last_price = trade.price
                trade_count += 1

                # Feed all algorithms
                algos.on_trade(trade)

                # Refresh display at configured interval
                now = time.monotonic()
                if now - last_display >= display_interval:
                    last_display = now
                    uptime = now - start_time
                    _render_dashboard(
                        symbol, last_price, trade_count,
                        uptime, algos, gate, use_signals,
                    )

        except websockets.ConnectionClosed:
            logger.warning("WebSocket closed, reconnecting…")
            await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live paper trader — Binance real-time aggTrades + all signals"
    )
    parser.add_argument("--symbol",   default="BTCUSDT",
                        help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--equity",   type=float, default=10_000.0,
                        help="Starting paper equity in USDT (default: 10000)")
    parser.add_argument("--no-signals", action="store_true",
                        help="Disable 19-dim scorecard (faster)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Signal scorecard re-eval interval seconds (default: 5)")
    parser.add_argument("--display",  type=float, default=1.0,
                        help="Dashboard refresh interval seconds (default: 1)")
    parser.add_argument("--spot",     action="store_true",
                        help="Use Binance Spot instead of Futures WebSocket")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    use_signals = not args.no_signals
    futures = not args.spot

    print(f"Starting live paper trader: {args.symbol}  "
          f"{'Futures' if futures else 'Spot'}  "
          f"equity=${args.equity:,.0f}  "
          f"signals={'ON' if use_signals else 'OFF'}")
    print("Connecting…\n")

    try:
        asyncio.run(_run(
            symbol=args.symbol.upper(),
            equity=args.equity,
            use_signals=use_signals,
            signal_interval=args.interval,
            futures=futures,
            display_interval=args.display,
        ))
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


if __name__ == "__main__":
    main()
