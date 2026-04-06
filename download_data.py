"""Download aggTrades data for backtesting — Binance or Bybit

Binance base URL: https://data.binance.vision/data/futures/um/daily/aggTrades
Bybit base URL:   https://public.bybit.com/trading/

Usage:
  # Download Binance futures data (default)
  python download_data.py --symbol BTCUSDT --days 30 --start 2024-01-01

  # Download Bybit spot data (matches live trader)
  python download_data.py --source bybit --symbol BTCUSDT SOLUSDT --days 30

  # Download Bybit for all watchlist symbols
  python download_data.py --source bybit --preset watchlist --days 30

  # Download multiple symbols from Binance
  python download_data.py --symbol BTCUSDT ETHUSDT ADAUSDT --days 3 --start 2024-01-15

  # Download entire months (smaller Binance monthly files)
  python download_data.py --symbol BTCUSDT --months 2024-01 2024-02
"""

import argparse
import csv
import gzip
import io
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger("hft")

# Coins with historically low or near-zero funding rates on Binance Futures.
LOW_FUNDING_COINS = [
    "USDCUSDT",   # USDC/USDT — near-zero funding, very stable
    "BUSDUSDT",   # BUSD/USDT — near-zero funding
    "BTCUSDT",    # BTC — most liquid, funding usually ±0.01%
    "ETHUSDT",    # ETH — second most liquid
    "ADAUSDT",    # ADA (Cardano) — often low funding
    "DOTUSDT",    # DOT (Polkadot)
    "LINKUSDT",   # LINK (Chainlink)
    "LTCUSDT",    # LTC (Litecoin) — often low funding
    "XRPUSDT",    # XRP — large cap, balanced
    "BNBUSDT",    # BNB — Binance native token
]

# Top 5 by volume
TOP5_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

# DeFi tokens with good liquidity
DEFI_COINS = [
    "UNIUSDT",    # Uniswap
    "AAVEUSDT",   # Aave
    "CRVUSDT",    # Curve
    "MKRUSDT",    # Maker
    "COMPUSDT",   # Compound
    "SNXUSDT",    # Synthetix
    "SUSHIUSDT",  # SushiSwap
    "YFIUSDT",    # yearn.finance
    "1INCHUSDT",  # 1inch
    "DYDXUSDT",   # dYdX
]

# Layer-1 / Layer-2 blockchains
LAYER1_COINS = [
    "BTCUSDT",    # Bitcoin
    "ETHUSDT",    # Ethereum
    "SOLUSDT",    # Solana
    "AVAXUSDT",   # Avalanche
    "MATICUSDT",  # Polygon (MATIC)
    "ATOMUSDT",   # Cosmos
    "NEARUSDT",   # NEAR Protocol
    "ALGOUSDT",   # Algorand
    "FTMUSDT",    # Fantom
    "DOTUSDT",    # Polkadot
]

# High-volume altcoins good for scalping
ALTCOIN_SCALP = [
    "DOGEUSDT",   # Dogecoin — high volume, volatile
    "SHIBUSDT",   # Shiba Inu
    "PEPEUSDT",   # PEPE
    "FLOKIUSDT",  # Floki
    "XRPUSDT",    # XRP
    "ADAUSDT",    # Cardano
    "TRXUSDT",    # TRON
    "LTCUSDT",    # Litecoin
    "BCHUSDT",    # Bitcoin Cash
    "ETCUSDT",    # Ethereum Classic
]

# Bybit live trader watchlist — matches _WATCHLIST in bybit_trader.py
BYBIT_WATCHLIST = [
    "SOLUSDT", "BTCUSDT", "ETHUSDT", "LINKUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT",
]

PRESETS = {
    "low-funding": LOW_FUNDING_COINS,
    "top5": TOP5_COINS,
    "defi": DEFI_COINS,
    "layer1": LAYER1_COINS,
    "altcoin": ALTCOIN_SCALP,
    "watchlist": BYBIT_WATCHLIST,   # Bybit live trader symbols
}


def parse_date(s: str) -> Tuple[int, int, int]:
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"Date must be YYYY-MM-DD, got: {s}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29
        return 28
    return 30


def advance_date(start: Tuple[int, int, int], days: int) -> Tuple[int, int, int]:
    y, m, d = start
    for _ in range(days):
        dim = days_in_month(y, m)
        d += 1
        if d > dim:
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    return y, m, d


def unix_to_ymd(unix: int) -> Tuple[int, int, int]:
    days = unix // 86400
    year = 1970 + (days // 365)
    day_of_year = days % 365
    month = min((day_of_year // 30 + 1), 12)
    day = min((day_of_year % 30 + 1), 28)
    return year, month, day


def download_file(url: str, path: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            data = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}")
    except urllib.error.URLError as e:
        raise RuntimeError(str(e))

    path.write_bytes(data)
    return len(data)


def parse_month(s: str) -> Tuple[int, int]:
    """Parse YYYY-MM into (year, month)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Month must be YYYY-MM, got: {s}")
    return int(parts[0]), int(parts[1])


# ── Bybit public data download ────────────────────────────────────────────────
# Bybit hosts trade data at https://public.bybit.com/trading/{symbol}/{symbol}{YYYY-MM-DD}.csv.gz
# Format: timestamp(ns), symbol, side, size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional

BYBIT_BASE_URL = "https://public.bybit.com/trading"


def download_bybit_day(symbol: str, year: int, month: int, day: int,
                       output: Path) -> int:
    """Download one day of Bybit trade data. Returns bytes downloaded."""
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    filename  = f"{symbol}{date_str}.csv.gz"
    url       = f"{BYBIT_BASE_URL}/{symbol}/{filename}"
    out_path  = output / f"{symbol}-trades-{date_str}.csv.gz"

    if out_path.exists():
        logger.debug("  [skip] %s (already exists)", out_path.name)
        return 0  # 0 = skipped

    nbytes = download_file(url, out_path)
    return nbytes


def load_bybit_csv_gz(path: Path) -> List[Tuple[float, float, float]]:
    """Load Bybit trade data. Returns list of (timestamp_ms, price, qty)."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_ms = float(row["timestamp"]) / 1e6  # ns → ms
                price = float(row["price"])
                size  = float(row["size"])
                rows.append((ts_ms, price, size))
            except (KeyError, ValueError):
                pass
    return rows


def convert_bybit_to_binance_format(bybit_path: Path, out_path: Path) -> int:
    """Convert Bybit .csv.gz to Binance aggTrades zip so existing code works.

    Output columns: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
    """
    rows = load_bybit_csv_gz(bybit_path)
    if not rows:
        return 0

    csv_rows = []
    for i, (ts_ms, price, qty) in enumerate(rows):
        csv_rows.append([i, price, qty, i, i, int(ts_ms), "False"])

    # Write CSV inside zip
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["agg_trade_id", "price", "quantity",
                "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"])
    w.writerows(csv_rows)

    date_part = bybit_path.stem  # e.g. BTCUSDT2024-01-15
    # Extract symbol and date from filename like SOLUSDT2024-01-15
    stem = bybit_path.stem  # SOLUSDT2024-01-15 (no .csv)
    inner_name = stem + ".csv"

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, buf.getvalue())

    return len(csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Download aggTrades data for HFT backtesting (Binance or Bybit)",
    )
    parser.add_argument("--symbol", nargs="+", default=[],
                        help="Trading symbols (e.g., BTCUSDT ETHUSDT ADAUSDT)")
    parser.add_argument("--preset",
                        help=f"Preset: {' | '.join(PRESETS)}")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to download (daily files)")
    parser.add_argument("--months", nargs="+",
                        help="Download monthly files (e.g., 2024-01 2024-02)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD), defaults to 7 days ago")
    parser.add_argument("--output", default="./data",
                        help="Output directory for downloaded files")
    parser.add_argument("--list-presets", action="store_true",
                        help="Show all available presets and their coins")
    parser.add_argument("--source", default="binance", choices=["binance", "bybit"],
                        help="Data source: binance (futures) or bybit (spot). Default: binance")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.list_presets:
        for name, coins in PRESETS.items():
            logger.info("\n%s:", name)
            for c in coins:
                logger.info("  %s", c)
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.preset:
        if args.preset not in PRESETS:
            logger.error("Unknown preset '%s'. Available: %s", args.preset, ' | '.join(PRESETS))
            sys.exit(1)
        symbols = list(PRESETS[args.preset])
    elif args.symbol:
        symbols = args.symbol
    else:
        symbols = ["BTCUSDT"]

    downloaded = 0
    failed = 0

    # ── Bybit download mode ───────────────────────────────────────────────────
    if args.source == "bybit":
        if args.start:
            start_date = parse_date(args.start)
        else:
            days_ago   = int(time.time()) - args.days * 86400
            start_date = unix_to_ymd(days_ago)

        bybit_dir = output / "bybit_raw"
        bybit_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Bybit source: downloading %d symbol(s) x %d day(s)",
                    len(symbols), args.days)
        logger.info("Start date  : %04d-%02d-%02d", *start_date)

        for symbol in symbols:
            for day_offset in range(args.days):
                year, month, day_num = advance_date(start_date, day_offset)
                date_str  = f"{year:04d}-{month:02d}-{day_num:02d}"
                gz_path   = bybit_dir / f"{symbol}-trades-{date_str}.csv.gz"
                zip_path  = output / f"{symbol}-aggTrades-{date_str}.zip"

                # Skip if final zip already exists
                if zip_path.exists():
                    logger.debug("  [skip] %s", zip_path.name)
                    downloaded += 1
                    continue

                logger.info("  Downloading %s %s ...", symbol, date_str)
                try:
                    nbytes = download_bybit_day(symbol, year, month, day_num, bybit_dir)
                    if nbytes == 0 and gz_path.exists():
                        pass  # was already cached
                    elif nbytes > 0:
                        logger.info("    Raw: %.1f MB", nbytes / 1_048_576)
                    # Convert to Binance-compatible zip
                    n_rows = convert_bybit_to_binance_format(gz_path, zip_path)
                    if n_rows > 0:
                        logger.info("    Converted: %d trades → %s", n_rows, zip_path.name)
                        downloaded += 1
                    else:
                        logger.warning("    Empty or failed conversion for %s", gz_path.name)
                        failed += 1
                except Exception as e:
                    logger.error("  FAILED %s %s: %s", symbol, date_str, e)
                    failed += 1
                    for p in [gz_path, zip_path]:
                        if p.exists() and p.stat().st_size == 0:
                            p.unlink()

        logger.info("Bybit download complete: %d files, %d failed", downloaded, failed)
        logger.info("Files saved to: %s", output)
        logger.info("To run backtest: python run_signal_backtest.py --data-dir %s", output)
        return

    # ── Binance download mode (original) ─────────────────────────────────────
    # Monthly download mode
    if args.months:
        base_url = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
        logger.info("Downloading %d symbol(s) x %d month(s) (monthly files)",
                    len(symbols), len(args.months))
        for symbol in symbols:
            for month_str in args.months:
                year, month = parse_month(month_str)
                filename = f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
                url = f"{base_url}/{symbol}/{filename}"
                output_path = output / filename

                if output_path.exists():
                    logger.debug("  [skip] %s (already exists)", filename)
                    downloaded += 1
                    continue

                logger.info("  Downloading %s ...", filename)
                try:
                    nbytes = download_file(url, output_path)
                    logger.info("  OK (%.1f MB)", nbytes / 1_048_576)
                    downloaded += 1
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    failed += 1
                    if output_path.exists():
                        output_path.unlink()
    else:
        # Daily download mode
        if args.start:
            start_date = parse_date(args.start)
        else:
            days_ago = int(time.time()) - 7 * 86400
            start_date = unix_to_ymd(days_ago)

        logger.info("Downloading %d symbol(s) x %d day(s)", len(symbols), args.days)
        logger.info("Start date: %04d-%02d-%02d", *start_date)

        base_url = "https://data.binance.vision/data/futures/um/daily/aggTrades"

        for symbol in symbols:
            date = start_date
            for day in range(args.days):
                year, month, day_num = advance_date(date, day)
                filename = f"{symbol}-aggTrades-{year:04d}-{month:02d}-{day_num:02d}.zip"
                url = f"{base_url}/{symbol}/{filename}"
                output_path = output / filename

                if output_path.exists():
                    logger.debug("  [skip] %s (already exists)", filename)
                    downloaded += 1
                    continue

                logger.info("  Downloading %s ...", filename)
                try:
                    nbytes = download_file(url, output_path)
                    logger.info("  OK (%.1f MB)", nbytes / 1_048_576)
                    downloaded += 1
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    failed += 1
                    if output_path.exists():
                        output_path.unlink()

            date = start_date  # reset for next symbol

    logger.info("Downloaded: %d files, Failed: %d", downloaded, failed)
    logger.info("To run backtest: python main.py --dir ./data --algo all")


if __name__ == "__main__":
    main()
