"""Download Binance aggTrades data for backtesting

Base URL: https://data.binance.vision/?prefix=data/futures/um/monthly/aggTrades

Usage:
  # Download 1 day
  python download_data.py --symbol BTCUSDT --days 1 --start 2024-01-15

  # Download 7 days (a week)
  python download_data.py --symbol BTCUSDT --days 7 --start 2024-01-15

  # Download multiple symbols
  python download_data.py --symbol BTCUSDT ETHUSDT ADAUSDT --days 3 --start 2024-01-15

  # Download all low-funding-rate coins for 7 days
  python download_data.py --preset low-funding --days 7 --start 2024-01-15

  # Download entire months (smaller Binance monthly files)
  python download_data.py --symbol BTCUSDT --months 2024-01 2024-02

  # Download DeFi coins for 30 days
  python download_data.py --preset defi --days 30 --start 2024-01-01
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple
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

PRESETS = {
    "low-funding": LOW_FUNDING_COINS,
    "top5": TOP5_COINS,
    "defi": DEFI_COINS,
    "layer1": LAYER1_COINS,
    "altcoin": ALTCOIN_SCALP,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Download Binance aggTrades data for HFT backtesting",
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
