#!/usr/bin/env python3
"""Convert Binance aggTrades zip files to Parquet for fast loading.

Parquet + CuPy GPU resampling is ~4.7x faster than CSV parsing.

Usage:
    python convert_to_parquet.py                    # convert all data/*.zip
    python convert_to_parquet.py --symbol SOLUSDT   # specific symbol
    python convert_to_parquet.py --workers 8        # parallel conversion
    python convert_to_parquet.py --check            # verify existing parquets
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def convert_zip_to_parquet(zip_path: Path, out_dir: Optional[Path] = None) -> Path:
    """Convert a single aggTrades zip file to Parquet (Snappy compressed).

    Parquet file is placed next to the zip (or in out_dir).
    Returns the parquet path.
    """
    if out_dir is None:
        out_dir = zip_path.parent

    parquet_path = out_dir / (zip_path.stem + ".parquet")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(z.namelist()[0]) as inner:
            text = io.TextIOWrapper(inner, encoding="utf-8")
            reader = csv.reader(text)
            header = next(reader)

            # Detect column positions (handle 'qty' vs 'quantity')
            try:
                price_col = header.index("price")
                qty_col   = header.index("qty") if "qty" in header else header.index("quantity")
                time_col  = header.index("transact_time")
                maker_col = header.index("is_buyer_maker")
            except ValueError:
                price_col, qty_col, time_col, maker_col = 1, 2, 5, 6

            prices, qtys, times, makers = [], [], [], []
            for row in reader:
                try:
                    prices.append(float(row[price_col]))
                    qtys.append(float(row[qty_col]))
                    times.append(int(row[time_col]))
                    makers.append(row[maker_col].strip().lower() in ("true", "1"))
                except (ValueError, IndexError):
                    continue

    table = pa.table(
        {
            "price":          pa.array(prices,  type=pa.float32()),
            "qty":            pa.array(qtys,    type=pa.float32()),
            "transact_time":  pa.array(times,   type=pa.int64()),
            "is_buyer_maker": pa.array(makers,  type=pa.bool_()),
        }
    )
    pq.write_table(table, parquet_path, compression="snappy")
    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert aggTrades zip to Parquet")
    parser.add_argument("--data-dir",  default="./data")
    parser.add_argument("--out-dir",   default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--symbol",    nargs="+",
                        help="Symbols to convert (default: all)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert even if parquet already exists")
    parser.add_argument("--check",     action="store_true",
                        help="Verify existing parquet files (read and count rows)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else None

    # Find zip files
    if args.symbol:
        zips = []
        for sym in args.symbol:
            zips.extend(sorted(data_dir.glob(f"{sym}-aggTrades-*.zip")))
    else:
        zips = sorted(data_dir.glob("*-aggTrades-*.zip"))

    if not zips:
        print("No zip files found.")
        sys.exit(1)

    # Filter already-converted (unless --overwrite)
    if not args.overwrite:
        to_convert = []
        already = 0
        for z in zips:
            pq_path = (out_dir or z.parent) / (z.stem + ".parquet")
            if pq_path.exists():
                already += 1
            else:
                to_convert.append(z)
        if already:
            print(f"  {already} files already converted (use --overwrite to redo)")
        zips = to_convert

    if not zips and not args.check:
        print("Nothing to convert.")
        return

    if args.check:
        print(f"\nVerifying parquet files in {out_dir or data_dir} ...\n")
        pqs = sorted((out_dir or data_dir).glob("*-aggTrades-*.parquet"))
        total_rows = 0
        for pq_path in pqs:
            try:
                t = pq.read_table(pq_path, columns=["price"])
                n = len(t)
                total_rows += n
                print(f"  {pq_path.name:<55} {n:>10,} rows  OK")
            except Exception as e:
                print(f"  {pq_path.name:<55} ERROR: {e}")
        print(f"\n  Total: {len(pqs)} files, {total_rows:,} rows")
        return

    print(f"\nConverting {len(zips)} zip files to Parquet (Snappy) ...")
    print(f"  Workers: {args.workers}  Output: {out_dir or 'same dir as input'}\n")

    done = 0
    errors = 0

    def _convert(zip_path: Path) -> Tuple[Path, int]:
        pq_path = convert_zip_to_parquet(zip_path, out_dir)
        t = pq.read_table(pq_path, columns=["price"])
        return pq_path, len(t)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_convert, z): z for z in zips}
        for fut in as_completed(futures):
            zip_path = futures[fut]
            done += 1
            try:
                pq_path, n_rows = fut.result()
                size_kb = pq_path.stat().st_size // 1024
                print(f"  [{done:>4}/{len(zips)}] {zip_path.name:<50} "
                      f"{n_rows:>10,} rows  {size_kb:>6}KB  -> {pq_path.name}",
                      flush=True)
            except Exception as e:
                errors += 1
                print(f"  [{done:>4}/{len(zips)}] ERROR: {zip_path.name}: {e}",
                      flush=True)

    print(f"\nDone: {done - errors} converted, {errors} errors")
    if errors == 0:
        print("\nTo use Parquet in walk_forward.py, it will auto-detect .parquet files")
        print("alongside .zip files and use them automatically (4.7x faster loading).")


if __name__ == "__main__":
    main()
