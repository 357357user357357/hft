#!/usr/bin/env python3
"""Generate labeled training data from historical prices.

Usage:
    python3 generate_training_data.py

Output:
    research/training_data.jsonl - JSONL file with samples
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import math signal modules
from poincare_trading import poincare_analysis
from whitehead_signal import whitehead_analysis
from hecke_operators import HeckeAlgebra
from polar_features import PolarExtractor, PolarSignalGenerator
from frenet_serret import FrenetFrame, FrenetFrameConfig


def compute_all_signals(prices: List[float], timestamps: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compute all mathematical signals for a price window.

    Args:
        prices: List of prices (min 40 for valid analysis)
        timestamps: Optional list of timestamps in ms

    Returns:
        Dict with all signal labels
    """
    if len(prices) < 40:
        return None

    result = {}

    # 1. Poincaré topology
    try:
        poincare = poincare_analysis(prices, embed_dim=3, subsample=60)
        result["poincare"] = {
            "regime": poincare.regime,
            "score": poincare.poincare_score,
            "beta1": poincare.beta1,
            "beta2": poincare.beta2,
            "simply_connected": poincare.simply_connected,
            "mean_ricci": poincare.mean_ricci,
            "neg_ricci_frac": poincare.neg_ricci_frac,
        }
    except Exception as e:
        result["poincare"] = {
            "regime": "neutral",
            "score": 0.0,
            "beta1": 0,
            "beta2": 0,
            "simply_connected": True,
            "mean_ricci": 0.0,
            "neg_ricci_frac": 0.5,
        }

    # 2. Whitehead torsion
    try:
        whitehead = whitehead_analysis(prices, embed_dim=3, subsample=60)
        result["whitehead"] = {
            "signal": whitehead.signal,
            "torsion_ratio": whitehead.torsion.torsion_ratio,
            "is_simple": whitehead.torsion.is_simple,
            "beta1_changes": whitehead.torsion.beta1_changes,
            "beta2_changes": whitehead.torsion.beta2_changes,
            "num_regimes": whitehead.reeb.num_regimes,
        }
    except Exception as e:
        result["whitehead"] = {
            "signal": "same_regime",
            "torsion_ratio": 0.0,
            "is_simple": True,
            "beta1_changes": 0,
            "beta2_changes": 0,
            "num_regimes": 1,
        }

    # 3. Hecke L-function
    try:
        hecke = HeckeAlgebra(max_n=15, weight=2)
        hecke.set_eigenvalues_from_prices(prices)
        l_value = hecke.l_function_value(prices, s=0.5)
        zeta_sig = hecke.zeta_signal(prices)
        result["hecke"] = {
            "l_value_real": l_value.real,
            "l_value_imag": l_value.imag,
            "zeta_signal": zeta_sig if zeta_sig else "moderate",
            "eigenvalues": hecke.eigenvalues,
        }
    except Exception as e:
        result["hecke"] = {
            "l_value_real": 0.0,
            "l_value_imag": 0.0,
            "zeta_signal": "moderate",
            "eigenvalues": {},
        }

    # 4. Polar features
    try:
        polar_ext = PolarExtractor(tau=10, price_scale=prices[0] if prices[0] > 0 else 1.0)
        polar_features = polar_ext.extract(prices)
        if polar_features and len(polar_features) > 0:
            result["polar"] = {
                "mean_r": float(np.mean([f.r for f in polar_features])),
                "mean_theta": float(np.mean([f.theta for f in polar_features])),
                "mean_dr_dt": float(np.mean([f.dr_dt for f in polar_features])),
                "mean_dtheta_dt": float(np.mean([f.dtheta_dt for f in polar_features])),
                "std_r": float(np.std([f.r for f in polar_features])),
                "std_theta": float(np.std([f.theta for f in polar_features])),
            }
        else:
            result["polar"] = {"mean_r": 0, "mean_theta": 0, "mean_dr_dt": 0, "mean_dtheta_dt": 0, "std_r": 0, "std_theta": 0}
    except Exception as e:
        result["polar"] = {"mean_r": 0, "mean_theta": 0, "mean_dr_dt": 0, "mean_dtheta_dt": 0, "std_r": 0, "std_theta": 0}

    # 5. Frenet-Serret frame
    try:
        frenet = FrenetFrame(config=FrenetFrameConfig())
        frames = frenet.analyze(prices)
        if frames and len(frames) > 0:
            result["frenet"] = {
                "mean_curvature": float(np.mean([f.curvature for f in frames])),
                "mean_torsion": float(np.mean([f.torsion for f in frames])),
                "mean_normal_x": float(np.mean([f.normal[0] for f in frames])),
                "mean_normal_y": float(np.mean([f.normal[1] for f in frames])),
                "mean_normal_z": float(np.mean([f.normal[2] for f in frames])),
            }
        else:
            result["frenet"] = {"mean_curvature": 0, "mean_torsion": 0, "mean_normal_x": 0, "mean_normal_y": 0, "mean_normal_z": 0}
    except Exception as e:
        result["frenet"] = {"mean_curvature": 0, "mean_torsion": 0, "mean_normal_x": 0, "mean_normal_y": 0, "mean_normal_z": 0}

    # 6. Future return (for supervised learning - what happens after this window?)
    # This is a placeholder - in real use, you'd look at future prices
    if len(prices) > 1:
        result["future_return"] = (prices[-1] - prices[0]) / prices[0] * 100
    else:
        result["future_return"] = 0.0

    return result


def generate_dataset(data_dir: Path, output_path: Path, window_size: int = 100, stride: int = 50):
    """Generate dataset from all CSV/ZIP files in data_dir.

    Args:
        data_dir: Directory with price data (CSV/ZIP files)
        output_path: Path to save JSONL output
        window_size: Number of prices per sample
        stride: Step size between windows
    """
    from data import load_agg_trades_csv

    all_samples = []

    data_files = list(data_dir.glob("*.zip")) + list(data_dir.glob("*.csv"))
    if not data_files:
        print(f"No data files found in {data_dir}")
        print("Please download data first: python3 download_data.py --symbol BTCUSDT --days 7")
        return

    for file in data_files:
        print(f"Processing {file.name}...")
        try:
            trades = load_agg_trades_csv(file)
        except Exception as e:
            print(f"  Error loading {file.name}: {e}")
            continue

        prices = [t.price for t in trades]
        timestamps = [t.transact_time for t in trades]

        print(f"  {len(trades)} trades, generating windows...")

        # Sliding window
        num_windows = (len(prices) - window_size) // stride + 1
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            window_prices = prices[start_idx:end_idx]
            window_ts = timestamps[start_idx:end_idx]

            if len(window_prices) < window_size:
                continue

            # Skip if prices are all the same (no volatility)
            if max(window_prices) - min(window_prices) < 1e-6:
                continue

            signals = compute_all_signals(window_prices, window_ts)
            if signals is None:
                continue

            sample = {
                "input_prices": window_prices,
                "input_timestamps": window_ts,
                "metadata": {
                    "source_file": file.name,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "min_price": min(window_prices),
                    "max_price": max(window_prices),
                },
                "labels": signals,
            }
            all_samples.append(sample)

            if len(all_samples) % 1000 == 0:
                print(f"    Generated {len(all_samples)} samples...")

    # Save as JSONL
    print(f"\nSaving {len(all_samples)} samples to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Done! Generated {len(all_samples)} samples")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    data_dir = Path("/home/nikolas/Documents/hft/data")
    output_path = Path("/home/nikolas/Documents/hft/research/training_data.jsonl")

    print("=== Generating Training Data for Math Signal AI ===")
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_path}")
    print()

    generate_dataset(data_dir, output_path, window_size=100, stride=50)
