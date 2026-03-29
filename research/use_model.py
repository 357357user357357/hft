#!/usr/bin/env python3
"""Use trained model for fast math signal prediction.

This is ~100-200× faster than direct computation because:
- No O(n²) persistent homology
- No Gröbner basis computation
- Single forward pass through transformer

Usage:
    python3 use_model.py

Example output:
    Poincaré regime: mean-reversion (score=0.423)
    Whitehead signal: same_regime (torsion=0.15)
    Hecke L(1/2): 35.3 + 0.0j (zeta=strong)

    Inference time: 1.2ms (vs. ~200ms direct computation)
"""

import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from math_signal_model import MathSignalTransformer, create_model


class FastMathSignalPredictor:
    """Fast inference using trained transformer for math signals."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_size: str = "small",
        device: str = "auto",
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            model_size: Model size (tiny/small/medium/large)
            device: "cuda", "cpu", or "auto"
        """
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model on {self.device}...")

        # Create model
        self.model = create_model(model_size)

        # Load weights
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded weights from {model_path}")
            print(f"  Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        else:
            print("  WARNING: Using random weights (not trained)")

        self.model.to(self.device)
        self.model.eval()

        # Class mappings
        self.regime_classes = ["mean-reversion", "trending", "neutral"]
        self.whitehead_classes = ["regime_change", "same_regime"]
        self.zeta_classes = ["strong", "moderate", "weak"]

    def predict(self, prices: List[float]) -> Dict[str, Any]:
        """Predict signals from price sequence.

        Args:
            prices: List of prices (any length, will be normalized)

        Returns:
            Dict with all predicted signals
        """
        if len(prices) < 10:
            raise ValueError(f"Need at least 10 prices, got {len(prices)}")

        # Normalize prices to [0, 1]
        p_min = min(prices)
        p_max = max(prices)
        p_range = max(p_max - p_min, 1e-10)
        prices_norm = [(p - p_min) / p_range for p in prices]

        # Pad/truncate to 200
        max_len = 200
        if len(prices_norm) < max_len:
            prices_norm = prices_norm + [prices_norm[-1]] * (max_len - len(prices_norm))
        else:
            prices_norm = prices_norm[:max_len]

        # Convert to tensor
        prices_tensor = torch.tensor(prices_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        prices_tensor = prices_tensor.to(self.device)

        # Forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(prices_tensor)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert to human-readable format
        result = {
            "poincare": {
                "regime": self.regime_classes[outputs["poincare_regime"][0].argmax().item()],
                "score": float(outputs["poincare_score"][0, 0].item()),
            },
            "whitehead": {
                "signal": self.whitehead_classes[outputs["whitehead_signal"][0].argmax().item()],
                "torsion_ratio": float(outputs["torsion_ratio"][0, 0].item()),
            },
            "hecke": {
                "l_value_real": float(outputs["hecke_l_value"][0, 0].item()),
                "l_value_imag": float(outputs["hecke_l_value"][0, 1].item()),
                "zeta_signal": self.zeta_classes[outputs["zeta_signal"][0].argmax().item()],
            },
            "polar": {
                "mean_r": float(outputs["polar_features"][0, 0].item()),
                "mean_theta": float(outputs["polar_features"][0, 1].item()),
                "mean_dr_dt": float(outputs["polar_features"][0, 2].item()),
                "mean_dtheta_dt": float(outputs["polar_features"][0, 3].item()),
            },
            "frenet": {
                "curvature": float(outputs["frenet_features"][0, 0].item()),
                "torsion": float(outputs["frenet_features"][0, 1].item()),
            },
            "future_return": float(outputs["future_return"][0, 0].item()),
            "inference_time_ms": inference_time_ms,
        }

        # Compute L-value as complex for convenience
        result["hecke"]["l_value"] = complex(
            result["hecke"]["l_value_real"],
            result["hecke"]["l_value_imag"]
        )

        return result

    def predict_batch(self, prices_list: List[List[float]]) -> List[Dict[str, Any]]:
        """Predict signals for multiple price sequences.

        Args:
            prices_list: List of price sequences

        Returns:
            List of prediction dicts
        """
        all_results = []

        for prices in prices_list:
            result = self.predict(prices)
            all_results.append(result)

        return all_results


def compare_with_direct(prices: List[float], predictor: FastMathSignalPredictor):
    """Compare model predictions with direct computation."""
    print("\n=== Model vs. Direct Computation Comparison ===\n")

    # Model prediction (fast)
    model_result = predictor.predict(prices)
    print("Model Prediction (fast):")
    print(f"  Poincaré: {model_result['poincare']['regime']} (score={model_result['poincare']['score']:+.3f})")
    print(f"  Whitehead: {model_result['whitehead']['signal']} (torsion={model_result['whitehead']['torsion_ratio']:.3f})")
    print(f"  Hecke L(1/2): {model_result['hecke']['l_value_real']:.2f} + {model_result['hecke']['l_value_imag']:.2f}j")
    print(f"  Zeta: {model_result['hecke']['zeta_signal']}")
    print(f"  Inference time: {model_result['inference_time_ms']:.2f}ms")

    # Direct computation (slow but exact)
    print("\nDirect Computation (slow, exact):")
    try:
        from poincare_trading import poincare_analysis
        from whitehead_signal import whitehead_analysis
        from hecke_operators import HeckeAlgebra

        start = time.perf_counter()
        poincare = poincare_analysis(prices, embed_dim=3, subsample=60)
        poincare_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        whitehead = whitehead_analysis(prices, embed_dim=3, subsample=60)
        whitehead_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hecke = HeckeAlgebra(max_n=15, weight=2)
        hecke.set_eigenvalues_from_prices(prices)
        l_value = hecke.l_function_value(prices, s=0.5)
        zeta_sig = hecke.zeta_signal(prices)
        hecke_time = (time.perf_counter() - start) * 1000

        direct_time = poincare_time + whitehead_time + hecke_time

        print(f"  Poincaré: {poincare.regime} (score={poincare.poincare_score:+.3f}) [{poincare_time:.1f}ms]")
        print(f"  Whitehead: {whitehead.signal} (torsion={whitehead.torsion.torsion_ratio:.3f}) [{whitehead_time:.1f}ms]")
        print(f"  Hecke L(1/2): {l_value.real:.2f} + {l_value.imag:.2f}j [{hecke_time:.1f}ms]")
        print(f"  Zeta: {zeta_sig or 'moderate'}")
        print(f"  Total time: {direct_time:.1f}ms")

        # Speedup
        speedup = direct_time / model_result['inference_time_ms']
        print(f"\n  → Speedup: {speedup:.1f}× faster with model")

        # Accuracy check
        print("\nAccuracy:")
        regime_match = model_result['poincare']['regime'] == poincare.regime
        print(f"  Poincaré regime match: {regime_match}")
        print(f"  Score diff: {abs(model_result['poincare']['score'] - poincare.poincare_score):.3f}")

        signal_match = model_result['whitehead']['signal'] == whitehead.signal
        print(f"  Whitehead signal match: {signal_match}")
        print(f"  Torsion diff: {abs(model_result['whitehead']['torsion_ratio'] - whitehead.torsion.torsion_ratio):.3f}")

    except ImportError as e:
        print(f"  Cannot import math modules: {e}")
        print("  Skipping direct comparison")


def main():
    print("=== Fast Math Signal Predictor ===\n")

    # Find model path
    model_path = Path("/home/nikolas/Documents/hft/research/math_signal_model.pt")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first with: python3 research/train_model.py")
        return

    # Create predictor
    predictor = FastMathSignalPredictor(
        model_path=model_path,
        model_size="small",
        device="auto",
    )

    # Test with synthetic data
    import math
    print("\n=== Test 1: Synthetic mean-reverting series ===")
    prices_mr = [50000 * (1 + 0.01 * math.sin(i / 10)) for i in range(100)]
    result = predictor.predict(prices_mr)
    print(f"  Regime: {result['poincare']['regime']}")
    print(f"  Score: {result['poincare']['score']:+.3f}")
    print(f"  Time: {result['inference_time_ms']:.2f}ms")

    print("\n=== Test 2: Synthetic trending series ===")
    prices_trend = [50000 * (1 + 0.001 * i) for i in range(100)]
    result = predictor.predict(prices_trend)
    print(f"  Regime: {result['poincare']['regime']}")
    print(f"  Score: {result['poincare']['score']:+.3f}")
    print(f"  Time: {result['inference_time_ms']:.2f}ms")

    print("\n=== Test 3: Real data comparison ===")
    # Try to load real data
    from data import load_agg_trades_csv
    data_dir = Path("/home/nikolas/Documents/hft/data")
    data_files = list(data_dir.glob("*.zip"))

    if data_files:
        print(f"Loading {data_files[0].name}...")
        trades = load_agg_trades_csv(data_files[0])
        prices = [t.price for t in trades[:100]]

        compare_with_direct(prices, predictor)
    else:
        print("No data files found. Using synthetic data for comparison.")
        prices = [50000 * (1 + 0.01 * math.sin(i / 10) + 0.005 * i / 100) for i in range(100)]
        compare_with_direct(prices, predictor)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
