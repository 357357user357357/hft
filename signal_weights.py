"""Centralized signal weights — single source of truth.

All signal weighting across the codebase uses these constants.
Import from here instead of hardcoding weights in multiple places.
"""

from __future__ import annotations
from typing import Dict

# =============================================================================
# INSTRUMENT INDEX WEIGHTS (19 dimensions, sum = 1.0)
# Used by: instrument_index.py, signal_gate.py
# =============================================================================

INSTRUMENT_INDEX_WEIGHTS: Dict[str, float] = {
    # Math / Topology (42%)
    "topology":      0.08,
    "torsion":       0.05,
    "algebraic":     0.05,
    "geometry":      0.05,
    "polar":         0.05,
    "number_theory": 0.03,
    "graph":         0.03,
    "spectral":      0.04,
    "fel":           0.03,
    "quaternion":    0.04,

    # Classic Finance (37%)
    "hurst":         0.07,
    "volatility":    0.05,
    "order_flow":    0.07,
    "volume_profile":0.04,
    "microstructure":0.03,
    "momentum":      0.07,
    "autocorr":      0.07,
    "funding":       0.04,

    # Simons SDE Models (11%)
    "simons":        0.11,
}

# Verify weights sum to 1.0
assert abs(sum(INSTRUMENT_INDEX_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Weights sum to {sum(INSTRUMENT_INDEX_WEIGHTS.values())}, not 1.0"


# =============================================================================
# AGENT INTEGRATION WEIGHTS (6 sources, sum = 1.0)
# Used by: agents/integration.py
# These combine math signals with LLM agent signals
# =============================================================================

AGENT_INTEGRATION_WEIGHTS: Dict[str, float] = {
    # Math signals (50%)
    "poincare":  0.25,
    "whitehead": 0.15,
    "hecke":     0.10,

    # LLM agent signals (50%)
    "sentiment": 0.20,
    "news":      0.15,
    "fundamental": 0.15,
}

# Verify weights sum to 1.0
assert abs(sum(AGENT_INTEGRATION_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Weights sum to {sum(AGENT_INTEGRATION_WEIGHTS.values())}, not 1.0"


# =============================================================================
# REGIME DETECTION WEIGHTS (for weighted regime scoring)
# Used by: regime_detector.py
# =============================================================================

REGIME_WEIGHTS: Dict[str, float] = {
    "genus":        0.40,   # Genus count (low = MR, high = trending)
    "k_ratio":      0.25,   # K_2 / K_0 ratio (high = trending)
    "discrepancy":  0.20,   # Max discrepancy (high = MR)
    "min_generator":0.15,   # Minimum generator size (high = trending)
}


# =============================================================================
# COMPOSITE SUPER-WEIGHTS
# When combining instrument_index (19-dim) with agent_integration (6-source)
# =============================================================================

# Default super-weights for combining the two systems
COMPOSITE_SUPER_WEIGHTS: Dict[str, float] = {
    "instrument_index":  0.70,  # 70% from 19-dimension scorecard
    "agent_integration": 0.30,  # 30% from LLM agents
}


def get_instrument_index_weight(dimension: str) -> float:
    """Get weight for a specific instrument index dimension."""
    return INSTRUMENT_INDEX_WEIGHTS.get(dimension, 0.0)


def get_agent_integration_weight(source: str) -> float:
    """Get weight for a specific agent integration source."""
    return AGENT_INTEGRATION_WEIGHTS.get(source, 0.0)


def get_all_math_weights() -> Dict[str, float]:
    """Get all math/topology related weights from both systems."""
    # From instrument index
    math_dims = {
        "topology": INSTRUMENT_INDEX_WEIGHTS["topology"],
        "torsion": INSTRUMENT_INDEX_WEIGHTS["torsion"],
        "algebraic": INSTRUMENT_INDEX_WEIGHTS["algebraic"],
        "geometry": INSTRUMENT_INDEX_WEIGHTS["geometry"],
        "polar": INSTRUMENT_INDEX_WEIGHTS["polar"],
        "number_theory": INSTRUMENT_INDEX_WEIGHTS["number_theory"],
        "graph": INSTRUMENT_INDEX_WEIGHTS["graph"],
        "spectral": INSTRUMENT_INDEX_WEIGHTS["spectral"],
        "fel": INSTRUMENT_INDEX_WEIGHTS["fel"],
        "quaternion": INSTRUMENT_INDEX_WEIGHTS["quaternion"],
        # From agent integration
        "poincare": AGENT_INTEGRATION_WEIGHTS["poincare"],
        "whitehead": AGENT_INTEGRATION_WEIGHTS["whitehead"],
        "hecke": AGENT_INTEGRATION_WEIGHTS["hecke"],
    }
    return math_dims


def describe_weights() -> str:
    """Return human-readable weight summary."""
    lines = [
        "=" * 60,
        "SIGNAL WEIGHTS SUMMARY",
        "=" * 60,
        "",
        "Instrument Index (19 dimensions):",
        f"  Math/Topology:    {sum(v for k, v in INSTRUMENT_INDEX_WEIGHTS.items() if k in ['topology', 'torsion', 'algebraic', 'geometry', 'polar', 'number_theory', 'graph', 'spectral', 'fel', 'quaternion']):.0%}",
        f"  Classic Finance:  {sum(v for k, v in INSTRUMENT_INDEX_WEIGHTS.items() if k in ['hurst', 'volatility', 'order_flow', 'volume_profile', 'microstructure', 'momentum', 'autocorr', 'funding']):.0%}",
        f"  Simons SDE:       {INSTRUMENT_INDEX_WEIGHTS['simons']:.0%}",
        "",
        "Agent Integration (6 sources):",
        f"  Math signals:     {sum(v for k, v in AGENT_INTEGRATION_WEIGHTS.items() if k in ['poincare', 'whitehead', 'hecke']):.0%}",
        f"  LLM agents:       {sum(v for k, v in AGENT_INTEGRATION_WEIGHTS.items() if k in ['sentiment', 'news', 'fundamental']):.0%}",
        "",
        "Composite Super-Weights:",
        f"  Instrument Index: {COMPOSITE_SUPER_WEIGHTS['instrument_index']:.0%}",
        f"  Agent Integration:{COMPOSITE_SUPER_WEIGHTS['agent_integration']:.0%}",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_weights())
