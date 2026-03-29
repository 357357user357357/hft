# Polar Coordinates in Trading: Research Review

## Overview

This review examines the potential of **polar coordinate systems** for representing market data as vectors rather than scalars, and their integration with AI trading systems like TradingAgents.

---

## 1. Current State: Scalar Representations

Most HFT systems today use **scalar representations**:

```
Price: P(t) ∈ ℝ
Returns: r(t) = (P(t) - P(t-1)) / P(t-1)
Volatility: σ(t) = std(r(t-w:t))
```

**Limitations:**
- Loses directional information (momentum phase)
- Treats price as 1D when market dynamics are inherently multi-dimensional
- Differentiation of scalars loses geometric structure
- Hard to capture rotational patterns (cycles, spirals)

---

## 2. Polar Coordinate Representation

### 2.1 Basic Formulation

Transform price-time into polar coordinates:

```
Given reference point (P₀, t₀):
  r = √((P - P₀)² + α(t - t₀)²)    [radial distance]
  θ = arctan(α(t - t₀) / (P - P₀))  [angular position]
```

Where α is a scaling factor to normalize time and price units.

### 2.2 Phase Space Reconstruction

More powerful: embed price in **phase space** using delay coordinates:

```
x(t) = P(t)
y(t) = P(t - τ)  or  y(t) = dP/dt (momentum)

Then convert to polar:
  r(t) = √(x² + y²)       [amplitude/magnitude]
  θ(t) = arctan2(y, x)    [phase angle]
```

**Key insight:** Market regimes manifest as distinct trajectories in (r, θ) space:
- **Mean-reversion**: Circular/elliptical orbits (θ cycles regularly)
- **Trending**: Spiral outward (r increases monotonically)
- **Breakout**: Sudden θ jump with r expansion

---

## 3. Vector vs Scalar Differentiation

### 3.1 Scalar Differentiation (Current)

```python
# Standard approach
price_change = P(t) - P(t-1)
momentum = dP/dt
acceleration = d²P/dt²
```

**Problem:** Each derivative loses information. A 2nd derivative tells you nothing about the original price level.

### 3.2 Vector Differentiation (Polar)

```python
# Position vector in phase space
z(t) = r(t) · e^(i·θ(t))  [complex representation]

# Derivative preserves structure
dz/dt = (dr/dt) · e^(i·θ) + i·r·(dθ/dt) · e^(i·θ)
      = e^(i·θ) · (dr/dt + i·r·dθ/dt)

# Radial velocity: dr/dt (expansion/contraction)
# Angular velocity: dθ/dt (rotation speed)
```

**Advantages:**
- Single complex derivative encodes **both** magnitude change AND rotation
- Can detect **spiral patterns** (trending with momentum rotation)
- Preserves geometric structure through transformations

### 3.3 Practical Trading Signals

| Signal | Scalar Approach | Polar/Vector Approach |
|--------|-----------------|----------------------|
| Momentum | dP/dt | r·dθ/dt (angular momentum) |
| Volatility | std(dP/P) | std(dr/r) + std(dθ) |
| Regime change | Threshold on P | θ discontinuity detection |
| Cycle detection | FFT on returns | d²θ/dt² zero-crossings |

---

## 4. Integration with AI Systems (TradingAgents)

### 4.1 TradingAgents Architecture

From the GitHub repo, TradingAgents uses:
- Multi-agent collaboration (fundamental, technical, sentiment analysts)
- LLM-based reasoning over market data
- Decision aggregation for trades

### 4.2 Polar Coordinates Enhancement

**Current input to agents:**
```
{price: 50234.5, change_pct: 1.2, volume: 1.5B, ...}
```

**Enhanced vector input:**
```
{
  # Polar features
  radial_distance: 1.05,        # r/r_ref
  phase_angle: 0.34,            # θ in radians
  angular_velocity: 0.12,       # dθ/dt
  radial_velocity: 0.03,        # dr/dt

  # Derived
  spiral_coefficient: 0.28,     # (dr/dt) / (r·dθ/dt)
  phase_coherence: 0.87,        # How circular vs chaotic
}
```

### 4.3 Agent Specialization by Coordinate Type

| Agent Type | Scalar Features | Vector/Polar Features |
|------------|-----------------|----------------------|
| **Trend Agent** | Moving averages, ADX | r(t) monotonicity, spiral detection |
| **Mean-Reversion Agent** | Bollinger Bands, RSI | θ periodicity, orbit stability |
| **Breakout Agent** | Volume spikes, ATR | θ discontinuity, dr/dt surge |
| **Market Maker Agent** | Bid-ask spread, order flow | Phase space density, attractor proximity |

---

## 5. Mathematical Foundations

### 5.1 Complex Log Returns

Instead of standard log returns:

```
Standard: r(t) = log(P(t)/P(t-1))

Complex: z(t) = log(P(t)/P(t-1)) + i·θ(t)
where θ(t) = arg(volume_buy - volume_sell) or similar
```

### 5.2 Quaternion Extension (4D)

For multi-asset portfolios:

```
q = w + xi + yj + zk

w = portfolio return
x = BTC momentum
y = ETH momentum
z = correlation term
```

Enables modeling **rotations in portfolio space** (sector rotation, asset rotation).

### 5.3 Differential Geometry Connection

The **Frenet-Serret frame** for price curves:

```
T (tangent) = direction of price movement
N (normal) = direction of curvature (mean-reversion force)
B (binormal) = torsion (regime change indicator)

Curvature κ = |dT/ds|  →  How "curved" the price path
Torsion τ = |dB/ds|    →  How much it's twisting out of plane
```

**Trading interpretation:**
- High κ + low τ → Strong mean-reversion (circular motion)
- Low κ + any τ → Trending (straight or helical)
- Sudden τ spike → Regime change

---

## 6. Implementation Roadmap

### Phase 1: Basic Polar Features
```python
# Add to fel_signal.py or new polar_features.py

def polar_embed(prices: List[float], tau: int = 10) -> List[Tuple[float, float]]:
    """Convert price series to (r, θ) trajectory."""
    result = []
    for t in range(tau, len(prices)):
        x = prices[t]
        y = prices[t] - prices[t - tau]  # momentum proxy
        r = math.sqrt(x*x + y*y)
        theta = math.atan2(y, x)
        result.append((r, theta))
    return result
```

### Phase 2: Vector-Aware Algorithms
```python
# Modify Vector algorithm to use polar features

class VectorBacktest:
    def __init__(self, config, use_polar=True):
        self.use_polar = use_polar
        # ...

    def _detect_signal(self, prices):
        if self.use_polar:
            r, theta = self._polar_features(prices)
            # Detect spiral patterns, phase jumps
            return self._polar_signal(r, theta)
        else:
            return self._scalar_signal(prices)
```

### Phase 3: AI Integration
```python
# Agent prompt enhancement

POLAR_CONTEXT = """
Market Phase Space Analysis:
- Current radial distance: {r:.3f} ({direction})
- Phase angle: {theta:.3f} rad ({phase})
- Angular velocity: {dtheta:.3f} ({rotation_speed})
- Spiral coefficient: {spiral:.3f} ({spiral_type})

Interpretation: {ai_readable_summary}
"""
```

---

## 7. Research Directions

### 7.1 Open Questions

1. **Optimal reference point**: What should (P₀, t₀) be?
   - Rolling window center?
   - All-time high/low?
   - Volume-weighted centroid?

2. **Time scaling factor α**: How to normalize time vs price?
   - Fixed (e.g., 1% price = 1 hour)?
   - Adaptive based on volatility?
   - Learned from data?

3. **Multi-scale polar analysis**:
   - Short-term (τ=10) for entry timing
   - Long-term (τ=1000) for regime detection
   - How to combine?

### 7.2 Connection to Existing Work

| Field | Concept | Trading Application |
|-------|---------|---------------------|
| **Signal Processing** | Analytic signal, Hilbert transform | Instantaneous phase/frequency |
| **Chaos Theory** | Strange attractors, Lyapunov exponents | Predictability horizon |
| **Topological Data Analysis** | Persistent homology | Cycle detection robustness |
| **Quantum Mechanics** | Wave function phase | Market "coherence" measures |

---

## 8. Recommended Next Steps

1. **Prototype polar feature extraction** (1-2 days)
   - Add `polar_features.py` module
   - Test on historical BTC/ETH data
   - Visualize (r, θ) trajectories for different regimes

2. **Backtest polar signals** (3-5 days)
   - Simple strategies: buy when θ crosses 0, sell at π
   - Compare against scalar baselines
   - Measure Sharpe, max DD, turnover

3. **Integrate with FelSemigroupSignal** (2-3 days)
   - Use polar features as input to semigroup detection
   - Hypothesis: polar embedding improves genus estimation

4. **AI agent experiments** (1-2 weeks)
   - Fine-tune open-source trading LLM with polar features
   - A/B test: scalar vs vector input
   - Measure decision quality, not just PnL

---

## 9. Key Takeaways

1. **Vectors > Scalars** for capturing market geometry
2. **Polar coordinates** naturally separate amplitude (r) from phase (θ)
3. **Differentiation in polar** preserves structure (unlike scalar derivatives)
4. **AI agents** can reason more effectively with geometric features
5. **Implementation is tractable** - start with simple polar embed, iterate

---

## References

1. **TradingAgents** - https://github.com/TauricResearch/TradingAgents
2. **Hilbert Transform in Trading** - Ehlers, J.F. "Rocket Science for Traders"
3. **Phase Space Reconstruction** - Kantz & Schreiber, "Nonlinear Time Series Analysis"
4. **Complex Log Returns** - Various quant finance papers on Fourier methods
5. **Frenet-Serret in Finance** - Limited literature, emerging area

---

*Document created: 2026-03-29*
*Author: HFT Research Team*
