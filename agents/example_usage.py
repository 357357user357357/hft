#!/usr/bin/env python3
"""Example: Using TradingAgents with CMP 50HX 10GB (local GPU).

This example shows how to:
1. Run sentiment analysis locally via Ollama (Mistral-Nemo-12B)
2. Integrate with Poincaré/Whitehead/Hecke math signals
3. Make a full trading decision

Prerequisites:
1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Pull model: ollama pull mistral-nemo:12b
3. Start server: ollama serve
4. Run this example: python3 agents/example_usage.py (from repo root)
"""

import sys
import math

# This script runs from repo root, imports work directly
from agents import (
    LLMConfig,
    SentimentAgent,
    SignalIntegrator,
    create_llm_config_for_hardware,
)


def main():
    print("=== TradingAgents Example (CMP 50HX 10GB) ===\n")

    # 1. Configure for CMP 50HX (10GB VRAM, all weights fit)
    print("Configuring for CMP 50HX 10GB...")
    config = create_llm_config_for_hardware("cmp50hx")
    print(f"  Provider: {config.provider.value}")
    print(f"  Model: {config.ollama_model}")
    print(f"  (Make sure 'ollama serve' is running)")
    print()

    # 2. Create sentiment agent (works with Ollama)
    print("Initializing SentimentAgent...")
    try:
        sentiment_agent = SentimentAgent(config)

        # Test with sample social media posts
        sample_posts = [
            "BTC breaking out! $100k incoming #bitcoin",
            "Fed hints at rate pause, crypto markets rally",
            "Just sold all my crypto, too much risk imo",
            "Technical analysis shows strong support at $48k",
        ]

        print("Analyzing sample social media posts...")
        sentiment = sentiment_agent.analyze_text(sample_posts, symbol="BTCUSDT")
        print(f"  Sentiment score: {sentiment.sentiment_score:.2f}")
        print(f"  Magnitude: {sentiment.sentiment_magnitude:.2f}")
        print(f"  Recommendation: {sentiment.recommendation}")
        print(f"  Credibility: {sentiment.credibility_score:.2f}")
        print()

    except ImportError as e:
        print(f"  Skipping LLM agent (missing dependency): {e}")
        print("  Install with: pip install ollama")
        sentiment = None
        print()

    # 3. Test math signals (no LLM required, always works)
    print("Testing mathematical signals (Poincaré/Whitehead/Hecke)...")

    # Generate synthetic price data for testing
    base_price = 50000
    prices = [base_price * (1 + 0.01 * math.sin(i / 10)) for i in range(100)]

    integrator = SignalIntegrator(hecke_max_n=15, hecke_weight=2)

    print("  Running Poincaré topology analysis...")
    poincare = integrator.get_poincare_report()
    if poincare:
        print(f"    Regime: {poincare['regime']}")
        print(f"    Score: {poincare['poincare_score']:.3f}")
        print(f"    β₁ (loops): {poincare['beta1']}")

    print("  Running Whitehead torsion analysis...")
    whitehead = integrator.get_whitehead_report()
    if whitehead:
        print(f"    Signal: {whitehead['signal']}")
        print(f"    Torsion ratio: {whitehead['torsion_ratio']:.3f}")
        print(f"    Simply connected: {whitehead['poincare_check']}")

    print("  Computing Hecke L-function...")
    l_value = integrator.hecke_algebra.l_function_value(prices, s=0.5)
    zeta_sig = integrator.hecke_algebra.zeta_signal(prices)
    print(f"    L(1/2) = {l_value}")
    print(f"    Zeta signal: {zeta_sig}")
    print()

    # 4. Full integrated signal (math only, no LLM)
    print("Computing integrated signal (math + topology)...")
    combined = integrator.get_combined_signal(
        symbol="BTCUSDT",
        prices=prices,
        current_price=base_price,
        sentiment=None,  # No LLM
        news=None,
        fundamental=None,
        initial_equity=10000.0,
    )

    print(f"  Action: {combined.action}")
    print(f"  Confidence: {combined.confidence:.2f}")
    print(f"  Size: ${combined.size_usdt:.2f}")
    print(f"  Entry: ${combined.entry_price:.2f}")
    print(f"  Stop: ${combined.stop_loss:.2f}")
    print(f"  Target: ${combined.take_profit:.2f}")
    print(f"  Latency: {combined.latency_ms:.1f}ms")
    print()

    print("Signal sources:")
    for src, score in combined.signal_sources.items():
        print(f"  {src}: {score:+.3f}")
    print()

    print("=== Example Complete ===")
    print()
    print("Next steps:")
    print("1. Rent 2×RTX 3090 for DeepSeek-Math-V2-70B")
    print("   ./scripts/start_vllm_2x3090.sh")
    print()
    print("2. Or use cloud API (Claude, GPT-4)")
    print("   export ANTHROPIC_API_KEY=sk-...")
    print("   config = create_llm_config_for_hardware('cloud')")
    print()
    print("3. Integrate with live data:")
    print("   - Twitter API for sentiment")
    print("   - News API for headlines")
    print("   - Alpha Vantage for fundamentals")


if __name__ == "__main__":
    main()
