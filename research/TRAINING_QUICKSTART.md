# AI Training Quickstart for Math Signals

Train a transformer model to predict mathematical trading signals ~100× faster than direct computation.

---

## Overview

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1. Generate data | `python3 generate_training_data.py` | 5-30 min | `training_data.jsonl` |
| 2. Train model | `python3 train_model.py` | 1-6 hours | `math_signal_model.pt` |
| 3. Run inference | `python3 use_model.py` | ~1ms/sample | Signal predictions |

---

## Step 1: Generate Training Data

```bash
cd /home/nikolas/Documents/hft

# Download data if you haven't already
python3 download_data.py --symbol BTCUSDT --days 7

# Generate training labels (this computes all math signals)
python3 research/generate_training_data.py
```

**Expected output:**
```
Processing BTCUSDT-aggTrades-2024-01.zip...
  500,000 trades, generating windows...
    Generated 1000 samples...
    Generated 5000 samples...
    ...
Done! Generated 45,230 samples
File size: 156.3 MB
```

---

## Step 2: Train the Model

### On CMP 50HX 10GB (CPU fallback, slow)

```bash
# Install PyTorch (CPU only for CMP 50HX if CUDA issues)
pip3 install torch --break-system-packages

# Train (will take ~4-6 hours on CPU)
python3 research/train_model.py --model_size small --epochs 50
```

### On 2×RTX 3090 (GPU-accelerated, fast)

```bash
# Install PyTorch with CUDA
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Train (~1-2 hours with GPU)
python3 research/train_model.py --model_size medium --epochs 50 --batch_size 64
```

### On Cloud (fastest, ~$0.50-2)

```bash
# Rent GPU on RunPod/Vast.ai/Lambda
# Upload repo, then:
python3 research/train_model.py --model_size large --epochs 100 --batch_size 128
```

**Expected training output:**
```
=== Training Math Signal Transformer ===
Device: cuda
Train samples: 40,707
Val samples: 4,523
Model size: small
Total parameters: 524,288

Epoch 1/50 (45.2s)
  Train loss: 1.2345
  Val accuracy: 0.623
    Poincaré: 0.645
    Whitehead: 0.612
    Zeta: 0.612
  → Saved best model

...

Epoch 50/50
  Val accuracy: 0.891
  → Saved best model (val_acc=0.891)

Training complete!
Best validation accuracy: 0.891
```

---

## Step 3: Use the Trained Model

```bash
python3 research/use_model.py
```

**Expected output:**
```
=== Fast Math Signal Predictor ===

Loading model on cuda...
  Loaded weights from research/math_signal_model.pt
  Validation accuracy: 0.891

=== Test 1: Synthetic mean-reverting series ===
  Regime: mean-reversion
  Score: +0.423
  Time: 1.23ms

=== Test 2: Synthetic trending series ===
  Regime: trending
  Score: -0.312
  Time: 1.18ms

=== Model vs. Direct Computation Comparison ===

Model Prediction (fast):
  Poincaré: mean-reversion (score=+0.423)
  Whitehead: same_regime (torsion=0.15)
  Hecke L(1/2): 35.33 + 0.00j
  Zeta: strong
  Inference time: 1.23ms

Direct Computation (slow, exact):
  Poincaré: mean-reversion (score=+0.441) [89.2ms]
  Whitehead: same_regime (torsion=0.18) [45.3ms]
  Hecke L(1/2): 35.33 + 0.00j [23.1ms]
  Zeta: strong
  Total time: 157.6ms

  → Speedup: 128.1× faster with model

Accuracy:
  Poincaré regime match: True
  Score diff: 0.018
  Whitehead signal match: True
  Torsion diff: 0.030
```

---

## Model Sizes

| Size | Parameters | VRAM Needed | Train Time (2×3090) | Expected Accuracy |
|------|------------|-------------|---------------------|-------------------|
| tiny | ~100k | 2 GB | 30 min | 75-80% |
| small | ~500k | 4 GB | 1-2 hours | 85-90% |
| medium | ~2M | 8 GB | 3-4 hours | 88-92% |
| large | ~8M | 16 GB | 6-8 hours | 90-94% |

**Recommendation:** Start with `small`, upgrade if you need more accuracy.

---

## Integration with Trading

```python
# In your main.py or algorithm
from research.use_model import FastMathSignalPredictor

# Initialize once
predictor = FastMathSignalPredictor(
    model_path=Path("research/math_signal_model.pt"),
    model_size="small",
)

# On each price update (takes ~1ms)
result = predictor.predict(recent_prices)

if result["poincare"]["regime"] == "mean-reversion" and result["poincare"]["score"] > 0.5:
    # Use mean-reversion strategy
    config = ShotConfig(
        distance_pct=0.05,  # Tight entries
        buffer_pct=0.03,
        tp_pct=0.04,        # Quick profits
    )
elif result["poincare"]["regime"] == "trending":
    # Use trending strategy
    config = ShotConfig(
        distance_pct=0.12,  # Wider entries
        buffer_pct=0.06,
        tp_pct=0.15,        # Let profits run
    )
```

---

## Fine-Tuning Pre-Trained LLMs

If you want an LLM to **reason about** these signals (not just predict):

### Option A: LoRA Fine-Tuning (Low-Rank Adaptation)

```bash
# Install
pip3 install peft transformers accelerate bitsandbytes --break-system-packages

# Fine-tune Mistral-7B or Llama-3-8B
python3 research/finetune_lora.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.3 \
  --data_path research/training_data.jsonl \
  --output_dir research/finetuned_math_llm
```

### Option B: RAG (Retrieval-Augmented Generation)

```python
# research/rag_for_signals.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Store signal computations + interpretations
# Query: "What does Poincaré score > 0.5 indicate?"
# Retrieve: Historical examples + regime outcomes
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--model_size tiny`
- Enable gradient checkpointing (edit `train_model.py`)

### "Training is too slow on CPU"
- Rent GPU on RunPod (~$0.40/hr for RTX 3090)
- Or accept 4-6 hours on CPU for one-time training

### "Model accuracy is low (<70%)"
- Generate more training data (more days of price data)
- Increase training epochs
- Try larger model size

### "ImportError: No module named 'torch'"
```bash
pip3 install torch --break-system-packages
# Or with CUDA:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --break-system-packages
```

---

## Next Steps

1. **Generate data**: `python3 research/generate_training_data.py`
2. **Train model**: `python3 research/train_model.py --model_size small`
3. **Test inference**: `python3 research/use_model.py`
4. **Integrate**: Add `FastMathSignalPredictor` to your trading loop
5. **(Optional) Fine-tune LLM**: For natural language reasoning about signals

---

## Files Created

```
research/
├── TRAINING_QUICKSTART.md       # This file
├── train_ai_for_math_signals.md # Full documentation
├── generate_training_data.py    # Step 1: Generate labels
├── math_signal_model.py         # Step 2a: Model architecture
├── train_model.py               # Step 2b: Training loop
├── use_model.py                 # Step 3: Inference
├── training_data.jsonl          # Generated dataset (after Step 1)
└── math_signal_model.pt         # Trained model (after Step 2)
```
