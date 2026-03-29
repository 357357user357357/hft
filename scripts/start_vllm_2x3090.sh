#!/bin/bash
# Start vLLM server for 2×RTX 3090 (48GB total VRAM)
# DeepSeek-Math-V2-70B-Instruct with tensor parallelism

set -e

MODEL="deepseek-ai/DeepSeek-Math-V2-70B-Instruct"
PORT=8000
HOST="0.0.0.0"
TP=2  # Tensor parallel across 2 GPUs

echo "=== Starting vLLM for 2×RTX 3090 ==="
echo "Model: $MODEL"
echo "Tensor Parallel: $TP"
echo "Port: $PORT"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Less than 2 GPUs detected. Tensor parallel may not work optimally."
fi

# Install vLLM if not present
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip3 install vllm --break-system-packages
fi

# Start vLLM server
# Key parameters for 70B model on 2×3090:
# - tensor-parallel-size=2: Split model across 2 GPUs
# - max-model-len=4096: Context window (reduce if OOM)
# - gpu-memory-utilization=0.95: Use 95% of VRAM
# - enforce-eager: Better for HFT low-latency

python3 -m vllm.entrypoints.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --dtype float16 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256

# Alternative: If you get OOM, try these adjustments:
# --max-model-len 2048          # Reduce context
# --gpu-memory-utilization 0.90 # Use less VRAM headroom
# --quantization fp8            # If supported (Hopper only)
