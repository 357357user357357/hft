#!/bin/bash
# Start Ollama server for CMP 50HX 10GB (all weights in VRAM)
# Uses Mistral-Nemo-12B (~8GB Q4) or similar small models

set -e

MODEL="mistral-nemo:12b"
PORT=11434
HOST="0.0.0.0"

echo "=== Starting Ollama for CMP 50HX 10GB ==="
echo "Model: $MODEL"
echo "Port: $PORT"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull model if not present
echo "Pulling model: $MODEL"
ollama pull $MODEL

# Start Ollama server
# Ollama automatically loads weights into VRAM
# For CMP 50HX 10GB, models up to ~12B Q4 fit entirely in VRAM

export OLLAMA_HOST="$HOST:$PORT"
export OLLAMA_ORIGINS="*"

echo "Starting Ollama serve..."
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
sleep 3

# Test connection
echo "Testing connection..."
ollama list

echo ""
echo "=== Ollama server running ==="
echo "API endpoint: http://localhost:$PORT"
echo "Model: $MODEL"
echo ""
echo "Test query:"
echo "  curl http://localhost:$PORT/api/chat -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "Press Ctrl+C to stop"

wait $OLLAMA_PID
