#!/bin/bash

# Simple SLURM script for vLLM server + evaluation
# Usage: sbatch simple_vllm_eval.sh <model_name>
#    or: srun -G 1 -C L40S --mem=128G simple_vllm_eval.sh Qwen/Qwen2.5-VL-3B-Instruct

#SBATCH --job-name=vllm-eval
#SBATCH --gpus=1
#SBATCH --constraint=L40S
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -e

# Get model name from argument
MODEL_NAME=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
PORT=8000

echo "Starting vLLM evaluation with model: $MODEL_NAME"

# 1. Get node name
NODE_NAME=$(hostname)
echo "Allocated node: $NODE_NAME"

# 2. Set up environment
source .venv/bin/activate

# 3. Start vLLM in background
echo "Starting vLLM server..."
uv run vllm serve "$MODEL_NAME" --host 0.0.0.0 --port $PORT &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
while ! curl -s http://$NODE_NAME:$PORT/health >/dev/null 2>&1; do
    sleep 5
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server failed to start"
        exit 1
    fi
done
echo "Server ready at http://$NODE_NAME:$PORT"

# TODO: make it easier to pass script arguments
# 4. Run evaluation script
echo "Running evaluation..."
python scripts/evaluate.py \
    --client_type vllm \
    --model_name "$MODEL_NAME" \
    --hostname "$NODE_NAME" \
    --board_format pgn

# uncomment to sanity-check if script above does not work
# curl -X POST http://$NODE_NAME:$PORT/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt": "What is the capital of France?", "max_tokens": 100, "temperature": 0.7}'

# Cleanup
echo "Cleaning up..."
kill $VLLM_PID 2>/dev/null || true

echo "Done!"
