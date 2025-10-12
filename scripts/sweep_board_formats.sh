#!/bin/bash

# Simple sweep script for board format configurations
# Usage: ./scripts/sweep_board_formats.sh MODEL_NAME CLIENT_TYPE [HOSTNAME] [PROMPT_CONFIG]

MODEL_NAME=${1:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
CLIENT_TYPE=${2:-"vllm"}
HOSTNAME=${3:-""}
PROMPT_CONFIG=${4:-"basic"}

# Build hostname option if provided
HOSTNAME_OPT=""
if [ ! -z "$HOSTNAME" ]; then
    HOSTNAME_OPT="--hostname $HOSTNAME"
fi

echo "🚀 Starting board format sweep"
echo "📊 Model: $MODEL_NAME"
echo "🔧 Client: $CLIENT_TYPE"
echo "⚙️  Prompt Config: $PROMPT_CONFIG"
echo ""

# Board format configurations
configs=(
    "fen"
    "ascii"
    "pgn"
    "png"
    "png fen"
    "png ascii"
    "png pgn"
)

for config in "${configs[@]}"; do
    echo "🔄 Running: $config"
    
    # Build board format arguments
    board_args=""
    for format in $config; do
        board_args="$board_args --board_format $format"
    done
    
    # Run evaluation
    python scripts/evaluate.py \
        --model_name "$MODEL_NAME" \
        --client_type "$CLIENT_TYPE" \
        $HOSTNAME_OPT \
        --prompt_config "$PROMPT_CONFIG" \
        $board_args
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $config"
    else
        echo "❌ Failed: $config"
    fi
    echo ""
done

echo "🎉 Sweep completed!"
