#!/bin/bash

# Comprehensive sweep script for prompt configurations and board formats
# Usage: ./scripts/sweep_prompt_configs.sh MODEL_NAME CLIENT_TYPE [HOSTNAME]

MODEL_NAME=${1:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
CLIENT_TYPE=${2:-"vllm"}
HOSTNAME=${3:-""}

# Build hostname option if provided
HOSTNAME_OPT=""
if [ ! -z "$HOSTNAME" ]; then
    HOSTNAME_OPT="--hostname $HOSTNAME"
fi

echo "🚀 Starting comprehensive prompt config and board format sweep"
echo "📊 Model: $MODEL_NAME"
echo "🔧 Client: $CLIENT_TYPE"
echo ""

# Prompt configurations (from configs/ directory)
prompt_configs=(
    "basic"
    "cot"
    "detailed_cot"
    "few_shot"
)

# Board format configurations
board_formats=(
    "fen"
    "ascii"
    "pgn"
    "png"
    "png fen"
    "png ascii"
    "png pgn"
)

# Nested loop: for each prompt config, test all board formats
for prompt_config in "${prompt_configs[@]}"; do
    echo "🎯 Starting prompt config: $prompt_config"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for board_format in "${board_formats[@]}"; do
        echo "🔄 Running: $prompt_config + $board_format"
        
        # Build board format arguments
        board_args=""
        for format in $board_format; do
            board_args="$board_args --board_format $format"
        done
        
        # Run evaluation
        python scripts/evaluate.py \
            --model_name "$MODEL_NAME" \
            --client_type "$CLIENT_TYPE" \
            $HOSTNAME_OPT \
            --prompt_config "$prompt_config" \
            $board_args
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $prompt_config + $board_format"
        else
            echo "❌ Failed: $prompt_config + $board_format"
        fi
        echo ""
    done
    
    echo "🎉 Finished all board formats for: $prompt_config"
    echo ""
done

echo "🎉 Sweep completed!"
