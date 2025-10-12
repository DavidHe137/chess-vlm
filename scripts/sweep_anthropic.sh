#!/bin/bash

# Comprehensive sweep script for prompt configurations and board formats
# Usage: ./scripts/sweep_anthropic.sh MODEL_NAME

MODEL_NAME=${1:-"claude-3-5-sonnet-20241022"}

echo "🚀 Starting comprehensive prompt config and board format sweep"
echo "📊 Model: $MODEL_NAME"
echo "🔧 Client: Anthropic"
echo ""

# Prompt configurations (from configs/ directory)
prompt_configs=(
    "basic"
    "cot"
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
        python scripts/evaluate_anthropic.py \
            --model_name "$MODEL_NAME" \
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
