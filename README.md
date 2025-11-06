
# Chess VLM

A toy analysis of the visual reasoning priors of Vision Language Models (VLMs) using chess puzzles.

## Overview

This project evaluates how well Vision Language Models can solve chess puzzles by analyzing visual board representations. The analysis uses the Lichess chess puzzles dataset to test VLM capabilities in spatial reasoning, pattern recognition, and strategic thinking.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
uv sync
```

**Note:** Always run scripts from the project root directory.

## Dataset

This project uses the [Lichess chess puzzles dataset](https://huggingface.co/datasets/Lichess/chess-puzzles) from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("Lichess/chess-puzzles", split="train")
```

### Puzzle Structure

Each puzzle contains the following fields:

```python
{
    'PuzzleId': '00008',                    # Unique puzzle identifier
    'FEN': 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24',  # Board position
    'Moves': 'f2g3 e6e7 b2b1 b3c1 b1c1 h6c1',  # Solution move sequence
    'Rating': 1807,                         # Puzzle difficulty rating
    'RatingDeviation': 75,                  # Rating uncertainty
    'Popularity': 95,                       # Community popularity score
    'NbPlays': 8585,                        # Number of times played
    'Themes': ['crushing', 'hangingPiece', 'long', 'middlegame'],  # Puzzle categories
    'GameUrl': 'https://lichess.org/787zsVup/black#48'  # Source game URL
}
```

## Evaluation

The evaluation process uses `scripts/evaluate.py` to test Vision Language Models on chess puzzles. For local model evaluation using vLLM, the process requires two steps:

### Step 1: Start vLLM Server (GPU Node)

Allocate a GPU node and start the vLLM server:

```bash
uv run vllm serve <model_name>
```

This starts a vLLM server on port 8000 that the evaluation script will connect to.

### Step 2: Run Evaluation (Another Node)

On a separate node, run the evaluation script:

```bash
uv run scripts/evaluate.py \
    --client_type vllm \
    --model_name <model_name> \
    --hostname <gpu_node_hostname> \
    --prompt_config basic \
    --board_format fen
```

### Evaluation Options

- `--client_type`: Choose `"openrouter"` or `"vllm"` (required)
- `--model_name`: Model identifier (required)
- `--hostname`: Hostname of the GPU node running vLLM (required for `vllm` client type)
- `--prompt_config`: Prompt configuration name (corresponds to a YAML file in `configs/`), e.g., `"basic"` or `"cot"` (default: `"basic"`)
- `--board_format`: Board representation format(s). Can specify multiple times. Options: `"ascii"`, `"fen"`, `"pgn"`, `"png"` (required)
- `--batch_size`: Number of concurrent puzzle evaluations (default: 100)

### Examples

**Single format (text only):**
```bash
# On GPU node
uv run vllm serve Qwen/Qwen2.5-VL-3B-Instruct

# On another node
uv run scripts/evaluate.py \
    --client_type vllm \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --hostname atl1-1-02-003-20-1 \
    --prompt_config basic \
    --board_format fen
```

**Combined formats (image + text):**
```bash
# On another node
uv run scripts/evaluate.py \
    --client_type vllm \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --hostname atl1-1-02-003-20-1 \
    --prompt_config basic \
    --board_format png \
    --board_format fen
```

The evaluation script processes puzzles asynchronously, logs progress, and saves results to a timestamped file with detailed metrics including accuracy, parse errors, and illegal moves.

### Board Formats

The `--board_format` option can be specified multiple times to include both image and text representations in the prompt. Each format is added as a separate message:

- **`png`**: Visual board representation as an image
- **`fen`**: Forsyth-Edwards Notation (e.g., `r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24`)
- **`ascii`**: ASCII art board representation
- **`pgn`**: Portable Game Notation (move history)

**Examples:**

```bash
# Single format (image only)
--board_format png

# Single format (text only)
--board_format fen

# Multiple formats (image + text)
--board_format png --board_format fen

# Multiple formats (image + multiple text formats)
--board_format png --board_format fen --board_format ascii
```

When multiple formats are specified, the model receives all of them in sequence, allowing it to use both visual and textual information to solve the puzzle.

## Prompt Configurations

Prompt configurations are defined in YAML files in the `configs/` directory. Each configuration file specifies:

- **System prompt**: Instructions for the model
- **Instruction template**: Template for puzzle instructions (supports `{last_move}` and `{themes}` placeholders)
- **Question template**: Template for the move question (supports `{side_to_move}` placeholder)
- **Model parameters**: Configuration for model inference (e.g., `max_completion_tokens`, `temperature`)
- **In-context examples**: Optional few-shot examples
- **Move parsing**: Format and tags for extracting moves from responses

### Available Configurations

- `basic`: Simple prompt without chain-of-thought reasoning
- `cot`: Chain-of-thought reasoning prompt
- `detailed_cot`: Extended chain-of-thought with more detailed instructions
- `few_shot`: Few-shot learning with in-context examples
- `kagi`: Custom configuration

### Creating Custom Configurations

To create a new prompt configuration, add a YAML file to the `configs/` directory with the following structure:

```yaml
name: "my_config"
description: "Description of the configuration"

system_prompt: |
  SYSTEM: You are an expert chess player...

instruction_template: |
  This is a chess puzzle. Your opponent just played {last_move}...

question_template: |
  {side_to_move} to move. What is the next best move?

model_params:
  max_completion_tokens: 1000
  temperature: 0.0

in_context_examples: []

move_format: "algebraic"
move_tags: ["<move>", "</move>"]
```

The configuration can then be used by specifying its name (without the `.yaml` extension) in the `--prompt_config` option.