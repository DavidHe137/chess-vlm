
# Chess VLM

A toy analysis of the visual reasoning priors of Vision Language Models (VLMs) using chess puzzles.

## Overview

This project evaluates how well Vision Language Models can solve chess puzzles by analyzing visual board representations. The analysis uses the Lichess chess puzzles dataset to test VLM capabilities in spatial reasoning, pattern recognition, and strategic thinking.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the analysis scripts:
   ```bash
   # Prepare the dataset
   uv run ./scripts/prepare_data.py
   
   # Run evaluation
   uv run ./scripts/run_eval.py
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

### Generating Board Images

Convert FEN positions to visual board representations:

```python
import chessboard_image as cbi

# Generate a PIL image of the chess board
board_image = cbi.generate_pil(dataset[0]["FEN"], size=400)
```

## Project Structure

```
chess-vlm/
├── scripts/
│   ├── prepare_data.py    # Data preprocessing
│   └── run_eval.py        # VLM evaluation
├── notebooks/
│   └── exploration.ipynb  # Analysis and visualization
├── results/
│   ├── puzzles.json       # Processed puzzle data
│   └── results.json       # Evaluation results
└── README.md
```

## Results

Evaluation results and processed data are stored in the `results/` directory for analysis and visualization.