# VLM Spatial Reasoning Research: Chess Puzzle Evaluation
## Research Proposal & Implementation Guide

---

## Executive Summary

**Core Question**: Do Vision-Language Models (VLMs) have better spatial reasoning than Language Models (LLMs)?

**Approach**: Use chess puzzles as a testbed because they require genuine spatial reasoning, have clear ground truth, and existing datasets.

**Key Insight**: If VLMs truly leverage visual spatial structure (not just pattern matching), they should outperform LLMs on chess puzzles - and this advantage should grow with training and be robust to visual variations.

---

## Research Questions

**RQ1**: Do VLM vision encoders provide spatial inductive biases that improve chess puzzle solving beyond what LLMs achieve with text-based board representations?

**RQ2**: Is any VLM advantage due to better board parsing (perception) or better spatial reasoning over the parsed representation?

---

## Hypotheses

**H1**: VLMs will outperform LLMs on chess puzzles when both use comparable board representations (image vs. optimal text format)

**H2**: The VLM advantage will increase with training because visual spatial structure is a stronger inductive bias than text-based patterns for learning chess tactics

---

## Dataset

We're using the Hugging Face Lichess chess puzzles dataset:

```python
from datasets import load_dataset
dset = load_dataset("Lichess/chess-puzzles", split="train")

# Example puzzle structure:
{
    'PuzzleId': '00008',
    'FEN': 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24',
    'Moves': 'f2g3 e6e7 b2b1 b3c1 b1c1 h6c1',  # Solution sequence
    'Rating': 1807,  # Puzzle difficulty
    'RatingDeviation': 75,
    'Popularity': 95,
    'NbPlays': 8585,
    'Themes': ['crushing', 'hangingPiece', 'long', 'middlegame'],
    'GameUrl': 'https://lichess.org/787zsVup/black#48'
}
```

**Sampling Strategy**:
- Use 1000 puzzles stratified by difficulty (Rating field)
- Difficulty bins: 1200-1500, 1500-1800, 1800-2100, 2100-2400
- 250 puzzles per bin
- Consider filtering by date or using Popularity field to check for data contamination

**Board Visualization**:
```python
import chessboard_image as cbi
board_image = cbi.generate_pil(dset[0]["FEN"], size=400)
```

---

## Experimental Design

### Experiment 1: Zero-Shot Evaluation (PRIMARY RESULT)

**Goal**: Establish baseline performance gap between VLMs and LLMs.

**Models to Test**:
- **VLMs**: GPT-4V, Claude Sonnet 4.5, Gemini 1.5 Pro, (optional: LLaVA, Qwen-VL for open source)
- **LLMs**: Same models in text-only mode (GPT-4, Claude Sonnet, Gemini Pro)

**Input Formats**:
- **VLMs**: Board images generated via `chessboard_image` (400x400px, standard colors)
- **LLMs**: Test multiple text representations and use the best:
  - FEN notation (directly from dataset)
  - ASCII board (from `python-chess` library)
  - Natural language description ("White: King on e1, Rook on a1...")
  - Structured list notation

**Prompt Variations** (ablation):
1. **Minimal**: "You are playing [White/Black]. Find the best move."
2. **Chain-of-Thought**: "Think step-by-step about the position. What are the key features? What is the best move?"
3. **Structured**: "First, describe what you see. Second, identify threats and tactics. Third, give your move."

**Metrics**:
- **Primary**: First move accuracy (does the model's first suggested move match the first move in the solution?)
- **Secondary**: Full sequence accuracy (for multi-move puzzles, does it solve completely?)
- Stratify results by difficulty rating
- Track by puzzle themes (fork, pin, skewer, etc.)

**Key Implementation Note**: 
- Keep prompts structurally identical for VLMs and LLMs (just swap image for text)
- Parse move from model response (handle algebraic notation, UCI notation, natural language)
- Need robust move parsing since models may output "Knight to f7", "Nf7", "knight f7", etc.

## Key Technical Considerations

### Move Parsing
Models will output moves in various formats:
- Standard algebraic: "Nf7", "Qxe4+"
- UCI notation: "e2e4", "g1f3"
- Natural language: "Knight to f7", "Take the pawn with the queen"

Need robust parsing that:
- Handles all common formats
- Validates moves are legal in the position
- Maps to the solution format in the dataset

Recommend using `python-chess` library for move validation.

### Evaluation Metrics Details

**First Move Accuracy**:
```
correct = (predicted_first_move == solution_first_move)
accuracy = correct_count / total_puzzles
```

**Full Sequence Accuracy** (for multi-move puzzles):
```
# Some puzzles require multiple moves to solve
# Check if model's full sequence matches solution
correct = (predicted_sequence == solution_moves)
```
---

## Open Questions to Resolve

1. **Contamination**: How do we verify models haven't seen these specific puzzles during training? Use popularity as proxy?

2. **Move format**: Should we standardize model outputs to UCI notation for easier comparison?

3. **Partial credit**: If model gets the right tactical idea but wrong square (e.g., "Qf7" vs "Qf8"), do we count it?

4. **Multi-move puzzles**: Do we evaluate move-by-move or full sequence? Some puzzles are 3-5 moves deep.

5. **Side to move**: Dataset has mixed positions (White to move / Black to move). Do we need to specify this in the prompt? Does the FEN notation make this clear?

6. **Temperature**: What sampling temperature should we use? 0 for deterministic? 0.3 for slight variation?