#!/usr/bin/env python3
"""Prepare chess puzzle data for evaluation."""

import os
import json
import chess
from datasets import load_dataset

def main():
    # Load puzzles
    dataset = load_dataset("Lichess/chess-puzzles", split="train")
    
    # Take first 100 puzzles for simplicity
    puzzles = []
    for i, puzzle in enumerate(dataset.select(range(100))):
        board = chess.Board(puzzle["FEN"])
        
        # Create prompt
        prompt = f"""You are a chess expert. Find the best move.

Side to move: {"White" if board.turn else "Black"}

Board:
{board}

Give your answer in UCI format (e.g., e2e4, g1f3, e7e8q for promotion).

Best move:"""
        
        # Store puzzle data
        puzzles.append({
            "id": i,
            "lichess_id": puzzle["PuzzleId"],
            "prompt": prompt,
            "solution": puzzle["Moves"].split()[0],  # First move only
            "rating": puzzle["Rating"]
        })
    
    # Save to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/puzzles.json", "w") as f:
        json.dump(puzzles, f, indent=2)
    
    print(f"Prepared {len(puzzles)} puzzles -> results/puzzles.json")

if __name__ == "__main__":
    main()
