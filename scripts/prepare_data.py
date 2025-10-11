#!/usr/bin/env python3
"""Prepare chess puzzle data for evaluation."""
import os
import json
import click
import chess
import chess.svg
from datasets import load_dataset
from tqdm import tqdm
import cairosvg

def fen_to_png(
    board: str,
    out_path: str = "board.png",
    size: int = 512,
    flipped: bool = False,
    lastmove: tuple[int,int] | None = None,  # e.g. (chess.E2, chess.E4)
    arrows: list[tuple[int,int]] | None = None,  # e.g. [(chess.E2, chess.E4)]
):
    svg = chess.svg.board(
        board=board,
        coordinates=True,
        flipped=flipped,
        lastmove=lastmove,
    )
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=out_path,
        output_width=size,
        output_height=size,
    )
    return out_path

important_themes = {
    'bodenMate' : {
        'prompt' : 'In this puzzle you will deliver checkmate by using two bishops',
        'puzzles' : list()
    },
    'hangingPiece' : {
        'prompt' : 'In this puzzle there is an unprotected piece. Identify it and capture it',
        'puzzles' : list()
    },
    'smotheredMate' : {
        'prompt' : 'In this puzzle you will deliver checkmate by using a Knight',
        'puzzles' : list()
    },
    'attackingF2F7' : {
        'prompt' : 'In this puzzle there is a weakness in an important square. Identify it and punish it',
        'puzzles' : list()
    },
    'backRankMate' : {
        'prompt' : 'In this puzzle you will checkmate by exploiting the open row to the king',
        'puzzles' : list()
    },
    'doubleBishopMate' : {
        'prompt' : 'In this puzzle you will deliver checkmate by using two bishops',
        'puzzles' : list()
    },
    'doubleCheck' : {
        'prompt' : 'In this puzzle you need to deliver a double check',
        'puzzles' : list()
    },
    'dovetailMate' : {
        'prompt' : 'In this puzzle you will deliver checkmate by using your queen',
        'puzzles' : list()
    },
    'anastasiaMate' : {
        'prompt' : 'In this puzzle you will use a piece with your knight to deliver checkmate',
        'puzzles' : list()
    },
    'arabianMate' : {
        'prompt' : 'In this puzzle you will use a piece with your knight to deliver checkmate',
        'puzzles' : list()
    },
    'discoveredAttack' : {
        'prompt' : 'In this puzzle there is the possibility to deliver a discovered attack',
        'puzzles' : list()
    },
    'killBoxMate' : {
        'prompt' : 'In this puzzle you can trap the king with queen and rook in a 3x3 square to deliver checkmate',
        'puzzles' : list()
    }
}

@click.command()
@click.option('--make_first_move', is_flag=True, default=False)
def main(make_first_move):
    dataset = load_dataset("Lichess/chess-puzzles", split="train")
    dataset = dataset.filter(lambda puzzle : 'oneMove' in puzzle['Themes'])
    print(dataset)
    themes = set()
    for x in tqdm(dataset):
        themes.update(x['Themes'])
        for theme in x['Themes']:
            if theme in important_themes:
                important_themes[theme]['puzzles'].append(x)

    for theme, theme_puzzles in important_themes.items():
        puzzles = []
        os.makedirs(f'pngs/{theme}/', exist_ok=True)
        for i, puzzle in enumerate(theme_puzzles['puzzles']):
            first_move, target_move = puzzle['Moves'].split()
            board = chess.Board(puzzle["FEN"])
            if make_first_move:
                board.push_uci(first_move)
            png_file_name = f'pngs/{theme}/puzzle_{i}.png'
            fen_to_png(board,png_file_name)
        
            # Create prompt
            starting_prompt = f"You are a chess expert. Your opponent just played {first_move}. {theme_puzzles['prompt']}. Find the best move.\n\nSide to move: {"White" if board.turn else "Black"}\n"
            end_prompt = " Give your answer in UCI format (e.g., e2e4, g1f3, e7e8q for promotion).\n\nBest move:"
            prompt_with_board = starting_prompt + f"\n Board:\n{board}\n" + end_prompt
            prompt_without_board = starting_prompt + end_prompt

            
            # Store puzzle data
            puzzles.append({
                "id": i,
                "lichess_id": puzzle["PuzzleId"],
                "prompt-with-board": prompt_with_board,
                "prompt-with-out-board": prompt_without_board,
                "solution": target_move,
                "rating": puzzle["Rating"],
                "png-file-name": png_file_name
            })
    
        # Save to JSON
        os.makedirs("results", exist_ok=True)
        file_name = f"results/puzzles_{theme}.json"
        with open(file_name, "w") as f:
            json.dump(puzzles, f, indent=2)
        
        print(f"Prepared {len(puzzles)} puzzles -> {file_name}")

if __name__ == "__main__":
    main()
