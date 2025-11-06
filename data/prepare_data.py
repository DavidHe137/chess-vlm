#!/usr/bin/env python3
"""Prepare chess puzzle data for evaluation."""
import os
import random
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

@click.group()
def main():
    pass

@main.command()
@click.option('--make_first_move', is_flag=True, default=False)
@click.option('--seed', type=int, default=42)
def create_themed_dataset(make_first_move, seed):
    random.seed(seed)
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
        train_set, eval_set, test_set = [], [], []
        for i, puzzle in tqdm(enumerate(theme_puzzles['puzzles']), total=len(theme_puzzles['puzzles'])):
            try:
                first_move, target_move = puzzle['Moves'].split()
            except:
                print(f"Error splitting moves for puzzle {puzzle['PuzzleId']}")
                print(puzzle['Moves'])
                continue
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
            puzzle_data = {
                "id": i,
                "lichess_id": puzzle["PuzzleId"],
                "prompt-with-board": prompt_with_board,
                "prompt-with-out-board": prompt_without_board,
                "solution": target_move,
                "rating": puzzle["Rating"],
                "png-file-name": png_file_name
            }
            u = random.random()
            if u < 0.8:
                train_set.append(puzzle_data)
            elif u < 0.9:
                eval_set.append(puzzle_data)
            else:
                test_set.append(puzzle_data)
    
        # Save to JSON
        os.makedirs(f"data/themed_prompts/{theme}", exist_ok=True)
        save_dataset(train_set, f"data/themed_prompts/{theme}/train.json")
        save_dataset(eval_set, f"data/themed_prompts/{theme}/eval.json")
        save_dataset(test_set, f"data/themed_prompts/{theme}/test.json")
        print(f"Prepared {len(train_set)} puzzles -> {f"data/themed_prompts/{theme}/train.json"}")
        print(f"Prepared {len(eval_set)} puzzles -> {f"data/themed_prompts/{theme}/eval.json"}")
        print(f"Prepared {len(test_set)} puzzles -> {f"data/themed_prompts/{theme}/test.json"}")

def save_dataset(dataset, file_name):
    with open(file_name, "w") as f:
        json.dump(dataset, f, indent=2)

@click.command()
@click.option('--folder_name', type=str, required=True, help='The folder in which the previous command created the themed datasets')
def create_full_dataset(folder_name):
    train_set, eval_set, test_set = [], [], []
    for file in os.listdir(folder_name):
        if file.endswith('.json'):
            with open(os.path.join(folder_name, file), 'r') as f:
                dataset = json.load(f)
            if file.endswith('train.json'):
                train_set.extend(dataset)
            elif file.endswith('eval.json'):
                eval_set.extend(dataset)
            elif file.endswith('test.json'):
                test_set.extend(dataset)
    save_dataset(train_set, os.path.join(folder_name, 'train.json'))
    save_dataset(eval_set, os.path.join(folder_name, 'eval.json'))
    save_dataset(test_set, os.path.join(folder_name, 'test.json'))
    print(f"Created train dataset with {len(train_set)} puzzles")
    print(f"Created eval dataset with {len(eval_set)} puzzles")
    print(f"Created test dataset with {len(test_set)} puzzles")
    
if __name__ == "__main__":
    main()