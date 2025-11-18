#!/usr/bin/env python3
"""Prepare chess puzzle data for evaluation."""
import os
import random
import json
import click
import chess
import chess.svg
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import cairosvg
from pprint import pprint
import requests
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import chess
from src.prompts import PromptFormatter
from src.puzzle import Puzzle


def push_moves_to_board(example, board=None):
    """Push moves to the board"""
    if board is None:
        board = chess.Board(example["FEN"])
    board.push_uci(example["Moves"].split()[0])
    example["FEN"] = board.fen()
    first_move = example["Moves"].split()[0]
    example["Moves"] = " ".join(example["Moves"].split()[1:])
    example["Full_Moves_SAN"] = first_move
    return example, board

def get_unicode_board(example, board=None):
    """Get Unicode board from FEN"""
    if board is None:
        board = chess.Board(example["FEN"])
    example["Unicode_Board"] = board.unicode()
    return example

def get_ascii_board(example, board=None):
    """Get ASCII board from FEN"""
    if board is None:
        board = chess.Board(example["FEN"])
    example["ASCII_Board"] = str(board)
    return example

def get_board_state(example, board=None):
    """Get board state and move from FEN"""
    if board is None:
        board = chess.Board(example["FEN"])
    example["Board_State"] = board.fen()
    example["Board_State_ASCII"] = str(board)
    return example

def get_all_valid_moves(example, board=None):
    """Get all valid moves from the board"""
    if board is None:
        board = chess.Board(example["FEN"])
    example["All_Valid_Moves"] = [move.uci() for move in board.legal_moves]
    return example

def get_png_file_name(example, directory="Lichess/one-move-chess-puzzles/pngs", board=None):
    """Generate PNG from board and save it with puzzle ID as filename"""
    if board is None:
        board = chess.Board(example["FEN"])
    svg = chess.svg.board(board)
    # Ensure directory exists before writing
    os.makedirs(directory, exist_ok=True)
    puzzle_id = example["PuzzleId"]
    png_filename = os.path.join(directory, f"{puzzle_id}.png")
    # cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_filename)
    example["png_file_name"] = png_filename
    return example

def transform_example(example, make_first_move, directory, prompt_formatter: PromptFormatter):
    """Transform example"""
    board = chess.Board(example["FEN"])
    if make_first_move:
        example, board = push_moves_to_board(example, board)
    example = get_png_file_name(example, directory=directory, board=board)
    example = get_unicode_board(example, board)
    example = get_ascii_board(example, board)
    example = get_board_state(example, board)
    example = get_all_valid_moves(example, board)
    example = format_lichess_puzzles(example, prompt_formatter)
    return example

def format_lichess_puzzles(example, prompt_formatter: PromptFormatter):
    """Format lichess puzzles"""
    # Adding unnecessary fields to the example
    # import ipdb; ipdb.set_trace()
    # example = example.__dict__
    example["PuzzleUrl"] = None
    example["GameUrl"] = None
    example["Puzzle_Moves_SAN"] = None
    example["GameId"] = None
    puzzle = Puzzle.from_dataset_row(example)
    messages = prompt_formatter.format_messages(puzzle, ["png"], 
        config_name="basic", 
        image_path=example["png_file_name"] if "png_file_name" in example else None)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"<move>{example['Moves']}</move>"}]})
    example["messages"] = messages
    return example

@click.group()
def main():
    pass

@main.command()
@click.option('--make_first_move', is_flag=True, default=True)
@click.option('--seed', type=int, default=42)
@click.option('--directory', type=str, default="Lichess/one-move-chess-puzzles")
@click.option('--num_proc', type=int, default=None, help="Number of processes for parallel processing. Defaults to number of CPU cores.")
def create_dataset(make_first_move, seed, directory, num_proc):
    if num_proc is None:
        num_proc = os.cpu_count() or 1
    prompt_formatter = PromptFormatter()
    dataset = load_dataset("Lichess/chess-puzzles", split="train")
    dataset = dataset.filter(
        lambda puzzle : 'oneMove' in puzzle['Themes'], 
        num_proc=num_proc,
        desc="Filtering one-move puzzles"
    )

    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.map(
        transform_example, 
        fn_kwargs={
            "make_first_move": make_first_move, 
            "directory": os.path.join(directory, "pngs"),
            "prompt_formatter": prompt_formatter
        }, 
        num_proc=num_proc,
        desc="Transforming puzzles"
    )
    dataset = dataset.filter(lambda x: len(x['Moves'].split()) == 1, num_proc=num_proc, desc="Filtering one-move puzzles")
    for elem in dataset:
        assert len(elem['Moves'].split()) == 1, f"Puzzle {elem['PuzzleId']} has {len(elem['Moves'].split())} moves {elem['Moves']}"

    print(json.dumps(dataset[0], indent=4) + "\n")
    n = len(dataset)

    train_set = dataset.select(range(int(n * 0.8)))
    eval_set = dataset.select(range(int(n * 0.8), int(n * 0.9)))
    test_set = dataset.select(range(int(n * 0.9), n))

    dataset_dict = DatasetDict({
        "train": train_set,
        "eval": eval_set,
        "test": test_set
    })

    dataset_dir = os.path.join(directory) 
    dataset_dict.save_to_disk(dataset_dir)
    print(f"Prepared {len(dataset)} puzzles -> {dataset_dir}")

if __name__ == "__main__":
    main()