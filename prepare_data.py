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

import requests
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import chess


def push_moves_to_board(example):
    """Push moves to the board"""
    board = chess.Board(example["FEN"])
    board.push_uci(example["Moves"].split()[0])
    example["FEN"] = board.fen()
    example["Moves"] = " ".join(example["Moves"].split()[1:])
    return example

def get_game_id(game_url):
    """
    extract the game id from the lichess url
    https://lichess.org/l6AejDMO#105
    https://lichess.org/787zsVup/black#48
    """
    return game_url.split("/")[3].split("#")[0]

def uci_to_san_from_fen(example):
    """Convert UCI moves to SAN notation from a given FEN position"""
    board = chess.Board(example["FEN"])
    san_moves = []
    
    for uci in example["Moves"].split():
        move = chess.Move.from_uci(uci)
        san_moves.append(board.san(move))
        board.push(move)
    
    return {"Puzzle_Moves_SAN": " ".join(san_moves)}

def get_game_pgn(game_ids) -> requests.Response:
    url = f"https://lichess.org/api/games/export/_ids"
    params = {"pgnInJson": "true"}
    headers = {"Accept": "application/x-ndjson"}
    response = requests.post(url, data=",".join(game_ids), params=params, headers=headers)
    return response

def get_unicode_board(example):
    """Get Unicode board from FEN"""
    board = chess.Board(example["FEN"])
    example["Unicode_Board"] = board.unicode()
    return example

def get_ascii_board(example):
    """Get ASCII board from FEN"""
    board = chess.Board(example["FEN"])
    example["ASCII_Board"] = str(board)
    return example

def get_board_state(example):
    """Get board state and move from FEN"""
    board = chess.Board(example["FEN"])
    example["Board_State"] = board.fen()
    example["Board_State_ASCII"] = str(board)
    return example

def get_all_valid_moves(example):
    """Get all valid moves from the board"""
    board = chess.Board(example["FEN"])
    example["All_Valid_Moves"] = [move.uci() for move in board.legal_moves]
    return example

def get_png_file_name(example, directory="Lichess/one-move-chess-puzzles/pngs"):
    """Generate PNG from board and save it with puzzle ID as filename"""
    board = chess.Board(example["FEN"])
    svg = chess.svg.board(board)
    # Ensure directory exists before writing
    os.makedirs(directory, exist_ok=True)
    puzzle_id = example["PuzzleId"]
    png_filename = os.path.join(directory, f"{puzzle_id}.png")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_filename)
    example["png_file_name"] = png_filename
    return example

@click.group()
def main():
    pass

@main.command()
@click.option('--make_first_move', is_flag=True, default=True)
@click.option('--seed', type=int, default=42)
@click.option('--directory', type=str, default="Lichess/one-move-chess-puzzles")
def create_dataset(make_first_move, seed, directory):
    dataset = load_dataset("Lichess/chess-puzzles", split="train")
    dataset = dataset.select(range(3000))
    dataset = dataset.filter(lambda puzzle : 'oneMove' in puzzle['Themes'])

    dataset = dataset.shuffle(seed=seed)
    if make_first_move:
        dataset = dataset.map(push_moves_to_board)
    dataset = dataset.map(uci_to_san_from_fen)
    dataset = dataset.map(get_unicode_board)
    dataset = dataset.map(get_ascii_board)
    dataset = dataset.map(get_board_state)
    dataset = dataset.map(get_all_valid_moves)
    png_directory = os.path.join(directory, "pngs")
    dataset = dataset.map(get_png_file_name, fn_kwargs={"directory": png_directory})
    for elem in dataset:
        assert len(elem['Moves'].split()) == 1, f"Puzzle {elem['PuzzleId']} has {len(elem['Moves'].split())} moves"

    # Pretty print json
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