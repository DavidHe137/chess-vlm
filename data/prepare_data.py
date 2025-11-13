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

import requests
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
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

def main():
    pass

@main.command()
@click.option('--make_first_move', is_flag=True, default=True)
@click.option('--seed', type=int, default=42)
def create_dataset(make_first_move, seed):
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
    for elem in dataset:
        assert len(elem['Moves'].split()) == 1, f"Puzzle {elem['PuzzleId']} has {len(elem['Moves'].split())} moves"

    print(dataset[0])
    n = len(dataset)
    train_set = dataset.select(range(int(n * 0.8)))
    eval_set = dataset.select(range(int(n * 0.8), int(n * 0.9)))
    test_set = dataset.select(range(int(n * 0.9), n))
    train_set.save_to_disk("Lichess/one-move-chess-puzzles-train")
    eval_set.save_to_disk("Lichess/one-move-chess-puzzles-eval")
    test_set.save_to_disk("Lichess/one-move-chess-puzzles-test")
    print(f"Prepared {len(train_set)} puzzles -> Lichess/one-move-chess-puzzles-train")
    print(f"Prepared {len(eval_set)} puzzles -> Lichess/one-move-chess-puzzles-eval")
    print(f"Prepared {len(test_set)} puzzles -> Lichess/one-move-chess-puzzles-test")

if __name__ == "__main__":
    main()