import chess
from enum import Enum
import re
from .puzzle import Puzzle

class SessionStatus(Enum):
    ACTIVE = "active"
    CORRECT = "correct" # provided all correct moves
    INCORRECT = "incorrect" # provided incorrect move
    INVALID = "invalid" # provided invalid move

class PuzzleSession:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.status = SessionStatus.ACTIVE
        self.current_board = chess.Board(puzzle.fen)
        self.played_moves = []
        self.correct_moves = " ".split(puzzle.puzzle_moves_san) # TODO: should refactor away

    def submit_move(self, move: str):
        try:
            self.current_board.push_san(move)
        except (ValueError) as e:
            self.status = SessionStatus.INVALID
            self.failure_reason = str(e)
        
        self.played_moves.append(move)
        if move == self.correct_moves[len(self.played_moves)]:
            if len(self.played_moves) == len(self.correct_moves):
                self.status = SessionStatus.CORRECT
            else:
                assert self.status == SessionStatus.ACTIVE
        else:
            self.status = SessionStatus.INCORRECT
            self.failure_reason = f"Incorrect move: {move}"
    
    def parse_move(response):
        # NOTE: can be extended to support other move formats
        match = re.search(r'<move>(.*?)</move>', response)
        return match.group(1) if match else None
    
    def get_next_move(self):
        assert self.status == SessionStatus.ACTIVE, "Cannot get next move if session is not active"
        assert len(self.played_moves) > 0 and len(self.played_moves) % 2 == 0, "Cannot get next move if there are no played moves or if the number of played moves is not even"
        return self.correct_moves[len(self.played_moves)]

    def get_turn_response(self):
        move = self.get_next_move()
        return f"That's correct. Now, {move}"