import chess
from enum import Enum
import re
from .puzzle import Puzzle, BoardFormat
from typing import List
from .prompts import PromptFormatter

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
        self.correct_moves = puzzle.puzzle_moves_san.split()
        self.formatter = PromptFormatter()
        self.chat_history = []

    def submit_move(self, move: str):
        try:
            self.current_board.push_san(move)
        except (ValueError) as e:
            self.status = SessionStatus.INVALID
            self.failure_reason = str(e)
        
        self.played_moves.append(move)
        if move == self.correct_moves[len(self.played_moves) - 1]:
            if len(self.played_moves) == len(self.correct_moves):
                self.status = SessionStatus.CORRECT
            else:
                assert self.status == SessionStatus.ACTIVE
        else:
            self.status = SessionStatus.INCORRECT
            self.failure_reason = f"Incorrect move: {move}"
    
    def parse_move(self, response: str) -> str:
        return PromptFormatter.parse_move(response)
    
    def get_next_move(self):
        assert self.status == SessionStatus.ACTIVE, "Cannot get next move if session is not active"
        assert len(self.played_moves) > 0 and len(self.played_moves) % 2 == 0, "Cannot get next move if there are no played moves or if the number of played moves is not even"
        return self.correct_moves[len(self.played_moves)]

    def get_turn_response(self):
        move = self.get_next_move()
        return f"That's correct. {self.puzzle.get_side_to_respond()} plays {move}. What's your next move?"
    
    def get_prompt_messages(self, board_formats: List[str], prompt_type: str = "basic"):
        if not self.chat_history:  # First call - initialize with system + user prompt
            messages = self.formatter.format_messages(self.puzzle, board_formats, prompt_type)
            self.chat_history = messages.copy()
            return messages
        else:  # Return current chat history
            return self.chat_history.copy()
    
    def add_assistant_response(self, response: str):
        """Add assistant response to chat history"""
        self.chat_history.append({"role": "assistant", "content": response})
    
    def add_user_message(self, message: str):
        """Add user message to chat history"""
        self.chat_history.append({"role": "user", "content": message})
    
    def get_session_result(self):
        return {
            "puzzle_id": self.puzzle.puzzle_id,
            "status": self.status.value,
            "played_moves": self.played_moves,
            "correct_moves": self.correct_moves,
            "failure_reason": getattr(self, 'failure_reason', None),
            "chat_history": self.chat_history
        }