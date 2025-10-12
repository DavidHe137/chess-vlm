from dataclasses import dataclass
from typing import Optional, List
import chessboard_image as cbi
from enum import Enum
import chess
import PIL.Image
from .prompts import template_registry

class Player(Enum):
    WHITE = "w"
    BLACK = "b"

"""
example of a puzzle row:
'PuzzleId': '00008',
 'FEN': 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24',
 'Moves': 'f2g3 e6e7 b2b1 b3c1 b1c1 h6c1',
 'Rating': 1807,
 'RatingDeviation': 75,
 'Popularity': 95,
 'NbPlays': 8585,
 'Themes': ['crushing', 'hangingPiece', 'long', 'middlegame'],
 'OpeningTags': None,
 'PuzzleUrl': 'https://lichess.org/training/00008',
 'GameUrl': 'https://lichess.org/787zsVup/black#48',
 'Puzzle_Moves_SAN': 'Bxg3 Rxe7 Qb1+ Nc1 Qxc1+ Qxc1',
 'GameId': '787zsVup',
 'Full_Moves_SAN': 'e4 c6 f4 d5 ex
 """
@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    moves: str
    to_move: Player
    move_number: int
    rating: int
    rating_deviation: int
    popularity: int
    nb_plays: int
    themes: List[str]
    opening_tags: Optional[str]
    puzzle_url: str
    game_url: str
    puzzle_moves_san: str
    game_id: str
    full_moves_san: str
    
    @classmethod
    def from_dataset_row(cls, row: dict):
        """Convert dataset row to Puzzle"""
        to_move = Player(row["FEN"].split()[-5])
        move_number = int(row["FEN"].split()[-1])

        return cls(
            puzzle_id=row["PuzzleId"],
            fen=row["FEN"],
            to_move=to_move,
            move_number=move_number,
            moves=row["Moves"],
            rating=row["Rating"],
            rating_deviation=row["RatingDeviation"],
            popularity=row["Popularity"],
            nb_plays=row["NbPlays"],
            themes=row["Themes"],
            opening_tags=row["OpeningTags"],
            puzzle_url=row["PuzzleUrl"],
            game_url=row["GameUrl"],
            puzzle_moves_san=row["Puzzle_Moves_SAN"],
            game_id=row["GameId"],
            full_moves_san=row["Full_Moves_SAN"]
        )
    
    def get_board_png(self, size: int = 400) -> PIL.Image.Image:
        """Get board PNG"""
        return cbi.generate_pil(self.fen, size=size)
    
    def get_board_ascii(self) -> str:
        """Get ASCII representation of the board"""
        board = chess.Board(self.fen)
        return str(board)
    
    def get_move_history_pgn(self) -> str:
        """Get move history in PGN format"""
        moves_played = self._get_moves_played()
        if not moves_played:
            return "Game start"
        
        # Format moves in PGN style
        pgn_parts = []
        move_number = 1
        
        for i, move in enumerate(moves_played):
            if i % 2 == 0:  # White's move
                pgn_parts.append(f"{move_number}. {move}")
            else:  # Black's move
                pgn_parts.append(f"{move}")
                move_number += 1
        
        return " ".join(pgn_parts)
    
    def get_side_to_move(self) -> str:
        """Get the side to move as a readable string"""
        return "White" if self.to_move == Player.WHITE else "Black"
    
    def get_san_moves(self) -> List[str]:
        """Get puzzle moves in SAN format"""
        return self.puzzle_moves_san.split()
    
    def _get_moves_played(self) -> list[str]:
        """Get the number of moves played"""
        num_moves = (self.move_number - 1) * 2 + (1 if self.to_move == Player.BLACK else 0)
        return self.full_moves_san.split()[:num_moves]
    
    def generate_prompt(self, 
                       board_representation: str = "fen",
                       instruction_style: str = "minimal",
                       theme: Optional[str] = None,
                       include_system: bool = True,
                       include_context: bool = False,
                       include_output_format: bool = True,
                       last_move: Optional[str] = None) -> str:
        """Generate a flexible prompt using templates
        
        Args:
            board_representation: "fen", "ascii", or "pgn"
            instruction_style: "minimal", "cot", or "structured"
            theme: Optional theme name for theme-specific prompts
            include_system: Whether to include system prompt
            include_context: Whether to include context about last move
            include_output_format: Whether to include output format instructions
            last_move: Last move played (for context)
        
        Returns:
            Complete formatted prompt string
        """
        # Prepare variables for template formatting
        variables = {
            "fen": self.fen,
            "ascii_board": self.get_board_ascii(),
            "move_history": self.get_move_history_pgn(),
            "side_to_move": self.get_side_to_move(),
        }
        
        if last_move:
            variables["last_move"] = last_move
        
        return template_registry.compose_prompt(
            board_representation=board_representation,
            instruction_style=instruction_style,
            theme=theme,
            include_system=include_system,
            include_context=include_context,
            include_output_format=include_output_format,
            **variables
        )
    
    def generate_numbered_moves_prompt(self) -> tuple[str, str]:
        """Legacy method for backward compatibility - generates PGN-style prompt
        
        Returns:
            tuple: (prompt, label) where prompt contains the game moves up to the puzzle position
                   and label is the first puzzle move
        """
        # get the moves played up to the puzzle position
        full_moves = self._get_moves_played()
        prompt_parts = []
        move_number = 1
        
        for i, move in enumerate(full_moves):
            if i % 2 == 0:  # White's move
                prompt_parts.append(f"{move_number}. {move}")
            else:  # Black's move
                prompt_parts.append(f"{move_number}... {move}")
                move_number += 1
        
        # Add the final move number for the puzzle position
        if len(full_moves) % 2 == 0:  # Next move is white's
            prompt_parts.append(f"{move_number}.")
        else:  # Next move is black's
            prompt_parts.append(f"{move_number}...")
        
        prompt = "Prompt: \"" + " ".join(prompt_parts) + "\""
        
        # The label is the first move of the puzzle solution
        first_puzzle_move = self.get_san_moves()[0]
        label = f"Label: \" {first_puzzle_move}\""
        
        return prompt, label
    
