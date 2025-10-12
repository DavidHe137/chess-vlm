from dataclasses import dataclass
from typing import Optional, List
import chessboard_image as cbi
from enum import Enum
import chess
import PIL.Image

class Player(Enum):
    WHITE = "w"
    BLACK = "b"

class BoardFormat(Enum):
    FEN = "fen"
    ASCII = "ascii"
    PGN = "pgn"
    PNG = "png"

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

    def get_board_fen(self) -> str:
        """Get board FEN"""
        return self.fen
    
    def get_board_png(self, size: int = 400) -> PIL.Image.Image:
        """Get board PNG"""
        return cbi.generate_pil(self.fen, size=size)
    
    def get_board_ascii(self) -> str:
        """Get ASCII representation of the board"""
        board = chess.Board(self.fen)
        return str(board)
    
    def get_board_pgn(self) -> str:
        """Get move history in PGN format"""
        moves_played = self.get_moves_played()
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

    def get_board(self, board_format: BoardFormat) -> str:
        """Get board in the specified format"""
        if board_format == BoardFormat.FEN:
            return self.get_board_fen()
        elif board_format == BoardFormat.ASCII:
            return self.get_board_ascii()
        elif board_format == BoardFormat.PGN:
            return self.get_board_pgn()
        elif board_format == BoardFormat.PNG:
            return self.get_board_png()
    
    # TODO: clean this up, maybe use some attribute?
    def get_side_to_move(self) -> str:
        """Get the side to move as a readable string"""
        return "White" if self.to_move == Player.WHITE else "Black"

    def get_side_to_respond(self) -> str:
        """Get the side to respond as a readable string"""
        return "Black" if self.to_move == Player.WHITE else "White"
    
    def get_moves_played(self) -> list[str]:
        """Get the number of moves played"""
        num_moves = (self.move_number - 1) * 2 + (1 if self.to_move == Player.BLACK else 0)
        return self.full_moves_san.split()[:num_moves]