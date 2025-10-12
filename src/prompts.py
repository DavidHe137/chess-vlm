from src.puzzle import Puzzle, BoardFormat
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

EXAMPLE_PROMPT = """
{instruction}
{board}
{side_to_move} to move. What is the next best move?
"""

class Example:
    def __init__(self, Puzzle: Puzzle):
        self.puzzle = Puzzle
 
    def get_prompt_messages(self, board_format: BoardFormat) -> list[dict]:
        moves_played = self.puzzle.get_moves_played()

        #TODO: custom instructions for themes
        instruction = f"This is a chess puzzle. Your opponent just played {moves_played[-1]}. The themes of this puzzle are {', '.join(self.puzzle.themes)}. Find the best move to solve the puzzle."
        board = self.puzzle.get_board(board_format)
        side_to_move = self.puzzle.get_side_to_move()

        messages 
        base64_image = encode_image("chess-board.png")
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }

        return EXAMPLE_PROMPT.format(instruction=instruction, board=board, side_to_move=side_to_move)


class Prompt:
    def __init__(self, system_prompt: str, in_context_examples: list[Example], example: Example):
        self.system_prompt = system_prompt
        self.in_context_examples = in_context_examples
        self.example = example
    
    def format_for_openai(self):
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        for in_context_example in self.in_context_examples:
            messages.append({"role": "user", "content": in_context_example})
            messages.append({"role": "assistant", "content": in_context_example.solution})

        messages.append({"role": "user", "content": self.example})
        return messages

BASIC_SYSTEM_PROMPT = """
SYSTEM: You are an expert chess player that solves tactical puzzles. When analyzing positions:
- Use algebraic notation (e.g., e4, Nf3, O-O)
- Always end your response with your move in this exact format:
  <move>e4</move>
"""

COT_SYSTEM_PROMPT = """
SYSTEM: You are an expert chess player that solves tactical puzzles. When analyzing positions:
- Think step-by-step about threats, tactics, and strategic ideas
- Explain your reasoning before giving the move
- Use algebraic notation (e.g., e4, Nf3, O-O)
- Always end your response with your move in this exact format:
  <move>e4</move>
"""

SYSTEM_PROMPTS = {
    "basic": BASIC_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
}