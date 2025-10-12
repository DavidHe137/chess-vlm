from dataclasses import dataclass
from typing import List, Optional
from src.puzzle import Puzzle, BoardFormat
import base64
import re

def encode_image(image_path_or_buffer):
    if isinstance(image_path_or_buffer, str):
        # File path
        with open(image_path_or_buffer, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Buffer (PIL Image or BytesIO)
        import io
        if hasattr(image_path_or_buffer, 'save'):
            # PIL Image
            buffer = io.BytesIO()
            image_path_or_buffer.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            # BytesIO buffer
            return base64.b64encode(image_path_or_buffer.getvalue()).decode('utf-8')

def coalesce_messages(messages: List[dict]) -> List[dict]:
    """Merge consecutive messages with same role and content type"""
    if not messages:
        return messages
    
    coalesced = [messages[0]]
    
    for msg in messages[1:]:
        last = coalesced[-1]
        
        # Same role and both string content?
        if (msg["role"] == last["role"] and 
            isinstance(msg["content"], str) and isinstance(last["content"], str)):
            # Merge content
            coalesced[-1]["content"] = f"{last['content']}\n\n{msg['content']}"
        else:
            # Different role or content type - keep separate
            coalesced.append(msg)
    
    return coalesced

@dataclass
class PromptTemplate:
    system_prompt: str
    user_template: str

class PromptFormatter:
    def __init__(self):
        self.templates = {
            "basic": PromptTemplate(
                system_prompt=BASIC_SYSTEM_PROMPT,
                user_template="{instruction}\n{board}\n{side_to_move} to move. What is the next best move?"
            ),
            "CoT": PromptTemplate(
                system_prompt=COT_SYSTEM_PROMPT,
                user_template="{instruction}\n{board}\n{side_to_move} to move. What is the next best move?"
            )
        }
    
    def format_messages(self, puzzle: Puzzle, board_formats: List[str], 
                       prompt_type: str = "basic", in_context_examples: Optional[List] = None) -> List[dict]:
        template = self.templates[prompt_type]
        moves_played = puzzle.get_moves_played()
        
        # Convert string formats to BoardFormat enums
        board_formats = [BoardFormat(fmt) for fmt in board_formats]
        
        instruction = f"This is a chess puzzle. Your opponent just played {moves_played[-1]}. The themes of this puzzle are {', '.join(puzzle.themes)}. Find the best move to solve the puzzle."
        side_to_move = puzzle.get_side_to_move()
        
        messages = [{"role": "system", "content": template.system_prompt}]
        
        if in_context_examples:
            for example in in_context_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add instruction message
        messages.append({"role": "user", "content": instruction})
        
        # Add each board format as separate message
        for fmt in board_formats:
            if fmt == BoardFormat.PNG:
                png_image = puzzle.get_board_png()
                img_base64 = encode_image(png_image)
                
                messages.append({
                    "role": "user", 
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]
                })
            else:
                board_repr = puzzle.get_board(fmt)
                messages.append({"role": "user", "content": f"{fmt.value.upper()}:\n{board_repr}"})
        
        # Add final question
        messages.append({"role": "user", "content": f"{side_to_move} to move. What is the next best move?"})
        
        messages = coalesce_messages(messages)
        return messages
    

    @staticmethod
    def parse_move(response: str) -> str:
        match = re.search(r'<move>(.*?)</move>', response)
        if not match:
            raise ValueError(f"Move not found in response: {response}")
        return match.group(1)

BASIC_SYSTEM_PROMPT = """
SYSTEM: You are an expert chess player that solves tactical puzzles. When analyzing positions:
- Use algebraic notation (e.g., e4, Nf3, O-O)
- Don't think at all. Just give the move.
- Always end your response with your move in this exact format:
  <move>e4</move>
"""

COT_SYSTEM_PROMPT = """
SYSTEM: You are an expert chess player that solves tactical puzzles. When analyzing positions:
- Think step-by-step about threats, tactics, and strategic ideas
- Explain your reasoning before giving the move
- Your response is limited to 1000 tokens, so be sure to give an answer in under 1000 tokens.
- Use algebraic notation (e.g., e4, Nf3, O-O)
- Always end your response with your move in this exact format:
  <move>e4</move>
"""

SYSTEM_PROMPTS = {
    "basic": BASIC_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
}