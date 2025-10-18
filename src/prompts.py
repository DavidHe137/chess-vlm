from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from src.puzzle import Puzzle, BoardFormat
import base64
import re
import yaml
import os

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
class PromptConfig:
    name: str
    description: str
    system_prompt: str
    instruction_template: str
    question_template: str
    model_params: Dict[str, Any]
    in_context_examples: List[Dict[str, str]]
    move_format: str
    move_tags: List[str]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PromptConfig':
        """Load prompt configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            name=data['name'],
            description=data['description'],
            system_prompt=data['system_prompt'],
            instruction_template=data['instruction_template'],
            question_template=data['question_template'],
            model_params=data.get('model_params', {}),
            in_context_examples=data.get('in_context_examples', []),
            move_format=data.get('move_format', 'algebraic'),
            move_tags=data.get('move_tags', ['<move>', '</move>'])
        )

class PromptFormatter:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.configs = {}
    
    def load_config(self, config_name: str) -> PromptConfig:
        """Load a prompt configuration from YAML file."""
        if config_name in self.configs:
            return self.configs[config_name]
        
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = PromptConfig.from_yaml(config_path)
        self.configs[config_name] = config
        return config
    
    def format_messages(self, puzzle: Puzzle, board_formats: List[str], 
                       config_name: str = "basic") -> List[dict]:
        """Format messages using YAML configuration."""
        config = self.load_config(config_name)
        moves_played = puzzle.get_moves_played()
        
        # Convert string formats to BoardFormat enums
        board_formats = [BoardFormat(fmt) for fmt in board_formats]
        
        side_to_move = puzzle.get_side_to_move()
        
        # Special handling for kagi format - uses FEN in instruction template
        if config_name == "kagi":
            fen = puzzle.get_board(BoardFormat.FEN)
            instruction = config.instruction_template.format(fen=fen)
            question = config.question_template.format(side_to_move=side_to_move.lower())
            
            messages = [{"role": "system", "content": config.system_prompt}]
            
            # Add in-context examples from config
            for example in config.in_context_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
            
            # For kagi, combine instruction and question into single message
            combined_message = f"{instruction}\n\n{question}"
            messages.append({"role": "user", "content": combined_message})
            
        else:
            # Standard format handling
            instruction = config.instruction_template.format(
                last_move=moves_played[-1],
                themes=', '.join(puzzle.themes)
            )
            
            question = config.question_template.format(side_to_move=side_to_move)
            
            messages = [{"role": "system", "content": config.system_prompt}]
            
            # Add in-context examples from config
            for example in config.in_context_examples:
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
            messages.append({"role": "user", "content": question})
        
        messages = coalesce_messages(messages)
        return messages

    @staticmethod
    def parse_move(response: str, move_tags: List[str] = None) -> str:
        if move_tags is None:
            move_tags = ['<move>', '</move>']
        
        start_tag, end_tag = move_tags
        # Escape special regex characters
        start_tag_escaped = re.escape(start_tag)
        end_tag_escaped = re.escape(end_tag)
        
        pattern = f'{start_tag_escaped}(.*?){end_tag_escaped}'
        match = re.search(pattern, response)
        if not match:
            raise ValueError(f"Move not found in response: {response}")
        return match.group(1)
