#!/usr/bin/env python3
"""
Mock vLLM-compatible HTTP server that always returns the correct chess move.
This server mimics the vLLM API format but extracts the correct move from the puzzle context
and returns it in the expected <move></move> format.
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import chess
from datasets import load_from_disk

# Load the puzzle dataset to get correct moves
print("Loading puzzle dataset...")
dataset = load_from_disk("data/Lichess/chess-puzzles-600-full-moves-balanced").sort("Rating").select(range(100))
puzzle_data = {}
for puzzle in dataset:
    puzzle_data[puzzle["PuzzleId"]] = {
        "fen": puzzle["FEN"],
        "moves": puzzle["Moves"].split(),
        "puzzle_moves_san": puzzle["Puzzle_Moves_SAN"].split(),
        "rating": puzzle["Rating"]
    }
print(f"Loaded {len(puzzle_data)} puzzles")

# API Models matching vLLM format
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_completion_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

app = FastAPI()

def extract_puzzle_id_from_messages(messages: List[Message]) -> Optional[str]:
    """Extract puzzle ID from the conversation messages by looking for FEN positions."""
    for message in messages:
        content = message.content
        
        # Look for FEN patterns in the content
        fen_pattern = r'([rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+)'
        fen_match = re.search(fen_pattern, content)
        
        if fen_match:
            fen = fen_match.group(1)
            # Find puzzle with matching FEN
            for puzzle_id, data in puzzle_data.items():
                if data["fen"] == fen:
                    return puzzle_id
    
    return None

def get_current_move_index(messages: List[Message], puzzle_id: str) -> int:
    """
    Determine which move in the sequence we should return based on conversation history.
    Returns the index of the next move to play (0-based).
    """
    if puzzle_id not in puzzle_data:
        return 0
    
    correct_moves = puzzle_data[puzzle_id]["puzzle_moves_san"]
    
    # Count how many moves have been played by looking at assistant responses
    moves_played = 0
    for message in messages:
        if message.role == "assistant":
            # Look for move tags in previous assistant responses
            move_match = re.search(r'<move>(.*?)</move>', message.content)
            if move_match:
                moves_played += 1
    
    # The next move index is the number of moves already played
    # But we need to account for the fact that the puzzle alternates between player and opponent
    # The player makes moves at indices 0, 2, 4, etc.
    return moves_played * 2

def get_correct_move(puzzle_id: str, move_index: int) -> str:
    """Get the correct move for the given puzzle and move index."""
    if puzzle_id not in puzzle_data:
        return "e4"  # Default fallback move
    
    correct_moves = puzzle_data[puzzle_id]["puzzle_moves_san"]
    
    if move_index >= len(correct_moves):
        return correct_moves[-1]  # Return last move if we've exceeded the sequence
    
    return correct_moves[move_index]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/v1/models")
async def list_models():
    """List available models (vLLM compatibility)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-chess-solver",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-server"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Mock chat completions endpoint that returns the correct chess move.
    """
    try:
        # Extract puzzle ID from messages
        puzzle_id = extract_puzzle_id_from_messages(request.messages)
        
        if not puzzle_id:
            # Fallback: return a generic move
            correct_move = "e4"
            print(f"Warning: Could not identify puzzle, using fallback move: {correct_move}")
        else:
            # Determine which move in the sequence to return
            move_index = get_current_move_index(request.messages, puzzle_id)
            correct_move = get_correct_move(puzzle_id, move_index)
            print(f"Puzzle {puzzle_id}, move index {move_index}: returning {correct_move}")
        
        # Format response with move tags as expected by the parser
        response_content = f"<move>{correct_move}</move>"
        
        # Create response in vLLM format
        response = ChatCompletionResponse(
            id=f"chatcmpl-mock-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=100,  # Mock values
                completion_tokens=10,
                total_tokens=110
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock vLLM server for chess puzzle testing")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()
    
    print("Starting mock vLLM server...")
    print("This server will return correct chess moves for puzzle evaluation testing.")
p    print(f"Server will be available at http://{args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Models endpoint: http://{args.host}:{args.port}/v1/models")
    
    uvicorn.run(app, host=args.host, port=args.port)
