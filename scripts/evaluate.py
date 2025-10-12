import os
import json
import click
from openai import OpenAI
from src.puzzle import Puzzle
from datasets import load_from_disk
from src.puzzle_session import PuzzleSession, SessionStatus
from tqdm.auto import tqdm

supported_models = {
    "openrouter": [
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.0-flash-exp:free",
    ],
    "vllm": [
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
}

def setup_client(client_type):
    if client_type == "openrouter":
        # OpenRouter configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise click.ClickException("OpenRouter API key required. Use --api_key or set OPENROUTER_API_KEY env var")
            
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    elif client_type == "vllm":
        client = OpenAI(
            base_url="http://localhost:8000/v1/completions",
        )
    return client

@click.command()
@click.option('--board_format', type=str, multiple=True, choices=["ascii", "fen", "pgn", "png"], 
              help="Board format(s) to use for evaluation. Can be specified multiple times.")
@click.option('--model_name', type=str)
@click.option('--client_type', type=str, choices=["openrouter", "vllm"], default="vllm",
              help="Client type: openrouter or vllm")
def main(board_format, model_name, client_type):
    """Run evaluation on prepared puzzles."""
    
    print(f"Model: {model_name}")
    print(f"Board formats: {list(board_format) if board_format else 'None specified'}")
    print(f"Client type: {client_type}")

    # TODO: Add evaluation logic here
    if model_name not in supported_models[client_type]:
        raise click.ClickException(f"Model {model_name} not supported for client type {client_type}")
    
    #TODO: add script logging
    client = setup_client(client_type)

    #TODO: first 5 examples for debugging
    dataset = load_from_disk("data/Lichess/chess-puzzles-3000-full-moves").select(range(5))
    puzzles = [Puzzle.from_dataset_row(puzzle) for puzzle in dataset]
    puzzle_sessions = [PuzzleSession(puzzle) for puzzle in puzzles]
    
    #TODO: make this async batched after finished debugging
    for puzzle_session in tqdm(puzzle_sessions):
        prompt_messages = puzzle_session.get_prompt_messages(board_format)
        while puzzle_session.status == SessionStatus.ACTIVE:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_messages
            )
            response_text = response.choices[0].message.content
            move = puzzle_session.parse_move(response_text)
            puzzle_session.submit_move(move)

            if puzzle_session.status != SessionStatus.ACTIVE:
                break

            #TODO: message dict formatting should go in one file
            prompt_messages.append({
                "role": "assistant",
                "content": response_text
            })

            prompt_messages.append({
                "role": "user",
                "content": puzzle_session.get_turn_response()
            })
    
    results = [puzzle_session.get_session_result() for puzzle_session in puzzle_sessions]
    os.makedirs(f"results", exist_ok=True)
    with open(f"results/{model_name}_{board_format}_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
