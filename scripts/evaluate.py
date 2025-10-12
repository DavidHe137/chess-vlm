import os
import json
import click
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from src.puzzle import Puzzle
from datasets import load_from_disk
from src.puzzle_session import PuzzleSession, SessionStatus
import requests
from tqdm.auto import tqdm

supported_models = {
    "openrouter": [
        "anthropic/claude-sonnet-4.5",
        "meta-llama/llama-4-maverick:free",
    ],
    "vllm": [
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
}

def setup_client(client_type, hostname):
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
            base_url=f"http://{hostname}:8000/v1",
            api_key=""
        )

        try:
            r = requests.get(f"http://{hostname}:8000/health")
            print(f"vLLM is healthy. {r}")
        except Exception as e:
            print("vLLM healthcheck failed")
            raise e

    return client

@click.command()
@click.option('--board_format', type=click.Choice(["ascii", "fen", "pgn", "png"]), multiple=True,
              help="Board format(s) to use for evaluation. Can be specified multiple times.", required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--client_type', type=click.Choice(["openrouter", "vllm"]),
              help="Client type: openrouter or vllm", required=True)
@click.option('--hostname', type=str,
              help="Hostname to use for evaluation. Only used for vllm client type.")
def main(board_format, model_name, client_type, hostname):
    """Run evaluation on prepared puzzles."""
    
    print(f"Model: {model_name}")
    print(f"Board formats: {list(board_format) if board_format else 'None specified'}")
    print(f"Client type: {client_type}")

    # TODO: Add evaluation logic here
    if model_name not in supported_models[client_type]:
        raise click.ClickException(f"Model {model_name} not supported for client type {client_type}")
    
    #TODO: add script logging
    client = setup_client(client_type, hostname)

    # Setup output file
    output_file = f"results/{model_name}_{board_format}_results.jsonl"
    print(f"Logging results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    #TODO: first 5 examples for debugging
    dataset = load_from_disk("data/Lichess/chess-puzzles-3000-full-moves").select(range(100))
    puzzles = [Puzzle.from_dataset_row(puzzle) for puzzle in dataset]
    puzzle_sessions = [PuzzleSession(puzzle) for puzzle in puzzles]
    
    #TODO: make this async batched after finished debugging
    with open(output_file, "w") as f:
        for puzzle_session in tqdm(puzzle_sessions):
            prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
            while puzzle_session.status == SessionStatus.ACTIVE:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt_messages,
                    temperature=0.0
                )
                response_text = response.choices[0].message.content
                puzzle_session.add_assistant_response(response_text)
                
                move = puzzle_session.parse_move(response_text)
                puzzle_session.submit_move(move)

                if puzzle_session.status != SessionStatus.ACTIVE:
                    break

                # Add user response to continue conversation
                user_response = puzzle_session.get_turn_response()
                puzzle_session.add_user_message(user_response)
                prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
            
            # Log result immediately after each puzzle is completed
            result = puzzle_session.get_session_result()
            json.dump(result, f)
            f.write('\n')
            f.flush()  # Ensure data is written to disk immediately

if __name__ == "__main__":
    main()
