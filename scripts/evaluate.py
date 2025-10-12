import os
import json
import click
import asyncio
from openai import AsyncOpenAI
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
        "Qwen/Qwen3-VL-30B-A3B-Instruct"
    ]
}

def setup_client(client_type, hostname):
    if client_type == "openrouter":
        # OpenRouter configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise click.ClickException("OpenRouter API key required. Use --api_key or set OPENROUTER_API_KEY env var")
            
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    elif client_type == "vllm":
        client = AsyncOpenAI(
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

async def process_puzzle_session(client, puzzle_session, model_name, board_format):
    """Process a single puzzle session asynchronously."""
    prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
    
    while puzzle_session.status == SessionStatus.ACTIVE:
        response = await client.chat.completions.create(
            model=model_name,
            messages=prompt_messages,
            max_completion_tokens=1000,
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
    
    return puzzle_session.get_session_result()

async def process_puzzle_with_semaphore(semaphore, client, puzzle_session, model_name, board_format):
    """Process a single puzzle session with semaphore limiting concurrency."""
    async with semaphore:
        return await process_puzzle_session(client, puzzle_session, model_name, board_format)

@click.command()
@click.option('--board_format', type=click.Choice(["ascii", "fen", "pgn", "png"]), multiple=True,
              help="Board format(s) to use for evaluation. Can be specified multiple times.", required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--client_type', type=click.Choice(["openrouter", "vllm"]),
              help="Client type: openrouter or vllm", required=True)
@click.option('--hostname', type=str,
              help="Hostname to use for evaluation. Only used for vllm client type.")
@click.option('--batch_size', type=int, default=10,
              help="Batch size for concurrent processing. Default is 10.")
@click.option('--prompt_config', type=str, default="basic",
              help="Prompt configuration to use. Can be a legacy type (basic, CoT) or YAML config name.")
def main(board_format, model_name, client_type, hostname, batch_size, prompt_config):
    """Run evaluation on prepared puzzles."""
    asyncio.run(async_main(board_format, model_name, client_type, hostname, batch_size, prompt_config))

async def async_main(board_format, model_name, client_type, hostname, batch_size, prompt_config):
    """Async main function for evaluation."""
    
    print(f"Model: {model_name}")
    print(f"Board formats: {list(board_format) if board_format else 'None specified'}")
    print(f"Client type: {client_type}")
    print(f"Batch size: {batch_size}")
    print(f"Prompt config: {prompt_config}")

    if model_name not in supported_models[client_type]:
        raise click.ClickException(f"Model {model_name} not supported for client type {client_type}")
    
    client = setup_client(client_type, hostname)

    # Setup output file
    output_file = f"results/{model_name}_{board_format}_{prompt_config}_results.jsonl"
    print(f"Logging results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Clear the output file at the start
    with open(output_file, "w") as f:
        pass

    dataset = load_from_disk("data/Lichess/chess-puzzles-3000-full-moves").select(range(100))
    puzzles = [Puzzle.from_dataset_row(puzzle) for puzzle in dataset]
    puzzle_sessions = [PuzzleSession(puzzle, prompt_config=prompt_config) for puzzle in puzzles]
    
    # Create semaphore to limit concurrent puzzles
    semaphore = asyncio.Semaphore(batch_size)
    
    # Create all tasks
    tasks = [
        process_puzzle_with_semaphore(semaphore, client, puzzle_session, model_name, board_format)
        for puzzle_session in puzzle_sessions
    ]
    
    # Process tasks as they complete
    total_puzzles = len(tasks)
    with tqdm(total=total_puzzles, desc="Processing puzzles") as pbar:
        with open(output_file, "a") as f:
            for task in asyncio.as_completed(tasks):
                result = await task
                json.dump(result, f)
                f.write('\n')
                f.flush()
                pbar.update(1)
    
    print(f"Evaluation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
