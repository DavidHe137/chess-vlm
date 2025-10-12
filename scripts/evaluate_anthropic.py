import os
import json
import click
import asyncio
import sys
from datetime import datetime
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
load_dotenv()
from src.puzzle import Puzzle
from datasets import load_from_disk
from src.puzzle_session import PuzzleSession, SessionStatus, ParseError, IllegalMoveError
from src.logging_config import setup_evaluation_logging, log_evaluation_start, log_evaluation_end
from src.results_processor import create_results_structure, save_results
import requests
from tqdm.auto import tqdm

supported_models = [
    "claude-sonnet-4-5-20250929",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022", 
    "claude-3-opus-20240229",
]

def setup_client():
    """Setup Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.ClickException("Anthropic API key required. Set ANTHROPIC_API_KEY env var")
        
    return AsyncAnthropic(api_key=api_key)

def convert_messages_for_anthropic(messages):
    """Convert OpenAI-style messages to Anthropic format with prompt caching."""
    anthropic_messages = []
    system_message = None
    
    for msg in messages:
        if msg["role"] == "system":
            # Store system message separately for Anthropic
            system_message = msg["content"]
        elif msg["role"] in ["user", "assistant"]:
            # Handle image content for vision models
            if isinstance(msg["content"], list):
                content = []
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        # Convert OpenAI image format to Anthropic format
                        image_data = item["image_url"]["url"]
                        if image_data.startswith("data:image/png;base64,"):
                            base64_data = image_data.split(",")[1]
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data
                                }
                            })
                    else:
                        content.append(item)
                anthropic_messages.append({"role": msg["role"], "content": content})
            else:
                # Convert string content to proper format for consistency
                if isinstance(msg["content"], str):
                    anthropic_messages.append({
                        "role": msg["role"], 
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add cache_control to the first user message for prompt caching
    if anthropic_messages and anthropic_messages[0]["role"] == "user":
        if isinstance(anthropic_messages[0]["content"], str):
            # Convert string content to list format with cache_control
            anthropic_messages[0]["content"] = [
                {
                    "type": "text",
                    "text": anthropic_messages[0]["content"],
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        elif isinstance(anthropic_messages[0]["content"], list):
            # Add cache control to the last content item
            if anthropic_messages[0]["content"]:
                last_item = anthropic_messages[0]["content"][-1]
                if "type" not in last_item:
                    last_item["type"] = "text"
                last_item["cache_control"] = {"type": "ephemeral"}
    
    return system_message, anthropic_messages

async def process_puzzle_session(client, puzzle_session, model_name, board_format, logger):
    """Process a single puzzle session asynchronously."""
    puzzle_id = puzzle_session.puzzle.puzzle_id
    logger.debug(f"Starting puzzle {puzzle_id}")
    
    prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
    
    try:
        while puzzle_session.status == SessionStatus.ACTIVE:
            # Convert messages for Anthropic API
            system_message, anthropic_messages = convert_messages_for_anthropic(prompt_messages)
            
            # Use Anthropic's messages API
            response = await client.messages.create(
                model=model_name,
                system=system_message,
                messages=anthropic_messages,
                max_tokens=1000,
            )
            response_text = response.content[0].text
            puzzle_session.add_assistant_response(response_text)
            
            logger.debug(f"Puzzle {puzzle_id} - Model response: {response_text[:100]}...")
            
            move = puzzle_session.parse_move(response_text)
            puzzle_session.submit_move(move)
            
            logger.debug(f"Puzzle {puzzle_id} - Submitted move: {move}, Status: {puzzle_session.status.value}")

            if puzzle_session.status != SessionStatus.ACTIVE:
                break

            # Add user response to continue conversation
            user_response = puzzle_session.get_turn_response()
            puzzle_session.add_user_message(user_response)
            prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
                
    except ParseError as e:
        logger.debug(f"Puzzle {puzzle_id} - Parse error: {str(e)}")
        puzzle_session.status = SessionStatus.PARSE_ERROR
        puzzle_session.failure_reason = str(e)
        
    except IllegalMoveError as e:
        logger.debug(f"Puzzle {puzzle_id} - Illegal move: {str(e)}")
        puzzle_session.status = SessionStatus.ILLEGAL_MOVE
        puzzle_session.failure_reason = str(e)
        
    except Exception as e:
        logger.error(f"Unexpected system error processing puzzle {puzzle_id}: {e}")
        puzzle_session.status = SessionStatus.SYSTEM_ERROR
        puzzle_session.failure_reason = f"Unexpected processing error: {str(e)}"
    
    result = puzzle_session.get_session_result()
    logger.debug(f"Completed puzzle {puzzle_id} - Status: {result['status']}")
    return result

async def process_puzzle_with_semaphore(semaphore, client, puzzle_session, model_name, board_format, logger):
    """Process a single puzzle session with semaphore limiting concurrency."""
    async with semaphore:
        return await process_puzzle_session(client, puzzle_session, model_name, board_format, logger)

@click.command()
@click.option('--board_format', type=click.Choice(["ascii", "fen", "pgn", "png"]), multiple=True,
              help="Board format(s) to use for evaluation. Can be specified multiple times.", required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--batch_size', type=int, default=100,
              help="Batch size for concurrent processing. Default is 100.")
@click.option('--prompt_config', type=str, default="basic",
              help="Prompt configuration to use. Can be a legacy type (basic, CoT) or YAML config name.")
def main(board_format, model_name, batch_size, prompt_config):
    """Run evaluation on prepared puzzles."""
    asyncio.run(async_main(board_format, model_name, batch_size, prompt_config))

async def async_main(board_format, model_name, batch_size, prompt_config):
    """Async main function for evaluation."""
    start_time = datetime.now()
    
    # Reconstruct command for logging
    command_parts = [
        "python", "scripts/evaluate_anthropic.py",
        "--model_name", model_name,
        "--prompt_config", prompt_config,
        "--batch_size", str(batch_size)
    ]
    for fmt in board_format:
        command_parts.extend(["--board_format", fmt])
    command = " ".join(command_parts)
    
    # Setup logging
    logger, log_path = setup_evaluation_logging(model_name, prompt_config, board_format)
    
    if model_name not in supported_models:
        logger.error(f"Model {model_name} not supported")
        raise click.ClickException(f"Model {model_name} not supported. Supported models: {', '.join(supported_models)}")
    
    client = setup_client()
    
    # dataset = load_from_disk("data/Lichess/chess-puzzles-600-full-moves-balanced")
    dataset = load_from_disk("data/Lichess/chess-puzzles-3000-full-moves").select(range(100))
    puzzles = [Puzzle.from_dataset_row(puzzle) for puzzle in dataset]
    puzzle_sessions = [PuzzleSession(puzzle, prompt_config=prompt_config) for puzzle in puzzles]
    
    total_puzzles = len(puzzle_sessions)
    
    # Log evaluation start
    log_evaluation_start(
        logger, model_name, prompt_config, board_format, 
        "anthropic", batch_size, total_puzzles, command
    )
    
    # Create semaphore to limit concurrent puzzles
    semaphore = asyncio.Semaphore(batch_size)
    
    # Create all tasks
    tasks = [
        process_puzzle_with_semaphore(semaphore, client, puzzle_session, model_name, board_format, logger)
        for puzzle_session in puzzle_sessions
    ]
    
    # Process tasks as they complete
    results = []
    logger.info(f"Starting evaluation of {total_puzzles} puzzles...")
    
    with tqdm(total=total_puzzles, desc="Processing puzzles", disable=False) as pbar:
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                pbar.update(1)
                
                # Log progress every 10 puzzles
                if len(results) % 10 == 0:
                    logger.debug(f"Completed {len(results)}/{total_puzzles} puzzles")
                    
            except Exception as e:
                logger.error(f"Error processing puzzle: {e}")
                pbar.update(1)
    
    end_time = datetime.now()
    
    # Create structured results
    results_data = create_results_structure(
        results, model_name, prompt_config, board_format,
        "anthropic", batch_size, None, command, start_time, end_time
    )
    
    # Save results
    results_file = save_results(results_data, model_name, board_format, prompt_config, start_time)
    
    # Log completion
    log_evaluation_end(logger, results_data['config']['evaluation_duration'], results_data['summary'])
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Logs saved to: {log_path}")
    
    print(f"\n✅ Evaluation completed!")
    print(f"📊 Accuracy: {results_data['summary']['accuracy']:.2%}")
    print(f"🎯 First Move Accuracy: {results_data['summary']['first_move_accuracy']:.2%}")
    print(f"✅ Correct: {results_data['summary']['correct']}")
    print(f"❌ Incorrect: {results_data['summary']['incorrect']}")
    print(f"🔍 Parse Errors: {results_data['summary']['parse_error']}")
    print(f"⚠️  Illegal Moves: {results_data['summary']['illegal_move']}")
    if results_data['summary']['system_error'] > 0:
        print(f"💥 System Errors: {results_data['summary']['system_error']}")
    if results_data['summary']['legacy_invalid'] > 0:
        print(f"❓ Legacy Invalid: {results_data['summary']['legacy_invalid']}")
    print(f"📁 Results: {results_file}")
    print(f"📝 Logs: {log_path}")

if __name__ == "__main__":
    main()
