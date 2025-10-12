import os
import json
import click
import asyncio
import sys
import time
from datetime import datetime
from anthropic import Anthropic
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
        
    return Anthropic(api_key=api_key)

def convert_messages_for_anthropic(messages, add_cache_control=False):
    """Convert OpenAI-style messages to Anthropic format with optional prompt caching."""
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
    
    # Add cache_control to the first user message for prompt caching (only if requested)
    if add_cache_control and anthropic_messages and anthropic_messages[0]["role"] == "user":
        if isinstance(anthropic_messages[0]["content"], list) and anthropic_messages[0]["content"]:
            # Add cache control to the last content item
            last_item = anthropic_messages[0]["content"][-1]
            if "type" not in last_item:
                last_item["type"] = "text"
            last_item["cache_control"] = {"type": "ephemeral"}
    
    return system_message, anthropic_messages

def create_batch_request(puzzle_sessions, model_name, board_format, logger):
    """Create a batch request for initial puzzle responses."""
    batch_requests = []
    
    for puzzle_session in puzzle_sessions:
        puzzle_id = puzzle_session.puzzle.puzzle_id
        logger.debug(f"Creating batch request for puzzle {puzzle_id}")
        
        try:
            prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
            system_message, anthropic_messages = convert_messages_for_anthropic(prompt_messages, add_cache_control=True)
            
            batch_requests.append({
                "custom_id": puzzle_id,
                "params": {
                    "model": model_name,
                    "system": system_message,
                    "messages": anthropic_messages,
                    "max_tokens": 1000,
                }
            })
        except Exception as e:
            logger.error(f"Error creating batch request for puzzle {puzzle_id}: {e}")
            puzzle_session.status = SessionStatus.SYSTEM_ERROR
            puzzle_session.failure_reason = f"Batch request creation error: {str(e)}"
    
    return batch_requests

def wait_for_batch_completion(client, batch_id, logger, poll_interval=10):
    """Wait for batch to complete and return results."""
    logger.info(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        batch_status = client.messages.batches.retrieve(batch_id)
        logger.debug(f"Batch status: {batch_status.processing_status}")
        
        if batch_status.processing_status == "ended":
            logger.info(f"Batch {batch_id} completed successfully")
            return batch_status
        elif batch_status.processing_status in ["failed", "canceled", "expired"]:
            logger.error(f"Batch {batch_id} failed with status: {batch_status.processing_status}")
            raise Exception(f"Batch processing failed: {batch_status.processing_status}")
        
        time.sleep(poll_interval)

def process_batch_results(client, batch_status, puzzle_sessions_dict, logger):
    """Process batch results and update puzzle sessions."""
    results_url = batch_status.results_url
    if not results_url:
        raise Exception("No results URL available for completed batch")
    
    # Download and parse results with authentication
    import requests
    headers = {
        "x-api-key": client.api_key,
        "anthropic-version": "2023-06-01"
    }
    response = requests.get(results_url, headers=headers)
    response.raise_for_status()
    
    active_sessions = []  # Sessions that need more responses
    completed_results = []  # Completed puzzle results
    
    for line in response.text.strip().split('\n'):
        if not line:
            continue
            
        result = json.loads(line)
        custom_id = result["custom_id"]
        puzzle_session = puzzle_sessions_dict[custom_id]
        
        try:
            if "error" in result:
                logger.error(f"Batch error for puzzle {custom_id}: {result['error']}")
                puzzle_session.status = SessionStatus.SYSTEM_ERROR
                puzzle_session.failure_reason = f"Batch API error: {result['error']}"
                completed_results.append(puzzle_session.get_session_result())
                continue
            
            # Process successful response
            response_content = result["result"]["message"]["content"]
            if isinstance(response_content, list) and response_content:
                response_text = response_content[0]["text"]
            else:
                response_text = str(response_content)
            
            puzzle_session.add_assistant_response(response_text)
            logger.debug(f"Puzzle {custom_id} - Model response: {response_text[:100]}...")
            
            # Parse and submit move
            move = puzzle_session.parse_move(response_text)
            puzzle_session.submit_move(move)
            
            logger.debug(f"Puzzle {custom_id} - Submitted move: {move}, Status: {puzzle_session.status.value}")
            
            if puzzle_session.status == SessionStatus.ACTIVE:
                # Need more responses - add to active sessions
                user_response = puzzle_session.get_turn_response()
                puzzle_session.add_user_message(user_response)
                active_sessions.append(puzzle_session)
            else:
                # Puzzle completed
                completed_results.append(puzzle_session.get_session_result())
                
        except ParseError as e:
            logger.debug(f"Puzzle {custom_id} - Parse error: {str(e)}")
            puzzle_session.status = SessionStatus.PARSE_ERROR
            puzzle_session.failure_reason = str(e)
            completed_results.append(puzzle_session.get_session_result())
            
        except IllegalMoveError as e:
            logger.debug(f"Puzzle {custom_id} - Illegal move: {str(e)}")
            puzzle_session.status = SessionStatus.ILLEGAL_MOVE
            puzzle_session.failure_reason = str(e)
            completed_results.append(puzzle_session.get_session_result())
            
        except Exception as e:
            logger.error(f"Unexpected error processing puzzle {custom_id}: {e}")
            puzzle_session.status = SessionStatus.SYSTEM_ERROR
            puzzle_session.failure_reason = f"Unexpected processing error: {str(e)}"
            completed_results.append(puzzle_session.get_session_result())
    
    return active_sessions, completed_results

def process_remaining_sessions_individually(client, active_sessions, model_name, board_format, logger):
    """Process remaining active sessions individually (non-batch)."""
    completed_results = []
    
    for puzzle_session in tqdm(active_sessions, desc="Processing remaining puzzles"):
        puzzle_id = puzzle_session.puzzle.puzzle_id
        logger.debug(f"Processing remaining turns for puzzle {puzzle_id}")
        
        try:
            while puzzle_session.status == SessionStatus.ACTIVE:
                prompt_messages = puzzle_session.get_prompt_messages(list(board_format))
                system_message, anthropic_messages = convert_messages_for_anthropic(prompt_messages)
                
                # Use regular messages API for follow-up responses
                response = client.messages.create(
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
        completed_results.append(result)
    
    return completed_results


@click.command()
@click.option('--board_format', type=click.Choice(["ascii", "fen", "pgn", "png"]), multiple=True,
              help="Board format(s) to use for evaluation. Can be specified multiple times.", required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--batch_size', type=int, default=100,
              help="Batch size for Anthropic batch API processing. Default is 100.")
@click.option('--prompt_config', type=str, default="basic",
              help="Prompt configuration to use. Can be a legacy type (basic, CoT) or YAML config name.")
def main(board_format, model_name, batch_size, prompt_config):
    """Run evaluation on prepared puzzles."""
    run_main(board_format, model_name, batch_size, prompt_config)

def run_main(board_format, model_name, batch_size, prompt_config):
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
    
    # Process puzzles in batches
    results = []
    logger.info(f"Starting batch evaluation of {total_puzzles} puzzles...")
    
    # Create batch request
    logger.info("Creating batch request...")
    batch_requests = create_batch_request(puzzle_sessions, model_name, board_format, logger)
    
    if not batch_requests:
        logger.error("No valid batch requests created")
        return
    
    # Submit batch
    logger.info(f"Submitting batch with {len(batch_requests)} requests...")
    message_batch = client.messages.batches.create(requests=batch_requests)
    batch_id = message_batch.id
    logger.info(f"Batch submitted with ID: {batch_id}")
    
    # Wait for batch completion
    batch_status = wait_for_batch_completion(client, batch_id, logger)
    
    # Process batch results
    logger.info("Processing batch results...")
    puzzle_sessions_dict = {session.puzzle.puzzle_id: session for session in puzzle_sessions}
    active_sessions, batch_results = process_batch_results(client, batch_status, puzzle_sessions_dict, logger)
    
    results.extend(batch_results)
    logger.info(f"Batch processing completed: {len(batch_results)} puzzles finished, {len(active_sessions)} need more responses")
    
    # Process remaining active sessions individually
    if active_sessions:
        logger.info(f"Processing {len(active_sessions)} remaining puzzles individually...")
        remaining_results = process_remaining_sessions_individually(client, active_sessions, model_name, board_format, logger)
        results.extend(remaining_results)
    
    logger.info(f"All puzzles completed: {len(results)} total results")
    
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
