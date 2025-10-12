import logging
import os
from datetime import datetime
from typing import Optional

def setup_evaluation_logging(
    model_name: str,
    prompt_config: str, 
    board_format: tuple,
    log_level: str = "INFO"
) -> tuple[logging.Logger, str]:
    """
    Set up logging for evaluation runs.
    
    Returns:
        tuple: (logger, log_file_path)
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model name for filename (replace slashes with underscores)
    clean_model = model_name.replace("/", "_")
    board_format_str = "_".join(board_format)
    
    # Create log filename
    log_filename = f"{timestamp}_{clean_model}_{prompt_config}_{board_format_str}.log"
    log_path = os.path.join("logs", log_filename)
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"evaluation_{timestamp}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler (only for INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

def log_evaluation_start(
    logger: logging.Logger,
    model_name: str,
    prompt_config: str,
    board_format: tuple,
    client_type: str,
    batch_size: int,
    total_puzzles: int,
    command: str,
    git_commit: Optional[str] = None
):
    """Log evaluation start with configuration details."""
    logger.info("="*60)
    logger.info("EVALUATION STARTED")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt Config: {prompt_config}")
    logger.info(f"Board Formats: {list(board_format)}")
    logger.info(f"Client Type: {client_type}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Total Puzzles: {total_puzzles}")
    logger.info(f"Command: {command}")
    if git_commit:
        logger.info(f"Git Commit: {git_commit}")
    logger.info("="*60)

def log_evaluation_end(
    logger: logging.Logger,
    duration: str,
    summary: dict
):
    """Log evaluation completion with summary."""
    logger.info("="*60)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Duration: {duration}")
    logger.info(f"Total Puzzles: {summary['total_puzzles']}")
    logger.info(f"Accuracy: {summary['accuracy']:.2%}")
    logger.info(f"First Move Accuracy: {summary['first_move_accuracy']:.2%}")
    logger.info(f"Correct: {summary['correct']}")
    logger.info(f"Incorrect: {summary['incorrect']}")
    logger.info(f"Invalid: {summary['invalid']}")
    logger.info("="*60)
