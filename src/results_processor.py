import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from puzzle results."""
    total_puzzles = len(results)
    correct = sum(1 for r in results if r['status'] == 'correct')
    incorrect = sum(1 for r in results if r['status'] == 'incorrect')
    invalid = sum(1 for r in results if r['status'] == 'invalid')
    
    first_move_correct = sum(1 for r in results if r.get('first_move_correct', False))
    
    accuracy = correct / total_puzzles if total_puzzles > 0 else 0.0
    first_move_accuracy = first_move_correct / total_puzzles if total_puzzles > 0 else 0.0
    
    return {
        "total_puzzles": total_puzzles,
        "correct": correct,
        "incorrect": incorrect,
        "invalid": invalid,
        "accuracy": accuracy,
        "first_move_accuracy": first_move_accuracy
    }

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def format_duration(start_time: datetime, end_time: datetime) -> str:
    """Format duration as HH:MM:SS."""
    duration = end_time - start_time
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_results_structure(
    results: List[Dict[str, Any]],
    model_name: str,
    prompt_config: str,
    board_formats: tuple,
    client_type: str,
    batch_size: int,
    hostname: Optional[str],
    command: str,
    start_time: datetime,
    end_time: datetime
) -> Dict[str, Any]:
    """Create the structured results format."""
    summary = calculate_summary(results)
    git_commit = get_git_commit()
    duration = format_duration(start_time, end_time)
    
    config = {
        "model_name": model_name,
        "prompt_config": prompt_config,
        "board_formats": list(board_formats),
        "client_type": client_type,
        "batch_size": batch_size,
        "timestamp": start_time.isoformat(),
        "evaluation_duration": duration,
        "command": command
    }
    
    if hostname:
        config["hostname"] = hostname
    
    if git_commit:
        config["git_commit"] = git_commit
    
    return {
        "summary": summary,
        "config": config,
        "results": results
    }

def save_results(
    results_data: Dict[str, Any],
    model_name: str,
    board_formats: tuple,
    prompt_config: str,
    timestamp: datetime
) -> str:
    """Save results to JSON file and return the file path."""
    # Clean model name for filename
    clean_model = model_name.replace("/", "_")
    board_format_str = "_".join(board_formats)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    filename = f"{clean_model}_{board_format_str}_{prompt_config}_{timestamp_str}.json"
    filepath = os.path.join("results", filename)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return filepath
