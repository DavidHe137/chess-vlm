#!/usr/bin/env python3
"""
Script to aggregate all summary results from chess puzzle evaluations into a pandas DataFrame.
"""

import json
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, List, Any

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract board format, prompt config, and timestamp.
    Expected formats:
    - {board_format}_{prompt_config}_{timestamp}.json (e.g., ascii_basic_20251012_190131.json)
    - png_{board_format}_{prompt_config}_{timestamp}.json (e.g., png_pgn_cot_20251012_190546.json)
    - png_{prompt_config}_{timestamp}.json (e.g., png_basic_20251012_190146.json)
    
    Prompt configs can be: basic, cot, detailed_cot, few_shot
    Board formats can be: ascii, fen, pgn, png, png_ascii, png_fen, png_pgn
    """
    # Remove .json extension
    name = filename.replace('.json', '')
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) < 3:
        # Fallback for malformed filenames
        return {
            'board_format': parts[0] if len(parts) > 0 else "unknown",
            'prompt_config': parts[1] if len(parts) > 1 else "unknown",
            'timestamp': '_'.join(parts[2:]) if len(parts) > 2 else "unknown"
        }
    
    # Known prompt configs (including multi-word ones)
    known_prompt_configs = {'basic', 'cot', 'detailed_cot', 'few_shot'}
    # Known board formats that can follow png_
    known_board_formats = {'ascii', 'fen', 'pgn'}
    
    # Check if it starts with 'png'
    if parts[0] == 'png':
        if len(parts) >= 4 and parts[1] in known_board_formats:
            # Format: png_{board_format}_{prompt_config}_{timestamp}
            # Need to handle multi-word prompt configs
            board_format = f"png_{parts[1]}"
            
            # Check for multi-word prompt configs
            if len(parts) >= 5 and f"{parts[2]}_{parts[3]}" in known_prompt_configs:
                # detailed_cot or few_shot
                prompt_config = f"{parts[2]}_{parts[3]}"
                timestamp = '_'.join(parts[4:])
            else:
                # single word prompt config
                prompt_config = parts[2]
                timestamp = '_'.join(parts[3:])
        else:
            # Format: png_{prompt_config}_{timestamp} (png only)
            board_format = "png"
            
            # Check for multi-word prompt configs
            if len(parts) >= 4 and f"{parts[1]}_{parts[2]}" in known_prompt_configs:
                # detailed_cot or few_shot
                prompt_config = f"{parts[1]}_{parts[2]}"
                timestamp = '_'.join(parts[3:])
            else:
                # single word prompt config
                prompt_config = parts[1]
                timestamp = '_'.join(parts[2:])
    else:
        # Format: {board_format}_{prompt_config}_{timestamp}
        board_format = parts[0]
        
        # Check for multi-word prompt configs
        if len(parts) >= 4 and f"{parts[1]}_{parts[2]}" in known_prompt_configs:
            # detailed_cot or few_shot
            prompt_config = f"{parts[1]}_{parts[2]}"
            timestamp = '_'.join(parts[3:])
        else:
            # single word prompt config
            prompt_config = parts[1]
            timestamp = '_'.join(parts[2:])
    
    return {
        'board_format': board_format,
        'prompt_config': prompt_config,
        'timestamp': timestamp
    }

def extract_data_from_json(file_path: Path) -> Dict[str, Any]:
    """Extract relevant data from a single JSON result file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract summary metrics
        summary = data.get('summary', {})
        config = data.get('config', {})
        
        # Parse filename for additional metadata
        filename_info = parse_filename(file_path.name)
        
        # Combine all relevant information
        result = {
            # File metadata
            'file_path': str(file_path),
            'filename': file_path.name,
            'model_provider': file_path.parent.parent.name,  # e.g., 'Qwen', 'anthropic'
            'model_name': file_path.parent.name,  # e.g., 'Qwen3-VL-30B-A3B-Instruct'
            
            # Parsed from filename
            'board_format_parsed': filename_info['board_format'],
            'prompt_config_parsed': filename_info['prompt_config'],
            'timestamp_parsed': filename_info['timestamp'],
            
            # Summary metrics
            'total_puzzles': summary.get('total_puzzles', 0),
            'correct': summary.get('correct', 0),
            'incorrect': summary.get('incorrect', 0),
            'invalid': summary.get('invalid', 0),
            'parse_error': summary.get('parse_error', 0),
            'illegal_move': summary.get('illegal_move', 0),
            'system_error': summary.get('system_error', 0),
            'legacy_invalid': summary.get('legacy_invalid', 0),
            'accuracy': summary.get('accuracy', 0.0),
            'first_move_accuracy': summary.get('first_move_accuracy', 0.0),
            
            # Config information
            'model_name_config': config.get('model_name', ''),
            'prompt_config_config': config.get('prompt_config', ''),
            'board_formats_config': ', '.join(config.get('board_formats', [])),
            'client_type': config.get('client_type', ''),
            'batch_size': config.get('batch_size', 0),
            'timestamp_config': config.get('timestamp', ''),
            'evaluation_duration': config.get('evaluation_duration', ''),
            'hostname': config.get('hostname', ''),
            'git_commit': config.get('git_commit', ''),
        }
        
        # Calculate additional metrics
        if result['total_puzzles'] > 0:
            result['correct_rate'] = result['correct'] / result['total_puzzles']
            result['incorrect_rate'] = result['incorrect'] / result['total_puzzles']
            result['invalid_rate'] = result['invalid'] / result['total_puzzles']
            result['parse_error_rate'] = result['parse_error'] / result['total_puzzles']
            result['illegal_move_rate'] = result['illegal_move'] / result['total_puzzles']
        else:
            result['correct_rate'] = 0.0
            result['incorrect_rate'] = 0.0
            result['invalid_rate'] = 0.0
            result['parse_error_rate'] = 0.0
            result['illegal_move_rate'] = 0.0
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def aggregate_results(results_dir: str = "/home/hice1/dhe83/scratch/chess-vlm/results") -> pd.DataFrame:
    """
    Aggregate all JSON result files into a pandas DataFrame.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        pandas.DataFrame with aggregated results
    """
    results_path = Path(results_dir)
    all_data = []
    
    # Find all JSON files recursively
    json_files = list(results_path.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        data = extract_data_from_json(json_file)
        if data is not None:
            all_data.append(data)
    
    if not all_data:
        print("No valid data found!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by model provider, model name, board format, and prompt config
    df = df.sort_values([
        'model_provider', 
        'model_name', 
        'board_format_parsed', 
        'prompt_config_parsed'
    ]).reset_index(drop=True)
    
    print(f"Successfully aggregated {len(df)} results!")
    return df

def save_results(df: pd.DataFrame, output_file: str = "aggregated_results.csv"):
    """Save the aggregated results to a CSV file."""
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path.absolute()}")

def create_pivot_tables_by_model(df: pd.DataFrame):
    """Create and display pivot tables for accuracy metrics, broken down by model."""
    
    # Get unique models
    unique_models = df['model_name'].unique()
    
    all_accuracy_pivots = {}
    all_first_move_pivots = {}
    
    for model in unique_models:
        model_df = df[df['model_name'] == model]
        
        print("\n" + "="*120)
        print(f"ACCURACY PIVOT TABLE FOR {model} (Prompt Styles × Board Formats)")
        print("="*120)
        
        # Create accuracy pivot table for this model
        accuracy_pivot = model_df.pivot_table(
            values='accuracy', 
            index='prompt_config_parsed', 
            columns='board_format_parsed', 
            aggfunc='mean'
        )
        
        # Format as percentage and display
        accuracy_formatted = accuracy_pivot.multiply(100).round(1)
        print(accuracy_formatted.to_string(na_rep='--'))
        
        print("\n" + "="*120)
        print(f"FIRST MOVE ACCURACY PIVOT TABLE FOR {model} (Prompt Styles × Board Formats)")
        print("="*120)
        
        # Create first move accuracy pivot table for this model
        first_move_pivot = model_df.pivot_table(
            values='first_move_accuracy', 
            index='prompt_config_parsed', 
            columns='board_format_parsed', 
            aggfunc='mean'
        )
        
        # Format as percentage and display
        first_move_formatted = first_move_pivot.multiply(100).round(1)
        print(first_move_formatted.to_string(na_rep='--'))
        
        # Store for saving later
        all_accuracy_pivots[model] = accuracy_pivot
        all_first_move_pivots[model] = first_move_pivot
    
    return all_accuracy_pivots, all_first_move_pivots

def create_pivot_tables(df: pd.DataFrame):
    """Create and display pivot tables for accuracy metrics."""
    
    # Debug: Show some sample values
    print("\nDEBUG: Sample accuracy values from dataframe:")
    sample_data = df[['filename', 'model_name', 'board_format_parsed', 'prompt_config_parsed', 'accuracy', 'first_move_accuracy']].head(10)
    print(sample_data.to_string(index=False))
    
    # Show model-specific pivot tables
    model_accuracy_pivots, model_first_move_pivots = create_pivot_tables_by_model(df)
    
    print("\n" + "="*100)
    print("OVERALL ACCURACY PIVOT TABLE (Prompt Styles × Board Formats) - AVERAGED ACROSS ALL MODELS")
    print("="*100)
    
    # Create overall accuracy pivot table
    accuracy_pivot = df.pivot_table(
        values='accuracy', 
        index='prompt_config_parsed', 
        columns='board_format_parsed', 
        aggfunc='mean'
    )
    
    # Format as percentage and display
    accuracy_formatted = accuracy_pivot.multiply(100).round(1)
    print(accuracy_formatted.to_string(na_rep='--'))
    
    print("\n" + "="*100)
    print("OVERALL FIRST MOVE ACCURACY PIVOT TABLE (Prompt Styles × Board Formats) - AVERAGED ACROSS ALL MODELS")
    print("="*100)
    
    # Create overall first move accuracy pivot table
    first_move_pivot = df.pivot_table(
        values='first_move_accuracy', 
        index='prompt_config_parsed', 
        columns='board_format_parsed', 
        aggfunc='mean'
    )
    
    # Format as percentage and display
    first_move_formatted = first_move_pivot.multiply(100).round(1)
    print(first_move_formatted.to_string(na_rep='--'))
    
    return accuracy_pivot, first_move_pivot, model_accuracy_pivots, model_first_move_pivots

def print_summary(df: pd.DataFrame):
    """Print a summary of the aggregated results."""
    print("\n" + "="*80)
    print("SUMMARY OF AGGREGATED RESULTS")
    print("="*80)
    
    print(f"Total experiments: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique board formats: {df['board_format_parsed'].nunique()}")
    print(f"Unique prompt configs: {df['prompt_config_parsed'].nunique()}")
    
    print("\nModels:")
    for model in df['model_name'].unique():
        count = len(df[df['model_name'] == model])
        print(f"  - {model}: {count} experiments")
    
    print("\nBoard formats:")
    for fmt in sorted(df['board_format_parsed'].unique()):
        count = len(df[df['board_format_parsed'] == fmt])
        print(f"  - {fmt}: {count} experiments")
    
    print("\nPrompt configs:")
    for config in sorted(df['prompt_config_parsed'].unique()):
        count = len(df[df['prompt_config_parsed'] == config])
        print(f"  - {config}: {count} experiments")
    
    print("\nAccuracy Statistics:")
    print(f"  - Mean accuracy: {df['accuracy'].mean():.3f}")
    print(f"  - Max accuracy: {df['accuracy'].max():.3f}")
    print(f"  - Min accuracy: {df['accuracy'].min():.3f}")
    
    print(f"\nFirst Move Accuracy Statistics:")
    print(f"  - Mean first move accuracy: {df['first_move_accuracy'].mean():.3f}")
    print(f"  - Max first move accuracy: {df['first_move_accuracy'].max():.3f}")
    print(f"  - Min first move accuracy: {df['first_move_accuracy'].min():.3f}")
    
    print("\nTop 5 performing experiments (by accuracy):")
    top_5 = df.nlargest(5, 'accuracy')[['model_name', 'board_format_parsed', 'prompt_config_parsed', 'accuracy', 'first_move_accuracy']]
    print(top_5.to_string(index=False))
    
    # Create and display pivot tables
    accuracy_pivot, first_move_pivot = create_pivot_tables(df)
    
    return accuracy_pivot, first_move_pivot

def main():
    """Main function to run the aggregation script."""
    print("Chess Puzzle Results Aggregation Script")
    print("="*50)
    
    # Aggregate results
    df = aggregate_results()
    
    if df.empty:
        print("No results to aggregate!")
        return
    
    # Print summary and get pivot tables
    accuracy_pivot, first_move_pivot, model_accuracy_pivots, model_first_move_pivots = print_summary(df)
    
    # Save results
    save_results(df)
    
    # Save overall pivot tables as separate CSV files
    accuracy_pivot.to_csv("accuracy_pivot_table_overall.csv")
    first_move_pivot.to_csv("first_move_accuracy_pivot_table_overall.csv")
    
    # Save model-specific pivot tables
    for model_name, pivot in model_accuracy_pivots.items():
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        pivot.to_csv(f"accuracy_pivot_table_{safe_model_name}.csv")
    
    for model_name, pivot in model_first_move_pivots.items():
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        pivot.to_csv(f"first_move_accuracy_pivot_table_{safe_model_name}.csv")
    
    print("Pivot tables saved as:")
    print("  - accuracy_pivot_table_overall.csv")
    print("  - first_move_accuracy_pivot_table_overall.csv")
    for model_name in model_accuracy_pivots.keys():
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        print(f"  - accuracy_pivot_table_{safe_model_name}.csv")
        print(f"  - first_move_accuracy_pivot_table_{safe_model_name}.csv")
    
    # Also save as Excel for easier analysis
    try:
        with pd.ExcelWriter("aggregated_results.xlsx") as writer:
            df.to_excel(writer, sheet_name='All Results', index=False)
            accuracy_pivot.to_excel(writer, sheet_name='Overall Accuracy Pivot')
            first_move_pivot.to_excel(writer, sheet_name='Overall First Move Pivot')
            
            # Add model-specific sheets
            for model_name, pivot in model_accuracy_pivots.items():
                safe_model_name = model_name.replace('/', '_').replace(' ', '_')
                sheet_name = f'Acc_{safe_model_name}'[:31]  # Excel sheet name limit
                pivot.to_excel(writer, sheet_name=sheet_name)
            
            for model_name, pivot in model_first_move_pivots.items():
                safe_model_name = model_name.replace('/', '_').replace(' ', '_')
                sheet_name = f'FMA_{safe_model_name}'[:31]  # Excel sheet name limit
                pivot.to_excel(writer, sheet_name=sheet_name)
                
        print("Results also saved as Excel file with multiple sheets: aggregated_results.xlsx")
    except ImportError:
        print("Note: Install openpyxl to save Excel files (pip install openpyxl)")
    
    print(f"\nDataFrame shape: {df.shape}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
