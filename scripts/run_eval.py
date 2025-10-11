#!/usr/bin/env python3
"""Run evaluation on prepared puzzles."""

import os
import json
import click
import re
from vllm import LLM, SamplingParams

def parse_move(response, solution):
    """Parse UCI move from response."""
    import re
    # Look for UCI pattern (e.g., e2e4, g1f3, e7e8q)
    uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
    matches = re.findall(uci_pattern, response.lower())
    return solution.lower() in matches

# TODO : Have more models
def get_model(name):
    model = LLM(name)
    return model


@click.command()
@click.option('--file_name', type=str, default=None)
@click.option('--out_file', type=str, default=None)
@click.option('--model_name', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
@click.option('--prompt_with_board', is_flag=True, default=False)
def main(file_name, out_file, model_name, prompt_with_board):
    assert file_name is not None, 'The file name cannot be None'
    assert out_file is not None, 'The out file name cannot be None'
    # Load puzzles
    with open(file_name) as f:
        puzzles = json.load(f)
    
    # Initialize model
    model = get_model(model_name)
    params = SamplingParams(temperature=0, max_tokens=500)
    
    # Get prompts
    prompts = [p["prompt-with-board" if prompt_with_board else "prompt-with-out-board"] for p in puzzles]
    
    # Generate responses
    print("Generating responses...")
    outputs = model.generate(prompts, params)
    
    # Evaluate
    correct = 0
    results = []
    
    for puzzle, output in zip(puzzles, outputs):
        response = output.outputs[0].text
        is_correct = parse_move(response, puzzle["solution"])
        
        results.append({
            "id": puzzle["id"],
            "correct": is_correct,
            "response": response,
            "solution": puzzle["solution"]
        })
        
        if is_correct:
            correct += 1
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/{out_file}", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    accuracy = correct / len(puzzles)
    print(f"Accuracy: {correct}/{len(puzzles)} = {accuracy:.2%}")

if __name__ == "__main__":
    main()
