#!/usr/bin/env python3
"""Run evaluation on prepared puzzles."""

import os
import json
import re
from vllm import LLM, SamplingParams

def parse_move(response, solution):
    """Parse UCI move from response."""
    import re
    # Look for UCI pattern (e.g., e2e4, g1f3, e7e8q)
    uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
    matches = re.findall(uci_pattern, response.lower())
    return solution.lower() in matches

def main():
    # Load puzzles
    with open("results/puzzles.json") as f:
        puzzles = json.load(f)
    
    # Initialize model
    model = LLM("Qwen/Qwen2.5-VL-3B-Instruct")
    params = SamplingParams(temperature=0, max_tokens=500)
    
    # Get prompts
    prompts = [p["prompt"] for p in puzzles]
    
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
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    accuracy = correct / len(puzzles)
    print(f"Accuracy: {correct}/{len(puzzles)} = {accuracy:.2%}")

if __name__ == "__main__":
    main()
