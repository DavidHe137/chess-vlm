import os
import json
import click
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset, load_from_disk
from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer
import pdb
import wandb
from src.prompts import PromptFormatter
from src.puzzle import Puzzle, BoardFormat
from PIL import Image
from peft import PeftModel
import numpy as np
import tqdm


from qwentest import create_format_function, recursive_apply, remove_null_url
@click.command()
@click.option("--source", type=str, required=True)
@click.option("--dataset_name", default="Lichess/one-move-chess-puzzles", type=str, required=True)
@click.option("--board_formats", default="png", type=str, 
              help="Comma-separated list of board formats (e.g., 'png,ascii,fen'). Options: png, ascii, fen, pgn")
@click.option("--prompt_config", default="basic", type=click.Choice(["basic", "cot", "detailed_cot", "few_shot"]), required=True)
@click.option("--model_id", default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=True)
@click.option("--is_base", default=False, type=bool, required=False)
@click.option("--output_dir", default=None, type=str, required=False)
def eval_hf(source, dataset_name, board_formats, prompt_config, model_id, is_base, output_dir):
    board_format_list = [fmt.strip().lower() for fmt in board_formats.split(",")]
    if output_dir is None:
        output_dir = source
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate board formats
    valid_formats = {"png", "ascii", "fen", "pgn"}
    for fmt in board_format_list:
        if fmt not in valid_formats:
            raise click.ClickException(f"Invalid board format: {fmt}. Valid options: {', '.join(valid_formats)}")
    
    click.echo(f"Using board formats: {board_format_list}")
    click.echo(f"Using prompt config: {prompt_config}")

    # Initialize PromptFormatter
    prompt_formatter = PromptFormatter()
    
    # Create format function with captured parameters
    format_func = create_format_function(
        board_formats=board_format_list,
        prompt_config=prompt_config,
        prompt_formatter=prompt_formatter,
        include_valid_moves=False,
        is_eval=True
    )

    # Load dataset
    dataset = load_from_disk(dataset_name)
    train_dset = dataset["train"].select(range(30000))
    eval_dset = dataset["eval"].select(range(1000))

    train_dset = train_dset.map(format_func, num_proc=os.cpu_count())
    eval_dset = eval_dset.map(format_func, num_proc=os.cpu_count())
    click.echo(f"Loaded {len(train_dset)} training samples and {len(eval_dset)} eval samples")
    
    # Test transform function directly on a raw sample first
    try:
        raw_sample = train_dset[0]
        click.echo(f"Testing transform on raw sample. Raw sample keys: {list(raw_sample.keys())}")
        test_result = format_func(raw_sample)
        click.echo(test_result)
        click.echo(f"Transform function test successful. Result keys: {list(test_result.keys())}")
        if "messages" not in test_result:
            raise ValueError("Transform function did not return 'messages' key")
        click.echo(f"Messages length: {len(test_result['messages'])}")
    except Exception as e:
        click.echo(f"Error testing transform function directly: {e}")
        import traceback
        click.echo(traceback.format_exc())
        raise
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id, padding_side='left')
    if not is_base:
        model = PeftModel.from_pretrained(model, source)

    # Add eos_token attribute to processor for SFTTrainer compatibility
    processor.eos_token = processor.tokenizer.eos_token
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        # texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        # image_inputs = [process_vision_info()[0] for example in examples]
        texts = []
        image_inputs = []
        for example in examples:
            less_image_pad = recursive_apply(example["messages"], lambda x: x, remove_null_url)
            texts.append(processor.apply_chat_template(less_image_pad, tokenize=False, add_generation_prompt=True))
            image_inputs.append(process_vision_info(less_image_pad)[0])
            
    
        # Tokenize the texts and process the images
        # import ipdb; ipdb.set_trace()
        if 'png' in board_format_list:
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        else:
            batch = processor.tokenizer(texts, return_tensors='pt', padding=True)
        batch["correct_move"] = [e['correct_move'] for e in examples]
        return batch
    from torch.utils.data import DataLoader
    eval_dl = DataLoader(eval_dset, batch_size=8, collate_fn=collate_fn)
    first_move_acc = []
    def extract(move_str):
        return move_str.split('</move>')[0].split("<move>")[-1]
    if os.path.exists(os.path.join(output_dir, "logs.jsonl")):
        os.remove(os.path.join(output_dir, "logs.jsonl"))# del old logs if exist
    for batch in tqdm.tqdm(eval_dl):
        # run eval on the data
        # import ipdb; ipdb.set_trace()
        correct_moves = batch.pop('correct_move')

        generated_ids = model.generate(**batch.to(model.device), max_new_tokens=20)
        input_strs = processor.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens = True)
        generated_answers = processor.tokenizer.batch_decode(generated_ids[:, batch['input_ids'].size(1):])
        batch_first_move_acc = [gold_move == extract(move_str) for gold_move, move_str in zip(correct_moves, generated_answers)]
        first_move_acc.extend(batch_first_move_acc)
        with open(os.path.join(output_dir, "logs.jsonl"), 'a') as fout:
            for i in range(batch['input_ids'].size(0)):
                fout.write(json.dumps({"first_move_acc": batch_first_move_acc[i], 
                                       "generated_answer": generated_answers[i], 
                                       "input_str": input_strs[i],
                                       "correct_move": correct_moves[i],
                                       }) + "\n")
    with open(os.path.join(output_dir, "final_log.txt"), 'w') as fout:
        fout.write(f"acc: {np.mean(first_move_acc)}")
    print("acc", np.mean(first_move_acc))
if __name__ == "__main__":
    eval_hf()