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
 
SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions about the image.
"""
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample['messages'])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

def create_format_function(board_formats, prompt_config, prompt_formatter, include_valid_moves, include_ascii_board):
    """
    Create a format function that uses PromptFormatter to format messages.
    This is a closure that captures the board_formats and prompt_config.
    """
    def format_lichess_puzzles(sample):
        """
        Format dataset sample into messages format for Qwen2VL using PromptFormatter.
        
        This function processes individual samples (not batches).
        Note: Sometimes set_transform may pass data in a format where values are lists,
        so we unwrap single-item lists to handle this edge case.
        """
        try:
            # Convert to dict if it's an Arrow object or other type
            if not isinstance(sample, dict):
                sample = dict(sample)
            
            # Unwrap any single-item lists (handles edge case where set_transform passes batch-like format)
            # This is a safety check - set_transform should pass individual samples, but sometimes
            # the dataset formatting can wrap values in lists
            sample_unwrapped = {}
            for key, value in sample.items():
                if isinstance(value, list):
                    # If it's a list, take the first element (should be single-item for set_transform)
                    sample_unwrapped[key] = value[0] if len(value) > 0 else value
                else:
                    sample_unwrapped[key] = value
            sample = sample_unwrapped
            
            # Ensure required fields exist for Puzzle.from_dataset_row
            if "PuzzleUrl" not in sample or sample["PuzzleUrl"] is None:
                sample["PuzzleUrl"] = None
                sample["GameUrl"] = None
                sample["Puzzle_Moves_SAN"] = None
                sample["GameId"] = None
            
            puzzle = Puzzle.from_dataset_row(sample)
            image_path = sample.get("png_file_name", None)
            
            # Format messages using PromptFormatter
            messages = prompt_formatter.format_messages(
                puzzle=puzzle,
                board_formats=board_formats,
                config_name=prompt_config,
                image_path=image_path,
                encode_images=False,
                for_qwen=True,
            )
        
            new_messages = []
            new_messages.append(messages[0])
            for msg, prev in zip(messages[1:], messages[:-1]):
                if msg["role"] == prev["role"]:
                    new_messages[-1]["content"].extend(msg["content"])
                else:
                    new_messages.append(msg)
            
            # Add assistant response with the move
            moves = sample.get("Moves", "")
            move_text = moves.split()[0] if moves else ""
            if include_ascii_board:
                board_statement = f"The current board position is: {sample['ASCII_Board']} . "
            else:
                board_statement = ""
            new_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"{board_statement}The next best move is: <move>{move_text}</move>"
                }]
            })

            return {"messages": new_messages, "images": [Image.open(image_path)]}
        except Exception as e:
            # Log error and re-raise with more context
            import traceback
            error_msg = f"Error formatting sample (PuzzleId: {sample.get('PuzzleId', 'unknown')}): {str(e)}\n{traceback.format_exc()}"
            raise RuntimeError(error_msg) from e
    
    return format_lichess_puzzles




def recursive_apply(node, child_apply_fn, all_apply_fn):
    new_node = all_apply_fn(node)
    if isinstance(new_node, dict):
        return_node = dict()
        for k in new_node.keys():
            return_node[k] = recursive_apply(new_node[k], child_apply_fn, all_apply_fn)
        return return_node
    elif isinstance(new_node, list):
        return_node = list()
        for li in new_node:
            return_node.append(recursive_apply(li, child_apply_fn, all_apply_fn))
        return return_node
    else:
        return child_apply_fn(new_node)
def remove_null_url(d):
    if isinstance(d, dict):
        if d.get('image', None) is None:
            return {k: v for k, v in d.items() if k != "image"}
    return d

@click.command()
@click.option("--dataset_name", default="Lichess/one-move-chess-puzzles", type=str, required=True)
@click.option("--board_formats", default="png", type=str, 
              help="Comma-separated list of board formats (e.g., 'png,ascii,fen'). Options: png, ascii, fen, pgn")
@click.option("--prompt_config", default="basic", type=click.Choice(["basic", "cot", "detailed_cot", "few_shot"]), required=True)
@click.option("--model_id", default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=True)
@click.option("--include_valid_moves", is_flag=True, default=False)
@click.option("--include_ascii_board", is_flag=True, default=False)
@click.option("--output_dir", default="qwen2-7b-instruct-trl-sft-ChartQA", type=str, required=True)
@click.option("--num_train_epochs", default=1, type=int, help="Number of training epochs")
@click.option("--regenerate_messages", default=False, type=bool, help="Force regeneration of messages even if they exist in dataset")
@click.option("--logging_steps", default=1, type=int, help="Steps interval for logging")
@click.option("--eval_steps", default=100, type=int, help="Steps interval for evaluation")
@click.option("--save_steps", default=100, type=int, help="Steps interval for saving")
def sft(dataset_name, board_formats, prompt_config, model_id, include_valid_moves, include_ascii_board, output_dir, num_train_epochs, regenerate_messages, logging_steps, eval_steps, save_steps):
    """
    Train a Qwen2 VL model on a dataset.
    """
    
    # Parse board formats from comma-separated string
    board_format_list = [fmt.strip().lower() for fmt in board_formats.split(",")]
    
    # Validate board formats
    valid_formats = {"png", "ascii", "fen", "pgn"}
    for fmt in board_format_list:
        if fmt not in valid_formats:
            raise click.ClickException(f"Invalid board format: {fmt}. Valid options: {', '.join(valid_formats)}")
    
    click.echo(f"Using board formats: {board_format_list}")
    click.echo(f"Using prompt config: {prompt_config}")
    click.echo(f"Training steps - logging: {logging_steps}, eval: {eval_steps}, save: {save_steps}")

    # Initialize PromptFormatter
    prompt_formatter = PromptFormatter()
    
    # Create format function with captured parameters
    format_func = create_format_function(
        board_formats=board_format_list,
        prompt_config=prompt_config,
        prompt_formatter=prompt_formatter,
        include_valid_moves=include_valid_moves,
        include_ascii_board=include_ascii_board
    )

    # Load dataset
    dataset = load_from_disk(dataset_name)
    train_dset = dataset["train"].select(range(30000))
    eval_dset = dataset["eval"].select(range(1000))

    train_dset = train_dset.map(format_func, num_proc=os.cpu_count())
    eval_dset = eval_dset.map(format_func, num_proc=os.cpu_count())
    # import ipdb; ipdb.set_trace()
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
    
    # Load model and tokenizer
    click.echo("loading model")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    # Add eos_token attribute to processor for SFTTrainer compatibility
    processor.eos_token = processor.tokenizer.eos_token

    # BitsAndBytesConfig int-4 config
    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        max_length=None,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        # Logging and evaluation
        logging_steps=logging_steps,  # Steps interval for logging
        eval_steps=eval_steps,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=save_steps,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=True,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
    )

    wandb.init(
        project="chess-vlm",
        name=f"chess-vlm-sft-{board_formats}",
        config=training_args,
    )
    # self.data_collator input_ids, labels, attention_mask self.args.dataloader_num_workers
    # def custom_data_collator_fn(samples):
    #     # (27, '<')
    #     # (3397, 'move')
    #     # (29, '>') # find the second to last occurance of this one.
    #     # (70, 'g')
    #     # (18, '3')
    #     # (70, 'g')
    #     # (23, '8')
    #     # (522, '</') # take the prediction to this, but do not include it.
    #     # (3397, 'move')
    #     # (29, '>')
    #     # (151645, '<|im_end|>') 
    #     # import ipdb; ipdb.set_trace()
    #     input_ids_l = []
    #     labels_l = []
    #     attention_mask_l = []
    #     max_len_sample = max(len(s['input_ids']) for s in samples)
    #     for sample in samples:
    #         excess_pad = max_len_sample - len(sample['input_ids'])
    #         input_ids_sample = sample['input_ids'] + [processor.tokenizer.eos_token_id] * excess_pad
    #         index_of_last_end_move = len(input_ids_sample) - input_ids_sample[::-1].index(3397) - 1
    #         index_beginning_move = len(input_ids_sample[:index_of_last_end_move]) - input_ids_sample[:index_of_last_end_move][::-1].index(3397)
    #         input_ids_sample_tensor = torch.tensor(input_ids_sample)
    #         label_sample_tensor = torch.full_like(input_ids_sample_tensor,fill_value=-100)
    #         # I adjusted this to be everything afert "assistant\n", which should be "<move>..."
    #         label_sample_tensor[index_beginning_move-1:] = input_ids_sample_tensor[index_beginning_move-1:]
    #         input_ids_l.append(input_ids_sample_tensor)
    #         labels_l.append(label_sample_tensor)
    #         attention_mask_l.append(torch.tensor([1] * len(sample['input_ids']) + [0] * excess_pad))
        
    #     out = {"input_ids": torch.stack(input_ids_l), 
    #            "labels": torch.stack(labels_l),
    #            "attention_mask": torch.stack(attention_mask_l)}
    #     return out
    # Create a data collator to encode text and image pairs
    # training_args.dataloader_num_workers = 0
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        # texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        # image_inputs = [process_vision_info()[0] for example in examples]
        texts = []
        image_inputs = []
        for example in examples:
            less_image_pad = recursive_apply(example["messages"], lambda x: x, remove_null_url)
            texts.append(processor.apply_chat_template(less_image_pad, tokenize=False))
            image_inputs.append(process_vision_info(less_image_pad)[0])
            
    
        # Tokenize the texts and process the images
        # import ipdb; ipdb.set_trace()
        if 'png' in board_format_list:
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        else:
            batch = processor.tokenizer(texts, return_tensors='pt', padding=True)
    
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100 #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        # import ipdb; ipdb.set_trace()
        # 77091 look for the last instance of assistant, and make everything before that and the next newline character all -100 labels.
        for i in range(labels.size(0)):
            label_i = labels[i].tolist()
            index_beginning_move = len(label_i) - label_i[::-1].index(77091) + 1
            labels[i,:index_beginning_move] = -100
        batch["labels"] = labels 
    
        return batch
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    sft()