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

def format_lichess_puzzles(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_MESSAGE
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["png_file_name"],
                    },
                    {
                        "type": "text",
                        "text": f"FEN: {sample['FEN']} with valid moves: {sample['All_Valid_Moves']}",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f'<move>{sample['Moves']}</move>',
                    }
                ],
            }
        ]
    }
    
@click.command()
@click.option("--dataset_name", default="Lichess/one-move-chess-puzzles", type=str, required=True)
@click.option("--board_format", default="png", type=click.Choice(["png", "ascii", "fen"]), required=True)
@click.option("--use_png", default=False, type=bool, required=True)
@click.option("--prompt_config", default="basic", type=str, required=True)
@click.option("--model_id", default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=True)
@click.option("--output_dir", default="qwen2-7b-instruct-trl-sft-ChartQA", type=str, required=True)
@click.option("--num_train_epochs", default=3, type=int, help="Number of training epochs")
def sft(dataset_name, board_format, use_png, prompt_config, model_id, output_dir, num_train_epochs):
    """
    Train a Qwen2 VL model on a dataset.
    """

    # Load dataset

    dataset = load_from_disk(dataset_name)
    train_dset = dataset["train"]
    eval_dset = dataset["eval"]
    # train_dset.set_transform(format_lichess_puzzles)
    # eval_dset.set_transform(format_lichess_puzzles)
    train_dset = train_dset.map(format_lichess_puzzles)
    eval_dset = eval_dset.map(format_lichess_puzzles)

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
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
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
        output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
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
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=True,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
    )

    wandb.init(
        project="qwen2-7b-instruct-trl-sft-ChartQA",
        name="qwen2-7b-instruct-trl-sft-ChartQA",
        config=training_args,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    sft()