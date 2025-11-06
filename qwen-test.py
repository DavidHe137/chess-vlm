import os
import json
import click
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
import trackio
from trl import SFTTrainer
import pdb


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

def load_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def get_formatted_dataset(folder_name):
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
                            "image": sample["png-file-name"],
                        },
                        {
                            "type": "text",
                            "text": sample['prompt-with-board'],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": sample['solution'],
                        }
                    ],
                }
            ]
        }
    
    # Get random .9 of the dataset
    # Get random elements from a dictionary
    train_dataset = load_json(os.path.join(folder_name, "train.json"))
    eval_dataset = load_json(os.path.join(folder_name, "eval.json"))
    test_dataset = load_json(os.path.join(folder_name, "test.json"))

    train_dataset = [format_lichess_puzzles(sample) for sample in train_dataset]
    eval_dataset = [format_lichess_puzzles(sample) for sample in eval_dataset]
    test_dataset = [format_lichess_puzzles(sample) for sample in test_dataset]
    
    # Convert lists to HuggingFace Dataset objects
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    
    return train_dataset, eval_dataset, test_dataset


@click.command()
@click.option("--dataset_json", default="data/themed_prompts/bodenMate/", type=str, required=True)
@click.option("--model_id", default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=True)
@click.option("--output_dir", default="qwen2-7b-instruct-trl-sft-ChartQA", type=str, required=True)
def sft(dataset_json, model_id, output_dir):
    """
    Train a Qwen2 VL model on a dataset.
    """

    # Load dataset
    train_dset, eval_dset, test_dset = get_formatted_dataset(dataset_json)

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
        num_train_epochs=3,  # Number of training epochs
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
        report_to="trackio",  # Reporting tool for tracking metrics
    )

    trackio.init(
        project="qwen2-7b-instruct-trl-sft-ChartQA",
        name="qwen2-7b-instruct-trl-sft-ChartQA",
        config=training_args,
        space_id=training_args.output_dir + "-trackio"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        peft_config=peft_config,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    sft()