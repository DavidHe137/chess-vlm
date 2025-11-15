import click


@click.group()
def main():
    pass

@main.command()
@click.option("--source", type=str, required=True)
@click.option("--dest", default="default", type=str, required=False)
@click.option('--model_id', default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=False)
def peft_to_full_model(source, dest="default", model_id="Qwen/Qwen2-VL-7B-Instruct"):
    import transformers
    import torch
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    import gc
    if dest == "default":
        dest = f"{source}_full"
    # import ipdb; ipdb.set_trace()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = PeftModel.from_pretrained(model, source)
    model = model.merge_and_unload()
    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    del model
    del processor
    gc.collect()
    return dest

# add support for serving a peft checkpoint and immediately deleting it afterwards
# uv run vllm serve Qwen/Qwen3-VL-8B-Instruct --port=8003 --generation-config vllm
@main.command()
@click.option("--source", type=str, required=True)
@click.option('--model_id', default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=False)
def serve(source, model_id):
    import os
    import shutil
    import subprocess
    import time
    import gc
    host = os.environ['HOSTNAME']
    port = 8000
    # full_model_dir = peft_to_full_model.callback(source=source, model_id=model_id)
    full_model_dir = f"{source}_full"
    subprocess.run(f"uv run python peft_to_full_model.py peft-to-full-model --source {source}".split(' '))
    # import ipdb; ipdb.set_trace()
    gc.collect()

    try:
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            full_model_dir,
            "--host",
            host,
            "--port",
            str(port),
        ]
        print(f"Starting vLLM server for model '{full_model_dir}' on {host}:{port}")
        server_process = subprocess.Popen(command)
        time.sleep(100)
        print("vLLM server started.")

        res = subprocess.run("uv run scripts/evaluate.py --client_type vllm --model_name Qwen/Qwen3-VL-8B-Instruct --hostname atl1-1-03-017-23-0 --prompt_config cot --board_format fen".split(' '))
        print("eval subprocess res", res)
        server_process.terminate()
        server_process.wait()
    finally:
        shutil.rmtree(full_model_dir) 

if __name__ == "__main__":
    main()