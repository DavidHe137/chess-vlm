import click
import os

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

import time
import requests
def wait_for_server_ready(host, port, timeout=200):
    """
    Waits for the vLLM server to become responsive.
    """
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("vLLM server is ready.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("vLLM server did not become ready in time.")
    return False
# add support for serving a peft checkpoint and immediately deleting it afterwards
# uv run vllm serve Qwen/Qwen3-VL-8B-Instruct --port=8003 --generation-config vllm
@main.command()
@click.option("--source", type=str, required=True)
@click.option('--model_id', default="Qwen/Qwen2-VL-7B-Instruct", type=str, required=False)
@click.option("--board_formats", default="png", type=str, 
              help="Comma-separated list of board formats (e.g., 'png,ascii,fen'). Options: png, ascii, fen, pgn")
@click.option("--prompt_config", default="basic", type=click.Choice(["basic", "cot", "detailed_cot", "few_shot"]), required=True)
def serve(source, model_id, board_formats, prompt_config):
    import os
    import shutil
    import subprocess
    import time
    import gc
    host = 'localhost'
    port = 8000
    # full_model_dir = peft_to_full_model.callback(source=source, model_id=model_id)
    full_model_dir = f"{source}_full"
    subprocess.run(f"uv run python peft_to_full_model.py peft-to-full-model --source {source} --model_id {model_id}".split(' '))
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
        try:
            wait_for_server_ready(host, port, timeout=200)
            print("vLLM server started.")
            board_formats_list = board_formats.split(",")
            board_formats_arg = "--board_format " + " --board_format ".join(board_formats_list)
            res = subprocess.run(f"uv run scripts/evaluate.py --client_type vllm --model_name {full_model_dir} --hostname localhost --prompt_config {prompt_config} {board_formats_arg}".split(' '))
            print("eval subprocess res", res)
        finally:
            server_process.terminate()
            server_process.wait()
    finally:
        shutil.rmtree(full_model_dir) 

if __name__ == "__main__":
    main()