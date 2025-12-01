from datetime import datetime
import argparse
import time
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pickle
import yaml
import datasets
from typing import Any, Optional
import os
import pandas as pd
import re

from utils.prompts import prompt_remove_answer


def create_cleaning_prompts(dataset: Any) -> tuple:
    chats = []
    traces = []

    for item in tqdm(dataset, total=len(dataset), desc="Creating cleaning prompts"):
        trace = item["reasoning_trace"]
        question = item["questions"]
        answer = item["predicted_answers"]
        traces.append(trace)

        user_input = f"<question>{question}</question>\n<answer>{answer}</answer>\n<trace>{trace}</trace>"
        chats.append([{
            "role": "system",
            "content": prompt_remove_answer,    
        }, {
            "role": "user",
            "content": user_input,
        }])

    return chats, traces

def generate_cleaned_traces(
    model_name: str,
    trace_csv_file: str,
    limit: Optional[int] = None,
    **kwargs
):
    dataset = datasets.load_dataset("csv", data_files=trace_csv_file, split="train")
    print(f"Loaded dataset with {len(dataset)} examples")
    if limit:
        dataset = dataset.select(range(limit))
        print(f"Limiting dataset to {limit} examples")

    print(f"Creating cleaning prompts for {len(dataset)} examples")
    chats, original_traces = create_cleaning_prompts(dataset)

    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", 512)
    seed = kwargs.pop("seed", 42)
    top_p = kwargs.pop("top_p", None)

    sampling_kwargs = {"max_tokens": max_tokens, "seed": seed}
    if temperature is not None:
        sampling_kwargs["temperature"] = temperature
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p

    params = SamplingParams(**sampling_kwargs)

    if kwargs.get("tensor_parallel_size") is None:
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            kwargs["tensor_parallel_size"] = ngpu

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # vLLM
    model_kwargs = {"model": model_name, "tokenizer": model_name, **filtered_kwargs}
    model = LLM(**model_kwargs)

    start_time = time.time()
    response = model.chat(chats, params, chat_template_kwargs={"enable_thinking": False})
    inference_time = time.time() - start_time

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Decode
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)

    cleaned_traces = []
    scores = []
    finished = []

    for out in tqdm(response, total=len(response), desc="Processing outputs"):
        decoded = tokenizer.decode(out.outputs[0].token_ids).strip()
        # Use regex to extract content between <cleaned_trace>...</cleaned_trace>
        match = re.search(r"<cleaned_trace>(.*?)</cleaned_trace>", decoded, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            print(f"No tags found in the decoded output: {decoded}")
            cleaned = decoded  # fallback to all of decoded if tags not found
        cleaned_traces.append(cleaned)
        scores.append(out.outputs[0].cumulative_logprob)
        finished.append(out.finished)

    output = {
        "metadata": {
            "model_name": model_name,
            "collected_on": str(datetime.now()),
            "num_gpus": torch.cuda.device_count(),
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "seed": seed,
            },
            "kwargs": {k: str(v) for k, v in filtered_kwargs.items()},
            "inference_time": inference_time,
        },
        "data": {
            "original_reasoning_trace": original_traces,
            "cleaned_reasoning_trace": cleaned_traces,
            "cumulative_logprob": scores,
            "finished": finished,
        }
    }

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean reasoning traces via vLLM")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--trace_csv_file", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    args = parser.parse_args()

    # Load YAML
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    call_kwargs = {}

    if isinstance(config.get("vllm_kwargs"), dict):
        filtered = {k: v for k, v in config["vllm_kwargs"].items() if v is not None}
        call_kwargs.update(filtered)

    for key in ["temperature", "max_tokens", "top_p", "seed"]:
        if key in config:
            call_kwargs[key] = config[key]

    if args.tensor_parallel_size is not None:
        call_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    if 'model_name' in config and args.model_name is None:
        args.model_name = config['model_name']
    assert args.model_name is not None, "Model name must be specified via --model_name or config file."

    output = generate_cleaned_traces(
        model_name=args.model_name,
        trace_csv_file=args.trace_csv_file,
        limit=args.limit,
        **call_kwargs
    )

    
    csv_dir = os.path.dirname(os.path.abspath(args.trace_csv_file))
    csv_name = os.path.basename(args.trace_csv_file).split(".")[0]
    output_dir = os.path.join(csv_dir, "cleaned_traces", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save PKL
    if args.limit:
        pkl_path = os.path.join(output_dir, f"cleaned_{csv_name}_{args.limit}.pkl")
    else:
        pkl_path = os.path.join(output_dir, f"cleaned_{csv_name}.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved PKL to {pkl_path}")

    # Save CSV
    orig_df = pd.read_csv(args.trace_csv_file)
    if args.limit:
        orig_df = orig_df.head(args.limit)

    if len(orig_df) != len(output["data"]["cleaned_reasoning_trace"]):
        raise ValueError("Row mismatch between dataset and model outputs.")

    orig_df["cleaned_reasoning_trace"] = output["data"]["cleaned_reasoning_trace"]

    if args.limit:
        csv_path = os.path.join(output_dir, f"cleaned_{csv_name}_{args.limit}.csv")
    else:
        csv_path = os.path.join(output_dir, f"cleaned_{csv_name}.csv")

    orig_df.to_csv(csv_path, index=False)
    print(f"Saved merged CSV to: {csv_path}")
