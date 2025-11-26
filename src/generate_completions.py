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
from typing import List, Dict, Any, Optional
import os

from utils.prompts import prompt_variant_completions_no_reasoning, prompt_variant_completions_reasoning

prompt_variants = {
    "completions_no_reasoning": prompt_variant_completions_no_reasoning,
    "completions_reasoning": prompt_variant_completions_reasoning,
}

def create_chats(
    dataset: Any,
    system_prompt: str,
    trace_dataset: Any,
    trace_len: float,
) -> tuple:
    """
    Create chat prompts for the dataset.
    
    Returns:
        Tuple of (chats, questions, answers)
    """
    chats = []
    questions = []
    answers = []
    
    assert len(dataset) == len(trace_dataset), "Dataset and trace dataset must have the same number of examples"
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Creating chats"):
        question, answer = item["question"], item["answer"]
        trace = trace_dataset[i]["reasoning_trace"]

        question = question + "\n\n<think>\n" + trace[:int(trace_len * len(trace))] + "\n</think>\n"

        chats.append([{
            "role": "system",
            "content": system_prompt,
        }, {
            "role": "user",
            "content": question,
        }])
        
        questions.append(question)
        answers.append(answer)
    
    return chats, questions, answers


def generate_completions(
    model_name: str,
    dataset_name: str,
    trace_csv_file: str,
    trace_len: float,
    limit: Optional[int] = None,
    dataset_split: Optional[str] = None,
    **kwargs
):
    # Load dataset with specified split
    dataset = datasets.load_dataset(dataset_name, "main", split=dataset_split)

    # Limit examples if specified
    if limit:
        print(f"Limiting dataset to {limit} examples")
        dataset = dataset.select(range(limit))
    
    prompt_variant = kwargs.pop("prompt_variant", "completions_no_reasoning")
    system_prompt = prompt_variants[prompt_variant]

    ### Create Chats ###
    print(f"Using system prompt: {system_prompt}")
    print(f"Loading trace dataset from {trace_csv_file}")
    # Get the path relative to this script's location
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # trace_csv_path = os.path.join(script_dir, "..", "outputs", "data", "Qwen_Qwen3-8B", "dataframes_default.csv")
    trace_dataset = datasets.load_dataset("csv", data_files=trace_csv_file, split="train")

    # Limit examples if specified
    if limit:
        print(f"Limiting dataset to {limit} examples")
        trace_dataset = trace_dataset.select(range(limit))

    chats, questions, answers = create_chats(dataset, system_prompt, trace_dataset, trace_len)
    
    # Extract sampling parameters
    temperature = kwargs.pop("temperature", None)
    max_tokens = kwargs.pop("max_tokens", 8192)
    seed = kwargs.pop("seed", 42)
    top_p = kwargs.pop("top_p", None)

    # Create sampling params
    sampling_kwargs = {"max_tokens": max_tokens, "seed": seed}
    if temperature is not None:
        sampling_kwargs["temperature"] = temperature
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    
    params = SamplingParams(**sampling_kwargs)

    if "tensor_parallel_size" not in kwargs or kwargs.get("tensor_parallel_size") is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            kwargs["tensor_parallel_size"] = num_gpus
            print(f"Auto-detected {num_gpus} GPUs, using tensor parallelism")
        elif "tensor_parallel_size" in kwargs:
            # Remove None value if only 1 GPU
            kwargs.pop("tensor_parallel_size")

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Initialize model
    model_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        **filtered_kwargs
    }

    model = LLM(**model_kwargs)

    print("Model loaded")
    print(f"Running inference on {len(chats)} prompts")

    start_time = time.time()
    response = model.chat(chats, params, chat_template_kwargs={"enable_thinking": False if prompt_variant == "completions_no_reasoning" else True})
    inference_time = time.time() - start_time

    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Throughput: {len(chats) / inference_time:.2f} examples/sec")
    
    torch.cuda.empty_cache()
    gc.collect()

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    
    completions = []
    finished = []
    
    for i, output in tqdm(enumerate(response), total=len(response), desc="Processing outputs"):
        completion = tokenizer.decode(output.prompt_token_ids + output.outputs[0].token_ids)
        completions.append(completion)

        finished.append(output.finished)
    
    output = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "collected_on": str(datetime.now()),
            "system_prompt": system_prompt,
            "num_gpus": torch.cuda.device_count(),
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "seed": seed,
            },
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "inference_time": inference_time,
            "throughput": len(chats) / inference_time if inference_time > 0 else 0,
        },
        "data": {
            "questions": questions,
            "completions": completions,
            "ground_truth_answers": answers,
            "finished": finished,
        }
    }
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate completions for evaluation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to load from Hugging Face.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='gsm8k',
        help="Name of the dataset ('gsm8k', 'math', 'gpqa').",
    )
    parser.add_argument(
        "--trace_csv_file",
        type=str,
        default=None,
        required=True,
        help="Path to the extracted trace dataset (CSV file) for reasoning traces.",
    )
    parser.add_argument(
        "--trace_len",
        type=float,
        default=None,
        required=True,
        help="Trace length to use  in the prompt. (0.5, 0.75, 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config file (YAML) with default generation parameters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: auto-detect).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Dataset split to use (e.g., 'train', 'test'). Default depends on dataset.",
    )
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default="default",
        help="Prompt variant to use (default: 'default').",
        choices=list(prompt_variants.keys()),
    )
    args = parser.parse_args()

    ### Loading YAML Config ###
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    call_kwargs = {}
    # Copy vllm kwargs if present, filtering out None values
    if isinstance(config.get("vllm_kwargs"), dict):
        # Filter out None values from config (null in YAML becomes None)
        filtered_vllm_kwargs = {k: v for k, v in config["vllm_kwargs"].items() if v is not None}
        call_kwargs.update(filtered_vllm_kwargs)
    
    # Sampling params
    if "temperature" in config:
        call_kwargs["temperature"] = config["temperature"]
    if "max_tokens" in config:
        call_kwargs["max_tokens"] = config["max_tokens"]
    if "top_p" in config:
        call_kwargs["top_p"] = config["top_p"]
    if "seed" in config:
        call_kwargs["seed"] = config["seed"]
    if "prompt_variant" in config:
        call_kwargs["prompt_variant"] = config["prompt_variant"]
    
    # Get dataset split from config or args
    dataset_split = args.dataset_split
    if dataset_split is None and "dataset_split" in config:
        dataset_split = config["dataset_split"]
    assert dataset_split in args.trace_csv_file, "Dataset split must be in the trace CSV file"
    # Override tensor_parallel_size from command line if provided
    if args.tensor_parallel_size is not None:
        call_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    if 'model_name' in config and args.model_name is None:
        args.model_name = config['model_name']
    if 'dataset_name' in config and args.dataset_name is None:
        args.dataset_name = config['dataset_name']
    
    assert args.model_name is not None, "Model name must be specified via --model_name or config file."
    assert args.dataset_name is not None, "Dataset name must be specified via --dataset_name or config file."
    
    output = generate_completions(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        trace_csv_file=args.trace_csv_file,
        trace_len=args.trace_len,
        limit=args.limit,
        dataset_split=dataset_split,
        **call_kwargs
    )
    
    # Save output
    base_dir = "/".join(args.trace_csv_file.split("/")[:-1])
    trace_prompt_type = "_".join(args.trace_csv_file.split("/")[-1].split("_")[1:]).split(".")[0]
    prompt_variant = call_kwargs["prompt_variant"]

    output_dir = f"{base_dir}/completions/{args.model_name}/length/trace_length_{args.trace_len}/trace_{prompt_variant}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.limit:
        output_path = os.path.join(
            output_dir,
            f"completions_{trace_prompt_type}_{args.limit}.pkl"
        )
    else:
        output_path = os.path.join(
            output_dir,
            f"completions_{trace_prompt_type}.pkl"
        )
    
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    
    print(f"Generated traces saved to {output_path}")
