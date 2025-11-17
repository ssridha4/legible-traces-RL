# # # ####################################################### # # #
# # #              Generate Reasoning Traces                  # # #
# # # ####################################################### # # #

# TODO: Add connections, APPS, MMLU
# TODO: More graceful dataset handling
# TODO: Investigate Magistral failures

from datetime import datetime
import random
import datasets
from dotenv import load_dotenv
import os
from fastapi import params
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import vllm
import pickle
import argparse
import yaml
import os.path as osp

from src.utils import math_parser
from src.utils.extractors import extract_answer, extract_trace
from src.utils.graders import grade_answer
from src.utils.prompts import REASONING_SYSTEM_PROMPTS, format_mcq

def load_dataset(dataset_name: str):
    """
    Dataset loader for supported datasets.
    """
    if dataset_name == 'math':
        dataset = datasets.load_from_disk("math_consolidated_test")
    elif dataset_name == 'gpqa':
        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    return dataset


def get_question_answer(dataset_name, item):
    """
    Given a dataset and an element, return the question prompt and (a) ground truth answer.
    Only supports 'math' and 'gpqa' for now.

    """
    if dataset_name == 'math':
        return item['problem'], math_parser.extract_answer(item['solution'])
    
    elif dataset_name == 'gpqa':
        question = item['Question']
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        # ground truth letter (after shuffle)
        correct_letter = "ABCD"[choices.index(item["Correct Answer"])]
        answer = correct_letter
        return f"{format_mcq(question, choices)}", answer
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

def generate_traces(
        model_name: str,
        dataset_name: str,
        verbose: bool = False,
        **kwargs
    ):
    """
    Generate reasoning traces to study as artifacts.
    Arguments:
    - model_name: str, name or path of the model to use.
    - dataset_name: str, name of the dataset to use ('math' or 'gpqa').
    - verbose: bool, whether to print progress information.
    - kwargs: additional keyword arguments to pass to the vLLM
    """

    ### Initialization ###
    print("KWARGS")
    print(kwargs)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    if verbose:
        print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    system_prompt = REASONING_SYSTEM_PROMPTS.get(dataset_name, "")
    if verbose:
        print(f"Dataset {dataset_name} loaded with {len(dataset)} examples.")
        if system_prompt != "":
            print(f"Using system prompt for {dataset_name}")


    ### Create Chats ###

    llama_user_prompt = {
        "math": "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem:\n\n",
    }
    chats = []
    questions = []
    answers = []
    if "llama" in model_name.lower() and dataset_name in llama_user_prompt:
            print("Using LLaMA user prompt format.")
    for item in tqdm(dataset, total=len(dataset), desc="Creating chats"):
        question, answer = get_question_answer(dataset_name, item)
        if "llama" in model_name.lower() and dataset_name in llama_user_prompt:
            chats.append([
                {
                    "role": "user",
                    "content": llama_user_prompt[dataset_name] + question,
                }
            ])
        else:
            chats.append([
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ])
        questions.append(question)
        answers.append(answer)
    
    ### Load Model and Run Inference ###

    model_path = model_name
    torch.cuda.empty_cache()
    gc.collect()
    
    if verbose:
        print(f"Loading model from {model_path}")

    temperature = kwargs.pop("temperature", None)
    max_tokens = kwargs.pop("max_tokens", 8192)
    if temperature is None:
        params = vllm.SamplingParams(max_tokens=max_tokens, seed=42)
    else:
        params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=42)
    print("PARAMS")
    print(params)

    model = vllm.LLM(model=model_path,
                     tokenizer=model_path,
                     hf_token=hf_token,
                     **kwargs
                    )
    if verbose:
        print("Model loaded")
        print(f"Running inference on {len(chats)} prompts")

    response = model.chat(chats, params)
    torch.cuda.empty_cache()
    gc.collect()

    if verbose:
        print("Inference completed")
    response = response[:len(dataset)] 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    completions = []
    traces = []
    extracted_answers = []
    scores = []
    finished = []
    
    for i, output in tqdm(enumerate(response), total=len(response), desc="Processing outputs"):
        completion = tokenizer.decode(output.prompt_token_ids + output.outputs[0].token_ids)
        trace = extract_trace(completion, model_name)
        extracted_answer = extract_answer(completion, dataset_name, model_name)
        score = grade_answer(dataset_name, extracted_answer, answers[i])
        
        completions.append(completion)
        traces.append(trace)
        extracted_answers.append(extracted_answer)
        scores.append(score)
        finished.append(output.finished)

    output = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "collected_on": str(datetime.now()),
            "system_prompt": system_prompt,
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        },
        "data": {
            "questions": questions,
            "completions": completions,
            "traces": traces,
            "extracted_answers": extracted_answers,
            "scores": scores,
            "ground_truth_answers": answers,
        }
    }
    if verbose:
        print("Finished!")
        
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to load from Hugging Face.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to load from disk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/traces/",
        help="Directory to save generated traces.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose logs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config file (YAML) with defaulty generation parameters.",
    )
    args = parser.parse_args()

    print("ARGS")
    print(args)
    ### Loading YAML Config ###

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
    call_kwargs = {}
    # copy vllm kwargs if present
    if isinstance(config.get("vllm_kwargs"), dict):
        call_kwargs.update(config["vllm_kwargs"])
    # sampling params
    if "temperature" in config:
        call_kwargs["temperature"] = config["temperature"]
    if "max_tokens" in config:
        call_kwargs["max_tokens"] = config["max_tokens"]

    print("CALL KWARGS")
    print(call_kwargs)

    if 'model_name' in config and args.model_name is None:
        args.model_name = config['model_name']
    if 'dataset_name' in config and args.dataset_name is None:
        args.dataset_name = config['dataset_name']
    assert args.model_name is not None, "Model name must be specified via --model_name or config file."
    assert args.dataset_name is not None, "Dataset name must be specified via --dataset_name or config file."

    output = generate_traces(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        verbose=args.verbose,
        **call_kwargs
    )

    output_dir = osp.join(args.output_dir, args.dataset_name.replace("/", "_"))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = osp.join(
        output_dir,
        f"traces_{args.model_name.replace('/', '_')}.pkl"
    )

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Generated traces saved to {output_path}")