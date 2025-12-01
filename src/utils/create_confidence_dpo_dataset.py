import argparse
import glob
import os
import pickle
import datasets
import json

from data_analysis import extract_predicted_answer, extract_ground_truth_answer

prompt_variants_llm = ["default", "numbered", "self_check", "structured"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create DPO dataset for evaluation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Name of the dataset to load from Hugging Face.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default=None,
        help="Name of the LLM to load from Hugging Face.",
    )
    parser.add_argument(
        "--slm_name",
        type=str,
        default=None,
        help="Name of the SLM to load from Hugging Face.",
    )
    parser.add_argument(
        "--trace_len",
        type=float,
        default=None,
        help="Trace length to use  in the prompt. (0.5, 0.75, 1)",
    )
    parser.add_argument(
        "--slm_prompt_variant",
        type=str,
        default=None,
        help="Prompt variant to use for the SLM  (completions_no_reasoning, completions_reasoning).",
        choices=["completions_no_reasoning", "completions_reasoning"],
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./outputs",
        help="Base directory",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Dataset split to use (e.g., 'train', 'test'). Default depends on dataset.",
    )
    args = parser.parse_args()

    extracted_traces_dir = os.path.join(args.base_dir, "extracted_traces", args.dataset_name, args.llm_name, args.dataset_split)
    completions_dir = os.path.join(extracted_traces_dir, "completions", args.slm_name, "length", f"trace_length_{args.trace_len}", f"trace_{args.slm_prompt_variant}")

    completions_files = glob.glob(os.path.join(completions_dir, "*.pkl"))
    extracted_traces_files = glob.glob(os.path.join(extracted_traces_dir, "*.csv"))

    completions_files = [f for f in completions_files if "no_reasoning" not in f.split("/")[-1]]
    completions_files = [f for f in completions_files if "logprobs" in f.split("/")[-1]]
    extracted_traces_files = [f for f in extracted_traces_files if "no_reasoning" not in f.split("/")[-1]]

    assert len(completions_files) == len(extracted_traces_files), f"Number of completions and extracted traces files must be the same: {len(completions_files)} != {len(extracted_traces_files)}"

    correct_llm_prompt_variant = []
    incorrect_llm_prompt_variant = []

    # ---- Load all PKL files ----
    all_slm_data = []
    for prompt_variant_llm in prompt_variants_llm:
        with open(f"{completions_dir}/completions_logprobs_{prompt_variant_llm}.pkl", "rb") as f:
            data = pickle.load(f)["data"]
            all_slm_data.append(data)
    
    # ---- Assert that all files have the same length ----
    lengths = [len(d["completions"]) for d in all_slm_data]
    assert len(set(lengths)) == 1, f"Completions Files have mismatched lengths: {lengths}"

    N = lengths[0]
    print(N)
    print("Completions Files have consistent length =", N)

    # ---- Traverse all PKL lists together ----
    for idx in range(N):
        # Extract the index-th element from each file
        items = [{"completions": d["completions"][idx], "ground_truth_answers": d["ground_truth_answers"][idx], "confidence": d["confidence"][idx]} for d in all_slm_data]   # each item is a dictionary

        predicted_answers = [extract_predicted_answer(item["completions"]) for item in items]
        ground_truth_answers = [extract_ground_truth_answer(item["ground_truth_answers"]) for item in items]
        correct = [a == b for a, b in zip(predicted_answers, ground_truth_answers)]

        current_correct_llm_prompt_variant = []
        current_incorrect_llm_prompt_variant = []
        for prompt_idx, is_correct in enumerate(correct):
            if is_correct:
                current_correct_llm_prompt_variant.append((prompt_variants_llm[prompt_idx], items[prompt_idx]["confidence"]))
            else:
                current_incorrect_llm_prompt_variant.append((prompt_variants_llm[prompt_idx], items[prompt_idx]["confidence"]))

        correct_llm_prompt_variant.append(current_correct_llm_prompt_variant)
        incorrect_llm_prompt_variant.append(current_incorrect_llm_prompt_variant)

    all_llm_data = {}
    for prompt_variant_llm in prompt_variants_llm:
        all_llm_data[prompt_variant_llm] = datasets.load_dataset("csv", data_files=os.path.join(extracted_traces_dir, f"extracted_traces_{prompt_variant_llm}.csv"), split="train")

    # ---- Assert that all files have the same length ----
    lengths = [len(all_llm_data[prompt_variant_llm]) for prompt_variant_llm in prompt_variants_llm]
    assert len(set(lengths)) == 1, f"Extracted Files have mismatched lengths: {lengths}"

    N_llm = lengths[0]
    assert N_llm == N, f"Extracted Files and Completions Files have mismatched lengths: {N_llm} != {N}"
    print("Extracted Files have consistent length =", N_llm)

    # ---- Traverse all CSV lists together ----
    dpo_data = []
    total_examples = 0
    skipped_examples = 0
    no_reasoning_examples = 0
    no_correct_examples = 0
    no_confidence_examples = 0
    for idx in range(N_llm):
        correct_prompt_variants, incorrect_prompt_variants = correct_llm_prompt_variant[idx], incorrect_llm_prompt_variant[idx]
        correct_llm_data = [(all_llm_data[prompt_variant_llm][idx], confidence) for prompt_variant_llm, confidence in correct_prompt_variants]
        incorrect_llm_data = [(all_llm_data[prompt_variant_llm][idx], confidence) for prompt_variant_llm, confidence in incorrect_prompt_variants]
        
        if len(correct_llm_data) > 0:
            # print(correct_llm_data[0].keys())
            confidence_scores = [confidence for _, confidence in correct_llm_data]
            chosen_idx = confidence_scores.index(max(confidence_scores))
            if len(set(confidence_scores)) == 1 and len(correct_llm_data) > 1:
                print(confidence_scores)
                print(correct_llm_data)
                print(f"Skipping example {idx} because it has equal confidence scores")
                no_confidence_examples += 1
                continue
            for i, (correct_data, confidence) in enumerate(correct_llm_data):
                if i == chosen_idx:
                    continue
                if len(correct_llm_data[chosen_idx][0]["reasoning_trace"]) == 0:
                    print(f"Skipping example {idx} because it has no reasoning trace")
                    no_reasoning_examples += 1
                    skipped_examples += 1
                    continue
                if len(correct_data["reasoning_trace"]) > 0:
                    dpo_data.append({
                            "prompt": correct_data["questions"],
                            "chosen": correct_llm_data[chosen_idx][0]["reasoning_trace"],
                            "rejected": correct_data["reasoning_trace"],
                            "chosen_score": correct_llm_data[chosen_idx][1],
                            "rejected_score": confidence,
                        })
                else:
                    print(f"Skipping example {idx} because it has no reasoning trace")
                    no_reasoning_examples += 1
                    skipped_examples += 1
                    continue
            for incorrect_data, confidence in incorrect_llm_data:
                if len(incorrect_data["reasoning_trace"]) > 0:
                    dpo_data.append({
                        "prompt": correct_llm_data[chosen_idx][0]["questions"],
                        "chosen": correct_llm_data[chosen_idx][0]["reasoning_trace"],
                        "rejected": incorrect_data["reasoning_trace"],
                        "chosen_score": correct_llm_data[chosen_idx][1],
                        "rejected_score": -confidence,
                    })
                else:
                    print(f"Skipping example {idx} because it has no reasoning trace")
                    no_reasoning_examples += 1
                    skipped_examples += 1
                    continue
        else:
            no_correct_examples += 1
            print(f"Skipping example {idx} because it has no correct LLM prompt variants")
            skipped_examples += 1
            continue
        
        total_examples += 1
    
    print(f"Total examples: {total_examples}")
    print(f"Skipped examples: {skipped_examples}")
    print(f"DPO data length: {len(dpo_data)}")
    print(f"No reasoning examples: {no_reasoning_examples}")
    print(f"No correct examples: {no_correct_examples}")
    print(f"No confidence examples: {no_confidence_examples}")
    # Save DPO data jsonl 
    dpo_data_path = os.path.join(completions_dir, "confidence_preference_data.jsonl")
    with open(dpo_data_path, "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved DPO data to {dpo_data_path}")
