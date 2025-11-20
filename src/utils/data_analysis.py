import pickle
import pandas as pd
import os
import pathlib
import re

def read_traces_to_dataframe(pickle_path):
    """
    Read a pickle file containing trace data and create a pandas DataFrame.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        pandas.DataFrame: DataFrame with columns for each key in data["data"]
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract the data dictionary
    if "data" in data:
        data_dict = data["data"]
        # Convert to DataFrame
        df = pd.DataFrame(data_dict)
        return df
    else:
        raise ValueError("No 'data' key found in pickle file")

def extract_reasoning_trace(completion):
    """
    Extract content between <think> and </think> tags.
    
    Args:
        completion: String containing the completion text
        
    Returns:
        str: Extracted reasoning trace, or None if not found
    """
    if pd.isna(completion) or not isinstance(completion, str):
        return None

    # Case 1: <think> ... </think> (normal)
    pattern_full = r'<think>(.*?)</think>'
    match = re.search(pattern_full, completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Case 2: <think> present but </think> missing
    pattern_open = r'<think>(.*)$'
    match = re.search(pattern_open, completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def extract_predicted_answer(completion):
    """
    Extract the final answer from an LLM completion.

    Rules:
    1. Only look after the </think> tag. If </think> is missing, return None.
    2. Normal case: final answer comes after ####.
       - Extract the token immediately after ####
       - Stop at whitespace, newline, or <|im_end|>
    3. Edge case: if #### is immediately followed by <|im_end|>,
       - Extract the token immediately before ####
    """
    if not isinstance(completion, str):
        return None

    # Find </think>
    end_think = re.search(r'</think>', completion)
    if not end_think:
        return None

    text_after = completion[end_think.end():].lstrip()

    # Extract answer from \boxed{ ... }
    ANS_RE = re.compile(r'\\boxed\{([^}]*)\}')
    INVALID_ANS = "[invalid]"


    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS
    return extract_answer(text_after)


def extract_ground_truth_answer(completion):
    """
    Extract the final answer from an the ground truth response.

    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"


    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS
    return extract_answer(completion)


def main(model_name):
    # root_dir = pathlib.Path("/data/user_data/ssridha4/legible-traces-RL")
    root_dir = pathlib.Path("/user_data/mdhawan/projects/categorization/code/external/legible-traces-RL")## "/data/user_data/ssridha4/legible-traces-RL"

    traces_dir = root_dir / "outputs" / "traces" / "gsm8k" / "test"

    # List of file variants to process
    variants = ["default", "numbered", "self_check", "structured"]

    # Dictionary to store dataframes
    dataframes = {}

    # Read each file and create a dataframe
    for variant in variants:
        filename = f"traces_{model_name}_{variant}.pkl"
        filepath = traces_dir / filename
        
        print(f"Reading {filename}...")
        df = read_traces_to_dataframe(str(filepath))
        dataframes[variant] = df
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()

    
    # Extract reasoning_trace and predicted_answers for each dataframe
    for variant, df in dataframes.items():
        print(f"Processing {variant}...")
        
        # Extract reasoning_trace
        df['reasoning_trace'] = df['completions'].apply(extract_reasoning_trace)
        
        # Extract predicted_answers
        df['predicted_answers'] = df['completions'].apply(extract_predicted_answer)
        
        print(f"  Extracted reasoning_trace: {df['reasoning_trace'].notna().sum()} / {len(df)}")
        print(f"  Extracted predicted_answers: {df['predicted_answers'].notna().sum()} / {len(df)}")
        print()
    
    # get the average length of the traces
    for variant, df in dataframes.items():
        print(f"Average length of traces for {variant}: {df['reasoning_trace'].apply(len).mean()}")
    
    # get the accuracy of the predicted answers
    for variant, df in dataframes.items():
        df['gt_answers'] = df['ground_truth_answers'].apply(extract_ground_truth_answer)
        df['correct'] = df['predicted_answers'] == df['gt_answers']
        accuracy = df['correct'].mean()
        print(f"Accuracy of predicted answers for {variant}: {accuracy}")

    
    output_dir = root_dir / "outputs" / "data" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, df in dataframes.items():
        output_path = output_dir / f"dataframes_{variant}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {variant} to {output_path}")

    
if __name__ == "__main__":
    model_name = "Qwen_Qwen3-8B"
    main(model_name)