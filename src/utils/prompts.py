"""
Prompt templates and formatting utilities for different datasets.
"""

default_prompt_gsm8k_cot = """You are a helpful assistant that solves grade school math problems step by step. Read the question and formulate a response. Please reason step by step.

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_variant_numbered = """You are a helpful assistant that solves grade-school math problems step by step. Read the question and produce a clear, numbered sequence of reasoning steps. Each step should be a short arithmetic/logical operation. At the end, give the final numeric answer on its own line.

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_variant_self_check = """You are a helpful assistant that solves grade-school math problems step by step. Solve the problem, then **review your work** and check each step for mistakes. If you find an error, correct it and show the corrected steps. Finally, present the final numeric answer on its own line. Please reason step by step.

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.

"""

prompt_variant_structured = """You are a helpful assistant that solves grade-school math problems using a goal-directed reasoning approach. First, restate the problem in your own words and identify the key quantities. Next, outline a short plan describing the intermediate goals needed to reach the solution. Then execute the plan step by step, showing intermediate results clearly.
Finally, present the final numeric answer on its own line. Please reason step by step.

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_variant_no_reasoning = """
You are a helpful assistant that solves grade school math problems. Read the question and compute the correct numeric answer. Do **NOT** provide any reasoning or intermediate steps, only provide the final answer. 

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_variant_completions_no_reasoning = """
You are a helpful assistant that solves grade school math problems. Using the given reasoning trace, compute the correct numeric answer. Do **NOT** provide any reasoning or intermediate steps, only provide the final answer. 

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_variant_completions_reasoning = """
You are a helpful assistant that solves grade school math problems. Using the given reasoning trace, compute the correct numeric answer. Continue the reasoning trace from the last step until you reach the final answer.

Formatting instructions:

Put your final answer within \\boxed{} at the end of your response.
"""

prompt_remove_answer = """
You are an AI assistant that cleans reasoning traces. 
Use the question and the answer given between <answer> and </answer> tags, and then remove the occurences of the answer from the trace. Remove the sentences having the answer from the trace, keep other steps and lines the same.

The question is given between <question> and </question> tags.
The answer is given between <answer> and </answer> tags.
The original trace is given between <trace> and </trace> tags. 

Formatting instructions:

Put your cleaned reasoning steps within <cleaned_trace> and </cleaned_trace> tags which doesn't contain the answer in any form.
"""