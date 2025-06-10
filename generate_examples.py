import random
import pandas as pd
from datasets import Dataset
from pyswip import Prolog
import re
import json

# System prompt from the original notebook
system_prompt = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""

def initialize_prolog():
    """Initialize a Prolog engine."""
    return Prolog()

def generate_arithmetic_logic_problem():
    """Generate a system of linear equations problem."""
    prolog = initialize_prolog()
    x = random.randint(1, 20)
    y = random.randint(1, 20)
    sum_xy = x + y
    diff_xy = x - y
    prompt = f"If x + y = {sum_xy} and x - y = {diff_xy}, find x and y."
    prolog_rules = f"equation(X, Y) :- X + Y = {sum_xy}, X - Y = {diff_xy}."
    prolog.assertz(prolog_rules)
    try:
        solution = list(prolog.query("equation(X, Y)"))[0]
        x_val, y_val = solution['X'], solution['Y']
        reasoning = (
            f"To solve x + y = {sum_xy} and x - y = {diff_xy}:\n"
            f"1. Add the equations: (x + y) + (x - y) = {sum_xy} + {diff_xy} => 2x = {sum_xy + diff_xy}.\n"
            f"2. Solve for x: x = {sum_xy + diff_xy}/2 = {x_val}.\n"
            f"3. Substitute x = {x_val} into x + y = {sum_xy}: {x_val} + y = {sum_xy} => y = {sum_xy - x_val}.\n"
            f"4. Thus, y = {y_val}."
        )
        solution_text = f"x = {x_val}, y = {y_val}"
        return {
            "prompt": prompt,
            "prolog_rules": prolog_rules,
            "reasoning": reasoning,
            "solution": solution_text,
            "type": "arithmetic_logic"
        }
    except Exception as e:
        return None  # Skip invalid problems

def generate_relational_reasoning_problem():
    """Generate a transitive relation problem."""
    prolog = initialize_prolog()
    names = random.sample(["Alice", "Bob", "Charlie", "David", "Eve"], 3)
    prompt = (
        f"{names[0]} is taller than {names[1]}. "
        f"{names[1]} is taller than {names[2]}. Who is the tallest?"
    )
    prolog_rules = (
        f"taller('{names[0]}', '{names[1]}').\n"
        f"taller('{names[1]}', '{names[2]}').\n"
        f"tallest(X) :- taller(X, Y), \\+ taller(_, X)."
    )
    for rule in prolog_rules.split("\n"):
        if rule.strip():
            prolog.assertz(rule)
    try:
        solution = list(prolog.query("tallest(X)"))[0]
        tallest = solution['X']
        reasoning = (
            f"To determine the tallest:\n"
            f"1. Given: {names[0]} is taller than {names[1]}, so {names[0]} > {names[1]}.\n"
            f"2. Given: {names[1]} is taller than {names[2]}, so {names[1]} > {names[2]}.\n"
            f"3. By transitivity, {names[0]} > {names[1]} > {names[2]}, so {names[0]} is taller than {names[2]}.\n"
            f"4. Since no one is taller than {names[0]}, {names[0]} is the tallest."
        )
        solution_text = tallest
        return {
            "prompt": prompt,
            "prolog_rules": prolog_rules,
            "reasoning": reasoning,
            "solution": solution_text,
            "type": "relational_reasoning"
        }
    except Exception as e:
        return None  # Skip invalid problems

def generate_constraint_satisfaction_problem():
    """Generate a simple scheduling problem."""
    prolog = initialize_prolog()
    tasks = ["Task A", "Task B"]
    durations = [random.randint(1, 5), random.randint(1, 5)]
    total_time = sum(durations) + random.randint(1, 3)
    prompt = (
        f"Schedule {tasks[0]} (duration {durations[0]} hours) and {tasks[1]} "
        f"(duration {durations[1]} hours) within {total_time} hours without overlap. "
        f"Find possible start times."
    )
    prolog_rules = (
        f"schedule(S1, S2) :- S1 >= 0, S2 >= 0, "
        f"S1 + {durations[0]} =< {total_time}, S2 + {durations[1]} =< {total_time}, "
        f"(S1 + {durations[0]} =< S2 ; S2 + {durations[1]} =< S1)."
    )
    prolog.assertz(prolog_rules)
    try:
        solutions = list(prolog.query("schedule(S1, S2)"))
        if not solutions:
            return None
        solution = random.choice(solutions)  # Pick one valid schedule
        s1, s2 = solution['S1'], solution['S2']
        reasoning = (
            f"To schedule {tasks[0]} and {tasks[1]}:\n"
            f"1. {tasks[0]} takes {durations[0]} hours, {tasks[1]} takes {durations[1]} hours, within {total_time} hours.\n"
            f"2. Ensure no overlap: {tasks[0]} starts at S1, ends at S1 + {durations[0]}; "
            f"{tasks[1]} starts at S2, ends at S2 + {durations[1]}.\n"
            f"3. Conditions: S1 + {durations[0]} ≤ S2 or S2 + {durations[1]} ≤ S1, and both end within {total_time} hours.\n"
            f"4. A valid schedule: S1 = {s1}, S2 = {s2} satisfies all constraints."
        )
        solution_text = f"{tasks[0]} starts at {s1}, {tasks[1]} starts at {s2}"
        return {
            "prompt": prompt,
            "prolog_rules": prolog_rules,
            "reasoning": reasoning,
            "solution": solution_text,
            "type": "constraint_satisfaction"
        }
    except Exception as e:
        return None  # Skip invalid problems

def format_for_grpo(sample):
    """Format a sample as Messages for GRPO."""
    if not sample:
        return None
    return {
        "Messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": (
                f"<start_working_out>{sample['reasoning']}<end_working_out>"
                f"<SOLUTION>{sample['solution']}</SOLUTION>"
            )}
        ],
        "prompt": sample["prompt"],
        "prolog_rules": sample["prolog_rules"],
        "reasoning": sample["reasoning"],
        "solution": sample["solution"],
        "type": sample["type"]
    }

def generate_dataset(num_samples=1000):
    """Generate a diverse Prolog-style dataset."""
    problem_generators = [
        generate_arithmetic_logic_problem,
        generate_relational_reasoning_problem,
        generate_constraint_satisfaction_problem
    ]
    samples = []
    target_per_type = num_samples // len(problem_generators)
    
    for generator in problem_generators:
        count = 0
        while count < target_per_type:
            sample = generator()
            formatted_sample = format_for_grpo(sample)
            if formatted_sample:
                samples.append(formatted_sample)
                count += 1
    
    # Fill remaining samples randomly
    while len(samples) < num_samples:
        generator = random.choice(problem_generators)
        sample = generator()
        formatted_sample = format_for_grpo(sample)
        if formatted_sample:
            samples.append(formatted_sample)
    
    # Convert to DataFrame and Dataset
    df = pd.DataFrame(samples)
    dataset = Dataset.from_pandas(df)
    
    # Add tokenized length for filtering (as in original notebook)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
    def compute_length(example):
        text = tokenizer.apply_chat_template(example["Messages"], tokenize=False)
        return {"N": len(tokenizer(text, add_special_tokens=False)["input_ids"])}
    dataset = dataset.map(compute_length)
    
    # Filter by sequence length (max_seq_length/2 = 1024, as in original)
    dataset = dataset.filter(lambda x: x["N"] <= 1024)
    
    return dataset

def save_dataset(dataset, output_path="prolog_style_dataset.json"):
    """Save dataset to JSON file."""
    dataset.to_json(output_path, orient="records", lines=True)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    # Generate dataset with 1000 samples
    dataset = generate_dataset(num_samples=1000)
    print(f"Generated dataset with {len(dataset)} samples")
    print("Sample types distribution:")
    print(pd.Series(dataset["type"]).value_counts())
    
    # Save dataset
    save_dataset(dataset)
    
    # Example of a sample
    print("\nExample sample:")
    sample = dataset[0]
    print(json.dumps(sample, indent=2))
