"""
Complete Minimal Example: Math + Logic GRPO Training
This is a self-contained example that can be run to test the integration
"""

import torch
import random
import re
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd

# === Part 1: Define all formatting and system components ===

# Formatting tags
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
logic_start = "<logical_reasoning>"
logic_end = "</logical_reasoning>"
step_start = "<step>"
step_end = "</step>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Unified system prompt
system_prompt = """You are given a problem that may require mathematical or logical reasoning.
Think about the problem and provide your working out.
For mathematical problems: Place calculations between <start_working_out> and <end_working_out>.
For logical problems: Place your reasoning between <logical_reasoning> and </logical_reasoning>, marking each inference step with <step> and </step>.
Then, provide your solution between <SOLUTION></SOLUTION>"""

# === Part 2: Problem Generators ===

def generate_math_problem():
    """Generate a simple math problem compatible with existing format."""
    x = random.randint(5, 25)
    y = random.randint(5, 25)
    
    problem = {
        "prompt": f"If x + y = {x + y} and x - y = {x - y}, what are the values of x and y?",
        "answer": f"x = {x}, y = {y}",
        "type": "math"
    }
    
    # Generate the expected completion
    reasoning = f"""{reasoning_start}
To solve this system of equations:
x + y = {x + y}
x - y = {x - y}

Adding the equations: 2x = {x + y + x - y}
So x = {x}

Substituting back: {x} + y = {x + y}
So y = {y}
{reasoning_end}"""
    
    problem["completion"] = f"{reasoning}\n{solution_start}{problem['answer']}{solution_end}"
    return problem

def generate_logic_problem():
    """Generate a simple logical reasoning problem."""
    names = random.sample(["Alice", "Bob", "Charlie", "David", "Eve"], 3)
    relation = random.choice(["taller than", "older than", "faster than"])
    
    problem = {
        "prompt": f"{names[0]} is {relation} {names[1]}. {names[1]} is {relation} {names[2]}. Who is the most {relation.split()[0]}?",
        "answer": names[0],
        "type": "logic"
    }
    
    # Generate the expected completion with logical steps
    reasoning = f"""{logic_start}
{step_start}Given: {names[0]} is {relation} {names[1]}{step_end}
{step_start}Given: {names[1]} is {relation} {names[2]}{step_end}
{step_start}By transitivity: {names[0]} > {names[1]} > {names[2]}{step_end}
{step_start}Therefore: {names[0]} is the most {relation.split()[0]}{step_end}
{logic_end}"""
    
    problem["completion"] = f"{reasoning}\n{solution_start}{problem['answer']}{solution_end}"
    return problem

def generate_hybrid_problem():
    """Generate a problem requiring both math and logic."""
    people = random.sample(["Alice", "Bob", "Charlie"], 3)
    base = random.randint(10, 30)
    
    problem = {
        "prompt": f"{people[0]} has ${base}. {people[1]} has twice as much as {people[0]}. "
                 f"{people[2]} has $15 more than {people[1]}. Who has the most money and how much?",
        "answer": f"{people[2]} with ${base * 2 + 15}",
        "type": "hybrid"
    }
    
    reasoning = f"""{logic_start}
{step_start}Given: {people[0]} has ${base}{step_end}
{step_start}Calculate: {people[1]} has 2 × ${base} = ${base * 2}{step_end}
{step_start}Calculate: {people[2]} has ${base * 2} + $15 = ${base * 2 + 15}{step_end}
{step_start}Compare amounts: ${base} < ${base * 2} < ${base * 2 + 15}{step_end}
{step_start}Therefore: {people[2]} has the most with ${base * 2 + 15}{step_end}
{logic_end}"""
    
    problem["completion"] = f"{reasoning}\n{solution_start}{problem['answer']}{solution_end}"
    return problem

# === Part 3: Reward Functions ===

def evaluate_format_unified(completions, **kwargs):
    """Unified format checker for all problem types."""
    scores = []
    
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        
        # Check for proper reasoning format
        has_math_format = reasoning_start in response and reasoning_end in response
        has_logic_format = logic_start in response and logic_end in response
        
        if has_math_format or has_logic_format:
            score += 2.0
            
            # Extra points for logical steps
            if has_logic_format and step_start in response:
                num_steps = response.count(step_start)
                score += min(num_steps * 0.5, 2.0)
        
        # Check for solution format
        if solution_start in response and solution_end in response:
            score += 1.0
        else:
            score -= 1.0
        
        scores.append(score)
    
    return scores

def evaluate_correctness_unified(prompts, completions, answer, **kwargs):
    """Unified correctness checker."""
    scores = []
    
    # Regex to extract solution
    solution_regex = re.compile(
        rf"{solution_start}(.*?){solution_end}",
        re.MULTILINE | re.DOTALL
    )
    
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        
        # Extract predicted answer
        match = solution_regex.search(response)
        if not match:
            scores.append(-2.0)
            continue
        
        predicted = match.group(1).strip()
        expected = str(answer[i]) if isinstance(answer, list) else str(answer)
        
        # Normalize for comparison
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Check various forms of correctness
        if predicted_lower == expected_lower:
            score = 5.0
        elif expected_lower in predicted_lower:
            score = 3.0
        elif all(word in predicted_lower for word in expected_lower.split()):
            score = 2.0
        else:
            score = -1.0
        
        scores.append(score)
    
    return scores

def evaluate_reasoning_quality(prompts, completions, **kwargs):
    """Evaluate the quality of reasoning."""
    scores = []
    
    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        
        # Get the prompt text
        if prompts and i < len(prompts):
            if isinstance(prompts[i], list):
                prompt_text = prompts[i][-1]["content"]
            else:
                prompt_text = str(prompts[i])
            
            # Extract key terms from prompt
            prompt_names = set(re.findall(r'\b[A-Z][a-z]+\b', prompt_text))
            prompt_numbers = set(re.findall(r'\b\d+\b', prompt_text))
            
            # Check reasoning section
            reasoning_match = re.search(
                rf"({logic_start}.*?{logic_end}|{reasoning_start}.*?{reasoning_end})",
                response, re.DOTALL
            )
            
            if reasoning_match:
                reasoning_text = reasoning_match.group(1)
                
                # Check if key terms appear in reasoning
                reasoning_names = set(re.findall(r'\b[A-Z][a-z]+\b', reasoning_text))
                reasoning_numbers = set(re.findall(r'\b\d+\b', reasoning_text))
                
                # Calculate relevance
                name_overlap = len(prompt_names & reasoning_names) / max(len(prompt_names), 1)
                number_overlap = len(prompt_numbers & reasoning_numbers) / max(len(prompt_numbers), 1)
                
                score += (name_overlap + number_overlap) * 2.0
                
                # Bonus for structured reasoning
                if step_start in reasoning_text:
                    score += 1.0
                
                # Length check (not too short, not too long)
                word_count = len(reasoning_text.split())
                if 30 <= word_count <= 200:
                    score += 0.5
        
        scores.append(min(score, 5.0))
    
    return scores

# === Part 4: Create Mixed Dataset ===

def create_mixed_dataset(num_samples=100):
    """Create a dataset with mixed problem types."""
    generators = [generate_math_problem, generate_logic_problem, generate_hybrid_problem]
    data = []
    
    for i in range(num_samples):
        # Rotate through problem types for balance
        generator = generators[i % len(generators)]
        problem = generator()
        
        # Format as messages
        formatted = {
            "Messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem["prompt"]},
                {"role": "assistant", "content": problem["completion"]}
            ],
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem["prompt"]}
            ],
            "answer": problem["answer"],
            "type": problem["type"]
        }
        data.append(formatted)
    
    # Convert to dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    return dataset

# === Part 5: Mock GRPO Training Loop ===

def mock_grpo_training_step(model, batch, reward_funcs):
    """Simulate one GRPO training step."""
    
    # Generate multiple completions per prompt
    num_generations = 4
    all_completions = []
    all_prompts = []
    all_answers = []
    
    for item in batch:
        prompts = [item["prompt"]] * num_generations
        answers = [item["answer"]] * num_generations
        
        # Mock generation (in real implementation, this would use the model)
        completions = []
        for _ in range(num_generations):
            # Simulate varying quality responses
            quality = random.choice(["good", "medium", "bad"])
            
            if quality == "good":
                # Use the ground truth completion
                completion = item["Messages"][2]["content"]
            elif quality == "medium":
                # Partial completion
                completion = item["Messages"][2]["content"].split(solution_start)[0] + \
                           f"{solution_start}Some answer{solution_end}"
            else:
                # Bad completion
                completion = "I don't know the answer."
            
            completions.append([{"content": completion}])
        
        all_completions.extend(completions)
        all_prompts.extend(prompts)
        all_answers.extend(answers)
    
    # Calculate rewards for all completions
    total_rewards = []
    
    for reward_func in reward_funcs:
        rewards = reward_func(
            prompts=all_prompts,
            completions=all_completions,
            answer=all_answers
        )
        total_rewards.append(rewards)
    
    # Sum rewards across all functions
    final_rewards = np.sum(total_rewards, axis=0)
    
    # Calculate advantages (GRPO style)
    # Group rewards by prompt
    rewards_per_prompt = []
    for i in range(0, len(final_rewards), num_generations):
        rewards_per_prompt.append(final_rewards[i:i+num_generations])
    
    # Compute advantages
    advantages = []
    for group_rewards in rewards_per_prompt:
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards) + 1e-8
        group_advantages = (group_rewards - mean_reward) / std_reward
        advantages.extend(group_advantages)
    
    return {
        "rewards": final_rewards.tolist(),
        "advantages": advantages,
        "mean_reward": np.mean(final_rewards)
    }

# === Part 6: Main Execution ===

def main():
    """Run the complete example."""
    print("=== Mixed Math + Logic GRPO Training Example ===\n")
    
    # Create dataset
    print("Creating mixed dataset...")
    dataset = create_mixed_dataset(num_samples=12)
    
    # Show dataset composition
    type_counts = pd.Series([item["type"] for item in dataset]).value_counts()
    print(f"Dataset composition:\n{type_counts}\n")
    
    # Show example problems
    print("Example problems:")
    for problem_type in ["math", "logic", "hybrid"]:
        example = next(item for item in dataset if item["type"] == problem_type)
        print(f"\n{problem_type.upper()}:")
        print(f"Q: {example['Messages'][1]['content']}")
        print(f"A: {example['answer']}")
    
    # Define reward functions
    reward_funcs = [
        evaluate_format_unified,
        evaluate_correctness_unified,
        evaluate_reasoning_quality
    ]
    
    # Simulate training
    print("\n=== Simulating GRPO Training ===")
    
    # Mock model (in practice, this would be your fine-tuned model)
    model = None
    
    # Training loop
    num_steps = 3
    batch_size = 4
    
    for step in range(num_steps):
        print(f"\nStep {step + 1}/{num_steps}")
        
        # Get batch
        batch_indices = random.sample(range(len(dataset)), batch_size)
        batch = [dataset[i] for i in batch_indices]
        
        # Run mock training step
        results = mock_grpo_training_step(model, batch, reward_funcs)
        
        print(f"Mean reward: {results['mean_reward']:.2f}")
        print(f"Reward range: [{min(results['rewards']):.2f}, {max(results['rewards']):.2f}]")
        
        # Show advantages for one prompt
        print(f"Example advantages: {results['advantages'][:4]}")
    
    print("\n=== Testing Reward Functions ===")
    
    # Test on specific examples
    test_cases = [
        {
            "prompt": [{"role": "user", "content": "Alice is taller than Bob. Bob is taller than Charlie. Who is tallest?"}],
            "completion": [{"content": f"{logic_start}\n{step_start}Alice > Bob{step_end}\n{step_start}Bob > Charlie{step_end}\n{step_start}Therefore: Alice is tallest{step_end}\n{logic_end}\n{solution_start}Alice{solution_end}"}],
            "answer": "Alice",
            "expected": "Good logical reasoning"
        },
        {
            "prompt": [{"role": "user", "content": "If x + y = 10 and x - y = 2, find x and y."}],
            "completion": [{"content": f"{reasoning_start}\nAdding: 2x = 12, so x = 6\nSubtracting: y = 4\n{reasoning_end}\n{solution_start}x = 6, y = 4{solution_end}"}],
            "answer": "x = 6, y = 4",
            "expected": "Good math reasoning"
        },
        {
            "prompt": [{"role": "user", "content": "Who is taller?"}],
            "completion": [{"content": "I don't know."}],
            "answer": "Unknown",
            "expected": "Bad response"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}: {test['expected']}")
        
        format_score = evaluate_format_unified([test["completion"]])[0]
        correct_score = evaluate_correctness_unified([test["prompt"]], [test["completion"]], [test["answer"]])[0]
        quality_score = evaluate_reasoning_quality([test["prompt"]], [test["completion"]])[0]
        
        total = format_score + correct_score + quality_score
        print(f"Scores - Format: {format_score:.1f}, Correct: {correct_score:.1f}, Quality: {quality_score:.1f}")
        print(f"Total: {total:.1f}")
    
    print("\n=== Integration Complete ===")
    print("\nNext steps:")
    print("1. Replace mock generation with actual model.generate()")
    print("2. Add this to your GRPO trainer")
    print("3. Monitor performance on both math and logic tasks")
    print("4. Adjust reward weights based on results")

if __name__ == "__main__":
    main()
