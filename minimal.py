import random
import re
from typing import List, Dict, Any

# --- 1. System Prompt and Formatting Tags ---

# A unified system prompt for both mathematical and logical reasoning tasks.
SYSTEM_PROMPT = """You are an expert in both mathematical and logical reasoning. 
Think step-by-step. For math problems, use <math_reasoning> tags. 
For logic problems, use <logic_reasoning> tags with <step> markers.
Finally, provide your answer inside <solution> tags."""

# Define the special tokens used for formatting the model's output.
LOGIC_START = "<logic_reasoning>"
LOGIC_END = "</logic_reasoning>"
STEP_START = "<step>"
STEP_END = "</step>"
MATH_START = "<math_reasoning>"
MATH_END = "</math_reasoning>"
SOLUTION_START = "<solution>"
SOLUTION_END = "</solution>"


# --- 2. Simplified Dataset Generation ---

def generate_problem() -> Dict[str, Any]:
    """
    Generates a simple reasoning problem, either arithmetic or logical.
    Each problem includes a prompt, the correct answer, and its type.
    """
    if random.choice([True, False]):
        # Generate an arithmetic problem (system of equations)
        x, y = random.randint(5, 25), random.randint(5, 25)
        return {
            "prompt": f"If x + y = {x + y} and x - y = {x - y}, what are the values of x and y?",
            "answer": f"x = {x}, y = {y}",
            "type": "arithmetic"
        }
    else:
        # Generate a logical reasoning problem (transitive relation)
        names = random.sample(["Alex", "Ben", "Chris"], 3)
        return {
            "prompt": f"{names[0]} is taller than {names[1]}. {names[1]} is taller than {names[2]}. Who is the tallest?",
            "answer": names[0],
            "type": "logic"
        }

# --- 3. Simplified Reward Evaluation ---

class PrologRewardEvaluator:
    """
    A simplified evaluator that provides multi-dimensional rewards for model outputs.
    It assesses structure, correctness, and reasoning quality.
    """

    def evaluate_structure(self, response: str, problem_type: str) -> float:
        """Rewards responses that use the correct formatting tags."""
        if problem_type == "logic":
            if LOGIC_START in response and LOGIC_END in response:
                # Bonus for using step markers
                return 2.0 + (0.5 * response.count(STEP_START))
            return -1.0  # Penalize for wrong structure
        if problem_type == "arithmetic":
            return 2.0 if MATH_START in response and MATH_END in response else -1.0
        return 0.0

    def evaluate_solution_correctness(self, response: str, correct_answer: str) -> float:
        """Rewards responses that have the correct final answer."""
        match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", response, re.DOTALL)
        if not match:
            return -2.0  # Penalize for missing solution tag

        predicted_answer = match.group(1).strip()
        return 5.0 if predicted_answer == correct_answer else -0.5

    def evaluate_reasoning_quality(self, response: str, prompt: str) -> float:
        """
        Rewards responses where the reasoning is relevant to the prompt.
        This is a simple proxy for checking if the reasoning makes sense.
        """
        # Extract key terms from the prompt (simple nouns and numbers)
        prompt_terms = set(re.findall(r'[A-Za-z]+|\d+', prompt))
        
        # Extract reasoning content
        logic_match = re.search(rf"{LOGIC_START}(.*?){LOGIC_END}", response, re.DOTALL)
        math_match = re.search(rf"{MATH_START}(.*?){MATH_END}", response, re.DOTALL)
        
        reasoning_content = ""
        if logic_match:
            reasoning_content = logic_match.group(1)
        elif math_match:
            reasoning_content = math_match.group(1)
        
        if not reasoning_content:
            return 0.0
            
        reasoning_terms = set(re.findall(r'[A-Za-z]+|\d+', reasoning_content))
        
        # Reward based on the overlap of terms between prompt and reasoning
        overlap = len(prompt_terms.intersection(reasoning_terms))
        return min(float(overlap) / 3.0, 1.0) # Normalize and cap the reward

    def get_total_reward(self, response: str, problem: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the total reward by summing scores from all evaluation heads.
        """
        structure_score = self.evaluate_structure(response, problem["type"])
        correctness_score = self.evaluate_solution_correctness(response, problem["answer"])
        quality_score = self.evaluate_reasoning_quality(response, problem["prompt"])

        total_score = structure_score + correctness_score + quality_score
        
        return {
            "structure": structure_score,
            "correctness": correctness_score,
            "quality": quality_score,
            "total": total_score
        }

# --- 4. Mock Fine-Tuning Loop ---

def run_end_to_end_example():
    """
    Simulates the fine-tuning process for a single problem.
    1. Generates a problem.
    2. Creates a set of mock model responses (good, bad, and mediocre).
    3. Evaluates each response using the multi-dimensional reward function.
    4. Selects the best response based on the total reward score.
    """
    print("--- [Step 1: Generating a Problem] ---")
    problem = generate_problem()
    print(f"Problem Type: {problem['type'].capitalize()}")
    print(f"Prompt: {problem['prompt']}")
    print(f"Expected Answer: {problem['answer']}\n")

    print("--- [Step 2: Simulating Model Responses] ---")
    
    # Create a set of mock responses to simulate a GRPO group
    mock_responses = []
    if problem['type'] == 'logic':
        names = re.findall(r'([A-Z][a-z]+)', problem['prompt'])
        mock_responses = [
            # Good response: correct structure, reasoning, and solution
            f"{LOGIC_START}\n{STEP_START}{names[0]} > {names[1]}{STEP_END}\n{STEP_START}{names[1]} > {names[2]}{STEP_END}\n{STEP_START}Therefore, by transitivity, {names[0]} is the tallest.{STEP_END}\n{LOGIC_END}\n{SOLUTION_START}{problem['answer']}{SOLUTION_END}",
            # Mediocre response: correct answer but poor structure
            f"I think the answer is {problem['answer']}. {SOLUTION_START}{problem['answer']}{SOLUTION_END}",
            # Bad response: wrong answer and wrong structure
            f"{MATH_START}Wrong reasoning type.{MATH_END}\n{SOLUTION_START}{names[2]}{SOLUTION_END}"
        ]
    else: # arithmetic
        x_val = int(re.search(r'x = (\d+)', problem['answer']).group(1))
        y_val = int(re.search(r'y = (\d+)', problem['answer']).group(1))
        mock_responses = [
            # Good response: correct structure, reasoning, and solution
            f"{MATH_START}\n(x+y) + (x-y) = {x_val+y_val} + {x_val-y_val}\n2x = {(x_val+y_val) + (x_val-y_val)}\nx = {x_val}\n{MATH_END}\n{SOLUTION_START}{problem['answer']}{SOLUTION_END}",
            # Mediocre response: correct answer but no reasoning
            f"{SOLUTION_START}{problem['answer']}{SOLUTION_END}",
            # Bad response: wrong answer and missing tags
            f"The answer is x = {y_val}, y = {x_val}. {SOLUTION_START}x = {y_val}, y = {x_val}{SOLUTION_END}"
        ]

    for i, resp in enumerate(mock_responses):
        print(f"[Response {i+1}]:\n{resp}\n")

    print("--- [Step 3: Evaluating Responses with Reward Function] ---")
    evaluator = PrologRewardEvaluator()
    evaluated_responses = []

    for i, response in enumerate(mock_responses):
        reward_scores = evaluator.get_total_reward(response, problem)
        evaluated_responses.append({"response": response, "scores": reward_scores})
        print(f"[Response {i+1} Scores]:")
        for key, value in reward_scores.items():
            print(f"  - {key.capitalize():<12}: {value:.2f}")
        print("-" * 20)

    print("\n--- [Step 4: Selecting the Best Response (GRPO Objective)] ---")
    # In a real GRPO implementation, these rewards would be used to calculate advantages
    # and update the policy. Here, we simply select the response with the highest total score.
    best_response = max(evaluated_responses, key=lambda x: x['scores']['total'])
    
    print(f"The response with the highest total reward ({best_response['scores']['total']:.2f}) is selected as the winner for this step.")
    print("\n[Winning Response]:")
    print(best_response['response'])
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    run_end_to_end_example()
