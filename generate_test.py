import pytest
import os
import json
from datasets import Dataset

# Import functions from the main script
from dataset_generator import (
    generate_arithmetic_logic_problem,
    generate_relational_reasoning_problem,
    generate_constraint_satisfaction_problem,
    format_for_grpo,
    generate_dataset,
    save_dataset,
    system_prompt
)

# --- Constants for Testing ---
EXPECTED_KEYS = ["prompt", "prolog_rules", "reasoning", "solution", "type"]
EXPECTED_FORMATTED_KEYS = ["Messages", "prompt", "prolog_rules", "reasoning", "solution", "type"]

# --- Unit Tests for Problem Generators ---

def test_generate_arithmetic_logic_problem():
    """Tests the structure and validity of the arithmetic logic problem generator."""
    problem = generate_arithmetic_logic_problem()
    assert problem is not None
    assert all(key in problem for key in EXPECTED_KEYS)
    assert problem["type"] == "arithmetic_logic"
    
    # Verify content logic
    # Example: "If x + y = 15 and x - y = 5, find x and y." -> solution must contain x = 10, y = 5
    x_val = int(re.search(r"x = (\d+)", problem["solution"]).group(1))
    y_val = int(re.search(r"y = (\d+)", problem["solution"]).group(1))
    sum_val = int(re.search(r"x \+ y = (\d+)", problem["prompt"]).group(1))
    diff_val = int(re.search(r"x - y = (-?\d+)", problem["prompt"]).group(1))
    assert x_val + y_val == sum_val
    assert x_val - y_val == diff_val

def test_generate_relational_reasoning_problem():
    """Tests the structure and validity of the relational reasoning problem generator."""
    problem = generate_relational_reasoning_problem()
    assert problem is not None
    assert all(key in problem for key in EXPECTED_KEYS)
    assert problem["type"] == "relational_reasoning"
    
    # Verify content logic: the first name in the prompt is always the tallest
    match = re.search(r"(\w+) is taller than", problem["prompt"])
    assert match is not None
    tallest_in_prompt = match.group(1)
    assert problem["solution"] == tallest_in_prompt

def test_generate_constraint_satisfaction_problem():
    """Tests the structure and validity of the constraint satisfaction problem generator."""
    problem = generate_constraint_satisfaction_problem()
    # This can sometimes fail to find a solution if random values align poorly.
    # We test that it returns either a valid dict or None.
    if problem:
        assert all(key in problem for key in EXPECTED_KEYS)
        assert problem["type"] == "constraint_satisfaction"
        assert "starts at" in problem["solution"]

# --- Unit Tests for Formatter ---

def test_format_for_grpo_structure_and_content():
    """Tests that the GRPO formatter creates the correct structure."""
    sample_problem = {
        "prompt": "Test prompt?",
        "reasoning": "Test reasoning.",
        "solution": "Test solution.",
        "prolog_rules": "test_rule.",
        "type": "test_type"
    }
    formatted = format_for_grpo(sample_problem)
    assert formatted is not None
    assert all(key in formatted for key in EXPECTED_FORMATTED_KEYS)
    
    # Check Messages structure
    messages = formatted["Messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system" and messages[0]["content"] == system_prompt
    assert messages[1]["role"] == "user" and messages[1]["content"] == "Test prompt?"
    assert messages[2]["role"] == "assistant"
    assert "<start_working_out>Test reasoning.<end_working_out>" in messages[2]["content"]
    assert "<SOLUTION>Test solution.</SOLUTION>" in messages[2]["content"]

def test_format_for_grpo_handles_none():
    """Tests that the formatter correctly handles None input."""
    assert format_for_grpo(None) is None

# --- Integration Tests for Dataset Generation ---

@pytest.fixture(scope="module")
def small_dataset():
    """Fixture to generate a small dataset once for all tests in this module."""
    return generate_dataset(num_samples=30, max_seq_length=2048)

def test_generate_dataset_output_type_and_columns(small_dataset):
    """Tests the overall dataset generation process for output type and structure."""
    assert isinstance(small_dataset, Dataset)
    assert len(small_dataset) > 0 and len(small_dataset) <= 30
    
    # Check that all expected columns are present
    expected_cols = set(EXPECTED_FORMATTED_KEYS + ["N"])
    assert set(small_dataset.column_names) == expected_cols

def test_generate_dataset_length_filtering(small_dataset):
    """Tests that the sequence length filtering is applied correctly."""
    max_len = 2048 / 2
    for length in small_dataset["N"]:
        assert length <= max_len

def test_generate_dataset_type_distribution(small_dataset):
    """Tests that the problem types are reasonably distributed."""
    counts = pd.Series(small_dataset["type"]).value_counts()
    assert "arithmetic_logic" in counts
    assert "relational_reasoning" in counts
    assert "constraint_satisfaction" in counts
    # Each type should have some samples
    assert all(count > 0 for count in counts)

# --- Tests for File I/O ---

def test_save_and_load_dataset(small_dataset, tmp_path):
    """Tests saving the dataset to a file and verifies its content."""
    output_path = tmp_path / "test_dataset.jsonl"
    
    # Save the dataset
    save_dataset(small_dataset, str(output_path))
    
    # Check that the file was created
    assert os.path.exists(output_path)
    
    # Read the file and verify content
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == len(small_dataset)
    
    # Check the first record
    first_record_from_file = json.loads(lines[0])
    first_record_from_dataset = small_dataset[0]
    
    # Compare a few key fields
    assert first_record_from_file["prompt"] == first_record_from_dataset["prompt"]
    assert first_record_from_file["type"] == first_record_from_dataset["type"]
    assert len(first_record_from_file["Messages"]) == 3
