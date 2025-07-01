# Enhanced Logic Problem Generator - Usage Guide

## Overview

The Enhanced Logic Problem Generator is a sophisticated Python tool for generating a wide variety of logic problems with configurable complexity, narrative styles, and evaluation strategies. It includes ground-truth solvers for verification and comprehensive evaluation capabilities.

## Installation

```python
# Import the main components
from generator_gpro import (
    EnhancedProblemGenerator,
    AdvancedEvaluator,
    LogicalSolver,
    NarrativeGenerator
)
```

## Basic Usage

### 1. Simple Problem Generation

```python
# Create a generator instance
generator = EnhancedProblemGenerator(seed=42)  # Optional seed for reproducibility

# Generate a random problem
problem = generator.generate_problem(complexity=5)

# Print the problem
print(problem['prompt'])
print(f"Problem type: {problem['reasoning_type']}")
```

### 2. Generating Specific Problem Types

```python
# Available problem types:
problem_types = [
    'transitive_chain',      # A > B > C reasoning
    'spatial_layout',        # Spatial positioning puzzles
    'deontic_reasoning',     # Obligations and permissions
    'quantitative_logic',    # Numerical relationships
    'logical_puzzle',        # Grid-based logic puzzles
    'modal_reasoning',       # Knowledge and belief
    'temporal_sequence',     # Time-based constraints
    'set_operations',        # Set theory problems
    'causal_network',        # Cause-effect reasoning
    'constraint_satisfaction' # CSP problems
]

# Generate a specific type
spatial_problem = generator.generate_problem(
    complexity=4,
    problem_type='spatial_layout'
)
```

### 3. Advanced Features

```python
# Generate with narrative style
narrative_problem = generator.generate_problem(
    complexity=5,
    problem_type='transitive_chain',
    use_narrative=True  # Converts to story format
)

# Generate with controlled ambiguity
ambiguous_problem = generator.generate_problem(
    complexity=4,
    include_ambiguity=True  # Adds vague elements
)

# Generate unsolvable problems
unsolvable = generator.generate_problem(
    complexity=5,
    solvable=False  # Creates contradictory/underspecified problems
)
```

## Problem Structure

Each generated problem contains:

```python
{
    'prompt': str,           # The problem statement/question
    'reasoning_type': str,   # Type of logical reasoning required
    'entities': List[str],   # Entities involved in the problem
    'relations': List[str],  # Logical relationships (if applicable)
    'question': str,         # The specific question to answer
    'solution': Any,         # Ground truth solution
    'metadata': {
        'complexity': int,
        'type': str,
        'id': str,
        'features': {
            'narrative': bool,
            'ambiguous': bool,
            'solvable': bool
        }
    }
}
```

## Using the Solver

```python
solver = LogicalSolver()

# Solve transitive chains
entities = ["Alice", "Bob", "Charlie"]
relations = [
    "Alice is taller than Bob",
    "Bob is taller than Charlie"
]
result = solver.solve_transitive_chain(entities, relations, "tall")

print(f"Tallest: {result['solution']['tallest']}")
print(f"Order: {result['solution']['ordering']}")
print(f"Consistent: {result['consistent']}")

# Solve spatial layouts
objects = ["book", "pen", "laptop"]
constraints = [
    "book is to the left of pen",
    "pen is to the left of laptop"
]
result = solver.solve_spatial_layout(objects, constraints)
```

## Evaluation Strategies

```python
evaluator = AdvancedEvaluator(generator)

# Generate a base problem
base_problem = generator.generate_problem(complexity=4)

# Apply different evaluation strategies
perturbed = evaluator._evaluate_with_perturbation(base_problem)
varied = evaluator._evaluate_with_semantic_variation(base_problem)
noisy = evaluator._evaluate_with_noise(base_problem)
incomplete = evaluator._evaluate_with_incomplete_info(base_problem)

# Create comprehensive evaluation set
eval_set = evaluator.create_comprehensive_evaluation_set(
    size=100,
    min_complexity=3
)
```

## Problem Types in Detail

### 1. Transitive Chains
```python
problem = generator.generate_problem(
    complexity=5,
    problem_type='transitive_chain'
)
# Generates: "A is taller than B, B is taller than C..."
```

### 2. Spatial Layout
```python
problem = generator.generate_problem(
    complexity=6,
    problem_type='spatial_layout'
)
# Generates: "X is to the left of Y, Z is between X and Y..."
```

### 3. Deontic Reasoning
```python
problem = generator.generate_problem(
    complexity=7,
    problem_type='deontic_reasoning'
)
# Generates: "Alice must submit the report, Bob may review it..."
```

### 4. Quantitative Logic
```python
problem = generator.generate_problem(
    complexity=5,
    problem_type='quantitative_logic'
)
# Generates: "A's age is 2 times B's age, B is 10 years older than C..."
```

### 5. Modal Reasoning
```python
problem = generator.generate_problem(
    complexity=6,
    problem_type='modal_reasoning'
)
# Generates: "Alice knows that Bob believes that..."
```

## Complexity Levels

- **1-2**: Very simple, minimal relations
- **3-4**: Basic problems suitable for beginners
- **5-6**: Moderate complexity with multiple constraints
- **7-8**: Advanced problems with complex relationships
- **9-10**: Expert level with maximum complexity

## Narrative Generation

```python
narrative_gen = NarrativeGenerator()

# Convert relations to narrative
relations = [
    "Alice owns the book",
    "Bob is taller than Charlie",
    "The book is red"
]
entities = ["Alice", "Bob", "Charlie"]

narrative = narrative_gen.generate_narrative(
    relations, entities, "logical_puzzle"
)
print(narrative)
# Output: "During a meeting, several observations were made. Alice owns the book. Meanwhile, Bob is taller than Charlie..."
```

## Caching and Performance

```python
# Generator with custom cache size
generator = EnhancedProblemGenerator(
    seed=42,
    cache_size=10000  # Default is 10000
)

# Problems are automatically cached to prevent duplicates
# Cache has TTL of 1 hour and LRU eviction
```

## Example: Complete Workflow

```python
# 1. Initialize components
generator = EnhancedProblemGenerator(seed=42)
solver = LogicalSolver()
evaluator = AdvancedEvaluator(generator)

# 2. Generate a problem
problem = generator.generate_problem(
    complexity=5,
    problem_type='transitive_chain',
    use_narrative=True
)

# 3. Display the problem
print("PROBLEM:")
print(problem['prompt'])
print(f"\nType: {problem['reasoning_type']}")

# 4. Extract components for solving
if problem['reasoning_type'] == 'transitive_chain':
    solution = solver.solve_transitive_chain(
        problem['entities'],
        problem['relations'],
        problem.get('relation_type', 'tall')
    )
    
    print(f"\nSOLUTION:")
    print(f"Ordering: {solution['solution']['ordering']}")
    print(f"Consistent: {solution['consistent']}")

# 5. Create variations for testing
varied = evaluator._evaluate_with_semantic_variation(problem)
print(f"\nVARIATION:")
print(varied['prompt'])
```

## Best Practices

1. **Set a seed** for reproducible problem generation during testing
2. **Start with lower complexity** and gradually increase
3. **Use narrative mode** for more natural language problems
4. **Verify solvability** using the included solvers
5. **Cache management** is automatic but can be configured
6. **Test robustness** using evaluation strategies

## Error Handling

The generator handles edge cases gracefully:
- Empty entity lists
- Contradictory constraints
- Invalid problem types
- Circular dependencies
- Memory limits

All methods return valid problem structures even in error cases.

## Testing Your Problems

```python
# Quick test suite for a problem type
def test_problem_type(problem_type, num_tests=10):
    successes = 0
    for i in range(num_tests):
        try:
            problem = generator.generate_problem(
                complexity=random.randint(3, 7),
                problem_type=problem_type
            )
            # Verify structure
            assert 'prompt' in problem
            assert 'reasoning_type' in problem
            assert problem['reasoning_type'] == problem_type
            successes += 1
        except Exception as e:
            print(f"Failed on test {i}: {e}")
    
    print(f"{problem_type}: {successes}/{num_tests} successful")

# Test all types
for ptype in ['transitive_chain', 'spatial_layout', 'logical_puzzle']:
    test_problem_type(ptype)
```

## Advanced Configuration

```python
# Custom entity pools
from generator_gpro import EntityPools

# Access predefined pools
people = EntityPools.PEOPLE_NAMES
objects = EntityPools.OBJECTS
colors = EntityPools.COLORS
actions = EntityPools.ACTIONS

# Generate with specific entities
problem = generator._generate_transitive_chain(complexity=4)
# Note: Direct access to internal methods for customization
```

This generator provides a robust framework for creating diverse logic problems suitable for testing reasoning capabilities, educational purposes, or AI evaluation tasks.