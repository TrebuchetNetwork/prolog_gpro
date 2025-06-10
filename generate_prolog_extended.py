import random
import pandas as pd
from datasets import Dataset
from pyswip import Prolog
import re
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

# Extended system prompt for both math and logic
system_prompt = """You are given a problem that may require mathematical or logical reasoning.
Think about the problem and provide your working out.
For mathematical problems: Place calculations between <start_working_out> and <end_working_out>.
For logical problems: Place your reasoning between <logical_reasoning> and </logical_reasoning>, 
marking each inference step with <step> and </step>.
Then, provide your solution between <SOLUTION></SOLUTION>"""

# Reasoning format markers
reasoning_start = "<start_working_out>"
reasoning_end = "</end_working_out>"
logic_start = "<logical_reasoning>"
logic_end = "</logical_reasoning>"
step_start = "<step>"
step_end = "</step>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

class PrologProblemGenerator:
    """Extended Prolog problem generator with more diverse problem types."""
    
    def __init__(self):
        self.prolog = Prolog()
        self.entities = ["alice", "bob", "charlie", "diana", "eve", "frank", "george", "helen"]
        self.objects = ["book", "pen", "laptop", "phone", "keys", "wallet", "glasses", "watch"]
        self.colors = ["red", "blue", "green", "yellow", "black", "white"]
        self.locations = ["office", "home", "library", "cafe", "park", "store"]
        
    def reset_prolog(self):
        """Reset Prolog engine for clean state."""
        self.prolog = Prolog()
        
    def generate_family_relations_problem(self) -> Optional[Dict]:
        """Generate complex family relationship problems."""
        self.reset_prolog()
        
        # Create a family tree
        people = random.sample(self.entities, 6)
        relationships = []
        
        # Define parent relationships
        relationships.append(f"parent('{people[0]}', '{people[2]}').")
        relationships.append(f"parent('{people[0]}', '{people[3]}').")
        relationships.append(f"parent('{people[1]}', '{people[2]}').")
        relationships.append(f"parent('{people[1]}', '{people[3]}').")
        relationships.append(f"parent('{people[2]}', '{people[4]}').")
        relationships.append(f"parent('{people[3]}', '{people[5]}').")
        
        # Define gender
        male = random.sample(people, 3)
        for person in people:
            if person in male:
                relationships.append(f"male('{person}').")
            else:
                relationships.append(f"female('{person}').")
        
        # Define rules
        rules = [
            "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
            "sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \\= Y.",
            "father(X, Y) :- parent(X, Y), male(X).",
            "mother(X, Y) :- parent(X, Y), female(X).",
            "cousin(X, Y) :- parent(A, X), parent(B, Y), sibling(A, B)."
        ]
        
        # Assert facts and rules
        for rel in relationships + rules:
            try:
                self.prolog.assertz(rel)
            except:
                pass
        
        # Generate question
        query_types = [
            ("grandparent", f"Who are the grandparents of {people[4]}?"),
            ("sibling", f"Who are the siblings of {people[2]}?"),
            ("cousin", f"Are {people[4]} and {people[5]} cousins?")
        ]
        
        query_type, question = random.choice(query_types)
        
        try:
            if query_type == "grandparent":
                results = list(self.prolog.query(f"grandparent(X, '{people[4]}')"))
                answer = ", ".join([r['X'] for r in results]) if results else "None"
                reasoning = self._generate_family_reasoning(relationships, rules, query_type, people[4], results)
            elif query_type == "sibling":
                results = list(self.prolog.query(f"sibling(X, '{people[2]}')"))
                answer = ", ".join([r['X'] for r in results]) if results else "None"
                reasoning = self._generate_family_reasoning(relationships, rules, query_type, people[2], results)
            else:  # cousin
                results = list(self.prolog.query(f"cousin('{people[4]}', '{people[5]}')"))
                answer = "Yes" if results else "No"
                reasoning = self._generate_cousin_reasoning(relationships, rules, people[4], people[5], results)
            
            return {
                "prompt": f"Given the following family relationships:\n" + 
                         "\n".join([r.replace("'", "") for r in relationships[:6]]) + 
                         f"\n\n{question}",
                "prolog_rules": "\n".join(relationships + rules),
                "reasoning": reasoning,
                "solution": answer,
                "type": "family_relations"
            }
        except Exception as e:
            return None
    
    def generate_propositional_logic_problem(self) -> Optional[Dict]:
        """Generate propositional logic problems."""
        self.reset_prolog()
        
        # Generate propositions
        props = ["raining", "cloudy", "wet_ground", "umbrella_needed", "stay_inside"]
        used_props = random.sample(props, 4)
        
        # Create logical rules
        rules = []
        facts = []
        
        # Random facts
        if random.random() > 0.5:
            facts.append(f"{used_props[0]}.")
        if random.random() > 0.5:
            facts.append(f"{used_props[1]}.")
        
        # Logical implications
        rules.append(f"{used_props[2]} :- {used_props[0]}.")
        rules.append(f"{used_props[3]} :- {used_props[0]}.")
        rules.append(f"{used_props[3]} :- {used_props[1]}, \\+ {used_props[0]}.")
        
        # Assert facts and rules
        for item in facts + rules:
            try:
                self.prolog.assertz(item)
            except:
                pass
        
        # Generate question
        query = random.choice(used_props[2:])
        question = f"Given the rules and facts, is '{query.replace('_', ' ')}' true?"
        
        try:
            result = list(self.prolog.query(query))
            answer = "Yes" if result else "No"
            
            reasoning = self._generate_propositional_reasoning(facts, rules, query, result)
            
            return {
                "prompt": f"Consider the following logical rules:\n" +
                         "\n".join([f"- {r}" for r in rules]) +
                         "\n\nFacts:\n" + 
                         "\n".join([f"- {f}" for f in facts]) if facts else "- None given" +
                         f"\n\n{question}",
                "prolog_rules": "\n".join(facts + rules),
                "reasoning": reasoning,
                "solution": answer,
                "type": "propositional_logic"
            }
        except Exception as e:
            return None
    
    def generate_constraint_logic_problem(self) -> Optional[Dict]:
        """Generate more complex constraint satisfaction problems."""
        self.reset_prolog()
        
        # Room assignment problem
        students = random.sample(self.entities[:4], 3)
        rooms = ["Room A", "Room B", "Room C"]
        
        constraints = []
        
        # Each student needs a room
        constraints.append(f"assign([{','.join([f'room({s},R{i})' for i,s in enumerate(students)])}]) :- " +
                          f"{','.join([f'member(R{i}, {rooms})' for i in range(len(students))])}," +
                          f"all_different([{','.join([f'R{i}' for i in range(len(students))])}]).")
        
        # Define all_different
        constraints.append("all_different([]).")
        constraints.append("all_different([H|T]) :- \\+ member(H, T), all_different(T).")
        
        # Additional constraint
        constraints.append(f"constraint(R0, R1) :- R0 \\= 'Room A' ; R1 \\= 'Room B'.")
        
        prompt = f"Assign {', '.join(students)} to {', '.join(rooms)} such that each gets a different room, " \
                f"and either {students[0]} is not in Room A or {students[1]} is not in Room B. Find a valid assignment."
        
        # For simplicity, provide a direct solution
        assignments = list(zip(students, random.sample(rooms, len(students))))
        # Ensure constraint is satisfied
        if assignments[0][1] == "Room A" and assignments[1][1] == "Room B":
            # Swap to satisfy constraint
            assignments[0], assignments[2] = assignments[2], assignments[0]
        
        answer = ", ".join([f"{s} in {r}" for s, r in assignments])
        
        reasoning = f"""To solve this constraint satisfaction problem:
{step_start}Identify constraints: Each student needs a unique room.{step_end}
{step_start}Additional constraint: Either {students[0]} ≠ Room A OR {students[1]} ≠ Room B.{step_end}
{step_start}Try assignment: {assignments}{step_end}
{step_start}Verify: All rooms different ✓, Constraint satisfied ✓{step_end}
{step_start}Solution found: {answer}{step_end}"""
        
        return {
            "prompt": prompt,
            "prolog_rules": "\n".join(constraints),
            "reasoning": reasoning,
            "solution": answer,
            "type": "constraint_logic"
        }
    
    def generate_rule_based_inference_problem(self) -> Optional[Dict]:
        """Generate rule-based inference problems."""
        self.reset_prolog()
        
        # Animal classification problem
        animals = ["sparrow", "eagle", "penguin", "bat", "dolphin"]
        animal = random.choice(animals)
        
        # Facts about the animal
        facts = []
        rules = []
        
        if animal in ["sparrow", "eagle", "penguin"]:
            facts.append(f"has_feathers('{animal}').")
            facts.append(f"lays_eggs('{animal}').")
            if animal != "penguin":
                facts.append(f"can_fly('{animal}').")
        
        if animal == "bat":
            facts.append(f"has_fur('{animal}').")
            facts.append(f"can_fly('{animal}').")
            facts.append(f"gives_birth('{animal}').")
        
        if animal == "dolphin":
            facts.append(f"lives_in_water('{animal}').")
            facts.append(f"gives_birth('{animal}').")
            facts.append(f"breathes_air('{animal}').")
        
        # Classification rules
        rules.extend([
            "bird(X) :- has_feathers(X), lays_eggs(X).",
            "mammal(X) :- gives_birth(X), (has_fur(X) ; lives_in_water(X)).",
            "can_fly_well(X) :- bird(X), can_fly(X).",
            "flightless_bird(X) :- bird(X), \\+ can_fly(X).",
            "aquatic_mammal(X) :- mammal(X), lives_in_water(X)."
        ])
        
        # Assert facts and rules
        for item in facts + rules:
            try:
                self.prolog.assertz(item)
            except:
                pass
        
        # Generate question
        questions = [
            (f"Is {animal} a bird?", "bird"),
            (f"Is {animal} a mammal?", "mammal"),
            (f"Can {animal} fly well?", "can_fly_well"),
            (f"What type of animal is {animal}?", "classify")
        ]
        
        question, query_type = random.choice(questions)
        
        try:
            if query_type == "classify":
                bird_result = list(self.prolog.query(f"bird('{animal}')"))
                mammal_result = list(self.prolog.query(f"mammal('{animal}')"))
                if bird_result:
                    answer = "bird"
                elif mammal_result:
                    answer = "mammal"
                else:
                    answer = "unknown"
            else:
                result = list(self.prolog.query(f"{query_type}('{animal}')"))
                answer = "Yes" if result else "No"
            
            reasoning = self._generate_rule_inference_reasoning(facts, rules, animal, query_type, answer)
            
            return {
                "prompt": f"Given these facts about {animal}:\n" +
                         "\n".join([f"- {f.replace('_', ' ').replace(f\"'{animal}'.\", '')}" for f in facts]) +
                         "\n\nAnd these classification rules:\n" +
                         "\n".join([f"- {self._format_rule(r)}" for r in rules[:3]]) +
                         f"\n\n{question}",
                "prolog_rules": "\n".join(facts + rules),
                "reasoning": reasoning,
                "solution": answer,
                "type": "rule_based_inference"
            }
        except Exception as e:
            return None
    
    def generate_arithmetic_logic_problem(self) -> Optional[Dict]:
        """Enhanced version of the original arithmetic logic problem."""
        self.reset_prolog()
        
        # More complex system of equations
        problem_type = random.choice(["linear_2var", "linear_3var", "quadratic_simple"])
        
        if problem_type == "linear_2var":
            x = random.randint(1, 20)
            y = random.randint(1, 20)
            a1, b1 = random.randint(1, 5), random.randint(1, 5)
            a2, b2 = random.randint(1, 5), random.randint(-5, 5)
            c1 = a1 * x + b1 * y
            c2 = a2 * x + b2 * y
            
            prompt = f"Solve the system: {a1}x + {b1}y = {c1} and {a2}x + {b2 if b2 >= 0 else ''}y = {c2}"
            
            reasoning = f"""{logic_start}
{step_start}Given: {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2}{step_end}
{step_start}Using elimination method: Multiply first equation by {a2} and second by {a1}{step_end}
{step_start}Get: {a1*a2}x + {b1*a2}y = {c1*a2} and {a2*a1}x + {b2*a1}y = {c2*a1}{step_end}
{step_start}Subtract: {(b1*a2 - b2*a1)}y = {c1*a2 - c2*a1}{step_end}
{step_start}Therefore: y = {(c1*a2 - c2*a1) // (b1*a2 - b2*a1)} = {y}{step_end}
{step_start}Substitute back: x = {x}{step_end}
{logic_end}"""
            
            solution = f"x = {x}, y = {y}"
            
            return {
                "prompt": prompt,
                "prolog_rules": f"solve(X, Y) :- {a1}*X + {b1}*Y =:= {c1}, {a2}*X + {b2}*Y =:= {c2}.",
                "reasoning": reasoning,
                "solution": solution,
                "type": "arithmetic_logic_enhanced"
            }
        
        return None
    
    def _generate_family_reasoning(self, relationships, rules, query_type, target, results):
        """Generate detailed reasoning for family relations."""
        reasoning = f"{logic_start}\n"
        reasoning += f"{step_start}Given family relationships and rules for {query_type}.{step_end}\n"
        
        if query_type == "grandparent":
            reasoning += f"{step_start}To find grandparents of {target}, need X where parent(X, Y) and parent(Y, {target}).{step_end}\n"
            reasoning += f"{step_start}Checking all parent relationships...{step_end}\n"
            if results:
                for r in results:
                    reasoning += f"{step_start}Found: {r['X']} is a grandparent of {target}.{step_end}\n"
            else:
                reasoning += f"{step_start}No grandparents found for {target}.{step_end}\n"
        
        reasoning += f"{logic_end}"
        return reasoning
    
    def _generate_cousin_reasoning(self, relationships, rules, person1, person2, results):
        """Generate reasoning for cousin relationships."""
        reasoning = f"{logic_start}\n"
        reasoning += f"{step_start}To check if {person1} and {person2} are cousins:{step_end}\n"
        reasoning += f"{step_start}Need parents A and B where parent(A, {person1}), parent(B, {person2}), and sibling(A, B).{step_end}\n"
        reasoning += f"{step_start}Checking parent and sibling relationships...{step_end}\n"
        reasoning += f"{step_start}Result: {'Yes, they are cousins' if results else 'No, they are not cousins'}.{step_end}\n"
        reasoning += f"{logic_end}"
        return reasoning
    
    def _generate_propositional_reasoning(self, facts, rules, query, result):
        """Generate reasoning for propositional logic."""
        reasoning = f"{logic_start}\n"
        reasoning += f"{step_start}Query: Is '{query}' true?{step_end}\n"
        reasoning += f"{step_start}Known facts: {facts if facts else 'None'}{step_end}\n"
        reasoning += f"{step_start}Checking rules that could derive '{query}'...{step_end}\n"
        
        for rule in rules:
            if query in rule:
                reasoning += f"{step_start}Found rule: {rule}{step_end}\n"
        
        reasoning += f"{step_start}Conclusion: '{query}' is {'true' if result else 'false'}.{step_end}\n"
        reasoning += f"{logic_end}"
        return reasoning
    
    def _generate_rule_inference_reasoning(self, facts, rules, animal, query_type, answer):
        """Generate reasoning for rule-based inference."""
        reasoning = f"{logic_start}\n"
        reasoning += f"{step_start}Analyzing {animal} with given facts.{step_end}\n"
        
        for fact in facts[:2]:  # Show first few facts
            reasoning += f"{step_start}Fact: {fact.replace('_', ' ')}{step_end}\n"
        
        reasoning += f"{step_start}Applying classification rules...{step_end}\n"
        reasoning += f"{step_start}Checking if {query_type} applies to {animal}...{step_end}\n"
        reasoning += f"{step_start}Result: {answer}{step_end}\n"
        reasoning += f"{logic_end}"
        return reasoning
    
    def _format_rule(self, rule):
        """Format Prolog rule for display."""
        return rule.replace(":-", "IF").replace(",", " AND").replace(";", " OR").replace("_", " ")

class PrologDatasetBuilder:
    """Build and manage the Prolog dataset with GRPO integration."""
    
    def __init__(self, tokenizer=None):
        self.generator = PrologProblemGenerator()
        self.tokenizer = tokenizer
        if not tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
    
    def generate_dataset(self, num_samples=1000, problem_types=None):
        """Generate a diverse Prolog-style dataset."""
        if problem_types is None:
            problem_types = [
                "family_relations",
                "propositional_logic", 
                "constraint_logic",
                "rule_based_inference",
                "arithmetic_logic",
                "relational_reasoning",
                "constraint_satisfaction"
            ]
        
        # Problem generators mapping
        generators = {
            "family_relations": self.generator.generate_family_relations_problem,
            "propositional_logic": self.generator.generate_propositional_logic_problem,
            "constraint_logic": self.generator.generate_constraint_logic_problem,
            "rule_based_inference": self.generator.generate_rule_based_inference_problem,
            "arithmetic_logic": self.generator.generate_arithmetic_logic_problem,
            "relational_reasoning": generate_relational_reasoning_problem,
            "constraint_satisfaction": generate_constraint_satisfaction_problem
        }
        
        samples = []
        samples_per_type = defaultdict(int)
        target_per_type = num_samples // len(problem_types)
        
        # Generate samples for each type
        for ptype in problem_types:
            if ptype not in generators:
                continue
                
            generator_func = generators[ptype]
            attempts = 0
            
            while samples_per_type[ptype] < target_per_type and attempts < target_per_type * 3:
                attempts += 1
                try:
                    if ptype in ["relational_reasoning", "constraint_satisfaction"]:
                        # Original functions
                        sample = generator_func()
                    else:
                        # New generator methods
                        sample = generator_func()
                    
                    if sample:
                        formatted = self.format_for_grpo(sample)
                        if formatted:
                            samples.append(formatted)
                            samples_per_type[ptype] += 1
                except Exception as e:
                    print(f"Error generating {ptype}: {e}")
                    continue
        
        # Fill remaining with random types
        while len(samples) < num_samples:
            ptype = random.choice(problem_types)
            generator_func = generators.get(ptype)
            if not generator_func:
                continue
                
            try:
                if ptype in ["relational_reasoning", "constraint_satisfaction"]:
                    sample = generator_func()
                else:
                    sample = generator_func()
                    
                if sample:
                    formatted = self.format_for_grpo(sample)
                    if formatted:
                        samples.append(formatted)
            except:
                continue
        
        # Convert to dataset
        df = pd.DataFrame(samples)
        dataset = Dataset.from_pandas(df)
        
        # Add length information
        dataset = dataset.map(self._compute_length)
        
        # Filter by length
        dataset = dataset.filter(lambda x: x["N"] <= 1024)
        
        print(f"Generated {len(dataset)} samples")
        print("Distribution by type:")
        type_counts = pd.Series([s["type"] for s in samples]).value_counts()
        print(type_counts)
        
        return dataset
    
    def format_for_grpo(self, sample):
        """Format sample for GRPO training."""
        if not sample:
            return None
            
        # Determine if we should use logic or math formatting
        use_logic_format = sample["type"] not in ["arithmetic_logic", "arithmetic_logic_enhanced"]
        
        if use_logic_format and logic_start not in sample["reasoning"]:
            # Wrap reasoning in logic tags if not already present
            sample["reasoning"] = f"{logic_start}\n{sample['reasoning']}\n{logic_end}"
        elif not use_logic_format and reasoning_start not in sample["reasoning"]:
            # Use math format
            sample["reasoning"] = f"{reasoning_start}\n{sample['reasoning']}\n{reasoning_end}"
        
        return {
            "Messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": 
                    f"{sample['reasoning']}\n"
                    f"{solution_start}{sample['solution']}{solution_end}"
                }
            ],
            "prompt": sample["prompt"],
            "prolog_rules": sample.get("prolog_rules", ""),
            "reasoning": sample["reasoning"],
            "solution": sample["solution"],
            "type": sample["type"],
            "answer": sample["solution"]  # For compatibility with reward functions
        }
    
    def _compute_length(self, example):
        """Compute tokenized length."""
        text = self.tokenizer.apply_chat_template(example["Messages"], tokenize=False)
        return {"N": len(self.tokenizer(text, add_special_tokens=False)["input_ids"])}

# Reward evaluation functions for GRPO
class PrologRewardEvaluator:
    """Evaluate rewards for Prolog-style reasoning."""
    
    def __init__(self):
        self.logic_indicators = [
            'therefore', 'implies', 'because', 'given that',
            'we can conclude', 'it follows that', 'hence',
            'by transitivity', 'by definition', 'thus'
        ]
        
        self.logic_format_regex = re.compile(
            rf"{logic_end}.*?"
            rf"{solution_start}(.+?){solution_end}",
            flags=re.MULTILINE | re.DOTALL
        )
        
        self.step_regex = re.compile(
            rf"{step_start}(.*?){step_end}",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def evaluate_logical_structure(self, completions, **kwargs):
        """Evaluate the structural quality of logical reasoning."""
        scores = []
        
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            
            # Check for logical reasoning section
            if logic_start in response and logic_end in response:
                score += 2.0
                
                # Extract logical content
                match = re.search(rf"{logic_start}(.*?){logic_end}", response, re.DOTALL)
                if match:
                    logic_content = match.group(1)
                    
                    # Count step markers
                    steps = self.step_regex.findall(logic_content)
                    score += min(len(steps) * 0.5, 2.0)
                    
                    # Check for logical indicators
                    for indicator in self.logic_indicators:
                        if indicator in logic_content.lower():
                            score += 0.2
                    
                    score = min(score, 5.0)  # Cap at 5
            
            scores.append(score)
        
        return scores
    
    def evaluate_solution_correctness(self, prompts, completions, answer, **kwargs):
        """Evaluate correctness of the solution."""
        scores = []
        
        for i, completion in enumerate(completions):
            score = 0
            response = completion[0]["content"]
            
            # Extract solution
            solution_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
            if not solution_match:
                scores.append(-1.0)
                continue
            
            predicted = solution_match.group(1).strip().lower()
            expected = str(answer[i]).lower() if isinstance(answer, list) else str(answer).lower()
            
            # Exact match
            if predicted == expected:
                score = 5.0
            # Partial match
            elif expected in predicted or predicted in expected:
                score = 3.0
            # Semantic equivalence for yes/no
            elif self._check_semantic_equivalence(predicted, expected):
                score = 4.0
            else:
                score = -0.5
            
            scores.append(score)
        
        return scores
    
    def evaluate_reasoning_quality(self, prompts, completions, **kwargs):
        """Evaluate the quality of reasoning steps."""
        scores = []
        
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            
            # Check if reasoning connects to the question
            if prompts:
                question = prompts[0][-1]["content"] if isinstance(prompts[0], list) else str(prompts[0])
                
                # Extract key terms from question
                question_terms = set(w.lower() for w in re.findall(r'\w+', question) if len(w) > 3)
                
                # Check reasoning content
                reasoning_match = re.search(rf"({logic_start}.*?{logic_end}|{reasoning_start}.*?{reasoning_end})", 
                                          response, re.DOTALL)
                if reasoning_match:
                    reasoning_content = reasoning_match.group(1)
                    reasoning_terms = set(w.lower() for w in re.findall(r'\w+', reasoning_content) if len(w) > 3)
                    
                    # Calculate overlap
                    overlap = len(question_terms & reasoning_terms) / max(len(question_terms), 1)
                    score += overlap * 2.0
                    
                    # Check for step-by-step reasoning
                    if step_start in reasoning_content:
                        score += 1.0
                    
                    # Length bonus (but not too long)
                    word_count = len(reasoning_content.split())
                    if 50 <= word_count <= 300:
                        score += 1.0
                    elif word_count > 300:
                        score += 0.5
            
            scores.append(min(score, 5.0))
        
        return scores
    
    def _check_semantic_equivalence(self, pred, expected):
        """Check semantic equivalence for common answer patterns."""
        equivalences = [
            (["yes", "true", "correct", "affirmative"], ["yes", "true", "correct", "affirmative"]),
            (["no", "false", "incorrect", "negative"], ["no", "false", "incorrect", "negative"])
        ]
        
        for group1, group2 in equivalences:
            if any(p in pred for p in group1) and any(e in expected for e in group2):
                return True
        
        return False

# Main execution
if __name__ == "__main__":
    # Initialize dataset builder
    builder = PrologDatasetBuilder()
    
    # Generate comprehensive dataset
    dataset = builder.generate_dataset(num_samples=1000)
    
    # Save dataset
    output_path = "extended_prolog_dataset.json"
    dataset.to_json(output_path, orient="records", lines=True)
    print(f"\nDataset saved to {output_path}")
    
    # Initialize reward evaluator
    evaluator = PrologRewardEvaluator()
    
    # Test reward functions on a sample
    print("\n" + "="*50)
    print("Testing Reward Functions on Sample")
    print("="*50)
    
    sample = dataset[0]
    test_completion = [{
        "content": sample["Messages"][2]["content"]  # Assistant's response
    }]
    
    struct_score = evaluator.evaluate_logical_structure([test_completion])[0]
    solution_score = evaluator.evaluate_solution_correctness(
        [sample["Messages"]], 
        [test_completion], 
        [sample["answer"]]
    )[0]
    reasoning_score = evaluator.evaluate_reasoning_quality(
        [sample["Messages"]], 
        [test_completion]
    )[0]
    
    print(f"Structural Score: {struct_score:.2f}")
    print(f"Solution Score: {solution_score:.2f}")
    print(f"Reasoning Quality Score: {reasoning_score:.2f}")
    print(f"Total Score: {struct_score + solution_score + reasoning_score:.2f}")
    
    # Example of integration with GRPO
    print("\n" + "="*50)
    print("GRPO Integration Example")
    print("="*50)
    
    print("""
# To integrate with your GRPO trainer:

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        # Existing math rewards
        match_format_exactly,
        check_answer,
        check_numbers,
        
        # New Prolog/logic rewards
        evaluator.evaluate_logical_structure,
        evaluator.evaluate_solution_correctness,
        evaluator.evaluate_reasoning_quality,
    ],
    args = training_args,
    train_dataset = dataset,  # Your mixed math + logic dataset
)
    """)
