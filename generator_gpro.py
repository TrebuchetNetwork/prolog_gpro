"""
Enhanced Sophisticated Logic Problem Generator
Incorporates spatial reasoning, deontic logic, ground-truth solvers,
narrative generation, and advanced evaluation strategies
"""

import random
import itertools
from enum import Enum
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import json
import re
from datetime import datetime, timedelta


class TransitiveChainGenerator:
    """A robust generator for transitive chain problems that ensures consistency."""

    def __init__(self):
        self.graph = defaultdict(set)  # A -> {B, C} means A > B and A > C
        self.reverse_graph = defaultdict(set)  # For efficient reachability checks
        self.equals = defaultdict(set)  # Equivalence classes

    def can_add_relation(self, a: str, b: str, relation: str) -> bool:
        """Check if adding a relation would create a contradiction."""
        if relation == "greater":
            # Check if b is already (transitively) greater than a
            if self._is_reachable(b, a):
                return False
            # Check if a equals b
            if b in self.equals.get(a, set()) or a in self.equals.get(b, set()):
                return False
        elif relation == "equal":
            # Check if a > b or b > a (directly or transitively)
            if self._is_reachable(a, b) or self._is_reachable(b, a):
                return False
        return True

    def _is_reachable(self, start: str, end: str) -> bool:
        """Check if end is reachable from start in the graph (DFS)."""
        if start == end:
            return False

        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)

            # Add direct descendants
            stack.extend(self.graph.get(node, set()))

            # IMPORTANT: Also check through equivalence classes
            # If node equals some other nodes, check their descendants too
            for equal in self.equals.get(node, set()):
                if equal not in visited:
                    stack.append(equal)
                    # Also add the descendants of the equal node
                    stack.extend(self.graph.get(equal, set()))

        return False

    def add_relation(self, a: str, b: str, relation: str) -> bool:
        """Add a relation if it's consistent. Returns True if added."""
        if not self.can_add_relation(a, b, relation):
            return False

        if relation == "greater":
            # Get all nodes equivalent to a and b
            a_equals = self.equals.get(a, {a}) | {a}
            b_equals = self.equals.get(b, {b}) | {b}

            # Add edges between all equivalent nodes
            for a_eq in a_equals:
                for b_eq in b_equals:
                    if a_eq != b_eq:
                        self.graph[a_eq].add(b_eq)
                        self.reverse_graph[b_eq].add(a_eq)

        elif relation == "equal":
            # Merge equivalence classes
            all_equal = (self.equals.get(a, {a}) | {a} |
                         self.equals.get(b, {b}) | {b})

            # Update equals for all nodes in the merged class
            for node in all_equal:
                self.equals[node] = all_equal - {node}


        return True

    def generate_consistent_chain(self, entities: List[str],
                                  relation_type: str,
                                  complexity: int) -> List[str]:
        """Generate a consistent set of relations."""
        self.graph.clear()
        self.reverse_graph.clear()
        self.equals.clear()

        relations = []

        # Ensure we don't create more relations than requested
        max_relations = min(complexity + 2, len(entities) * (len(entities) - 1) // 2)

        # Strategy 1: Start with a base chain
        shuffled = entities[:]
        random.shuffle(shuffled)

        # Create base chain with some gaps
        for i in range(min(len(shuffled) - 1, max_relations)):
            if self.add_relation(shuffled[i], shuffled[i + 1], "greater"):
                relations.append(f"{shuffled[i]} is {relation_type}er than {shuffled[i + 1]}")
                if len(relations) >= max_relations:
                    break

        # Strategy 2: Add some additional relations if we have room
        attempts = 0
        while len(relations) < max_relations and attempts < 50:
            attempts += 1
            a, b = random.sample(entities, 2)

            if self._is_reachable(a, b):
                # This relation is already implied, safe to add explicitly
                relations.append(f"{a} is {relation_type}er than {b}")
            elif not self._is_reachable(b, a) and self.add_relation(a, b, "greater"):
                relations.append(f"{a} is {relation_type}er than {b}")

        # Add equality relations only if complexity > 3 and we have room
        if complexity > 3 and len(relations) < max_relations:
            attempts = 0
            while attempts < 10 and len(relations) < max_relations:
                attempts += 1
                a, b = random.sample(entities, 2)
                if self.can_add_relation(a, b, "equal") and self.add_relation(a, b, "equal"):
                    relations.append(f"{a} is as {relation_type} as {b}")
                    break

        random.shuffle(relations)
        return relations[:max_relations]  # Ensure we don't exceed the limit


# === Enhanced Entity Pools ===

class EntityPools:
    """Extended entity pools with spatial and quantitative attributes."""

    # Existing pools remain the same...
    PEOPLE_NAMES = [
        "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry",
        "Isabella", "Jack", "Kate", "Liam", "Maria", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
        "Yara", "Zoe", "Ahmad", "Beatriz", "Chen", "Dmitri", "Elena", "Fatima",
        "Giovanni", "Hiroshi", "Ingrid", "Jamal", "Katya", "Luis", "Mei", "Nadia",
        "Omar", "Priya", "Rashid", "Sophia", "Tariq", "Ursula", "Vikram", "Wei",
        "Xenia", "Yuki", "Zara", "Aarav", "Bianca", "Carlos", "Diya", "Erik",
        "Fiona", "Gabriel", "Hannah", "Ivan", "Julia", "Kevin", "Luna", "Marco",
        "Nina", "Oscar", "Petra", "Quincy", "Rosa", "Stefan", "Tina", "Umar",
        "Vera", "William", "Xander", "Yasmin", "Zachary"
    ]

    OBJECTS = [
        "book", "pen", "laptop", "phone", "key", "wallet", "watch", "glass",
        "bottle", "chair", "table", "lamp", "mirror", "clock", "camera", "bag",
        "shoe", "hat", "coat", "ring", "necklace", "bracelet", "painting", "vase",
        "plant", "flower", "tree", "car", "bicycle", "motorcycle", "boat", "plane",
        "train", "bus", "house", "apartment", "office", "store", "restaurant",
        "hospital", "school", "library", "museum", "theater", "stadium", "park"
    ]

    COLORS = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "silver", "gold", "bronze", "turquoise", "teal",
        "magenta", "cyan", "lime", "olive", "navy", "maroon", "coral", "salmon",
        "crimson", "scarlet", "amber", "jade", "emerald", "sapphire", "ruby"
    ]

    # New pools for enhanced features
    SPATIAL_POSITIONS = ["left", "right", "above", "below", "center", "corner", "edge", "middle"]

    ACTIONS = ["attend", "complete", "submit", "review", "approve", "reject", "modify",
               "cancel", "postpone", "delegate", "authorize", "investigate"]

    OBLIGATIONS = ["must", "is obligated to", "is required to", "has a duty to", "is responsible for"]

    PERMISSIONS = ["may", "is permitted to", "is allowed to", "has the right to", "can choose to"]

    PROHIBITIONS = ["must not", "is forbidden from", "is prohibited from", "cannot", "is banned from"]

    QUANTITIES = list(range(1, 101))  # Numbers 1-100 for quantitative relations

    UNITS = ["meters", "kilometers", "feet", "miles", "kilograms", "pounds",
             "years", "months", "days", "hours", "dollars", "euros", "percent"]

    LOCATIONS = ["Paris", "London", "New York", "Tokyo", "Sydney", "Berlin", "Moscow", "Cairo", "Mumbai", "Beijing"]

    PROFESSIONS = ["doctor", "teacher", "engineer", "artist", "lawyer", "chef", "pilot", "writer", "musician", "scientist"]


# === Enhanced Logical Constructs ===

class RelationType(Enum):
    """Extended relation types including spatial and deontic."""
    COMPARISON = "comparison"
    POSSESSION = "possession"
    LOCATION = "location"
    KINSHIP = "kinship"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    CAUSAL = "causal"
    PREFERENCE = "preference"
    SPATIAL = "spatial"  # NEW: Relative spatial positioning
    DEONTIC = "deontic"  # NEW: Obligations and permissions
    QUANTITATIVE = "quantitative"  # NEW: Numerical comparisons


@dataclass
class Relation:
    """Enhanced relation with optional quantitative value."""
    subject: str
    predicate: str
    object: str
    relation_type: RelationType
    negated: bool = False
    certainty: float = 1.0
    value: Optional[float] = None  # NEW: For quantitative relations
    unit: Optional[str] = None  # NEW: Unit for the value


# === Ground Truth Solvers ===

class LogicalSolver:
    """Ground truth solver for various problem types with reasoning traces."""

    def _topological_sort_with_equals(self, graph: Dict[str, Set[str]],
                                      equals: Dict[str, Set[str]],
                                      all_nodes: Set[str]) -> List[str]:
        """Topological sort that keeps equal entities adjacent."""
        # Build equivalence classes
        visited_equals = set()
        equiv_classes = []

        for node in all_nodes:
            if node not in visited_equals:
                equiv_class = equals.get(node, set()) | {node}
                equiv_classes.append(sorted(equiv_class))  # Sort for consistency
                visited_equals.update(equiv_class)

        # Build graph between equivalence classes
        class_graph = defaultdict(set)
        node_to_class = {}

        for i, equiv_class in enumerate(equiv_classes):
            for node in equiv_class:
                node_to_class[node] = i

        for a in graph:
            for b in graph[a]:
                if a in node_to_class and b in node_to_class:
                    class_a = node_to_class[a]
                    class_b = node_to_class[b]
                    if class_a != class_b:
                        class_graph[class_a].add(class_b)

        # Topological sort on equivalence classes
        in_degree = {i: 0 for i in range(len(equiv_classes))}
        for i in class_graph:
            for j in class_graph[i]:
                in_degree[j] += 1

        queue = deque([i for i in range(len(equiv_classes)) if in_degree[i] == 0])
        result = []

        while queue:
            class_idx = queue.popleft()
            # Add all nodes from this equivalence class
            result.extend(equiv_classes[class_idx])

            for neighbor in class_graph.get(class_idx, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result if len(result) == len(all_nodes) else list(all_nodes)

    def solve_transitive_chain(self, entities: List[str], relations: List[str],
                                     relation_type: str) -> Dict[str, Any]:
        """Solve transitive reasoning with full trace - FULLY FIXED version."""
        from collections import defaultdict, deque

        # Build graph and equality sets
        graph = defaultdict(set)
        equals = defaultdict(set)

        trace = []
        trace.append(f"Building {relation_type} relationship graph...")

        # Collect all mentioned entities
        all_mentioned = set(entities) if entities else set()

        # First pass: parse all relations
        for rel in relations:
            # Parse relation
            if f"is {relation_type}er than" in rel:
                parts = rel.split(f" is {relation_type}er than ")
                if len(parts) == 2:
                    subject, obj = parts[0].strip(), parts[1].strip()
                    graph[subject].add(obj)
                    all_mentioned.add(subject)
                    all_mentioned.add(obj)
                    trace.append(f"Added: {subject} > {obj}")
            elif f"is as {relation_type} as" in rel:
                parts = rel.split(f" is as {relation_type} as ")
                if len(parts) == 2:
                    subject, obj = parts[0].strip(), parts[1].strip()
                    equals[subject].add(obj)
                    equals[obj].add(subject)
                    all_mentioned.update([subject, obj])
                    trace.append(f"Added equality: {subject} = {obj}")

        # Build complete equivalence classes (transitive closure of equals)
        trace.append("Building equivalence classes...")
        equiv_classes_map = {}  # node -> set of all equal nodes (including self)
        visited = set()

        for node in sorted(all_mentioned):  # Sort for deterministic behavior
            if node not in visited:
                # BFS to find all nodes equal to this one
                equiv_class = set()
                queue = deque([node])

                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        equiv_class.add(current)

                        # Add all nodes equal to current
                        for equal in equals.get(current, set()):
                            if equal not in visited:
                                queue.append(equal)

                # Store the equivalence class for each member
                for member in equiv_class:
                    equiv_classes_map[member] = equiv_class

                trace.append(f"Equivalence class: {sorted(equiv_class)}")

        # Apply equality constraints to the graph
        trace.append("Propagating relations through equality classes...")

        # Create a new graph with all implied edges
        new_graph = defaultdict(set)

        # For each original edge, add edges between all equivalent nodes
        for a, targets in graph.items():
            a_equiv = equiv_classes_map.get(a, {a})
            for b in targets:
                b_equiv = equiv_classes_map.get(b, {b})

                # Add edges from all nodes equivalent to a to all nodes equivalent to b
                for a_eq in a_equiv:
                    for b_eq in b_equiv:
                        if a_eq != b_eq:  # No self-loops
                            new_graph[a_eq].add(b_eq)
                            if (a_eq, b_eq) != (a, b):  # Only log new edges
                                trace.append(f"Propagated: {a_eq} > {b_eq} (from {a} > {b})")

        # Replace graph with the new one
        graph = new_graph

        # Compute transitive closure
        trace.append("Computing transitive closure...")
        changed = True
        iterations = 0
        max_iterations = len(all_mentioned) ** 2

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Create a copy to avoid modifying while iterating
            graph_snapshot = {k: set(v) for k, v in graph.items()}

            for a in graph_snapshot:
                for b in graph_snapshot[a]:
                    for c in graph_snapshot.get(b, set()):
                        if c not in graph[a] and a != c:
                            graph[a].add(c)
                            trace.append(f"Transitive: {a} > {b} > {c}, so {a} > {c}")
                            changed = True

        # Check for contradictions
        trace.append("Checking for contradictions...")
        for a in graph:
            if a in graph[a]:
                trace.append(f"CONTRADICTION: {a} cannot be {relation_type}er than itself!")
                return {
                    "solution": "The given information is contradictory.",
                    "trace": trace,
                    "consistent": False,
                    "graph": {k: sorted(list(v)) for k, v in graph.items()},
                    "equals": {k: sorted(list(v)) for k, v in equals.items()}
                }

        # Check for cycles
        for a in graph:
            for b in graph[a]:
                if a in graph.get(b, set()):
                    trace.append(f"CONTRADICTION: {a} > {b} and {b} > {a} cannot both be true!")
                    return {
                        "solution": "The given information is contradictory.",
                        "trace": trace,
                        "consistent": False,
                        "graph": {k: sorted(list(v)) for k, v in graph.items()},
                        "equals": {k: sorted(list(v)) for k, v in equals.items()}
                    }

        # Find extremes
        trace.append("Finding the extremes...")

        # Maximum: nodes with no incoming edges (considering equivalence classes)
        has_incoming = set()
        for source in graph:
            has_incoming.update(graph[source])

        # Filter out nodes that are "less than" others
        candidates_max = []
        for node in all_mentioned:
            if node not in has_incoming:
                # Check if any equivalent node has incoming edges
                equiv_class = equiv_classes_map.get(node, {node})
                if not any(eq_node in has_incoming for eq_node in equiv_class):
                    candidates_max.append(node)

        candidates_max = sorted(list(set(candidates_max)))

        # Minimum: nodes with no outgoing edges (considering equivalence classes)
        candidates_min = []
        for node in all_mentioned:
            equiv_class = equiv_classes_map.get(node, {node})
            # Check if any node in the equivalence class has outgoing edges
            if not any(eq_node in graph and graph[eq_node] for eq_node in equiv_class):
                candidates_min.append(node)

        candidates_min = sorted(list(set(candidates_min)))

        trace.append(f"Has incoming edges: {sorted(has_incoming)}")
        trace.append(f"Maximum candidates (no incoming): {candidates_max}")
        trace.append(f"Minimum candidates (no outgoing): {candidates_min}")

        # Create ordering with equality groups maintained
        ordering = self._topological_sort_with_equals(graph, equiv_classes_map, all_mentioned)

        solution = {
            f"{relation_type}est": candidates_max[0] if len(candidates_max) == 1 else candidates_max,
            f"least {relation_type}": candidates_min[0] if len(candidates_min) == 1 else candidates_min,
            "ordering": ordering
        }

        return {
            "solution": solution,
            "trace": trace,
            "consistent": True,
            "graph": {k: sorted(list(v)) for k, v in graph.items()},
            "equals": {k: sorted(list(v)) for k, v in equals.items()}
        }



    def _topological_sort(self, graph: Dict[str, Set[str]], all_nodes: Set[str]) -> List[str]:
        """Perform topological sort for ordering - FIXED version."""
        # FIX: Initialize in_degree for ALL nodes, not just those with outgoing edges
        in_degree = {node: 0 for node in all_nodes}

        # Count incoming edges
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
                else:
                    # Handle nodes that appear only as targets
                    in_degree[neighbor] = 1

        # Find nodes with no incoming edges
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Remove edges from this node
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # If we processed all nodes, return the order; otherwise there's a cycle
        return result if len(result) == len(all_nodes) else None

    def solve_spatial_layout(self, objects: List[str], constraints: List[str]) -> Dict[str, Any]:
        """Solve spatial arrangement with proper between constraint handling."""
        trace = []
        trace.append("Analyzing spatial constraints...")

        # Build constraint graph
        left_of = defaultdict(set)
        between_constraints = []

        # Parse constraints
        for constraint in constraints:
            if "is to the left of" in constraint:
                parts = constraint.split(" is to the left of ")
                if len(parts) == 2:
                    a, b = parts[0].strip(), parts[1].strip()
                    left_of[a].add(b)
                    trace.append(f"Constraint: {a} < {b}")
            elif "is to the right of" in constraint:
                parts = constraint.split(" is to the right of ")
                if len(parts) == 2:
                    a, b = parts[0].strip(), parts[1].strip()
                    left_of[b].add(a)  # Convert to left_of
                    trace.append(f"Constraint: {b} < {a}")
            elif "is between" in constraint and "and" in constraint:
                match = re.search(r"(.+) is between (.+) and (.+)", constraint)
                if match:
                    x, y, z = [s.strip() for s in match.groups()]
                    between_constraints.append((x, y, z))
                    trace.append(f"Constraint: {x} between {y} and {z}")

        # Helper function to check reachability
        def can_reach(start, end):
            visited = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node == end:
                    return True
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(left_of.get(node, set()))
            return False

        # Process between constraints
        for x, y, z in between_constraints:
            if can_reach(y, z):
                # y < z, so y < x < z
                left_of[y].add(x)
                left_of[x].add(z)
                trace.append(f"Inferred: {y} < {x} < {z}")
            elif can_reach(z, y):
                # z < y, so z < x < y
                left_of[z].add(x)
                left_of[x].add(y)
                trace.append(f"Inferred: {z} < {x} < {y}")
            else:
                # No order determined - make consistent arbitrary choice
                # Choose lexicographic order for consistency
                if y < z:
                    left_of[y].add(x)
                    left_of[x].add(z)
                    trace.append(f"Assumed: {y} < {x} < {z} (arbitrary)")
                else:
                    left_of[z].add(x)
                    left_of[x].add(y)
                    trace.append(f"Assumed: {z} < {x} < {y} (arbitrary)")

        # Compute transitive closure
        trace.append("Computing transitive closure...")
        changed = True
        iterations = 0
        max_iterations = len(objects) ** 2

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for a in list(left_of.keys()):
                for b in list(left_of[a]):
                    for c in left_of.get(b, set()):
                        if c not in left_of[a]:
                            left_of[a].add(c)
                            trace.append(f"Transitive: {a} < {b} < {c}, so {a} < {c}")
                            changed = True

        # Check for cycles
        for a in left_of:
            if a in left_of[a]:
                trace.append(f"ERROR: Cycle detected! {a} < {a}")
                return {"solution": None, "trace": trace, "consistent": False}

        # Topological sort
        all_objects = set(objects)
        in_degree = {obj: 0 for obj in all_objects}

        for a in left_of:
            for b in left_of[a]:
                if b in in_degree:
                    in_degree[b] += 1

        queue = deque([obj for obj in all_objects if in_degree[obj] == 0])
        result = []

        while queue:
            obj = queue.popleft()
            result.append(obj)

            for neighbor in left_of.get(obj, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if len(result) != len(all_objects):
            trace.append("ERROR: Could not find valid ordering")
            return {"solution": None, "trace": trace, "consistent": False}

        trace.append(f"Found valid ordering: {' < '.join(result)}")

        # Find middle position
        middle_idx = len(result) // 2
        middle_obj = result[middle_idx] if result else None
        trace.append(f"Middle position (index {middle_idx}): {middle_obj}")

        return {
            "solution": result,
            "trace": trace,
            "consistent": True,
            "middle": middle_obj
        }
    def _solve_with_constraint_propagation(self, objects: List[str],
                                                    left_of: Dict[str, Set[str]],
                                                    between_constraints: List[Tuple[str, str, str]],
                                                    trace: List[str]) -> Dict[str, Any]:
        """Enhanced constraint propagation that properly handles 'between' constraints."""

        # First, process between constraints to derive ordering
        for x, y, z in between_constraints:
            # Check if we know the order of y and z
            if z in left_of.get(y, set()):
                # y < z, so y < x < z
                left_of[y].add(x)
                left_of[x].add(z)
                trace.append(f"Inferred from between: {y} < {x} < {z}")
            elif y in left_of.get(z, set()):
                # z < y, so z < x < y
                left_of[z].add(x)
                left_of[x].add(y)
                trace.append(f"Inferred from between: {z} < {x} < {y}")
            else:
                # We don't know the order of y and z yet
                # Try to infer from other constraints

                # Check transitive relations
                y_before_z = False
                z_before_y = False

                # Use BFS to check reachability
                from collections import deque

                def is_before(a, b, graph):
                    """Check if a comes before b transitively."""
                    visited = set()
                    queue = deque([a])
                    while queue:
                        node = queue.popleft()
                        if node == b:
                            return True
                        if node in visited:
                            continue
                        visited.add(node)
                        queue.extend(graph.get(node, set()))
                    return False

                if is_before(y, z, left_of):
                    y_before_z = True
                elif is_before(z, y, left_of):
                    z_before_y = True

                if y_before_z:
                    left_of[y].add(x)
                    left_of[x].add(z)
                    trace.append(f"Inferred from between (transitive): {y} < {x} < {z}")
                elif z_before_y:
                    left_of[z].add(x)
                    left_of[x].add(y)
                    trace.append(f"Inferred from between (transitive): {z} < {x} < {y}")
                else:
                    # Still ambiguous - make a consistent choice
                    # Default: lexicographic order of y and z
                    if y < z:
                        left_of[y].add(x)
                        left_of[x].add(z)
                        trace.append(f"Assumed from between (default): {y} < {x} < {z}")
                    else:
                        left_of[z].add(x)
                        left_of[x].add(y)
                        trace.append(f"Assumed from between (default): {z} < {x} < {y}")

        # Now compute transitive closure
        trace.append("Computing transitive closure after between constraints...")
        changed = True
        iterations = 0
        max_iterations = len(objects) ** 2

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for a in list(left_of.keys()):
                for b in list(left_of[a]):
                    for c in left_of.get(b, set()):
                        if c not in left_of[a]:
                            left_of[a].add(c)
                            changed = True

        # Continue with topological sort...
        return self._complete_spatial_solution(objects, left_of, trace)

    def solve_deontic_logic(self, agents: List[str], statements: List[str],
                            rules: List[str], question: str) -> Dict[str, Any]:
        """Solve deontic logic problems about obligations and permissions."""
        trace = []
        trace.append("Analyzing deontic statements...")

        # Parse statements into structured form
        obligations = defaultdict(set)
        permissions = defaultdict(set)
        prohibitions = defaultdict(set)

        for statement in statements:
            # Parse obligations
            for obligation in EntityPools.OBLIGATIONS:
                if obligation in statement:
                    match = re.search(f"(.+) {obligation} (.+)", statement)
                    if match:
                        agent, action = match.groups()
                        obligations[agent.strip()].add(action.strip())
                        trace.append(f"Obligation: {agent} must {action}")

            # Parse permissions
            for permission in EntityPools.PERMISSIONS:
                if permission in statement:
                    match = re.search(f"(.+) {permission} (.+)", statement)
                    if match:
                        agent, action = match.groups()
                        permissions[agent.strip()].add(action.strip())
                        trace.append(f"Permission: {agent} may {action}")

            # Parse prohibitions
            for prohibition in EntityPools.PROHIBITIONS:
                if prohibition in statement:
                    match = re.search(f"(.+) {prohibition} (.+)", statement)
                    if match:
                        agent, action = match.groups()
                        prohibitions[agent.strip()].add(action.strip())
                        trace.append(f"Prohibition: {agent} must not {action}")

        # Apply deontic rules
        trace.append("Applying deontic principles...")

        # Rule: If obligated to X, then permitted to X
        for agent, obs in obligations.items():
            for action in obs:
                if action not in permissions[agent]:
                    permissions[agent].add(action)
                    trace.append(f"Derived: {agent} is permitted to {action} (from obligation)")

        # Check for conflicts
        trace.append("Checking for deontic conflicts...")
        conflicts = []

        for agent in agents:
            # Check obligation-prohibition conflicts
            common_op = obligations[agent] & prohibitions[agent]
            if common_op:
                conflicts.append(f"{agent} is both obligated and prohibited from: {common_op}")
                trace.append(f"CONFLICT: {agent} cannot be both obligated and prohibited from {common_op}")

        return {
            "solution": {
                "obligations": dict(obligations),
                "permissions": dict(permissions),
                "prohibitions": dict(prohibitions),
                "conflicts": conflicts
            },
            "trace": trace,
            "consistent": len(conflicts) == 0
        }



# === Narrative Generator ===

class NarrativeGenerator:
    """Convert logical facts into natural language narratives."""

    def __init__(self):
        self.transition_phrases = [
            "Meanwhile, ", "Additionally, ", "It was also noted that ",
            "Furthermore, ", "In addition, ", "Moreover, ", "Interestingly, ",
            "At the same time, ", "It turns out that ", "As it happens, "
        ]

        self.story_openers = [
            "During a {event}, several observations were made. ",
            "A group of {people} were discussing {topic}. ",
            "At the {location}, an interesting situation arose. ",
            "While examining the {context}, we discovered that ",
            "The following facts came to light during {activity}: "
        ]

    def generate_narrative(self, relations: List[str], entities: List[str],
                           problem_type: str) -> str:
        """Convert a list of relations into a narrative paragraph."""
        # Select appropriate story opener
        opener_template = random.choice(self.story_openers)

        context_options = {
            'event': ['meeting', 'conference', 'investigation', 'study', 'gathering'],
            'people': ['friends', 'colleagues', 'researchers', 'students', 'experts'],
            'topic': ['their findings', 'recent events', 'various attributes', 'the situation'],
            'location': ['office', 'laboratory', 'university', 'conference room'],
            'context': ['evidence', 'data', 'records', 'testimonies'],
            'activity': ['the review process', 'our analysis', 'the investigation']
        }

        # Fill in the opener
        opener = opener_template
        for key, options in context_options.items():
            if f"{{{key}}}" in opener:
                opener = opener.replace(f"{{{key}}}", random.choice(options))

        # Convert relations to narrative sentences
        narrative_parts = [opener]

        for i, relation in enumerate(relations):
            # Add transition phrase for non-first relations
            if i > 0 and random.random() > 0.3:
                relation = random.choice(self.transition_phrases) + relation[0].lower() + relation[1:]

            # Make the relation more narrative-like
            relation = self._narrativize_relation(relation)
            narrative_parts.append(relation)

        # Join with appropriate punctuation
        narrative = " ".join(narrative_parts)

        # Add concluding context
        if random.random() > 0.5:
            conclusions = [
                " These facts need to be considered together.",
                " The question remains about the full implications.",
                " From this information, certain conclusions can be drawn.",
                " This presents an interesting puzzle to solve."
            ]
            narrative += random.choice(conclusions)

        return narrative

    def _narrativize_relation(self, relation: str) -> str:
        """Convert a formal relation into a more narrative form."""
        # Add variety to common patterns
        replacements = [
            ("is taller than", ["stands taller than", "has greater height than", "towers over"]),
            ("is older than", ["has lived longer than", "was born before", "has more years than"]),
            ("owns", ["is the owner of", "possesses", "has in their possession"]),
            ("is located in", ["can be found in", "resides in", "is situated in"]),
            ("must", ["has an obligation to", "is required to", "needs to"]),
            ("may", ["has permission to", "is allowed to", "can choose to"])
        ]

        for original, alternatives in replacements:
            if original in relation and random.random() > 0.5:
                relation = relation.replace(original, random.choice(alternatives))

        return relation


# === Enhanced Problem Generator ===

class EnhancedProblemGenerator:
    """Problem generator with all advanced features."""

    def __init__(self, seed: Optional[int] = None, cache_size: int = 10000):
        if seed:
            random.seed(seed)
        self.entity_pool = EntityPools()
        self.solver = LogicalSolver()
        self.narrative_gen = NarrativeGenerator()
        self.cache_timestamps = {}  # For TTL
        self.problem_cache = {}
        self.max_cache_size = cache_size
        self.generated_count = 0

    def generate_problem(self,
                         complexity: int = 3,
                         problem_type: Optional[str] = None,
                         ensure_unique: bool = True,
                         use_narrative: bool = False,
                         include_ambiguity: bool = False,
                         solvable: bool = True) -> Dict[str, Any]:
        """Generate a problem with enhanced features - FIXED with cache management."""

        # Select problem type
        if problem_type is None:
            problem_type = self._select_problem_type(complexity)

        # Generate base problem
        problem_generators = {
            'transitive_chain': self._generate_transitive_chain,
            'spatial_layout': self._generate_spatial_layout,
            'deontic_reasoning': self._generate_deontic_reasoning,
            'quantitative_logic': self._generate_quantitative_logic,
            'logical_puzzle': self._generate_logical_puzzle,
            'constraint_satisfaction': self._generate_constraint_satisfaction,
            'modal_reasoning': self._generate_modal_reasoning,
            'temporal_sequence': self._generate_temporal_sequence,
            'set_operations': self._generate_set_operations,
            'causal_network': self._generate_causal_network,
            'unsolvable_problem': self._generate_unsolvable_problem
        }

        generator = problem_generators.get(problem_type, self._generate_transitive_chain)

        # Handle unsolvable problems
        if not solvable:
            problem = self._generate_unsolvable_problem(complexity)
        else:
            problem = generator(complexity)
            if problem_type == 'unsolvable_problem':
                problem_type = 'transitive_chain'
            if 'entities' not in problem:
                # Add default entities based on problem type WITHOUT regenerating
                if problem_type == 'deontic_reasoning':
                    problem['entities'] = problem.get('agents', [])
                elif problem_type == 'modal_reasoning':
                    problem['entities'] = problem.get('agents', [])
                elif problem_type == 'temporal_sequence':
                    problem['entities'] = [event['name'] for event in problem.get('events', [])]
                elif problem_type == 'set_operations':
                    problem['entities'] = list(problem.get('sets', {}).keys())
                elif problem_type == 'causal_network':
                    problem['entities'] = problem.get('variables', [])
                elif problem_type == 'constraint_satisfaction':
                    problem['entities'] = problem.get('variables', [])
                elif problem_type == 'logical_puzzle':
                    problem['entities'] = problem.get('categories', {}).get('people', [])
                else:
                    # Generic fallback entities
                    problem['entities'] = [f"Entity{i}" for i in range(max(3, complexity // 2))]



        # Apply narrative transformation if requested
        if use_narrative and 'relations' in problem:
            original_prompt = problem['prompt']
            narrative_version = self.narrative_gen.generate_narrative(
                problem['relations'],
                problem.get('entities', []),
                problem_type
            )
            problem['prompt'] = narrative_version + "\n\n" + problem.get('question', '')
            problem['original_prompt'] = original_prompt

        # Add controlled ambiguity if requested
        if include_ambiguity:
            problem = self._add_ambiguity(problem)

        # Ensure uniqueness
        if ensure_unique:
            problem_hash = self._hash_problem(problem)
            # FIX: Check if the problem already exists BEFORE adding it to the cache.
            if problem_hash in self.problem_cache:
                # If it exists, recurse to generate a different one.
                return self.generate_problem(complexity, problem_type, ensure_unique,
                                             use_narrative, include_ambiguity, solvable)
            # If the problem is unique, add it to the cache.
            self._manage_cache(problem_hash)

        # Add metadata
        problem['metadata'] = {
            'complexity': complexity,
            'type': problem_type,
            'id': f"PROB_{self.generated_count:06d}",
            'features': {
                'narrative': use_narrative,
                'ambiguous': include_ambiguity,
                'solvable': solvable
            },
            'timestamp': self.generated_count
        }

        self.generated_count += 1
        return problem

    def _manage_cache(self, problem_hash: str):
        """Implement bounded cache with TTL."""
        current_time = datetime.now()

        # Add to cache
        self.problem_cache[problem_hash] = True
        self.cache_timestamps[problem_hash] = current_time

        # Remove old entries (TTL = 1 hour)
        ttl = timedelta(hours=1)
        expired = [h for h, t in self.cache_timestamps.items()
                   if current_time - t > ttl]
        for h in expired:
            self.problem_cache.pop(h, None)
            self.cache_timestamps.pop(h, None)

        # Enforce size limit (LRU)
        if len(self.problem_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_hashes = sorted(self.cache_timestamps.items(),
                                   key=lambda x: x[1])
            to_remove = len(self.problem_cache) - self.max_cache_size
            for h, _ in sorted_hashes[:to_remove]:
                self.problem_cache.pop(h, None)
                self.cache_timestamps.pop(h, None)

    def _generate_spatial_layout(self, complexity: int) -> Dict[str, Any]:
        """Generate spatial reasoning problems."""
        num_objects = min(3 + complexity // 2, 8)
        objects = random.sample(self.entity_pool.OBJECTS[:20], num_objects)

        # Generate spatial constraints
        constraints = []
        spatial_predicates = [
            "is to the left of", "is to the right of", "is above", "is below",
            "is next to", "is between"
        ]

        # Create a valid arrangement first
        arrangement = list(objects)
        random.shuffle(arrangement)

        # Generate constraints based on the arrangement
        for i in range(complexity + 2):
            if i < len(arrangement) - 1:
                if random.random() < 0.7:
                    # Direct adjacency
                    constraints.append(f"{arrangement[i]} is to the left of {arrangement[i + 1]}")
                else:
                    # Skip some positions - FIX: Check bounds before skipping
                    if i + 2 < len(arrangement):
                        j = random.randint(i + 2, len(arrangement) - 1)
                        constraints.append(f"{arrangement[i]} is to the left of {arrangement[j]}")
                    else:
                        # Fall back to direct adjacency if can't skip
                        constraints.append(f"{arrangement[i]} is to the left of {arrangement[i + 1]}")

        # Add "between" constraints for higher complexity
        if complexity > 4 and len(arrangement) >= 3:
            for _ in range(complexity // 3):
                indices = sorted(random.sample(range(len(arrangement)), 3))
                constraints.append(
                    f"{arrangement[indices[1]]} is between {arrangement[indices[0]]} and {arrangement[indices[2]]}"
                )

        # Add some vertical constraints for 2D layouts
        if complexity > 6:
            vertical_objects = random.sample(objects, min(3, len(objects)))
            for i in range(len(vertical_objects) - 1):
                constraints.append(f"{vertical_objects[i]} is above {vertical_objects[i + 1]}")

        random.shuffle(constraints)

        # Generate question
        questions = [
            f"What is the complete left-to-right ordering of the objects?",
            f"Which object is in the middle position?",
            f"What objects are adjacent to {random.choice(objects)}?",
            f"Is {random.choice(objects)} to the left of {random.choice(objects)}?"
        ]

        if complexity > 5:
            questions.extend([
                "Draw a valid 2D layout satisfying all constraints.",
                "How many valid arrangements exist?",
                "What is the minimum number of moves needed to reverse the order?"
            ])

        question = random.choice(questions)

        # Solve it
        solution = self.solver.solve_spatial_layout(objects, constraints)

        return {
            'prompt': f"Given the following spatial arrangement constraints:\n" +
                      "\n".join(f"- {c}" for c in constraints) + f"\n\n{question}",
            'relations': constraints,
            'entities': objects,
            'question': question,
            'solution': solution['solution'],
            'solution_trace': solution['trace'],
            'reasoning_type': 'spatial_layout',
            'consistent': solution['consistent']
        }

    def _generate_deontic_reasoning(self, complexity: int) -> Dict[str, Any]:
        """Generate deontic logic problems - FIXED to ensure conditionals."""
        num_agents = min(3 + complexity // 3, 6)
        agents = random.sample(self.entity_pool.PEOPLE_NAMES, num_agents)
        actions = random.sample(self.entity_pool.ACTIONS, min(complexity + 2, len(self.entity_pool.ACTIONS)))

        # Generate deontic statements
        statements = []
        conditional_included = False

        for i in range(complexity + 3):
            agent = random.choice(agents)
            action = random.choice(actions)
            deontic_type = random.choice(['obligation', 'permission', 'prohibition'])

            if deontic_type == 'obligation':
                modal = random.choice(self.entity_pool.OBLIGATIONS)
                statement = f"{agent} {modal} {action}"
            elif deontic_type == 'permission':
                modal = random.choice(self.entity_pool.PERMISSIONS)
                statement = f"{agent} {modal} {action}"
            else:  # prohibition
                modal = random.choice(self.entity_pool.PROHIBITIONS)
                statement = f"{agent} {modal} {action}"

            # ENSURE conditions for higher complexity
            if complexity > 5 and (not conditional_included or random.random() < 0.3):
                condition = random.choice([
                    "if the deadline is met",
                    "unless there is an emergency",
                    "only during business hours",
                    "when authorized by a supervisor"
                ])
                statement += f" {condition}"
                conditional_included = True

            statements.append(statement)

        # Add rules
        rules = [
            "If someone is obligated to do something, they are permitted to do it.",
            "No one can be both obligated and prohibited from the same action.",
            "Permissions can be revoked, but obligations cannot."
        ]

        if complexity > 5:
            rules.extend([
                "If X is obligated to do A, and A requires B, then X is permitted to do B.",
                "Collective obligations apply to each member individually.",
                "Emergency situations override normal prohibitions."
            ])

        # Generate question
        questions = [
            f"What is {random.choice(agents)} obligated to do?",
            f"Is {random.choice(agents)} permitted to {random.choice(actions)}?",
            "Are there any conflicts in the given obligations and prohibitions?",
            "Who has the most obligations?"
        ]

        if complexity > 6:
            questions.extend([
                "Design a schedule where all agents can fulfill their obligations without conflicts.",
                "If we must minimize total actions, what is the optimal assignment?",
                "Which permissions are derived from obligations versus explicitly stated?"
            ])

        question = random.choice(questions)

        # Solve it
        solution = self.solver.solve_deontic_logic(agents, statements, rules, question)

        return {
            'prompt': self._format_deontic_problem(agents, statements, rules, question),
            'agents': agents,
            'actions': actions,
            'statements': statements,  # Always include
            'rules': rules,
            'question': question,
            'solution': solution['solution'],
            'solution_trace': solution['trace'],
            'reasoning_type': 'deontic_reasoning',
            'consistent': solution['consistent'],
            'entities': agents,  # For consistency
            'relations': statements  # For consistency
        }


    def _generate_quantitative_logic(self, complexity: int) -> Dict[str, Any]:
        """Generate problems mixing logical and quantitative reasoning."""
        num_entities = min(4 + complexity // 2, 8)
        entities = random.sample(self.entity_pool.PEOPLE_NAMES, num_entities)

        # Generate quantitative relations
        relations = []
        attributes = ['age', 'height', 'weight', 'salary', 'distance']
        attribute = random.choice(attributes)
        unit = {
            'age': 'years',
            'height': 'cm',
            'weight': 'kg',
            'salary': 'dollars',
            'distance': 'km'
        }[attribute]

        # Assign values
        true_values = {entity: random.randint(20, 100) for entity in entities}

        # Ensure we include ratios
        constraint_types = ['exact_difference', 'inequality', 'sum_constraint']
        if complexity >= 4:
            # Add ratio constraints with higher probability
            constraint_types.extend(['ratio', 'ratio'])
        if complexity >= 6:
            constraint_types.append('average_constraint')

        # Generate constraints
        ratio_included = False
        for i in range(complexity + 3):
            if not ratio_included and i >= complexity // 2:
                constraint_type = 'ratio'  # Force at least one ratio
                ratio_included = True
            else:
                constraint_type = random.choice(constraint_types)

            if constraint_type == 'exact_difference':
                e1, e2 = random.sample(entities, 2)
                diff = abs(true_values[e1] - true_values[e2])
                relations.append(f"The {attribute} difference between {e1} and {e2} is {diff} {unit}")

            elif constraint_type == 'ratio':
                e1, e2 = random.sample(entities, 2)
                if true_values[e2] > 0:  # Avoid division by zero
                    ratio = round(true_values[e1] / true_values[e2], 1)
                    relations.append(f"{e1}'s {attribute} is {ratio} times {e2}'s {attribute}")
                    ratio_included = True

            elif constraint_type == 'sum_constraint':
                group = random.sample(entities, random.randint(2, min(4, len(entities))))
                total = sum(true_values[e] for e in group)
                relations.append(f"The combined {attribute} of {', '.join(group)} is {total} {unit}")

            elif constraint_type == 'average_constraint':
                group = random.sample(entities, random.randint(3, min(5, len(entities))))
                avg = sum(true_values[e] for e in group) / len(group)
                relations.append(f"The average {attribute} of {', '.join(group)} is {avg:.1f} {unit}")

            else:  # inequality
                e1, e2 = random.sample(entities, 2)
                diff = random.randint(5, 20)
                if true_values[e1] > true_values[e2]:
                    relations.append(f"{e1}'s {attribute} is at least {diff} {unit} more than {e2}'s")

        # Generate question
        questions = [
            f"What is {random.choice(entities)}'s exact {attribute}?",
            f"Order all people by {attribute} from lowest to highest.",
            f"What is the maximum possible {attribute} for {random.choice(entities)}?",
            f"Is there a unique solution for everyone's {attribute}?"
        ]

        question = random.choice(questions)

        return {
            'prompt': f"Given the following {attribute} relationships:\n" +
                      "\n".join(f"- {r}" for r in relations) + f"\n\n{question}",
            'relations': relations,
            'entities': entities,
            'attribute': attribute,
            'unit': unit,
            'true_values': true_values,
            'question': question,
            'reasoning_type': 'quantitative_logic'
        }
    def _generate_unsolvable_problem(self, complexity: int) -> Dict[str, Any]:
        """Generate intentionally unsolvable or contradictory problems."""
        problem_types = ['contradictory', 'underspecified', 'impossible']
        selected_type = random.choice(problem_types)

        if selected_type == 'contradictory':
            # Create a problem with built-in contradictions
            entities = random.sample(self.entity_pool.PEOPLE_NAMES, 4)
            relations = [
                f"{entities[0]} is taller than {entities[1]}",
                f"{entities[1]} is taller than {entities[2]}",
                f"{entities[2]} is taller than {entities[0]}"  # Creates a cycle
            ]

            # Add more relations to obscure the contradiction
            for _ in range(complexity):
                e1, e2 = random.sample(entities, 2)
                relations.append(f"{e1} is taller than {e2}")

            random.shuffle(relations)

            question = "List all people in order from tallest to shortest."
            expected_answer = "This problem is contradictory. There is a circular relationship that makes a consistent ordering impossible."

        elif selected_type == 'underspecified':
            # Create a problem with insufficient information
            entities = random.sample(self.entity_pool.OBJECTS[:10], 5)
            colors = random.sample(self.entity_pool.COLORS[:5], 3)

            relations = [
                f"The {entities[0]} is {colors[0]}",
                f"The {entities[1]} is not {colors[1]}",
                f"Either the {entities[2]} or the {entities[3]} is {colors[2]}"
            ]

            question = f"What color is the {entities[4]}?"
            expected_answer = "There is insufficient information to determine the color. No constraints were given about this object."

        else:  # impossible
            # Create a problem with impossible constraints
            num_items = 5
            relations = [
                f"There are exactly {num_items} items in total",
                f"Each item is either red or blue",
                f"There are more red items than blue items",
                f"There are more blue items than red items",  # Contradiction
                f"No two adjacent items have the same color"
            ]

            question = "How many red items are there?"
            expected_answer = "This problem has no solution. The constraints are mutually exclusive."

        return {
            'prompt': f"Consider the following information:\n" +
                      "\n".join(f"- {r}" for r in relations) + f"\n\n{question}",
            'relations': relations,
            'question': question,
            'expected_answer': expected_answer,
            'reasoning_type': 'unsolvable_problem',
            'unsolvable_type': selected_type,
            'solvable': False
        }

    def _add_ambiguity(self, problem: Dict) -> Dict:
        """Add controlled ambiguity with proper grammar - FIXED version."""
        if 'relations' in problem:
            relations = problem['relations'][:]

            # Add redundant statements
            if len(relations) > 2:
                rel1 = random.choice(relations)

                # Create a grammatically correct paraphrase
                paraphrases = {
                    "is taller than": "has greater height than",
                    "owns": "is the owner of",
                    "is older than": "was born before",
                    "is to the left of": "is positioned left of"
                }

                for original, paraphrase in paraphrases.items():
                    if original in rel1:
                        redundant = rel1.replace(original, paraphrase)
                        relations.insert(random.randint(0, len(relations)), redundant)
                        break

            # Add uncertainty modifiers
            if random.random() < 0.3:
                idx = random.randint(0, len(relations) - 1)
                relations[idx] = "It is likely that " + relations[idx].lower()

            # FIX: Add vague quantifiers with proper grammar
            if random.random() < 0.3:
                vague_terms = ["some", "many", "most", "several", "a few"]
                idx = random.randint(0, len(relations) - 1)
                # Use regex to properly replace "The" at the beginning
                relations[idx] = re.sub(r'^The\s+', f"{random.choice(vague_terms).capitalize()} ",
                                        relations[idx], count=1)

            problem['relations'] = relations
            problem['has_ambiguity'] = True

        return problem

    def _select_problem_type(self, complexity: int) -> str:
        all_types = [
        'transitive_chain', 'spatial_layout', 'logical_puzzle',
        'deontic_reasoning', 'quantitative_logic', 'modal_reasoning',
        'temporal_sequence', 'constraint_satisfaction', 'set_operations',
        'causal_network'
        ]

        # Use a simple deterministic selection based on generated_count
        # This ensures the same sequence with the same seed
        type_index = (self.generated_count * 7 + complexity * 3) % len(all_types)
        return all_types[type_index]

    def _hash_problem(self, problem: Dict) -> str:
        """Generate a hash for problem uniqueness checking."""
        # Convert to JSON-safe format
        safe_problem = safe_json_serialize(problem)
        canonical = json.dumps(safe_problem, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _format_deontic_problem(self, agents: List[str], statements: List[str],
                                rules: List[str], question: str) -> str:
        """Format a deontic reasoning problem."""
        prompt = "Deontic Logic Problem:\n\n"
        prompt += f"Agents: {', '.join(agents)}\n\n"
        prompt += "Obligations, Permissions, and Prohibitions:\n"

        for i, statement in enumerate(statements, 1):
            prompt += f"{i}. {statement}\n"

        prompt += "\nDeontic Principles:\n"
        for i, rule in enumerate(rules, 1):
            prompt += f"- {rule}\n"

        prompt += f"\n{question}"
        return prompt

    def _generate_transitive_chain(self, complexity: int) -> Dict[str, Any]:
        """Generate transitive reasoning chain problems with guaranteed consistency."""
        # Scale entities properly - fewer for low complexity, slightly more for high
        if complexity == 1:
            num_entities = 2  # Just A > B
        elif complexity <= 2:
            num_entities = random.randint(3, 4)
        elif complexity <= 5:
            num_entities = random.randint(4, 5)
        else:
            num_entities = random.randint(4, 6)

        entities = random.sample(self.entity_pool.PEOPLE_NAMES, num_entities)
        relation_type = random.choice(['tall', 'old', 'fast', 'wealthy', 'strong'])

        # Use the robust generator
        chain_gen = TransitiveChainGenerator()
        if complexity == 1:
            # Just one relation for complexity 1
            relations = [f"{entities[0]} is {relation_type}er than {entities[1]}"]
        else:
            relations = chain_gen.generate_consistent_chain(entities, relation_type, complexity)

        if complexity <= 2:
            max_relations = complexity + 1
            relations = relations[:max_relations]

        # Ensure we have enough relations for the complexity
        target_relations = min(complexity + 2, num_entities * (num_entities - 1) // 2)

        attempts = 0
        while len(relations) < target_relations and attempts < 50:
            attempts += 1
            if complexity > 5 and random.random() < 0.3:
                # Try to add equality
                a, b = random.sample(entities, 2)
                if chain_gen.can_add_relation(a, b, "equal"):
                    chain_gen.add_relation(a, b, "equal")
                    relations.append(f"{a} is as {relation_type} as {b}")
            else:
                # Try to add ordering
                a, b = random.sample(entities, 2)
                if chain_gen.can_add_relation(a, b, "greater"):
                    chain_gen.add_relation(a, b, "greater")
                    relations.append(f"{a} is {relation_type}er than {b}")

        # Generate question
        question_templates = [
            f"Who is the {relation_type}est?",
            f"Who is the least {relation_type}?",
            f"Order all people by {relation_type} from highest to lowest.",
            f"Is {random.choice(entities)} {relation_type}er than {random.choice(entities)}?",
            f"How many people are {relation_type}er than {random.choice(entities)}?"
        ]

        if complexity > 5:
            question_templates.extend([
                "Is the given information consistent?",
                "Can you determine a complete ordering? If not, what additional information is needed?"
            ])

        question = random.choice(question_templates)

        # Solve to verify consistency
        solution_data = self.solver.solve_transitive_chain(entities, relations, relation_type)

        return {
            'prompt': f"Given the following information:\n" + "\n".join(
                f"- {r}" for r in relations) + f"\n\n{question}",
            'relations': relations,
            'question': question,
            'solution': solution_data['solution'],
            'solution_trace': solution_data['trace'],
            'consistent': solution_data['consistent'],
            'reasoning_type': 'transitive_chain',
            'entities': entities,
            'relation_type': relation_type
        }

    def _generate_logical_puzzle(self, complexity: int) -> Dict[str, Any]:
        """Generate grid-based logical puzzles."""
        # Categories
        categories = {
            'people': random.sample(self.entity_pool.PEOPLE_NAMES, 4),
            'colors': random.sample(self.entity_pool.COLORS, 4),
            'objects': random.sample(self.entity_pool.OBJECTS, 4),
            'locations': random.sample(self.entity_pool.LOCATIONS[:10], 4)
        }

        if complexity > 5:
            categories['professions'] = random.sample(self.entity_pool.PROFESSIONS, 4)

        # Generate clues
        clues = []
        clue_types_used = set()
        solution_grid = self._create_solution_grid(categories)


        # Force at least one of each basic type
        for clue_type in ['direct', 'negative']:
            for _ in range(5):  # Try up to 5 times
                clue = self._generate_clue(categories, solution_grid, clue_type, complexity)
                if clue:
                    clues.append(clue)
                    clue_types_used.add(clue_type)
                    break

        # Add more clues with variety
        for _ in range(complexity + 2):
            clue_type = random.choice(['direct', 'negative', 'relative', 'compound'])
            clue = self._generate_clue(categories, solution_grid, clue_type, complexity)
            if clue:
                clues.append(clue)
                clue_types_used.add(clue_type)





        # Calculate minimum clues needed
        num_categories = len(categories)
        num_items = len(categories['people'])
        # Need enough clues to uniquely determine the solution
        min_clues_needed = (num_categories - 1) * num_items

        # Generate key clues first
        key_clues = self._generate_key_clues(categories, solution_grid)
        clues.extend(key_clues)

        # Add additional clues to meet minimum
        target_clues = max(min_clues_needed // 2 + 1, complexity + 3)

        attempts = 0
        while len(clues) < target_clues and attempts < 50:
            attempts += 1
            clue_type = random.choice(['direct', 'negative', 'relative', 'compound'])
            clue = self._generate_clue(categories, solution_grid, clue_type, complexity)
            if clue and clue not in clues:
                clues.append(clue)

        random.shuffle(clues)

        # Generate question
        questions = [
            f"Who owns the {random.choice(categories['objects'])}?",
            f"What color is associated with {random.choice(categories['people'])}?",
            f"Match each person with their object, color, and location.",
            "Is there a unique solution? If so, provide it. If not, explain why."
        ]

        question = random.choice(questions)

        return {
            'prompt': self._format_logic_puzzle(categories, clues, question),
            'clues': clues,
            'categories': categories,
            'solution_grid': solution_grid,
            'question': question,
            'reasoning_type': 'logical_puzzle'
        }

    def _generate_constraint_satisfaction(self, complexity: int) -> Dict[str, Any]:
        """Generate constraint satisfaction problems."""
        # Variables
        num_variables = min(3 + complexity // 2, 8)
        variables = [f"X{i}" for i in range(num_variables)]

        # Domains
        domain_size = min(4 + complexity // 3, 10)
        domains = {var: list(range(1, domain_size + 1)) for var in variables}

        # Constraints
        constraints = []
        constraint_types = [
            'all_different',
            'sum_equals',
            'ordered',
            'adjacent_different',
            'at_least_one',
            'at_most_one',
            'implies',
            'arithmetic'
        ]

        num_constraints = complexity + 3
        for _ in range(num_constraints):
            c_type = random.choice(constraint_types[:min(len(constraint_types), complexity)])
            constraint = self._generate_csp_constraint(variables, domains, c_type)
            if constraint:
                constraints.append(constraint)

        # Special constraints for higher complexity
        if complexity > 6:
            # Add global constraints
            constraints.append(f"The sum of all variables must be {'even' if random.random() < 0.5 else 'odd'}")
            constraints.append(f"At least {random.randint(2, num_variables // 2)} variables must have the same value")

        # Generate question
        questions = [
            "Find all valid assignments for the variables.",
            "How many solutions exist?",
            "What is the minimum/maximum value for the sum of all variables?",
            f"Can {random.choice(variables)} equal {random.choice(list(domains[variables[0]]))}? Why or why not?"
        ]

        if complexity > 5:
            questions.extend([
                "Which constraint is the most restrictive?",
                "If we remove one constraint, how many more solutions would we have?",
                "Prove that the problem has a unique solution or show multiple solutions exist."
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_csp_problem(variables, domains, constraints, question),
            'variables': variables,
            'domains': domains,
            'constraints': constraints,
            'question': question,
            'reasoning_type': 'constraint_satisfaction',
            'entities': variables
        }

    def _generate_modal_reasoning(self, complexity: int) -> Dict[str, Any]:
        """Generate problems involving possibility, necessity, and belief."""
        agents = random.sample(self.entity_pool.PEOPLE_NAMES, min(3 + complexity // 3, 6))
        propositions = self._generate_propositions(complexity)

        # Modal statements
        modal_ops = ['knows that', 'believes that', 'thinks that']
        if complexity > 4:
            modal_ops.extend(['is certain that', 'doubts that', 'assumes that'])
        if complexity > 6:
            modal_ops.extend(['knows that X knows that', 'believes that Y believes that'])

        # Initialize statements list BEFORE using it
        statements = []
        negation_included = False

        for i in range(complexity + 2):
            agent = random.choice(agents)
            modal_op = random.choice(modal_ops)
            prop = random.choice(propositions)

            if 'X' in modal_op or 'Y' in modal_op:
                other_agent = random.choice([a for a in agents if a != agent])
                modal_op = modal_op.replace('X', other_agent).replace('Y', other_agent)

            statement = f"{agent} {modal_op} {prop}"

            # Ensure at least one negation
            if (not negation_included and i >= 1) or (random.random() < 0.2):
                statement = f"It is not the case that {statement}"
                negation_included = True

            statements.append(statement)

        # After generating statements, ensure nested modals for high complexity
        if complexity > 6:
            nested_found = any(" that " in s and s.count("that") >= 2 for s in statements)
            if not nested_found:
                agent1, agent2 = random.sample(agents, 2)
                prop = random.choice(propositions)
                nested_statement = f"{agent1} knows that {agent2} knows that {prop}"

                # Randomly decide if this should be negated
                if random.random() < 0.2:
                    nested_statement = f"It is not the case that {nested_statement}"

                statements.append(nested_statement)

        # Add epistemic rules
        rules = [
            "If someone knows something, then they believe it.",
            "If someone knows something, then it is true.",
            "If someone is certain about something, they know it."
        ]

        if complexity > 5:
            rules.extend([
                "If X knows that Y knows P, then X knows P.",
                "If everyone believes P, it doesn't necessarily mean P is true."
            ])

        # Generate question
        questions = [
            f"What does {random.choice(agents)} know for certain?",
            f"Is it possible that {random.choice(propositions)}?",
            "Who has the most accurate beliefs?",
            "Are there any contradictions in the agents' beliefs?"
        ]

        if complexity > 6:
            questions.extend([
                "Construct a possible world where all statements are true.",
                "What is the minimum number of facts needed to explain all beliefs?",
                f"If {random.choice(agents)} is always truthful, what can we deduce?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_modal_problem(agents, statements, rules, question),
            'agents': agents,
            'statements': statements,
            'rules': rules,
            'propositions': propositions,
            'question': question,
            'reasoning_type': 'modal_reasoning',
            'entities': agents
        }
    def _generate_temporal_sequence(self, complexity: int) -> Dict[str, Any]:
        """Generate temporal reasoning problems - FIXED."""
        events = self._generate_events(complexity)

        # Temporal relations
        relations = []
        rel_types = ['before', 'after', 'during', 'overlaps with']
        if complexity > 4:
            rel_types.extend(['starts when', 'ends when', 'throughout'])

        conditional_included = False

        # Generate temporal constraints
        for i in range(complexity * 2):
            e1, e2 = random.sample(events, 2)
            rel = random.choice(rel_types)
            relations.append(f"{e1['name']} {rel} {e2['name']}")

        # Add absolute time constraints
        if complexity > 3:
            for event in random.sample(events, min(len(events), complexity // 2)):
                time_constraint = random.choice([
                    f"happens on {random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])}",
                    f"starts at {random.randint(8, 17)}:00",
                    f"lasts {random.randint(1, 4)} hours",
                    f"must finish before {random.randint(12, 20)}:00"
                ])
                relations.append(f"{event['name']} {time_constraint}")

        # ENSURE complex temporal logic for high complexity
        if complexity > 6:
            if not conditional_included:
                # Force add conditional constraints
                conditional_constraints = [
                    f"If {events[0]['name']} happens, then {events[1]['name']} must happen within 2 hours",
                    f"If {events[0]['name']} is delayed, then {events[1]['name']} must be rescheduled"
                ]
                relations.append(random.choice(conditional_constraints))
                conditional_included = True

            # Add other complex constraints
            complex_constraints = [
                f"No more than {random.randint(2, 4)} events can happen simultaneously",
                f"There must be at least 30 minutes between any two events"
            ]
            relations.extend(random.sample(complex_constraints, min(len(complex_constraints), 1)))

        # Generate question
        questions = [
            "What is a valid ordering of all events?",
            f"When could {random.choice(events)['name']} happen?",
            "What is the minimum time needed to complete all events?",
            "Which events must happen on the same day?"
        ]

        if complexity > 5:
            questions.extend([
                "Is it possible to satisfy all constraints? If not, which ones conflict?",
                "What is the critical path of events?",
                "If we must minimize the total time span, how should we schedule the events?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_temporal_problem(events, relations, question),
            'events': events,
            'relations': relations,  # Always include
            'question': question,
            'reasoning_type': 'temporal_sequence',
            'entities': [e['name'] for e in events],  # For consistency
        }

    def _generate_set_operations(self, complexity: int) -> Dict[str, Any]:
        """Generate problems involving set theory and operations."""
        # Define sets
        num_sets = min(3 + complexity // 2, 6)
        set_names = [chr(65 + i) for i in range(num_sets)]  # A, B, C, ...

        # Universal set
        universal_set = set(range(1, 20 + complexity * 5))

        # Generate sets with specific properties
        sets = {}
        for name in set_names:
            size = random.randint(5, 15)
            sets[name] = set(random.sample(list(universal_set), size))

        # Generate relationships
        relationships = []

        # Basic relationships
        for _ in range(complexity):
            s1, s2 = random.sample(set_names, 2)
            rel_type = random.choice([
                'subset', 'disjoint', 'overlapping', 'equal'
            ])

            # Enforce relationship
            if rel_type == 'subset':
                sets[s2] = sets[s2].union(sets[s1])
                relationships.append(f"{s1} ⊆ {s2}")
            elif rel_type == 'disjoint':
                sets[s2] = sets[s2] - sets[s1]
                relationships.append(f"{s1} ∩ {s2} = ∅")
            elif rel_type == 'equal':
                sets[s2] = sets[s1].copy()
                relationships.append(f"{s1} = {s2}")
            else:
                relationships.append(f"{s1} and {s2} have {random.randint(1, 5)} common elements")

        # Complex operations
        if complexity > 4:
            operations = []
            for _ in range(complexity // 2):
                sets_involved = random.sample(set_names, random.randint(2, min(4, len(set_names))))
                op_type = random.choice(['union', 'intersection', 'difference', 'symmetric_difference'])

                if op_type == 'union':
                    op_str = ' ∪ '.join(sets_involved)
                elif op_type == 'intersection':
                    op_str = ' ∩ '.join(sets_involved)
                elif op_type == 'difference':
                    op_str = f"{sets_involved[0]} - ({' ∪ '.join(sets_involved[1:])})"
                else:
                    op_str = f"{sets_involved[0]} Δ {sets_involved[1]}"

                result_size = random.randint(0, 20)
                operations.append(f"|{op_str}| = {result_size}")

            relationships.extend(operations)

        # Generate question
        questions = [
            f"How many elements are in {random.choice(set_names)}?",
            f"What is {random.choice(set_names)} ∩ {random.choice(set_names)}?",
            "Which sets are disjoint?",
            f"Is {random.randint(1, 30)} in set {random.choice(set_names)}?"
        ]

        if complexity > 5:
            questions.extend([
                "What is the minimum number of elements that must be in the universal set?",
                "Find all possible values for |A ∪ B ∪ C|",
                "If we add one element to set A, how does it affect the other relationships?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_set_problem(set_names, relationships, question),
            'sets': {name: list(s) for name, s in sets.items()},
            'relationships': relationships,
            'question': question,
            'reasoning_type': 'set_operations'
        }

    def _generate_causal_network(self, complexity: int) -> Dict[str, Any]:
        """Generate causal reasoning problems."""
        # Generate events/variables
        num_vars = min(4 + complexity // 2, 8)
        variables = []

        var_types = ['event', 'condition', 'action', 'state']
        for i in range(num_vars):
            var_type = random.choice(var_types)
            if var_type == 'event':
                name = random.choice([
                    "rain", "traffic", "delay", "accident", "meeting", "alarm",
                    "power_outage", "celebration", "construction", "emergency"
                ])
            elif var_type == 'condition':
                name = random.choice([
                    "wet_roads", "crowded", "busy", "closed", "available",
                    "operational", "safe", "dangerous", "optimal", "critical"
                ])
            elif var_type == 'action':
                name = random.choice([
                    "leave_early", "take_alternate_route", "cancel_plans",
                    "call_ahead", "wait", "proceed", "stop", "continue"
                ])
            else:  # state
                name = random.choice([
                    "happy", "stressed", "late", "prepared", "successful",
                    "failed", "completed", "pending", "active", "inactive"
                ])

            variables.append(f"{name}_{i}")

        # Generate causal relationships
        causal_links = []

        # Direct causation
        for _ in range(complexity + 3):
            cause = random.choice(variables)
            effect = random.choice([v for v in variables if v != cause])

            link_type = random.choice([
                "causes", "prevents", "increases probability of",
                "decreases probability of", "is necessary for", "is sufficient for"
            ])

            if complexity > 5 and random.random() < 0.3:
                # Conditional causation
                condition = random.choice([v for v in variables if v not in [cause, effect]])
                causal_links.append(f"{cause} {link_type} {effect} only if {condition}")
            else:
                causal_links.append(f"{cause} {link_type} {effect}")

        # Add feedback loops for high complexity
        if complexity > 6:
            # Create a cycle
            cycle_vars = random.sample(variables, 3)
            for i in range(len(cycle_vars)):
                causal_links.append(
                    f"{cycle_vars[i]} influences {cycle_vars[(i + 1) % len(cycle_vars)]}"
                )

        # Observations
        observations = []
        for _ in range(random.randint(1, 3)):
            var = random.choice(variables)
            state = random.choice(["occurred", "did not occur", "is true", "is false"])
            observations.append(f"{var} {state}")

        # Generate question
        questions = [
            f"What could have caused {random.choice(variables)}?",
            f"What are the likely effects of {random.choice(variables)}?",
            "What is the root cause of the observed situation?",
            "Which intervention would be most effective?"
        ]

        if complexity > 5:
            questions.extend([
                "Are there any causal cycles? What are their implications?",
                "What is the minimal set of interventions needed to achieve a desired outcome?",
                "Given the observations, what can we infer about the unobserved variables?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_causal_problem(variables, causal_links, observations, question),
            'variables': variables,
            'causal_links': causal_links,
            'observations': observations,
            'question': question,
            'reasoning_type': 'causal_network'
        }

    def _generate_preference_ordering(self, complexity: int) -> Dict[str, Any]:
        """Generate preference and voting problems."""
        # Agents and choices
        num_agents = min(3 + complexity // 3, 7)
        num_choices = min(4 + complexity // 3, 8)

        agents = random.sample(self.entity_pool.PEOPLE_NAMES, num_agents)
        choices = random.sample(self.entity_pool.OBJECTS[:20], num_choices)

        # Generate preferences
        preferences = {}
        for agent in agents:
            # Each agent has a preference ordering
            if complexity > 5 and random.random() < 0.3:
                # Incomplete preferences
                pref_subset = random.sample(choices, random.randint(3, len(choices) - 1))
                preferences[agent] = random.sample(pref_subset, len(pref_subset))
            else:
                preferences[agent] = random.sample(choices, len(choices))

        # Generate preference statements
        statements = []
        for agent in agents:
            pref = preferences[agent]

            # Direct preferences
            for i in range(min(complexity, len(pref) - 1)):
                if random.random() < 0.7:
                    statements.append(f"{agent} prefers {pref[i]} to {pref[i + 1]}")
                else:
                    # Indirect statement
                    j = random.randint(i + 2, len(pref) - 1)
                    statements.append(f"{agent} prefers {pref[i]} to {pref[j]}")

            # Indifference statements for complexity
            if complexity > 4 and random.random() < 0.3:
                i1, i2 = random.sample(range(len(pref)), 2)
                statements.append(f"{agent} is indifferent between {pref[i1]} and {pref[i2]}")

        # Group preferences
        if complexity > 6:
            group_statements = [
                f"The majority prefers {random.choice(choices)} to {random.choice(choices)}",
                f"Everyone agrees that {random.choice(choices)} is better than {random.choice(choices)}",
                f"There is no consensus on the ranking of {random.choice(choices)}"
            ]
            statements.extend(random.sample(group_statements, min(2, len(group_statements))))

        random.shuffle(statements)

        # Generate question
        questions = [
            f"What is {random.choice(agents)}'s most preferred choice?",
            f"Who prefers {random.choice(choices)} the most?",
            "Is there a choice that everyone agrees is the worst?",
            "What would be the group's choice using majority rule?"
        ]

        if complexity > 5:
            questions.extend([
                "Is there a Condorcet winner (beats all others in pairwise comparisons)?",
                "Does the preference profile satisfy transitivity for all agents?",
                "What voting system would produce the fairest outcome?",
                "Are there any voting paradoxes in this scenario?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_preference_problem(agents, choices, statements, question),
            'agents': agents,
            'choices': choices,
            'preferences': preferences,
            'statements': statements,
            'question': question,
            'reasoning_type': 'preference_ordering'
        }

    def _generate_counterfactual(self, complexity: int) -> Dict[str, Any]:
        """Generate counterfactual reasoning problems."""
        # Base scenario
        scenario_types = [
            'historical_event', 'scientific_experiment', 'business_decision',
            'personal_choice', 'natural_phenomenon', 'social_situation'
        ]

        scenario_type = random.choice(scenario_types)
        scenario = self._create_scenario(scenario_type, complexity)

        # Actual outcome
        actual_outcome = scenario['outcome']

        # Counterfactual conditions
        counterfactuals = []
        for _ in range(min(complexity, 4)):
            cf_type = random.choice([
                'change_initial_condition',
                'remove_constraint',
                'add_intervention',
                'alter_timing',
                'change_participant'
            ])

            counterfactual = self._create_counterfactual_condition(scenario, cf_type)
            counterfactuals.append(counterfactual)

        # Causal relationships
        causal_factors = scenario['causal_factors']

        # Generate question
        questions = [
            f"If {counterfactuals[0]}, what would have happened?",
            "Which factor was most critical to the actual outcome?",
            "What is the minimal change needed to prevent the outcome?",
            "Under what conditions would the opposite outcome occur?"
        ]

        if complexity > 5:
            questions.extend([
                "Consider all counterfactuals: which leads to the most different outcome?",
                "Are there any counterfactuals that would lead to the same outcome? Why?",
                "What does this scenario teach us about causation vs correlation?",
                "Design an experiment to test the causal relationships."
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_counterfactual_problem(scenario, counterfactuals, question),
            'scenario': scenario,
            'counterfactuals': counterfactuals,
            'question': question,
            'reasoning_type': 'counterfactual'
        }

    def _generate_meta_reasoning(self, complexity: int) -> Dict[str, Any]:
        """Generate problems about reasoning itself."""
        # Reasoning scenarios
        num_scenarios = min(2 + complexity // 3, 5)
        reasoning_scenarios = []

        for i in range(num_scenarios):
            scenario_type = random.choice([
                'logical_argument', 'mathematical_proof', 'scientific_hypothesis',
                'legal_argument', 'philosophical_claim', 'everyday_reasoning'
            ])

            scenario = {
                'id': f'Argument {i + 1}',
                'type': scenario_type,
                'premises': self._generate_premises(scenario_type, random.randint(2, 4)),
                'conclusion': self._generate_conclusion(scenario_type),
                'reasoning_steps': self._generate_reasoning_steps(scenario_type, complexity)
            }

            # Add flaws for some scenarios
            if random.random() < 0.4 + (complexity * 0.05):
                flaw_type = random.choice([
                    'circular_reasoning', 'false_premise', 'invalid_inference',
                    'equivocation', 'missing_premise', 'overgeneralization',
                    'false_dichotomy', 'ad_hominem', 'straw_man'
                ])
                scenario['flaw'] = flaw_type
                scenario['reasoning_steps'] = self._inject_flaw(
                    scenario['reasoning_steps'], flaw_type
                )

            reasoning_scenarios.append(scenario)

        # Meta-level principles
        principles = [
            "A valid argument can have false premises",
            "A sound argument must be valid and have true premises",
            "Correlation does not imply causation",
            "Absence of evidence is not evidence of absence"
        ]

        if complexity > 5:
            principles.extend([
                "Self-reference can lead to paradoxes",
                "Gödel's incompleteness theorem applies to formal systems",
                "The map is not the territory",
                "All models are wrong, but some are useful"
            ])

        # Generate question
        questions = [
            "Which arguments are valid? Which are sound?",
            "Identify any logical fallacies in the arguments.",
            "Which argument is the strongest? Why?",
            "What additional information would strengthen each argument?"
        ]

        if complexity > 6:
            questions.extend([
                "Compare the reasoning methods used. Which is most appropriate for each domain?",
                "Can you construct a meta-argument about the nature of these arguments?",
                "What are the implicit assumptions in each argument?",
                "How would you test the conclusions empirically?"
            ])

        question = random.choice(questions)

        return {
            'prompt': self._format_meta_reasoning_problem(reasoning_scenarios, principles, question),
            'scenarios': reasoning_scenarios,
            'principles': principles,
            'question': question,
            'reasoning_type': 'meta_reasoning'
        }

    # === Helper Methods ===

    def _topological_sort_with_equals(self, graph: Dict[str, Set[str]],
                                      equiv_classes_map: Dict[str, Set[str]],
                                      all_nodes: Set[str]) -> List[str]:
        """Topological sort that keeps equal entities adjacent."""
        from collections import defaultdict, deque

        # Get unique equivalence classes
        equiv_classes = []
        processed = set()
        node_to_class = {}

        for node in sorted(all_nodes):
            if node not in processed:
                # Get this node's equivalence class
                equiv_class = equiv_classes_map.get(node, {node})
                processed.update(equiv_class)

                class_idx = len(equiv_classes)
                equiv_classes.append(sorted(equiv_class))

                # Map all members to this class index
                for member in equiv_class:
                    node_to_class[member] = class_idx

        # Build DAG of equivalence classes
        class_graph = defaultdict(set)

        for src in graph:
            src_class = node_to_class.get(src, -1)
            if src_class >= 0:
                for dst in graph[src]:
                    dst_class = node_to_class.get(dst, -1)
                    if dst_class >= 0 and src_class != dst_class:
                        class_graph[src_class].add(dst_class)

        # Topological sort on the class graph
        in_degree = [0] * len(equiv_classes)
        for src_class in class_graph:
            for dst_class in class_graph[src_class]:
                in_degree[dst_class] += 1

        # Find all nodes with in-degree 0
        queue = deque([i for i in range(len(equiv_classes)) if in_degree[i] == 0])
        result = []

        while queue:
            class_idx = queue.popleft()
            # Add all members of this equivalence class together
            result.extend(equiv_classes[class_idx])

            # Decrease in-degree of neighbors
            for neighbor in class_graph.get(class_idx, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Return the result if we processed all nodes, otherwise return sorted list
        return result if len(result) == len(all_nodes) else sorted(all_nodes)
    def _create_solution_grid(self, categories: Dict[str, List]) -> Dict:
        """Create a solution grid for logic puzzles."""
        # Create a valid assignment
        grid = {}
        cat_names = list(categories.keys())

        # Create permutations
        indices = list(range(len(categories[cat_names[0]])))

        for i, cat in enumerate(cat_names):
            perm = indices[i:] + indices[:i]  # Rotate
            grid[cat] = {categories[cat][j]: categories[cat_names[0]][perm[j]]
                         for j in range(len(indices))}

        return grid

    def _generate_clue(self, categories: Dict, solution_grid: Dict,
                       clue_type: str, complexity: int) -> Optional[str]:
        """Generate a clue for logic puzzles."""
        if clue_type == 'direct':
            cat1, cat2 = random.sample(list(categories.keys()), 2)
            item1 = random.choice(categories[cat1])
            # Find corresponding item in solution
            for item2 in categories[cat2]:
                if solution_grid[cat1].get(item1) == solution_grid[cat2].get(item2):
                    return f"The {item1} is associated with {item2}"

        elif clue_type == 'negative':
            cat1, cat2 = random.sample(list(categories.keys()), 2)
            item1 = random.choice(categories[cat1])
            # Find non-corresponding item
            wrong_items = [item for item in categories[cat2]
                           if solution_grid[cat1].get(item1) != solution_grid[cat2].get(item)]
            if wrong_items:
                item2 = random.choice(wrong_items)
                return f"The {item1} is NOT associated with {item2}"

        elif clue_type == 'relative':
            # FIX: Add actual relative clues
            cat = 'people'  # Use a category that has natural ordering
            if cat in categories and len(categories[cat]) >= 2:
                people = categories[cat]
                idx1, idx2 = sorted(random.sample(range(len(people)), 2))
                return f"{people[idx1]} sits to the left of {people[idx2]}"

        elif clue_type == 'compound':
            # FIX: Add actual compound clues
            if len(categories) >= 3:
                cats = random.sample(list(categories.keys()), 2)
                person = random.choice(categories['people'])
                item1 = random.choice(categories[cats[0]])
                item2 = random.choice(categories[cats[1]])
                # Check if they match in solution
                if (solution_grid['people'].get(person) == solution_grid[cats[0]].get(item1) and
                        solution_grid['people'].get(person) == solution_grid[cats[1]].get(item2)):
                    return f"{person} has both the {item1} and the {item2}"

        return None

    def _generate_key_clues(self, categories: Dict, solution_grid: Dict) -> List[str]:
        """Generate essential clues for solvability."""
        clues = []
        # Ensure at least one direct clue per category pair
        for cat1, cat2 in itertools.combinations(categories.keys(), 2):
            item1 = random.choice(categories[cat1])
            for item2 in categories[cat2]:
                if solution_grid[cat1].get(item1) == solution_grid[cat2].get(item2):
                    clues.append(f"The {item1} goes with {item2}")
                    break
        return clues

    def _format_logic_puzzle(self, categories: Dict, clues: List[str], question: str) -> str:
        """Format a logic puzzle for presentation."""
        prompt = "Solve this logic puzzle:\n\n"

        # List categories
        for cat_name, items in categories.items():
            prompt += f"{cat_name.capitalize()}: {', '.join(items)}\n"

        prompt += "\nClues:\n"
        for i, clue in enumerate(clues, 1):
            prompt += f"{i}. {clue}\n"

        prompt += f"\n{question}"
        return prompt

    def _generate_csp_constraint(self, variables: List[str], domains: Dict,
                                 constraint_type: str) -> Optional[str]:
        """Generate a constraint for CSP."""
        if constraint_type == 'all_different':
            vars = random.sample(variables, min(len(variables), random.randint(2, 4)))
            return f"All of {{{', '.join(vars)}}} must have different values"

        elif constraint_type == 'sum_equals':
            vars = random.sample(variables, random.randint(2, 3))
            target = random.randint(len(vars) * 2, len(vars) * max(list(domains.values())[0]))
            return f"{' + '.join(vars)} = {target}"

        elif constraint_type == 'ordered':
            vars = random.sample(variables, random.randint(2, 4))
            return f"{' < '.join(vars)}"

        return None

    def _format_csp_problem(self, variables: List[str], domains: Dict,
                            constraints: List[str], question: str) -> str:
        """Format a CSP for presentation."""
        prompt = "Constraint Satisfaction Problem:\n\n"
        prompt += f"Variables: {', '.join(variables)}\n"
        prompt += f"Domain for each variable: {{{', '.join(map(str, list(domains.values())[0]))}}}\n"
        prompt += "\nConstraints:\n"

        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"

        prompt += f"\n{question}"
        return prompt

    def _generate_propositions(self, complexity: int) -> List[str]:
        """Generate propositions for modal logic."""
        templates = [
            "the {object} is {color}",
            "{person} is in {location}",
            "the {event} happened",
            "{person} has the {object}",
            "it is {attribute}"
        ]

        propositions = []
        for _ in range(complexity + 2):
            template = random.choice(templates)
            prop = template.format(
                object=random.choice(self.entity_pool.OBJECTS[:10]),
                color=random.choice(self.entity_pool.COLORS[:10]),
                person=random.choice(self.entity_pool.PEOPLE_NAMES[:10]),
                location=random.choice(self.entity_pool.LOCATIONS[:10]),
                event=random.choice(['meeting', 'party', 'accident', 'celebration']),
                attribute=random.choice(['raining', 'sunny', 'late', 'early'])
            )
            propositions.append(prop)

        return propositions

    def _format_modal_problem(self, agents: List[str], statements: List[str],
                              rules: List[str], question: str) -> str:
        """Format a modal reasoning problem."""
        prompt = "Modal Reasoning Problem:\n\n"
        prompt += f"Agents: {', '.join(agents)}\n\n"
        prompt += "Statements:\n"

        for i, statement in enumerate(statements, 1):
            prompt += f"{i}. {statement}\n"

        prompt += "\nRules:\n"
        for i, rule in enumerate(rules, 1):
            prompt += f"- {rule}\n"

        prompt += f"\n{question}"
        return prompt

    def _generate_events(self, complexity: int) -> List[Dict]:
        """Generate events for temporal reasoning."""
        event_templates = [
            "meeting with {person}",
            "presentation about {topic}",
            "travel to {location}",
            "work on {project}",
            "break for {activity}",
            "call with {person}",
            "review of {document}",
            "training on {skill}"
        ]

        events = []
        for i in range(min(complexity + 2, 10)):
            template = random.choice(event_templates)
            event = {
                'name': template.format(
                    person=random.choice(self.entity_pool.PEOPLE_NAMES[:10]),
                    topic=random.choice(['sales', 'marketing', 'development', 'strategy']),
                    location=random.choice(self.entity_pool.LOCATIONS[:10]),
                    project=random.choice(['Project A', 'Project B', 'Project C']),
                    activity=random.choice(['lunch', 'coffee', 'exercise']),
                    document=random.choice(['report', 'proposal', 'contract']),
                    skill=random.choice(['leadership', 'communication', 'technical skills'])
                ),
                'duration': random.randint(30, 180),  # minutes
                'priority': random.choice(['high', 'medium', 'low'])
            }
            events.append(event)

        return events

    def _format_temporal_problem(self, events: List[Dict], relations: List[str],
                                 question: str) -> str:
        """Format a temporal reasoning problem."""
        prompt = "Temporal Scheduling Problem:\n\n"
        prompt += "Events:\n"

        for i, event in enumerate(events):
            prompt += f"- {event['name']} (duration: {event['duration']} minutes, priority: {event['priority']})\n"

        prompt += "\nTemporal Constraints:\n"
        for i, relation in enumerate(relations, 1):
            prompt += f"{i}. {relation}\n"

        prompt += f"\n{question}"
        return prompt

    def _format_set_problem(self, set_names: List[str], relationships: List[str],
                            question: str) -> str:
        """Format a set theory problem."""
        prompt = "Set Theory Problem:\n\n"
        prompt += f"Sets: {', '.join(set_names)}\n\n"
        prompt += "Relationships:\n"

        for i, rel in enumerate(relationships, 1):
            prompt += f"{i}. {rel}\n"

        prompt += f"\n{question}"
        return prompt

    def _format_causal_problem(self, variables: List[str], causal_links: List[str],
                               observations: List[str], question: str) -> str:
        """Format a causal reasoning problem."""
        prompt = "Causal Reasoning Problem:\n\n"
        prompt += "Variables:\n"
        for var in variables:
            prompt += f"- {var}\n"

        prompt += "\nCausal Relationships:\n"
        for i, link in enumerate(causal_links, 1):
            prompt += f"{i}. {link}\n"

        prompt += "\nObservations:\n"
        for obs in observations:
            prompt += f"- {obs}\n"

        prompt += f"\n{question}"
        return prompt

    def _format_preference_problem(self, agents: List[str], choices: List[str],
                                   statements: List[str], question: str) -> str:
        """Format a preference ordering problem."""
        prompt = "Preference and Choice Problem:\n\n"
        prompt += f"Decision makers: {', '.join(agents)}\n"
        prompt += f"Options: {', '.join(choices)}\n\n"
        prompt += "Preference statements:\n"

        for i, statement in enumerate(statements, 1):
            prompt += f"{i}. {statement}\n"

        prompt += f"\n{question}"
        return prompt

    def _create_scenario(self, scenario_type: str, complexity: int) -> Dict:
        """Create a scenario for counterfactual reasoning."""
        scenarios = {
            'historical_event': {
                'description': f"In 1969, {random.choice(['Apollo 11', 'a space mission'])} successfully landed on the moon",
                'causal_factors': [
                    'years of preparation',
                    'adequate funding',
                    'technological advancement',
                    'skilled astronauts',
                    'favorable weather'
                ],
                'outcome': 'successful moon landing'
            },
            'business_decision': {
                'description': f"Company X decided to {random.choice(['expand internationally', 'launch a new product', 'acquire a competitor'])}",
                'causal_factors': [
                    'market research',
                    'available capital',
                    'competitive pressure',
                    'leadership vision',
                    'economic conditions'
                ],
                'outcome': random.choice(['increased market share', 'financial loss', 'mixed results'])
            }
        }

        return scenarios.get(scenario_type, scenarios['historical_event'])

    def _create_counterfactual_condition(self, scenario: Dict, cf_type: str) -> str:
        """Create a counterfactual condition."""
        if cf_type == 'change_initial_condition':
            factor = random.choice(scenario['causal_factors'])
            return f"{factor} had been different"
        elif cf_type == 'remove_constraint':
            factor = random.choice(scenario['causal_factors'])
            return f"there had been no {factor}"
        else:
            return "an additional factor had been present"

    def _format_counterfactual_problem(self, scenario: Dict, counterfactuals: List[str],
                                       question: str) -> str:
        """Format a counterfactual reasoning problem."""
        prompt = "Counterfactual Reasoning:\n\n"
        prompt += f"Actual scenario: {scenario['description']}\n"
        prompt += f"Outcome: {scenario['outcome']}\n\n"
        prompt += "Key factors:\n"

        for factor in scenario['causal_factors']:
            prompt += f"- {factor}\n"

        prompt += "\nConsider these counterfactuals:\n"
        for i, cf in enumerate(counterfactuals, 1):
            prompt += f"{i}. What if {cf}?\n"

        prompt += f"\n{question}"
        return prompt

    def _generate_premises(self, scenario_type: str, num_premises: int) -> List[str]:
        """Generate premises for meta-reasoning."""
        premise_templates = {
            'logical_argument': [
                "All {A} are {B}",
                "Some {A} are {B}",
                "No {A} are {B}",
                "If {P} then {Q}",
                "{P} or {Q}",
                "Not {P}"
            ],
            'mathematical_proof': [
                "Let x be a {type} number",
                "Assume {statement}",
                "By definition, {property}",
                "From theorem {n}, we know {fact}",
                "Given that {equation}"
            ]
        }

        templates = premise_templates.get(scenario_type, premise_templates['logical_argument'])
        premises = []

        for _ in range(num_premises):
            template = random.choice(templates)
            premise = template.format(
                A=random.choice(['mammals', 'birds', 'students', 'cars']),
                B=random.choice(['animals', 'flying', 'intelligent', 'fast']),
                P=random.choice(['it rains', 'x > 0', 'the door is open']),
                Q=random.choice(['ground is wet', 'x² > 0', 'wind enters']),
                type=random.choice(['positive', 'prime', 'rational']),
                statement=random.choice(['x ≠ 0', 'the function is continuous']),
                property=random.choice(['derivative exists', 'set is closed']),
                n=random.randint(1, 10),
                fact=random.choice(['the limit exists', 'the series converges']),
                equation=random.choice(['f(x) = x²', 'a + b = c'])
            )
            premises.append(premise)

        return premises

    def _generate_conclusion(self, scenario_type: str) -> str:
        """Generate a conclusion for meta-reasoning."""
        conclusions = {
            'logical_argument': [
                "Therefore, {conclusion}",
                "We can conclude that {conclusion}",
                "It follows that {conclusion}"
            ],
            'mathematical_proof': [
                "Thus, {theorem} is proven",
                "QED: {statement}",
                "We have shown that {property}"
            ]
        }

        template = random.choice(conclusions.get(scenario_type, conclusions['logical_argument']))
        return template.format(
            conclusion=random.choice(['some A are C', 'P implies R', 'not all X are Y']),
            theorem=random.choice(['the statement', 'the proposition', 'the lemma']),
            statement=random.choice(['x = y', 'the function converges', 'the set is finite']),
            property=random.choice(['uniqueness', 'existence', 'continuity'])
        )

    def _generate_reasoning_steps(self, scenario_type: str, complexity: int) -> List[str]:
        """Generate reasoning steps for meta-reasoning."""
        num_steps = min(3 + complexity // 2, 7)
        steps = []

        step_templates = [
            "From premises {i} and {j}, we can derive {fact}",
            "Applying {rule} to {premise}",
            "By {method}, we get {result}",
            "This contradicts {statement}",
            "Assuming the opposite leads to {contradiction}"
        ]

        for i in range(num_steps):
            template = random.choice(step_templates)
            step = template.format(
                i=random.randint(1, 3),
                j=random.randint(1, 3),
                fact=random.choice(['a new statement', 'a contradiction', 'the result']),
                rule=random.choice(['modus ponens', 'contraposition', 'distribution']),
                premise=f'premise {random.randint(1, 3)}',
                method=random.choice(['substitution', 'induction', 'elimination']),
                result=random.choice(['the desired form', 'a simpler equation', 'the answer']),
                statement=f'premise {random.randint(1, 3)}',
                contradiction=random.choice(['impossibility', 'falsehood', 'absurdity'])
            )
            steps.append(f"Step {i + 1}: {step}")

        return steps

    def _inject_flaw(self, steps: List[str], flaw_type: str) -> List[str]:
        """Inject a logical flaw into reasoning steps."""
        if not steps:
            return steps

        flaw_position = random.randint(1, len(steps) - 1)

        if flaw_type == 'circular_reasoning':
            steps[flaw_position] = f"Step {flaw_position + 1}: Since we need to prove X, and X implies X, therefore X"
        elif flaw_type == 'false_premise':
            steps.insert(0, "Step 0: Assume that all birds can fly (including penguins)")
        elif flaw_type == 'invalid_inference':
            steps[
                flaw_position] = f"Step {flaw_position + 1}: Since some A are B, and some B are C, therefore all A are C"

        return steps

    def _format_meta_reasoning_problem(self, scenarios: List[Dict], principles: List[str],
                                       question: str) -> str:
        """Format a meta-reasoning problem."""
        prompt = "Reasoning Analysis Problem:\n\n"

        for scenario in scenarios:
            prompt += f"{scenario['id']} ({scenario['type']}):\n"
            prompt += "Premises:\n"
            for i, premise in enumerate(scenario['premises'], 1):
                prompt += f"  {i}. {premise}\n"
            prompt += "Reasoning:\n"
            for step in scenario['reasoning_steps']:
                prompt += f"  {step}\n"
            prompt += f"Conclusion: {scenario['conclusion']}\n\n"

        prompt += "Relevant principles:\n"
        for principle in principles:
            prompt += f"- {principle}\n"

        prompt += f"\n{question}"
        return prompt
# === Advanced Evaluation Strategies ===

class AdvancedEvaluator:
    """Enhanced evaluator with semantic robustness testing."""

    def __init__(self, generator):
        self.generator = generator

        self.evaluation_strategies = [
            self._evaluate_with_perturbation,
            self._evaluate_with_composition,
            self._evaluate_with_contradiction,
            self._evaluate_with_incomplete_info,
            self._evaluate_with_noise,
            self._evaluate_with_semantic_variation,
            self._evaluate_with_voice_transformation
        ]

        # Expanded synonym map to cover more cases
        self.synonym_map = {
            # Predicates
            "is taller than": ["has greater height than", "stands taller than", "towers over"],
            "is older than": ["has lived longer than", "was born before", "has more years than"],
            "is taller than": ["has greater height than", "stands taller than", "towers over"],
            "is older than": ["has lived longer than", "was born before", "has more years than"],
            "is faster than": ["has greater speed than", "moves quicker than", "outpaces"],
            "is wealthier than": ["has more money than", "is richer than", "has greater wealth than"],
            "is stronger than": ["has more strength than", "is more powerful than", "overpowers"],
            "owns": ["possesses", "has", "is the owner of"],
            "is located in": ["can be found in", "is in", "resides in"],
            "is to the left of": ["is positioned left of", "sits to the left of", "is leftward of"],

            # Modal operators
            "knows that": ["is certain that", "is aware that", "understands that"],
            "believes that": ["thinks that", "assumes that", "supposes that"],
            "thinks that": ["believes that", "considers that", "supposes that"],
            "is certain that": ["is sure that", "knows for certain that", "has no doubt that"],

            # Logical connectives
            "and": ["as well as", "in addition to", "along with", "furthermore"],
            "or": ["alternatively", "otherwise", "else"],
            "if": ["when", "in case", "provided that", "assuming"],
            "then": ["consequently", "therefore", "thus", "as a result"],

            # Common phrases
            "Given the following": ["Consider these", "Based on the following", "From these"],
            "information": ["facts", "statements", "data"],
            "Who": ["Which person", "What individual"],
            "What": ["Which", "What exactly"]
        }

    def _evaluate_with_semantic_variation(self, problem: Dict) -> Dict:
        """Test robustness to semantic variations with better coverage."""
        varied = problem.copy()

        if 'prompt' not in varied:
            return varied

        prompt = varied['prompt']
        original_prompt = prompt
        changes_made = []

        # Sort by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self.synonym_map.items(),
                              key=lambda x: len(x[0]), reverse=True)

        for original, synonyms in sorted_terms:
            if original.lower() in prompt.lower():
                synonym = random.choice(synonyms)

                # Use regex for case-insensitive replacement with word boundaries
                import re
                pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
                matches = list(pattern.finditer(prompt))

                if matches:
                    # Replace a random subset of occurrences
                    num_to_replace = random.randint(1, max(1, len(matches) // 2))
                    indices_to_replace = random.sample(range(len(matches)), num_to_replace)

                    # Replace from end to start to maintain indices
                    for i in sorted(indices_to_replace, reverse=True):
                        match = matches[i]
                        # Preserve the original case
                        if match.group()[0].isupper():
                            replacement = synonym[0].upper() + synonym[1:]
                        else:
                            replacement = synonym

                        prompt = prompt[:match.start()] + replacement + prompt[match.end():]
                        changes_made.append(f"'{match.group()}' → '{replacement}'")

        # Also vary sentence structure slightly
        if random.random() < 0.3:
            # Add discourse markers
            markers = ["Now, ", "Indeed, ", "Specifically, ", "In particular, "]
            marker = random.choice(markers)
            lines = prompt.split('\n')
            if len(lines) > 2:
                insert_idx = random.randint(1, len(lines) - 2)
                lines[insert_idx] = marker + lines[insert_idx]
                prompt = '\n'.join(lines)
                changes_made.append(f"Added '{marker.strip()}'")

        varied['prompt'] = prompt
        varied['original_prompt'] = original_prompt
        varied['evaluation_type'] = 'semantic_variation'
        varied['changes_applied'] = changes_made
        varied['num_changes'] = len(changes_made)

        return varied


    def _evaluate_with_perturbation(self, problem: Dict) -> Dict:
        """Modify problem slightly to test robustness."""
        perturbed = problem.copy()

        # Change entity names
        if 'entities' in problem:
            new_entities = random.sample(
                self.generator.entity_pool.PEOPLE_NAMES,
                len(problem['entities'])
            )
            # Replace in prompt
            for old, new in zip(problem['entities'], new_entities):
                perturbed['prompt'] = perturbed['prompt'].replace(old, new)
                if 'solution' in perturbed and isinstance(perturbed['solution'], str):
                    for old, new in zip(problem['entities'], new_entities):
                        perturbed['solution'] = perturbed['solution'].replace(old, new)

        # Add distractor information
        distractors = [
            "Note: Some information might be irrelevant.",
            "Additional context: The weather is sunny.",
            "Background: This problem was posed on a Tuesday."
        ]
        perturbed['prompt'] = random.choice(distractors) + "\n\n" + perturbed['prompt']

        return perturbed

    def _evaluate_with_composition(self, problem: Dict) -> Dict:
        """Combine multiple problems to test complex reasoning."""
        # Get another problem of same type
        problem2 = self.generator.generate_problem(
            complexity=problem['metadata']['complexity'],
            problem_type=problem['metadata']['type']
        )

        # Combine them
        combined = {
            'prompt': f"Part A:\n{problem['prompt']}\n\nPart B:\n{problem2['prompt']}\n\n"
                      f"Question: How do the solutions to Part A and Part B relate to each other?",
            'subproblems': [problem, problem2],
            'reasoning_type': 'composition',
            'metadata': problem['metadata']
        }

        return combined

    def _evaluate_with_contradiction(self, problem: Dict) -> Dict:
        """Introduce contradictions to test consistency checking."""
        contradictory = problem.copy()

        if 'relations' in problem or 'constraints' in problem:
            # Add contradictory information
            items = problem.get('relations', problem.get('constraints', []))
            if items and len(items) > 2:
                # Create contradiction
                item1, item2 = random.sample(items, 2)
                contradiction = f"However, we also know that {item1} contradicts {item2}"
                contradictory['prompt'] += f"\n\n{contradiction}"

                # Change question
                contradictory['prompt'] += "\n\nIs this information consistent? If not, identify the contradiction."

        return contradictory

    def _evaluate_with_incomplete_info(self, problem: Dict) -> Dict:
        """Remove information to test inference capabilities."""
        incomplete = problem.copy()

        # Remove some constraints/relations
        if 'relations' in problem:
            num_to_remove = max(1, len(problem['relations']) // 3)
            remaining = problem['relations'][:-num_to_remove]

            # Rebuild prompt
            lines = incomplete['prompt'].split('\n')
            new_lines = []
            removed_count = 0

            for line in lines:
                if any(rel in line for rel in problem['relations'][-num_to_remove:]):
                    new_lines.append("[Some information has been redacted]")
                    removed_count += 1
                else:
                    new_lines.append(line)

            incomplete['prompt'] = '\n'.join(new_lines)
            incomplete['prompt'] += f"\n\nNote: {removed_count} pieces of information have been redacted. "
            incomplete['prompt'] += "What can you still determine with certainty?"

        return incomplete

    def _evaluate_with_noise(self, problem: Dict) -> Dict:
        """Add irrelevant information to test focus."""
        noisy = problem.copy()

        # Add random facts
        noise_facts = [
            f"{random.choice(self.generator.entity_pool.PEOPLE_NAMES)} likes {random.choice(self.generator.entity_pool.COLORS)} color",
            f"The {random.choice(self.generator.entity_pool.OBJECTS)} is made of {random.choice(['wood', 'metal', 'plastic'])}",
            f"Yesterday was {random.choice(['sunny', 'rainy', 'cloudy'])}",
            f"The meeting is scheduled for {random.randint(1, 12)}:00 PM"
        ]

        # Insert noise throughout the problem
        lines = noisy['prompt'].split('\n')
        for _ in range(min(3, len(lines) // 3)):
            insert_pos = random.randint(1, len(lines) - 1)
            lines.insert(insert_pos, f"(Aside: {random.choice(noise_facts)})")

        noisy['prompt'] = '\n'.join(lines)
        noisy['prompt'] += "\n\nNote: Some information provided may not be relevant to solving the problem."

        return noisy

    def _identify_required_capabilities(self, problem: Dict) -> List[str]:
        """Identify what capabilities are needed to solve the problem."""
        capabilities = []

        # Check problem content
        prompt = problem.get('prompt', '')

        if 'transitive' in prompt or 'taller than' in prompt:
            capabilities.append('transitive_reasoning')
        if 'All' in prompt and 'are' in prompt:
            capabilities.append('syllogistic_reasoning')
        if 'causes' in prompt or 'leads to' in prompt:
            capabilities.append('causal_reasoning')
        if 'before' in prompt or 'after' in prompt:
            capabilities.append('temporal_reasoning')
        if '∩' in prompt or '∪' in prompt:
            capabilities.append('set_theory')
        if 'knows that' in prompt or 'believes that' in prompt:
            capabilities.append('modal_logic')
        if 'What if' in prompt:
            capabilities.append('counterfactual_reasoning')

        # Check for specific reasoning patterns
        if problem.get('reasoning_type'):
            capabilities.append(f"type_{problem['reasoning_type']}")

        # Check complexity indicators
        if problem.get('metadata', {}).get('complexity', 0) > 7:
            capabilities.append('advanced_reasoning')

        return list(set(capabilities))

    def _evaluate_with_voice_transformation(self, problem: Dict) -> Dict:
        """Transform between active and passive voice - FULLY FIXED."""
        transformed = problem.copy()

        if 'relations' in transformed:
            original_relations = transformed['relations'][:]
            new_relations = []

            for relation in original_relations:
                if random.random() < 0.5:  # Transform 50% of relations
                    transformed_rel = relation

                    # Handle various ownership patterns
                    if " owns " in relation and " nothing" not in relation:
                        parts = relation.split(" owns ", 1)
                        if len(parts) == 2:
                            transformed_rel = f"{parts[1]} is owned by {parts[0]}"

                    elif " knows " in relation:
                        parts = relation.split(" knows ", 1)
                        if len(parts) == 2:
                            transformed_rel = f"{parts[1]} is known by {parts[0]}"

                    elif " believes " in relation:
                        parts = relation.split(" believes ", 1)
                        if len(parts) == 2:
                            transformed_rel = f"{parts[1]} is believed by {parts[0]}"

                    new_relations.append(transformed_rel)
                else:
                    new_relations.append(relation)

            transformed['relations'] = new_relations

            # Rebuild prompt properly
            if 'prompt' in transformed:
                # Find and replace relations in the prompt
                new_prompt = transformed['prompt']

                # Try to find where relations appear in the prompt
                for i, orig_rel in enumerate(original_relations):
                    if orig_rel in new_prompt:
                        new_prompt = new_prompt.replace(orig_rel, new_relations[i])
                    elif f"- {orig_rel}" in new_prompt:
                        new_prompt = new_prompt.replace(f"- {orig_rel}", f"- {new_relations[i]}")

                transformed['prompt'] = new_prompt

            transformed['evaluation_type'] = 'voice_transformation'
            transformed['original_relations'] = original_relations

        return transformed



    def create_comprehensive_evaluation_set(self, size: int = 1000,
                                            min_complexity: int = 5) -> List[Dict]:
        """Create a comprehensive evaluation set with all strategies."""
        eval_set = []

        # Ensure we test all problem types
        problem_types = [
            'transitive_chain', 'spatial_layout', 'deontic_reasoning',
            'quantitative_logic', 'logical_puzzle', 'modal_reasoning',
            'unsolvable_problem'
        ]

        for i in range(size):
            # Rotate through problem types
            problem_type = problem_types[i % len(problem_types)]

            # Vary complexity
            complexity = random.randint(min_complexity, 10)

            # Generate base problem with various features
            use_narrative = random.random() < 0.3
            include_ambiguity = random.random() < 0.2
            solvable = random.random() > 0.1  # 10% unsolvable

            base_problem = self.generator.generate_problem(
                complexity=complexity,
                problem_type=problem_type if solvable else None,
                use_narrative=use_narrative,
                include_ambiguity=include_ambiguity,
                solvable=solvable
            )

            # Apply evaluation strategy
            strategy = self.evaluation_strategies[i % len(self.evaluation_strategies)]
            eval_problem = strategy(base_problem)

            # Add evaluation metadata
            eval_problem['eval_metadata'] = {
                'strategy': strategy.__name__,
                'base_complexity': complexity,
                'problem_id': f"EVAL_{i:05d}",
                'problem_type': problem_type,
                'features': {
                    'narrative': use_narrative,
                    'ambiguous': include_ambiguity,
                    'solvable': solvable
                }
            }

            eval_set.append(eval_problem)

        return eval_set

def safe_json_serialize(obj: Any, depth: int = 0, max_depth: int = 100) -> Any:
    """Convert non-JSON-serializable objects for hashing with recursion limit."""
    if depth > max_depth:
        return str(type(obj).__name__) + "_max_depth"

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v, depth + 1, max_depth) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item, depth + 1, max_depth) for item in obj]
    elif isinstance(obj, set):
        return sorted(safe_json_serialize(item, depth + 1, max_depth) for item in obj)
    elif isinstance(obj, Enum):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # Only serialize a simplified version of the object
        return {
            "_type": type(obj).__name__,
            "_data": safe_json_serialize(
                {k: v for k, v in obj.__dict__.items() if not k.startswith('_')},
                depth + 1,
                max_depth
            )
        }
    else:
        return str(type(obj).__name__)

# === Demonstration ===

def demonstrate_enhanced_generator():
    """Demonstrate all enhanced features."""
    print("=== Enhanced Logic Problem Generator ===\n")

    # Initialize
    generator = EnhancedProblemGenerator(seed=42)
    evaluator = AdvancedEvaluator(generator)

    # 1. Demonstrate spatial reasoning
    print("--- Spatial Reasoning Example ---")
    spatial_problem = generator.generate_problem(
        complexity=5,
        problem_type='spatial_layout'
    )
    print(f"Prompt:\n{spatial_problem['prompt']}\n")
    print(f"Solution: {spatial_problem['solution']}")
    print(f"Reasoning trace: {spatial_problem['solution_trace'][:3]}...")
    print("-" * 80)

    # 2. Demonstrate deontic logic
    print("\n--- Deontic Logic Example ---")
    deontic_problem = generator.generate_problem(
        complexity=6,
        problem_type='deontic_reasoning'
    )
    print(f"Prompt:\n{deontic_problem['prompt'][:400]}...\n")
    print(f"Conflicts detected: {deontic_problem['solution']['conflicts']}")
    print("-" * 80)

    # 3. Demonstrate narrative generation
    print("\n--- Narrative Version Example ---")
    narrative_problem = generator.generate_problem(
        complexity=4,
        problem_type='transitive_chain',
        use_narrative=True
    )
    print(f"Narrative prompt:\n{narrative_problem['prompt']}\n")
    print("-" * 80)

    # 4. Demonstrate unsolvable problem
    print("\n--- Unsolvable Problem Example ---")
    unsolvable = generator.generate_problem(
        complexity=5,
        solvable=False
    )
    print(f"Prompt:\n{unsolvable['prompt']}\n")
    print(f"Expected answer: {unsolvable['expected_answer']}")
    print(f"Type: {unsolvable['unsolvable_type']}")
    print("-" * 80)

    # 5. Demonstrate semantic variation
    print("\n--- Semantic Variation Example ---")
    base_problem = generator.generate_problem(complexity=4)
    varied = evaluator._evaluate_with_semantic_variation(base_problem)
    print(f"Original: {base_problem['prompt'][:200]}...")
    print(f"Varied: {varied['prompt'][:200]}...")
    print("-" * 80)

    # 6. Show statistics
    print("\n=== Generator Statistics ===")
    print(f"Total problems generated: {generator.generated_count}")
    print(f"Problem types available: 11")
    print(f"Evaluation strategies: {len(evaluator.evaluation_strategies)}")
    print(f"Entity pool sizes:")
    print(f"  - People names: {len(EntityPools.PEOPLE_NAMES)}")
    print(f"  - Actions: {len(EntityPools.ACTIONS)}")
    print(f"  - Spatial positions: {len(EntityPools.SPATIAL_POSITIONS)}")



def test_fixes():
    """Test the three main fixes to the generator."""
    print("=== Testing Generator Fixes ===\n")

    # Test 1: Spatial Reasoning Fix
    print("1. Testing Spatial Reasoning Fix")
    print("-" * 40)

    # Recreate the exact constraints from the output
    objects = ["bottle", "phone", "shoe", "book", "glass"]
    constraints = [
        "bottle is to the left of phone",
        "bottle is between shoe and book",
        "shoe is to the left of glass",
        "phone is to the left of book",
        "glass is to the left of bottle"
    ]

    solver = LogicalSolver()
    result = solver.solve_spatial_layout(objects, constraints)

    print(f"Constraints:")
    for c in constraints:
        print(f"  - {c}")
    print(f"\nSolution: {result['solution']}")
    print(f"Middle object: {result.get('middle', 'Not found')}")
    print(f"Consistent: {result['consistent']}")
    print(f"\nExpected: shoe < glass < bottle < phone < book")
    print(f"Expected middle: bottle")
    print()

    # Test 2: Transitive Chain Without Contradictions
    print("2. Testing Transitive Chain Generation")
    print("-" * 40)

    generator = EnhancedProblemGenerator(seed=42)

    # Generate multiple problems to check for contradictions
    contradictions_found = 0
    for i in range(5):
        problem = generator._generate_transitive_chain(complexity=5)

        if i == 0:  # Show first example
            print(f"Example problem:")
            print(f"Relations:")
            for rel in problem['relations']:
                print(f"  - {rel}")
            print(f"Question: {problem['question']}")
            print(f"Consistent: {problem['consistent']}")

        if not problem['consistent']:
            contradictions_found += 1

    print(f"\nContradictions found in 5 problems: {contradictions_found}")
    print(f"Expected: 0 (or very few)")
    print()

    # Test 3: Semantic Variation
    print("3. Testing Semantic Variation")
    print("-" * 40)

    # Create a test problem with known content
    test_problem = {
        'prompt': """Given the following information:
- Alice knows that the book is red
- Bob believes that Alice is older than Charlie
- Charlie thinks that it is raining
- David is certain that Bob owns the car

Who knows the most facts?""",
        'metadata': {'complexity': 5, 'type': 'modal_reasoning'}
    }

    evaluator = AdvancedEvaluator(generator)
    varied = evaluator._evaluate_with_semantic_variation(test_problem)

    print("Original prompt:")
    print(test_problem['prompt'][:150] + "...")
    print("\nVaried prompt:")
    print(varied['prompt'][:150] + "...")
    print(f"\nChanges applied: {varied.get('num_changes', 0)}")
    if 'changes_applied' in varied:
        for change in varied['changes_applied'][:5]:  # Show first 5 changes
            print(f"  - {change}")

    # Check if variation actually occurred
    if test_problem['prompt'] == varied['prompt']:
        print("\n⚠️ WARNING: No variation detected!")
    else:
        print("\n✓ Variation successful!")


# Run the tests
if __name__ == "__main__":
    test_fixes()