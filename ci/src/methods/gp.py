# ci/src/methods/gp.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import math
import operator

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class GeneticProgramming(BaseMethod):
    name = "GP"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "population_size": 200,        # 100-1000
            "generations": 50,             # 20-200
            "max_depth": 6,                # 3-10
            "crossover_rate": 0.9,         # 0.7-0.95
            "mutation_rate": 0.1,          # 0.05-0.2
            "function_set": ["+", "-", "*", "/"],
            "terminal_set": ["x", "constants"],
            "parsimony_coefficient": 0.001,  # 0-0.01
            "tournament_size": 3,          # برای انتخاب
            "elitism": 2,                  # تعداد نخبه‌ها
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "population_size": {"type": int, "min": 100, "max": 1000},
            "generations": {"type": int, "min": 20, "max": 200},
            "max_depth": {"type": int, "min": 3, "max": 10},
            "crossover_rate": {"type": float, "min": 0.7, "max": 0.95},
            "mutation_rate": {"type": float, "min": 0.05, "max": 0.2},
            "function_set": {"type": list},
            "terminal_set": {"type": list},
            "parsimony_coefficient": {"type": float, "min": 0.0, "max": 0.01},
            "tournament_size": {"type": int, "min": 2, "max": 10},
            "elitism": {"type": int, "min": 0, "max": 10},
        }

    # ---- Tree Node Structure ----
    class Node:
        def __init__(self, value: str, children: List[GP.Node] = None):
            self.value = value
            self.children = children or []
            self._evaluate_func = None
        
        def __repr__(self) -> str:
            if not self.children:
                return str(self.value)
            return f"({self.value} {' '.join(str(c) for c in self.children)})"
        
        def depth(self) -> int:
            """Return depth of tree"""
            if not self.children:
                return 1
            return 1 + max(c.depth() for c in self.children)
        
        def size(self) -> int:
            """Return total number of nodes"""
            if not self.children:
                return 1
            return 1 + sum(c.size() for c in self.children)
        
        def copy(self) -> GP.Node:
            """Deep copy of tree"""
            return GeneticProgramming.Node(
                self.value,
                [c.copy() for c in self.children]
            )
        
        def evaluate(self, x: Union[float, np.ndarray], 
                    constants: Dict[str, float] = None) -> Union[float, np.ndarray]:
            """Evaluate tree for given input x"""
            # Convert to numpy array for vectorized operations
            if isinstance(x, (int, float)):
                x_scalar = float(x)
                return_scalar = True
                x_arr = np.array([x_scalar])
            else:
                x_arr = np.asarray(x, dtype=float)
                return_scalar = False
            
            # Evaluate recursively
            if self.value == "x":
                result = x_arr
            elif self.value.replace('.', '', 1).isdigit() or \
                 (self.value[0] == '-' and self.value[1:].replace('.', '', 1).isdigit()):
                # Numeric constant
                const_val = float(self.value)
                result = np.full_like(x_arr, const_val)
            elif constants and self.value in constants:
                # Named constant
                result = np.full_like(x_arr, constants[self.value])
            else:
                # Function node
                child_results = [c.evaluate(x_arr, constants) for c in self.children]
                
                if self.value == "+":
                    result = child_results[0] + child_results[1]
                elif self.value == "-":
                    if len(child_results) == 1:
                        result = -child_results[0]  # Unary minus
                    else:
                        result = child_results[0] - child_results[1]
                elif self.value == "*":
                    result = child_results[0] * child_results[1]
                elif self.value == "/":
                    # Protected division
                    denom = child_results[1]
                    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                    result = child_results[0] / denom
                elif self.value == "sin":
                    result = np.sin(child_results[0])
                elif self.value == "cos":
                    result = np.cos(child_results[0])
                elif self.value == "exp":
                    result = np.exp(np.clip(child_results[0], -50, 50))
                elif self.value == "log":
                    # Protected log
                    arg = child_results[0]
                    arg = np.where(arg < 1e-10, 1e-10, arg)
                    result = np.log(arg)
                elif self.value == "sqrt":
                    # Protected sqrt
                    arg = child_results[0]
                    arg = np.where(arg < 0, 0, arg)
                    result = np.sqrt(arg)
                elif self.value == "abs":
                    result = np.abs(child_results[0])
                else:
                    raise ValueError(f"Unknown operator: {self.value}")
            
            return float(result[0]) if return_scalar else result
        
        def simplify(self) -> GP.Node:
            """Simplify tree (constant folding, remove redundant operations)"""
            if not self.children:
                return self.copy()
            
            simplified_children = [c.simplify() for c in self.children]
            
            # Try constant folding
            try:
                if all(c.value.replace('.', '', 1).isdigit() or 
                       (c.value[0] == '-' and c.value[1:].replace('.', '', 1).isdigit())
                       for c in simplified_children):
                    # All children are constants
                    const_vals = [float(c.value) for c in simplified_children]
                    
                    if self.value == "+":
                        result = sum(const_vals)
                    elif self.value == "-":
                        result = const_vals[0] - sum(const_vals[1:]) if len(const_vals) > 1 else -const_vals[0]
                    elif self.value == "*":
                        result = np.prod(const_vals)
                    elif self.value == "/":
                        result = const_vals[0] / np.prod(const_vals[1:]) if len(const_vals) > 1 else 1.0/const_vals[0]
                    elif self.value == "sin":
                        result = math.sin(const_vals[0])
                    elif self.value == "cos":
                        result = math.cos(const_vals[0])
                    elif self.value == "exp":
                        result = math.exp(const_vals[0])
                    else:
                        raise ValueError
                    
                    return GeneticProgramming.Node(f"{result:.6f}")
            except:
                pass
            
            return GeneticProgramming.Node(self.value, simplified_children)

    def solve(
        self,
        problem: Any,
        params: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        """
        Genetic Programming for symbolic regression, classification, etc.
        
        Expects problem to have:
        - X: Input data (n_samples, n_features) or (n_samples,)
        - y: Target values
        - Or specific methods for different problem types
        """
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        # Extract parameters
        pop_size = params["population_size"]
        generations = params["generations"]
        max_depth = params["max_depth"]
        cx_rate = params["crossover_rate"]
        mut_rate = params["mutation_rate"]
        function_set = params["function_set"]
        terminal_set = params["terminal_set"]
        parsimony = params["parsimony_coefficient"]
        tournament_size = params.get("tournament_size", 3)
        elitism = min(params.get("elitism", 2), pop_size)

        # ---- Problem Setup ----
        # Check problem type and get data
        X_data = None
        y_data = None
        problem_type = "unknown"
        
        if hasattr(problem, "X") and hasattr(problem, "y"):
            # Symbolic regression or classification
            X_data = np.asarray(problem.X)
            y_data = np.asarray(problem.y)
            
            if X_data.ndim == 1:
                X_data = X_data.reshape(-1, 1)
            
            problem_type = "regression" if y_data.dtype in [float, np.float32, np.float64] else "classification"
            n_features = X_data.shape[1]
            
            print(f"GP: {problem_type} problem with {len(X_data)} samples, {n_features} features")
        
        elif hasattr(problem, "evaluate_tree"):
            # Custom tree evaluation
            problem_type = "custom"
            n_features = getattr(problem, "n_features", 1)
            print(f"GP: Custom problem with evaluate_tree method")
        
        else:
            raise ValueError(
                "GP problem must have X and y attributes, "
                "or evaluate_tree method"
            )
        
        # ---- Tree Generation Functions ----
        def create_random_tree(depth: int, method: str = "grow") -> Node:
            """Create random tree using grow or full method"""
            if depth <= 1 or (method == "grow" and rng.random() < 0.3):
                # Terminal node
                term = rng.choice(terminal_set)
                if term == "x":
                    if n_features > 1:
                        # Multiple features: x0, x1, ...
                        feature_idx = rng.integers(0, n_features)
                        return self.Node(f"x{feature_idx}")
                    else:
                        return self.Node("x")
                elif term == "constants":
                    # Random constant between -5 and 5
                    const_val = rng.uniform(-5.0, 5.0)
                    return self.Node(f"{const_val:.4f}")
                else:
                    return self.Node(term)
            else:
                # Function node
                func = rng.choice(function_set)
                
                if func in ["sin", "cos", "exp", "log", "sqrt", "abs"]:
                    # Unary functions
                    child = create_random_tree(depth - 1, method)
                    return self.Node(func, [child])
                else:
                    # Binary functions
                    left = create_random_tree(depth - 1, method)
                    right = create_random_tree(depth - 1, method)
                    return self.Node(func, [left, right])
        
        def evaluate_tree_fitness(tree: Node) -> float:
            """Calculate fitness of a tree"""
            if problem_type == "regression":
                # Mean Squared Error
                y_pred = tree.evaluate(X_data)
                mse = np.mean((y_pred - y_data) ** 2)
                
                # Parsimony pressure: penalize large trees
                size_penalty = parsimony * tree.size()
                
                return mse + size_penalty
            
            elif problem_type == "classification":
                # Accuracy (for binary classification)
                y_pred_raw = tree.evaluate(X_data)
                y_pred = (y_pred_raw > 0.5).astype(int)
                accuracy = np.mean(y_pred == y_data)
                
                # Fitness = 1 - accuracy (to minimize)
                size_penalty = parsimony * tree.size()
                
                return (1.0 - accuracy) + size_penalty
            
            elif problem_type == "custom":
                # Use problem's custom evaluation
                return problem.evaluate_tree(tree)
            
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")
        
        # ---- Genetic Operators ----
        def tournament_selection(population: List[Node], fitnesses: List[float]) -> int:
            """Select individual using tournament selection"""
            contestants = rng.choice(len(population), size=tournament_size, replace=False)
            best_idx = contestants[0]
            best_fitness = fitnesses[best_idx]
            
            for idx in contestants[1:]:
                if fitnesses[idx] < best_fitness:  # Minimization
                    best_idx = idx
                    best_fitness = fitnesses[idx]
            
            return best_idx
        
        def crossover(parent1: Node, parent2: Node) -> Tuple[Node, Node]:
            """Subtree crossover"""
            def get_random_node(root: Node) -> Node:
                """Get random node from tree (biased toward internal nodes)"""
                nodes = []
                stack = [(root, 1)]  # (node, depth)
                
                while stack:
                    node, depth = stack.pop()
                    nodes.append((node, depth))
                    stack.extend([(c, depth + 1) for c in node.children])
                
                # Bias selection toward internal nodes (70% internal, 30% any)
                internal_nodes = [n for n, d in nodes if n.children]
                
                if internal_nodes and rng.random() < 0.7:
                    return rng.choice(internal_nodes)
                else:
                    return rng.choice([n for n, _ in nodes])
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            node1 = get_random_node(child1)
            node2 = get_random_node(child2)
            
            # Swap subtrees
            node1.value, node2.value = node2.value, node1.value
            node1.children, node2.children = node2.children, node1.children
            
            return child1, child2
        
        def mutation(tree: Node) -> Node:
            """Point mutation or subtree mutation"""
            mutant = tree.copy()
            
            # Collect all nodes
            nodes = []
            stack = [(mutant, None, 0)]  # (node, parent, child_index)
            
            while stack:
                node, parent, idx = stack.pop()
                nodes.append((node, parent, idx))
                for i, child in enumerate(node.children):
                    stack.append((child, node, i))
            
            if not nodes:
                return mutant
            
            # Select random node to mutate
            target, parent, child_idx = rng.choice(nodes)
            
            mutation_type = rng.choice(["point", "subtree", "shrink"], 
                                      p=[0.4, 0.4, 0.2])
            
            if mutation_type == "point":
                # Point mutation: change value of terminal or function
                if not target.children:  # Terminal
                    if target.value.startswith("x") or target.value.replace('.', '', 1).isdigit():
                        # Change constant or variable
                        if rng.random() < 0.5 and "constants" in terminal_set:
                            # New constant
                            new_const = rng.uniform(-5.0, 5.0)
                            target.value = f"{new_const:.4f}"
                        elif n_features > 1:
                            # Change variable index
                            new_idx = rng.integers(0, n_features)
                            target.value = f"x{new_idx}"
                else:  # Function
                    # Change function (same arity)
                    current_func = target.value
                    same_arity_funcs = []
                    for func in function_set:
                        if func in ["sin", "cos", "exp", "log", "sqrt", "abs"]:
                            arity = 1
                        else:
                            arity = 2
                        
                        if func != current_func and (
                            (arity == 1 and len(target.children) == 1) or
                            (arity == 2 and len(target.children) == 2)
                        ):
                            same_arity_funcs.append(func)
                    
                    if same_arity_funcs:
                        target.value = rng.choice(same_arity_funcs)
            
            elif mutation_type == "subtree":
                # Replace subtree with new random tree
                new_subtree = create_random_tree(
                    depth=rng.integers(1, max_depth // 2 + 1),
                    method=rng.choice(["grow", "full"])
                )
                
                if parent is not None:
                    parent.children[child_idx] = new_subtree
                else:
                    mutant = new_subtree
            
            elif mutation_type == "shrink":
                # Replace subtree with terminal
                if parent is not None:
                    new_terminal = create_random_tree(1, "grow")
                    parent.children[child_idx] = new_terminal
            
            return mutant
        
        # ---- Initial Population ----
        print(f"  Creating initial population of {pop_size} trees...")
        population = []
        for i in range(pop_size):
            method = rng.choice(["grow", "full"])
            tree = create_random_tree(max_depth, method)
            population.append(tree)
        
        # Initial evaluation
        fitnesses = [evaluate_tree_fitness(ind) for ind in population]
        best_idx = int(np.argmin(fitnesses))
        best_tree = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])
        
        history = [best_fitness]
        avg_sizes = [np.mean([ind.size() for ind in population])]
        
        print(f"  Initial best fitness: {best_fitness:.6f}")
        
        # ---- Main Evolution Loop ----
        for gen in range(1, generations + 1):
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[:elitism]
            elite_individuals = [population[i].copy() for i in elite_indices]
            elite_fitnesses = [fitnesses[i] for i in elite_indices]
            
            new_population = elite_individuals.copy()
            new_fitnesses = elite_fitnesses.copy()
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Selection
                parent1_idx = tournament_selection(population, fitnesses)
                parent2_idx = tournament_selection(population, fitnesses)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if rng.random() < cx_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if rng.random() < mut_rate:
                    child1 = mutation(child1)
                if rng.random() < mut_rate:
                    child2 = mutation(child2)
                
                # Control depth
                if child1.depth() <= max_depth and len(new_population) < pop_size:
                    new_population.append(child1)
                    new_fitnesses.append(evaluate_tree_fitness(child1))
                
                if child2.depth() <= max_depth and len(new_population) < pop_size:
                    new_population.append(child2)
                    new_fitnesses.append(evaluate_tree_fitness(child2))
            
            # Update population
            population = new_population[:pop_size]
            fitnesses = new_fitnesses[:pop_size]
            
            # Update best individual
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_tree = population[gen_best_idx].copy()
                print(f"  Generation {gen}: New best fitness = {best_fitness:.6f}")
            
            history.append(best_fitness)
            avg_sizes.append(np.mean([ind.size() for ind in population]))
            
            # Progress callback
            if progress_cb and gen % 5 == 0:
                progress_cb(gen, best_fitness, {
                    "avg_tree_size": avg_sizes[-1],
                    "best_tree_size": best_tree.size(),
                    "best_tree_depth": best_tree.depth(),
                    "generation": gen,
                })
        
        # Simplify best tree
        simplified_tree = best_tree.simplify()
        
        # Final evaluation
        if problem_type == "regression":
            y_pred = simplified_tree.evaluate(X_data)
            mse = np.mean((y_pred - y_data) ** 2)
            r2 = 1 - mse / np.var(y_data)
            
            metrics = {
                "mse": float(mse),
                "r2": float(r2),
                "tree_size": simplified_tree.size(),
                "tree_depth": simplified_tree.depth(),
                "problem_type": "regression",
            }
        elif problem_type == "classification":
            y_pred_raw = simplified_tree.evaluate(X_data)
            y_pred = (y_pred_raw > 0.5).astype(int)
            accuracy = np.mean(y_pred == y_data)
            
            metrics = {
                "accuracy": float(accuracy),
                "tree_size": simplified_tree.size(),
                "tree_depth": simplified_tree.depth(),
                "problem_type": "classification",
            }
        else:
            metrics = {
                "tree_size": simplified_tree.size(),
                "tree_depth": simplified_tree.depth(),
                "problem_type": problem_type,
            }
        
        print(f"\n  Final best tree: {simplified_tree}")
        print(f"  Tree size: {simplified_tree.size()}, depth: {simplified_tree.depth()}")
        print(f"  Best fitness: {best_fitness:.6f}")
        
        return MethodResult(
            method_name=self.name,
            best_solution=str(simplified_tree),  # Store as string
            best_fitness=best_fitness,
            history=history,
            iterations=generations,
            status="ok",
            metrics=metrics,
        )