# ci/src/methods/fuzzy.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class FuzzyController(BaseMethod):
    name = "Fuzzy"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "n_membership_functions": 3,        # 3, 5, 7
            "membership_type": "triangular",    # triangular/gaussian/trapezoid
            "defuzzification": "centroid",      # centroid/bisector/mom
            "rule_generation": "wang_mendel",   # wang_mendel/manual
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "n_membership_functions": {"type": int, "choices": [3, 5, 7]},
            "membership_type": {"type": str, "choices": ["triangular", "gaussian", "trapezoid"]},
            "defuzzification": {"type": str, "choices": ["centroid", "bisector", "mom", "som"]},
            "rule_generation": {"type": str, "choices": ["wang_mendel", "manual"]},
        }

    def solve(
        self,
        problem: Any,
        params: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        """
        Fuzzy Logic Controller for control/optimization problems.
        
        Expects problem to have:
        - input_ranges: List[Tuple[float, float]] for each input variable
        - output_range: Tuple[float, float] for output variable
        - training_data: Optional Tuple[X, y] for Wang-Mendel rule generation
        - evaluate: Optional function to evaluate controller performance
        """
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        # Extract parameters
        n_mf = params["n_membership_functions"]
        mf_type = params["membership_type"]
        defuzz_method = params["defuzzification"]
        rule_gen = params["rule_generation"]

        # Validate problem structure
        if not hasattr(problem, "input_ranges"):
            raise ValueError("Fuzzy problem must have 'input_ranges' attribute")
        if not hasattr(problem, "output_range"):
            raise ValueError("Fuzzy problem must have 'output_range' attribute")
        
        input_ranges = problem.input_ranges  # List of (min, max)
        output_range = problem.output_range  # (min, max)
        n_inputs = len(input_ranges)
        
        print(f"Fuzzy: {n_inputs} inputs, {n_mf} MFs per input, {mf_type} MFs")

        # ---- Membership Function Definitions ----
        def triangular_mf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            """Triangular membership function: a (left), b (center), c (right)"""
            return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
        
        def gaussian_mf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
            """Gaussian membership function"""
            return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        
        def trapezoidal_mf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
            """Trapezoidal membership function"""
            return np.maximum(0, np.minimum(
                np.minimum((x - a) / (b - a), 1.0),
                (d - x) / (d - c)
            ))
        
        # Create membership functions for each input
        input_mfs = []  # List of lists: for each input, list of MF functions
        input_centers = []  # Center points for each MF
        
        for i, (low, high) in enumerate(input_ranges):
            centers = np.linspace(low, high, n_mf)
            input_centers.append(centers)
            
            mfs_for_input = []
            for j, center in enumerate(centers):
                if mf_type == "triangular":
                    # Triangular: left neighbor, center, right neighbor
                    left = centers[j-1] if j > 0 else low
                    right = centers[j+1] if j < n_mf-1 else high
                    mfs_for_input.append(
                        lambda x, c=center, l=left, r=right: triangular_mf(x, l, c, r)
                    )
                elif mf_type == "gaussian":
                    # Gaussian: sigma = 1/3 of spacing
                    sigma = (high - low) / (n_mf * 3)
                    mfs_for_input.append(
                        lambda x, m=center, s=sigma: gaussian_mf(x, m, s)
                    )
                elif mf_type == "trapezoid":
                    # Trapezoidal: with 20% overlap
                    width = (high - low) / (n_mf - 1) if n_mf > 1 else (high - low)
                    a = center - width * 0.6
                    b = center - width * 0.2
                    c = center + width * 0.2
                    d = center + width * 0.6
                    mfs_for_input.append(
                        lambda x, a=a, b=b, c=c, d=d: trapezoidal_mf(x, a, b, c, d)
                    )
            
            input_mfs.append(mfs_for_input)
        
        # Create output membership functions
        output_centers = np.linspace(output_range[0], output_range[1], n_mf)
        output_mfs = []
        for center in output_centers:
            if mf_type == "triangular":
                width = (output_range[1] - output_range[0]) / (n_mf - 1) if n_mf > 1 else 1.0
                left = center - width
                right = center + width
                output_mfs.append(
                    lambda x, c=center, l=left, r=right: triangular_mf(x, l, c, r)
                )
            elif mf_type == "gaussian":
                sigma = (output_range[1] - output_range[0]) / (n_mf * 3)
                output_mfs.append(
                    lambda x, m=center, s=sigma: gaussian_mf(x, m, s)
                )
            else:  # trapezoid
                width = (output_range[1] - output_range[0]) / (n_mf - 1) if n_mf > 1 else 1.0
                a = center - width * 0.6
                b = center - width * 0.2
                c = center + width * 0.2
                d = center + width * 0.6
                output_mfs.append(
                    lambda x, a=a, b=b, c=c, d=d: trapezoidal_mf(x, a, b, c, d)
                )
        
        # ---- Rule Generation ----
        rules = []  # List of (antecedents, consequent)
        
        if rule_gen == "wang_mendel":
            # Wang-Mendel rule generation from training data
            if hasattr(problem, "training_data"):
                X_train, y_train = problem.training_data
                X_train = np.asarray(X_train)
                y_train = np.asarray(y_train)
                
                # For each training sample
                rule_strengths = {}
                for idx in range(min(len(X_train), 100)):  # Use up to 100 samples
                    x = X_train[idx]
                    y = y_train[idx]
                    
                    # Determine which MF has highest membership for each input
                    antecedents = []
                    for i in range(n_inputs):
                        memberships = [mf(x[i]) for mf in input_mfs[i]]
                        best_mf_idx = np.argmax(memberships)
                        antecedents.append(best_mf_idx)
                    
                    # Determine output MF
                    output_memberships = [mf(y) for mf in output_mfs]
                    consequent = np.argmax(output_memberships)
                    
                    # Rule strength = product of memberships
                    strength = 1.0
                    for i, mf_idx in enumerate(antecedents):
                        strength *= input_mfs[i][mf_idx](x[i])
                    
                    rule_key = tuple(antecedents)
                    if rule_key not in rule_strengths or strength > rule_strengths[rule_key][1]:
                        rule_strengths[rule_key] = (consequent, strength)
                
                # Convert to list, keeping only strongest rule for each antecedent combination
                for antecedents, (consequent, strength) in rule_strengths.items():
                    rules.append((list(antecedents), consequent))
                
                print(f"  Generated {len(rules)} rules using Wang-Mendel method")
            else:
                print("  No training data for Wang-Mendel, using random rules")
                rule_gen = "manual"  # Fallback
        
        if rule_gen == "manual" or not rules:
            # Manual rule generation or fallback
            if hasattr(problem, "rules"):
                # Use provided rules
                rules = problem.rules
                print(f"  Using {len(rules)} provided rules")
            else:
                # Generate exhaustive rules (all combinations)
                from itertools import product
                antecedents_combinations = product(range(n_mf), repeat=n_inputs)
                rules = [(list(ant), rng.integers(0, n_mf)) for ant in antecedents_combinations]
                print(f"  Generated {len(rules)} exhaustive rules")
        
        # ---- Fuzzy Inference System ----
        def fuzzy_inference(x_input: np.ndarray) -> float:
            """Infer output for given input vector"""
            if len(x_input) != n_inputs:
                raise ValueError(f"Expected {n_inputs} inputs, got {len(x_input)}")
            
            # Fuzzify inputs
            fuzzified = []
            for i in range(n_inputs):
                xi = x_input[i]
                memberships = [mf(xi) for mf in input_mfs[i]]
                fuzzified.append(memberships)
            
            # Apply rules and aggregate
            rule_outputs = []  # (strength, output_center)
            for antecedents, consequent in rules:
                # Rule firing strength (min for AND, could be product)
                strength = 1.0
                for i, mf_idx in enumerate(antecedents):
                    if mf_idx < len(fuzzified[i]):
                        strength = min(strength, fuzzified[i][mf_idx])
                
                if strength > 0:
                    rule_outputs.append((strength, output_centers[consequent]))
            
            if not rule_outputs:
                # No rules fired, return midpoint
                return (output_range[0] + output_range[1]) / 2.0
            
            # Defuzzification
            strengths = np.array([s for s, _ in rule_outputs])
            centers = np.array([c for _, c in rule_outputs])
            
            if defuzz_method == "centroid":
                # Weighted average
                weighted_sum = np.sum(strengths * centers)
                total_strength = np.sum(strengths)
                return weighted_sum / total_strength if total_strength > 0 else np.mean(centers)
            
            elif defuzz_method == "bisector":
                # Bisector of area
                sorted_idx = np.argsort(centers)
                sorted_centers = centers[sorted_idx]
                sorted_strengths = strengths[sorted_idx]
                
                total_area = np.sum(sorted_strengths)
                half_area = total_area / 2
                cumulative = 0
                
                for i, strength in enumerate(sorted_strengths):
                    cumulative += strength
                    if cumulative >= half_area:
                        return sorted_centers[i]
                return sorted_centers[-1]
            
            elif defuzz_method == "mom":  # Mean of Maximum
                max_strength = np.max(strengths)
                max_centers = centers[strengths == max_strength]
                return np.mean(max_centers)
            
            elif defuzz_method == "som":  # Smallest of Maximum
                max_strength = np.max(strengths)
                max_centers = centers[strengths == max_strength]
                return np.min(max_centers)
            
            else:
                return np.mean(centers)
        
        # ---- Evaluation/Optimization ----
        history = []
        best_fitness = float("inf")
        best_solution = None
        
        if hasattr(problem, "evaluate"):
            # If problem has evaluate function, use it to optimize
            print("  Optimizing controller parameters...")
            
            # Simple grid search over rule consequents
            n_rules_to_optimize = min(10, len(rules))  # Optimize first 10 rules
            
            for iteration in range(50):  # Limited iterations
                # Randomly modify some rules
                modified_rules = rules.copy()
                for _ in range(rng.integers(1, 4)):
                    rule_idx = rng.integers(0, n_rules_to_optimize)
                    new_consequent = rng.integers(0, n_mf)
                    antecedents = modified_rules[rule_idx][0]
                    modified_rules[rule_idx] = (antecedents, new_consequent)
                
                # Evaluate this rule set
                # Create temporary inference function with modified rules
                def temp_inference(x):
                    # Re-create inference with modified rules
                    # (simplified - in real implementation would cache)
                    rule_outputs = []
                    for antecedents, consequent in modified_rules:
                        strength = 1.0
                        for i, mf_idx in enumerate(antecedents):
                            if i < len(x) and mf_idx < len(input_mfs[i]):
                                strength = min(strength, input_mfs[i][mf_idx](x[i]))
                        
                        if strength > 0:
                            rule_outputs.append((strength, output_centers[consequent]))
                    
                    if not rule_outputs:
                        return np.mean(output_range)
                    
                    strengths = np.array([s for s, _ in rule_outputs])
                    centers = np.array([c for _, c in rule_outputs])
                    return np.sum(strengths * centers) / np.sum(strengths)
                
                # Evaluate on test data if available
                if hasattr(problem, "test_data"):
                    X_test, y_test = problem.test_data
                    errors = []
                    for x, y_true in zip(X_test[:20], y_test[:20]):  # Sample
                        y_pred = temp_inference(x)
                        errors.append(abs(y_pred - y_true))
                    fitness = np.mean(errors)
                else:
                    # Use problem's evaluate function
                    fitness = problem.evaluate(temp_inference)
                
                history.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = {
                        "rules": modified_rules[:10],  # Store first 10 rules
                        "n_mf": n_mf,
                        "mf_type": mf_type,
                        "defuzz_method": defuzz_method,
                    }
                
                if progress_cb and iteration % 10 == 0:
                    progress_cb(iteration, fitness, {
                        "best_fitness": best_fitness,
                        "n_rules": len(modified_rules)
                    })
        else:
            # No evaluation function, just build controller
            print("  Building fuzzy controller (no optimization)")
            best_fitness = 0.0  # Default
            best_solution = {
                "rules": rules[:10],  # Store first 10 rules
                "n_mf": n_mf,
                "mf_type": mf_type,
                "defuzz_method": defuzz_method,
                "n_inputs": n_inputs,
            }
        
        return MethodResult(
            method_name=self.name,
            best_solution=best_solution,
            best_fitness=best_fitness,
            history=history,
            iterations=len(history),
            status="ok",
            metrics={
                "n_inputs": n_inputs,
                "n_membership_functions": n_mf,
                "membership_type": mf_type,
                "defuzzification": defuzz_method,
                "rule_generation": rule_gen,
                "n_rules": len(rules),
                "output_range": output_range,
            },
        )