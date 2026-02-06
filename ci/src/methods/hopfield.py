# ci/src/methods/hopfield.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np

from .base import BaseMethod, ProgressCallback
from .result import MethodResult
from src.utils.seeding import set_global_seed


class HopfieldNetwork(BaseMethod):
    name = "Hopfield"

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        # مطابق PDF
        return {
            "max_iterations": 100,      # 50-500
            "threshold": 0.0,
            "async_update": True,
            "energy_threshold": 1e-6,
        }

    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        return {
            "max_iterations": {"type": int, "min": 50, "max": 500},
            "threshold": {"type": float},
            "async_update": {"type": bool},
            "energy_threshold": {"type": float, "min": 0.0, "max": 1.0},
        }

    def solve(
        self,
        problem: Any,
        params: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        """
        Hopfield Network for pattern recall and associative memory.
        
        Expects problem to have:
        - patterns: List[np.ndarray] (training patterns, values -1 or 1)
        - n_neurons: int
        - test_patterns: List[np.ndarray] (optional, patterns to recall)
        """
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        # Extract parameters
        max_iterations = params["max_iterations"]
        threshold = params["threshold"]
        async_update = params["async_update"]
        energy_threshold = params["energy_threshold"]

        # Validate problem structure
        if not hasattr(problem, "patterns"):
            raise ValueError("Hopfield problem must have 'patterns' attribute")
        
        patterns = problem.patterns  # List of training patterns
        if not patterns:
            raise ValueError("No patterns provided for training")
        
        # Convert patterns to numpy arrays
        patterns_np = [np.asarray(p).flatten() for p in patterns]
        n_neurons = len(patterns_np[0])
        
        # Verify all patterns have same length
        for i, p in enumerate(patterns_np):
            if len(p) != n_neurons:
                raise ValueError(f"Pattern {i} has length {len(p)}, expected {n_neurons}")
        
        # Check pattern values are bipolar (-1, 1)
        for i, p in enumerate(patterns_np):
            if not np.all(np.isin(p, [-1, 1])):
                raise ValueError(f"Pattern {i} contains values other than -1 or 1")

        # ---- Training: Hebbian learning rule ----
        print(f"Hopfield: Training with {len(patterns_np)} patterns, {n_neurons} neurons")
        
        # Initialize weight matrix (symmetric, zero diagonal)
        W = np.zeros((n_neurons, n_neurons), dtype=float)
        
        # Hebb rule: W = (1/N) * Σ x_i * x_j^T
        for pattern in patterns_np:
            x = pattern.reshape(-1, 1)  # column vector
            W += np.dot(x, x.T)
        
        W /= len(patterns_np)  # average
        np.fill_diagonal(W, 0.0)  # no self-connections
        
        # Test patterns (if provided) or use noisy versions of training patterns
        test_patterns = getattr(problem, "test_patterns", None)
        if test_patterns is None:
            # Create noisy versions of training patterns
            test_patterns = []
            for pattern in patterns_np:
                noisy = pattern.copy()
                # Flip 30% of bits
                flip_mask = rng.random(n_neurons) < 0.3
                noisy[flip_mask] *= -1
                test_patterns.append(noisy)
        
        # ---- Recall process ----
        history = []
        best_energy = float("inf")
        best_state = None
        recall_accuracies = []
        
        for pattern_idx, initial_state in enumerate(test_patterns):
            state = initial_state.copy().astype(float)
            prev_energy = float("inf")
            converged = False
            
            for iteration in range(1, max_iterations + 1):
                # Compute energy: E = -0.5 * ΣΣ w_ij * s_i * s_j
                energy = -0.5 * np.dot(state, np.dot(W, state))
                
                # Track best energy
                if energy < best_energy:
                    best_energy = energy
                    best_state = state.copy()
                
                history.append(energy)
                
                # Check convergence
                if iteration > 1 and abs(energy - prev_energy) < energy_threshold:
                    converged = True
                    break
                
                prev_energy = energy
                
                # Update neurons
                if async_update:
                    # Asynchronous: update neurons in random order
                    update_order = rng.permutation(n_neurons)
                    for i in update_order:
                        net_input = np.dot(W[i, :], state)
                        if net_input > threshold:
                            state[i] = 1.0
                        else:
                            state[i] = -1.0
                else:
                    # Synchronous: update all neurons at once
                    net_inputs = np.dot(W, state)
                    state = np.where(net_inputs > threshold, 1.0, -1.0)
                
                # Progress callback
                if progress_cb and iteration % 10 == 0:
                    progress_cb(iteration, energy, {
                        "pattern": pattern_idx,
                        "energy": energy,
                        "converged": converged
                    })
            
            # Calculate recall accuracy for this pattern
            if pattern_idx < len(patterns_np):
                target = patterns_np[pattern_idx]
                accuracy = np.mean(state == target)
                recall_accuracies.append(accuracy)
                print(f"  Pattern {pattern_idx}: Recall accuracy = {accuracy:.3f}")
        
        # Calculate metrics
        avg_recall_accuracy = np.mean(recall_accuracies) if recall_accuracies else 0.0
        
        return MethodResult(
            method_name=self.name,
            best_solution=best_state.tolist() if best_state is not None else None,
            best_fitness=best_energy,
            history=history,
            iterations=len(history),
            status="ok",
            metrics={
                "n_patterns": len(patterns_np),
                "n_neurons": n_neurons,
                "async_update": async_update,
                "avg_recall_accuracy": float(avg_recall_accuracy),
                "max_iterations": max_iterations,
                "energy_threshold": energy_threshold,
            },
        )