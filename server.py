"""
Fed-HARP Server Implementation.
Handles heterogeneous layer allocation, staleness tracking, and staleness-weighted aggregation logic.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional
import numpy as np
from models import LoRALinear, get_lora_layer_names


class FedHarpServer:
    """
    Fed-HARP Federated Learning Server.
    
    Core Functions:
    - Generate Allocation Map: 
        * Supports random allocation (initial phase)
        * Supports B-matrix sensitivity-based allocation (High Sensitivity First)
    - Track Staleness: Records how many rounds each layer of each client has not been updated.
    - Execute Staleness-Weighted Aggregation: Down-weights stale updates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_layers: Dict[str, LoRALinear],
        num_clients: int,
        allocation_ratio: float = 0.5,
        aggregation_lr: float = 1.0, 
        struct_to_data_rounds: int = 20,
        seed: int = 42,
        method: str = "fedharp",
        sensitivity_momentum: float = 0.8,
    ):
        """
        Initialize the server.
        
        Args:
            model: Global model instance.
            lora_layers: Dictionary of LoRA layers.
            num_clients: Total number of clients.
            allocation_ratio: Ratio of layers updated by each client (e.g., 0.5 means 50%).
            aggregation_lr: Server aggregation learning rate (eta). usually 1.0 for standard FedAvg logic.
            seed: Random allocation seed.
        """
        self.model = model
        self.lora_layers = lora_layers
        self.num_clients = num_clients
        self.allocation_ratio = allocation_ratio
        self.aggregation_lr = aggregation_lr
        self.struct_to_data_rounds = max(1, int(struct_to_data_rounds))
        self.seed = seed
        self.method = "fedharp" if str(method).lower() == "fedanon" else method
        
        # Get sorted list of layer names
        self.layer_names = get_lora_layer_names(lora_layers)
        self.num_layers = len(self.layer_names)
        
        # Staleness tracking table: staleness[client_id][layer_name] = rounds since last update
        self.staleness: Dict[int, Dict[str, int]] = {
            client_id: {layer_name: 0 for layer_name in self.layer_names}
            for client_id in range(num_clients)
        }
        
        # Store client sensitivities (B matrix norms)
        # client_sensitivities[client_id][layer_name] = float value
        self.client_sensitivities: Dict[int, Dict[str, float]] = {}
        
        # Current round
        self.current_round = 0

        # Record total selection counts per layer for fairness weighting
        self.layer_selection_counts: Dict[str, int] = {
            layer_name: 0 for layer_name in self.layer_names
        }

        self.client_layer_counts: Dict[int, int] = {}

        try:
            m = float(sensitivity_momentum)
        except Exception:
            m = 0.8
        if m < 0.0:
            m = 0.0
        if m >= 1.0:
            m = 0.99
        self.sensitivity_momentum = m
        
        # Set random seed
        np.random.seed(seed)
    
    def update_client_sensitivities(self, client_id: int, sensitivities: Dict[str, float]):
        """
        Receive and update client B-matrix sensitivity information.
        This information will be used for allocation in the next round.
        """
        prev = self.client_sensitivities.get(client_id, {})
        if prev is None or len(prev) == 0:
            self.client_sensitivities[client_id] = dict(sensitivities)
            return

        beta = float(getattr(self, "sensitivity_momentum", 0.2))
        eps = 1e-12
        merged: Dict[str, float] = {}

        keys = set(prev.keys()) | set(sensitivities.keys()) | set(self.layer_names)
        for k in keys:
            old_v = float(prev.get(k, 0.0) or 0.0)
            new_v = float(sensitivities.get(k, 0.0) or 0.0)
            if new_v <= eps:
                merged[k] = old_v
            elif old_v <= eps:
                merged[k] = new_v
            else:
                merged[k] = beta * old_v + (1.0 - beta) * new_v

        self.client_sensitivities[client_id] = merged

    def _get_layer_depth(self, layer_name: str) -> int:
        if "blocks." in layer_name:
            parts = layer_name.split("blocks.", 1)[1]
            idx_str = parts.split(".", 1)[0]
            if idx_str.isdigit():
                return int(idx_str)
        return 0

    def _get_structure_weight(self) -> float:
        t = max(0, self.current_round - 1)
        w = 1.0 - (t / float(self.struct_to_data_rounds))
        if w < 0.0:
            return 0.0
        if w > 1.0:
            return 1.0
        return w
    
    def _z_score_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-Score Normalization: Converts raw scores to standard normal distribution to amplify small differences.
        """
        if scores.size == 0:
            return scores
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Softmax converts scores to a probability distribution (Sum = 1).
        """
        if scores.size == 0:
            return scores
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # Prevent numerical overflow
        sum_exp = np.sum(exp_scores)
        if sum_exp == 0:
            return np.ones_like(scores) / scores.size
        return exp_scores / sum_exp
    
    def generate_allocation_map(self) -> Dict[int, Set[str]]:
        """
        Generate allocation map for the current round (Heterogeneous Allocation).
        
        Strategy:
        1. Structure Weight (Dominant in early phase): Prioritize layers closer to input + Uniform layer selection distribution.
        2. Data Weight (Increases with rounds): Norm of gradient changes in B at client side + Fairness weight.
        3. As rounds increase, data weight proportion rises; initial structure weight is 1.0.
        4. Use Z-Score normalization + Softmax to convert structure/data scores to probabilities, then apply geometric weighted averaging.
        """
        method = str(self.method).lower()
        if method in {"fedsalora", "fedsa-lora", "fedsa_lora", "flora"}:
            allocation_map = {client_id: set(self.layer_names) for client_id in range(self.num_clients)}
            for name in self.layer_names:
                self.layer_selection_counts[name] += self.num_clients
            return allocation_map

        suffixes = (".query", ".value", ".q_proj", ".v_proj")
        group_to_layers: Dict[str, List[str]] = {}
        for name in self.layer_names:
            group_id = name
            for suf in suffixes:
                if name.endswith(suf):
                    group_id = name[: -len(suf)]
                    break
            group_to_layers.setdefault(group_id, []).append(name)
        group_ids = sorted(group_to_layers.keys())

        if method == "fedra":
            rng = np.random.RandomState(int(self.seed) + int(getattr(self, "current_round", 1)) * 10007)
            allocation_map: Dict[int, Set[str]] = {}
            for client_id in range(self.num_clients):
                client_layer_budget = self.client_layer_counts.get(
                    client_id,
                    max(1, int(self.num_layers * self.allocation_ratio)),
                )
                max_budget = max(1, min(int(client_layer_budget), len(self.layer_names)))
                target_budget = int(max_budget)

                perm = rng.permutation(group_ids)
                chosen_groups: List[str] = []
                used = 0
                gid_sizes = {gid: len(group_to_layers.get(str(gid), [])) for gid in group_ids}
                for gid in perm:
                    members = group_to_layers.get(str(gid), [])
                    if len(members) == 0:
                        continue
                    if used + len(members) > target_budget:
                        continue
                    chosen_groups.append(str(gid))
                    used += len(members)
                    if used >= target_budget:
                        break

                if used < target_budget:
                    remaining = [str(gid) for gid in perm if str(gid) not in set(chosen_groups)]
                    remaining.sort(key=lambda g: int(gid_sizes.get(g, 10**9)))
                    for gid in remaining:
                        size = int(gid_sizes.get(gid, 0))
                        if size <= 0:
                            continue
                        if used + size > target_budget:
                            continue
                        chosen_groups.append(str(gid))
                        used += size
                        if used >= target_budget:
                            break

                if len(chosen_groups) == 0 and len(group_ids) > 0:
                    chosen_groups = [str(perm[0])]

                allocated_layers: Set[str] = set()
                for gid in chosen_groups:
                    allocated_layers.update(group_to_layers.get(str(gid), []))
                allocation_map[client_id] = allocated_layers
                for name in allocated_layers:
                    self.layer_selection_counts[name] += 1
            return allocation_map

        allocation_map = {}
        if method == "fedhello":
            is_stage1 = int(getattr(self, "current_round", 1)) <= int(getattr(self, "struct_to_data_rounds", 1))
            structure_weight = 1.0 if is_stage1 else 0.0
            data_weight = 0.0 if is_stage1 else 1.0
        elif method == "fedharp_g":
            structure_weight = 1.0
            data_weight = 0.0
        else:
            structure_weight = self._get_structure_weight()
            data_weight = 1.0 - structure_weight

        # 1) Calculate Global Structure Scores (Shared across all clients)
        struct_scores = []
        max_depth = max(self._get_layer_depth(gid) for gid in group_ids) if len(group_ids) > 0 else 0
        for gid in group_ids:
            member_counts = [self.layer_selection_counts.get(n, 0) for n in group_to_layers.get(gid, [])]
            count = float(np.mean(member_counts)) if len(member_counts) > 0 else 0.0
            uniform_weight = 1.0 / (1.0 + np.sqrt(count))
            depth = self._get_layer_depth(gid)
            if method == "fedhello":
                depth_idx = int(depth)
                decay = 0.9
                proximity_weight = float(decay) ** float(depth_idx)
                struct_scores.append(proximity_weight * uniform_weight)
            else:
                normalized_depth = (depth + 1) / (max_depth + 1)
                # Give higher weight to layers closer to input (small depth) and output (large depth)
                # Use inverted V-shape: proximity_weight = 1 - |2*normalized_depth - 1| ^ gamma
                gamma = 0.5  # Controls the steepness, smaller is steeper
                proximity_weight = 0.1 + 0.9 * np.abs(2.0 * normalized_depth - 1.0) ** gamma
                struct_scores.append(proximity_weight * uniform_weight)
        struct_scores = np.array(struct_scores, dtype=np.float64)
        # Softmax after Z-Score Normalization
        struct_probs = self._softmax(self._z_score_normalize(struct_scores))

        for client_id in range(self.num_clients):
            client_layer_budget = self.client_layer_counts.get(
                client_id,
                max(1, int(self.num_layers * self.allocation_ratio))
            )
            if len(group_ids) == 0:
                allocation_map[client_id] = set()
                continue

            denom = float(len(self.layer_names)) if len(self.layer_names) > 0 else 1.0
            ratio = float(client_layer_budget) / denom
            if ratio < 0.0:
                ratio = 0.0
            if ratio > 1.0:
                ratio = 1.0
            client_num_allocated = max(1, int(len(group_ids) * ratio))
            client_num_allocated = min(client_num_allocated, len(group_ids))

            client_sense = self.client_sensitivities.get(client_id, {})
            if method == "fedhello" and (client_sense is None or len(client_sense) == 0):
                merged: Dict[str, float] = {}
                for _, m in self.client_sensitivities.items():
                    if not m:
                        continue
                    for k, v in m.items():
                        merged[k] = float(merged.get(k, 0.0)) + float(v or 0.0)
                client_sense = merged

            # 2) Calculate Data Scores for this client
            data_scores = []
            for gid in group_ids:
                member_scores = []
                for n in group_to_layers.get(gid, []):
                    score = float(client_sense.get(n, 0.0))
                    if score <= 0.0:
                        score = 1e-12
                    member_scores.append(score)
                data_scores.append(float(np.mean(member_scores)) if len(member_scores) > 0 else 1e-12)
            data_scores = np.array(data_scores, dtype=np.float64)
            # Softmax after Z-Score Normalization
            data_probs = self._softmax(self._z_score_normalize(data_scores))

            if method == "fedharp_r":
                final_probs = None
            elif method == "fedhello":
                final_probs = struct_probs if structure_weight > 0.5 else data_probs
            else:
                log_final = structure_weight * np.log(struct_probs + 1e-20) + data_weight * np.log(data_probs + 1e-20)
                final_probs = np.exp(log_final)
                final_probs /= np.sum(final_probs)

            # 4) Sample based on probabilities
            if final_probs is None:
                selected_group_ids = np.random.choice(
                    group_ids,
                    size=client_num_allocated,
                    replace=False,
                )
            else:
                selected_group_ids = np.random.choice(
                    group_ids,
                    size=client_num_allocated,
                    replace=False,
                    p=final_probs,
                )
            allocated_layers: Set[str] = set()
            for gid in selected_group_ids:
                allocated_layers.update(group_to_layers.get(str(gid), []))
            allocation_map[client_id] = allocated_layers

            # 5) Update global selection counts
            for name in allocated_layers:
                self.layer_selection_counts[name] += 1

        return allocation_map
                
    
    def update_staleness(self, allocation_map: Dict[int, Set[str]]):
        """
        Update staleness counters.
        If a layer is allocated this round, staleness resets to 0; otherwise, staleness increments by 1.
        """
        for client_id in range(self.num_clients):
            allocated_layers = allocation_map.get(client_id, set())
            
            for layer_name in self.layer_names:
                if layer_name in allocated_layers:
                    # Client updated this layer this round, reset staleness
                    self.staleness[client_id][layer_name] = 0
                else:
                    # Client did not update this layer, increment staleness
                    self.staleness[client_id][layer_name] += 1
    
    def get_staleness_dampening(self, client_id: int, layer_name: str) -> float:
        """
        Calculate staleness dampening factor: alpha = 1 / sqrt(1 + tau)
        """
        tau = self.staleness[client_id][layer_name]
        return 1.0 / np.sqrt(1.0 + tau)
    
    def aggregate_matrices(
        self,
        client_updates_A: Dict[int, Dict[str, torch.Tensor]],
        client_updates_B: Optional[Dict[int, Dict[str, torch.Tensor]]],
        client_sample_counts: Dict[int, int],
        allocation_map: Dict[int, Set[str]]
    ):
        """
        Aggregate A/B matrix updates submitted by clients.
        
        Aggregation Logic (Layer-wise Weighted Average):
        For each layer j:
            Delta_j = (Sum_{k in Active} n_k * alpha_k * Delta_{k,j}) / (Sum_{k in Active} n_k)
        
        Args:
            client_updates_A: Mapping from Client ID to delta_A
            client_updates_B: Mapping from Client ID to delta_B (Optional)
            client_sample_counts: Mapping from Client ID to sample count (for weighting)
            allocation_map: Allocation map for the current round
        """
        if self.method == "flora":
            if client_updates_B is None:
                raise ValueError("flora requires client_updates_B")

            for layer_name in self.layer_names:
                pieces_A = []
                pieces_B = []
                for client_id, delta_A in client_updates_A.items():
                    allocated_layers = allocation_map.get(client_id, set())
                    if layer_name not in allocated_layers:
                        continue
                    a = delta_A.get(layer_name)
                    if a is None:
                        continue
                    b_map = client_updates_B.get(client_id)
                    if b_map is None:
                        continue
                    b = b_map.get(layer_name)
                    if b is None:
                        continue
                    pieces_A.append(a)
                    pieces_B.append(b)

                base_A = self.lora_layers[layer_name].get_A()
                base_B = self.lora_layers[layer_name].get_B()
                if len(pieces_A) == 0:
                    self.lora_layers[layer_name].set_A(torch.zeros((1, base_A.shape[1]), device=base_A.device, dtype=base_A.dtype))
                    self.lora_layers[layer_name].set_B(torch.zeros((base_B.shape[0], 1), device=base_B.device, dtype=base_B.dtype))
                    continue

                stacked_A = torch.cat(pieces_A, dim=0)
                stacked_B = torch.cat(pieces_B, dim=1)
                self.lora_layers[layer_name].set_A(stacked_A)
                self.lora_layers[layer_name].set_B(stacked_B)
            return
        method = str(self.method).lower()
        if method == "fedharp_a":
            aggregate_B = False
            use_staleness = False
            use_sample_weights = True
        elif method.startswith("fedharp") or method == "fedanon":
            aggregate_B = False
            use_staleness = True
            use_sample_weights = True
        elif method in {"fedsalora", "fedsa-lora", "fedsa_lora"}:
            aggregate_B = False
            use_staleness = False
            use_sample_weights = False
        elif method == "fedhello":
            aggregate_B = True
            use_staleness = False
            use_sample_weights = True
        elif method == "fedra":
            aggregate_B = True
            use_staleness = False
            use_sample_weights = False
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if method == "fedharp_a":
            weighted_updates_sum_A = {
                layer_name: torch.zeros_like(self.lora_layers[layer_name].get_A())
                for layer_name in self.layer_names
            }
            total_weights_A = {layer_name: 0.0 for layer_name in self.layer_names}

            for client_id, delta_A in client_updates_A.items():
                allocated_layers = allocation_map.get(client_id, set())
                for layer_name, delta in delta_A.items():
                    if layer_name not in allocated_layers:
                        continue
                    weighted_updates_sum_A[layer_name] += delta
                    total_weights_A[layer_name] += 1.0

            for layer_name, update_sum in weighted_updates_sum_A.items():
                denominator = total_weights_A[layer_name]
                if denominator <= 0:
                    continue
                avg_update = update_sum / denominator
                current_A = self.lora_layers[layer_name].get_A()
                new_A = current_A + self.aggregation_lr * avg_update
                self.lora_layers[layer_name].set_A(new_A)
            return

        weighted_updates_sum_A = {
            layer_name: torch.zeros_like(self.lora_layers[layer_name].get_A())
            for layer_name in self.layer_names
        }
        total_weights_A = {layer_name: 0.0 for layer_name in self.layer_names}

        weighted_updates_sum_B = None
        total_weights_B = None
        if aggregate_B:
            if client_updates_B is None:
                raise ValueError(f"{self.method} requires client_updates_B")
            weighted_updates_sum_B = {
                layer_name: torch.zeros_like(self.lora_layers[layer_name].get_B())
                for layer_name in self.layer_names
            }
            total_weights_B = {layer_name: 0.0 for layer_name in self.layer_names}

        for client_id, delta_A in client_updates_A.items():
            allocated_layers = allocation_map.get(client_id, set())
            n_k = float(client_sample_counts.get(client_id, 1.0)) if use_sample_weights else 1.0

            for layer_name, delta in delta_A.items():
                if layer_name not in allocated_layers:
                    continue
                alpha = self.get_staleness_dampening(client_id, layer_name) if use_staleness else 1.0
                weighted_updates_sum_A[layer_name] += (n_k * alpha) * delta
                total_weights_A[layer_name] += (n_k * alpha) if use_staleness else n_k

            if aggregate_B:
                b_map = client_updates_B.get(client_id, {})
                for layer_name, delta in b_map.items():
                    if layer_name not in allocated_layers:
                        continue
                    alpha = self.get_staleness_dampening(client_id, layer_name) if use_staleness else 1.0
                    weighted_updates_sum_B[layer_name] += (n_k * alpha) * delta
                    total_weights_B[layer_name] += (n_k * alpha) if use_staleness else n_k

        for layer_name, update_sum in weighted_updates_sum_A.items():
            denominator = total_weights_A[layer_name]
            if denominator <= 0:
                continue
            avg_update = update_sum / denominator
            current_A = self.lora_layers[layer_name].get_A()
            new_A = current_A + self.aggregation_lr * avg_update
            self.lora_layers[layer_name].set_A(new_A)

        if aggregate_B and weighted_updates_sum_B is not None and total_weights_B is not None:
            for layer_name, update_sum in weighted_updates_sum_B.items():
                denominator = total_weights_B[layer_name]
                if denominator <= 0:
                    continue
                avg_update = update_sum / denominator
                current_B = self.lora_layers[layer_name].get_B()
                new_B = current_B + self.aggregation_lr * avg_update
                self.lora_layers[layer_name].set_B(new_B)

    def aggregate_flora_stacking(
        self,
        client_updates_A: Dict[int, Dict[str, torch.Tensor]],
        client_updates_B: Dict[int, Dict[str, torch.Tensor]]
    ):
        for layer_name in self.layer_names:
            pieces_A = []
            pieces_B = []

            for client_id, A_map in client_updates_A.items():
                A = A_map.get(layer_name)
                if A is None:
                    continue
                B_map = client_updates_B.get(client_id)
                if B_map is None:
                    continue
                B = B_map.get(layer_name)
                if B is None:
                    continue
                pieces_A.append(A)
                pieces_B.append(B)

            if len(pieces_A) == 0:
                continue

            stacked_A = torch.cat(pieces_A, dim=0)
            stacked_B = torch.cat(pieces_B, dim=1)
            self.lora_layers[layer_name].set_A(stacked_A)
            self.lora_layers[layer_name].set_B(stacked_B)

    def aggregate_A_matrices(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sample_counts: Dict[int, int],
        allocation_map: Dict[int, Set[str]]
    ):
        return self.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=None,
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )
    
    def get_global_A_matrices(self) -> Dict[str, torch.Tensor]:
        """Get all global A matrices"""
        return {name: layer.get_A() for name, layer in self.lora_layers.items()}

    def get_global_B_matrices(self) -> Dict[str, torch.Tensor]:
        """Get all global B matrices"""
        return {name: layer.get_B() for name, layer in self.lora_layers.items()}
    
    def set_global_A_matrices(self, A_matrices: Dict[str, torch.Tensor]):
        """Set global A matrices (for initialization)"""
        for name, A in A_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_A(A)

    def set_global_B_matrices(self, B_matrices: Dict[str, torch.Tensor]):
        """Set global B matrices (for initialization)"""
        for name, B in B_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_B(B)
    
    def get_staleness_stats(self) -> Dict[str, float]:
        """Get staleness statistics"""
        all_staleness = []
        for client_staleness in self.staleness.values():
            all_staleness.extend(client_staleness.values())
        
        if len(all_staleness) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        all_staleness = np.array(all_staleness)
        return {
            'mean': float(np.mean(all_staleness)),
            'std': float(np.std(all_staleness)),
            'max': int(np.max(all_staleness))
        }
    
    def print_allocation_stats(self, allocation_map: Dict[int, Set[str]]):
        """Print allocation statistics for the current round"""
        print(f"\nRound {self.current_round} Allocation Stats:")
        print("-" * 60)
        
        # Count how many clients selected each layer
        layer_client_count = {layer_name: 0 for layer_name in self.layer_names}
        for client_id, allocated_layers in allocation_map.items():
            for layer_name in allocated_layers:
                layer_client_count[layer_name] += 1
        
        # Print staleness stats
        staleness_stats = self.get_staleness_stats()
        print(f"Layer Coverage: Min={min(layer_client_count.values())}, Max={max(layer_client_count.values())}")
        print(f"System Staleness: Mean={staleness_stats['mean']:.2f}, Max={staleness_stats['max']}")
        
        # Print fairness/selection count stats
        print("Fairness Counts (Top-5 most selected layers):")
        sorted_counts = sorted(self.layer_selection_counts.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_counts[:5]:
            print(f"  {name}: {count} times (Fairness weight: {1.0:.4f})")

        structure_weight = self._get_structure_weight()
        data_weight = 1.0 - structure_weight
        if self.client_sensitivities:
            print(f"Allocation Weights: Structure={structure_weight:.2f}, Data={data_weight:.2f} (Collected data from {len(self.client_sensitivities)} clients)")
        else:
            print(f"Allocation Weights: Structure={structure_weight:.2f}, Data={data_weight:.2f} ")
            
        print("-" * 60 + "\n")
    
    def start_round(self) -> Dict[int, Set[str]]:
        """Start a new round"""
        self.current_round += 1
        allocation_map = self.generate_allocation_map()
        if str(self.method).lower().startswith("fedharp") or str(self.method).lower() == "fedanon":
            self.update_staleness(allocation_map)
        return allocation_map


