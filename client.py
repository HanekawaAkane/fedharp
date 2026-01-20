import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Set, Optional, Tuple
import copy
from models import LoRALinear, get_lora_layer_names


class FedHarpClient:
    """
    Fed-HARP Federated Learning Client.

    Core Functions:
    - Native Privacy: B matrices are persisted locally and never uploaded.
    - Heterogeneous Allocation: Jointly train allocated layers.
    - Sensitivity Calculation: Compute B matrix sensitivity (L2 Norm) to guide server allocation.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        lora_layers: Dict[str, LoRALinear],
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),   
        lr: float = 0.001,
        warmup_steps: int = 10,
        warmup_lr: float = 0.0001,
        method: str = "fedharp"
    ):
        self.client_id = client_id
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.method = "fedharp" if str(method).lower() == "fedanon" else method
        
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        
        self.previously_frozen_layers: Set[str] = set()
        
        self.model.to(device)
        
        self.optimizer = None
    
    def get_all_A_matrices(self) -> Dict[str, torch.Tensor]:
        return {name: layer.get_A() for name, layer in self.lora_layers.items()}
    
    def get_all_B_matrices(self) -> Dict[str, torch.Tensor]:
        return {name: layer.get_B() for name, layer in self.lora_layers.items()}
    
    def set_A_matrices(self, A_matrices: Dict[str, torch.Tensor]):
        for name, A in A_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_A(A)

    def set_B_matrices(self, B_matrices: Dict[str, torch.Tensor]):
        for name, B in B_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_B(B)

    def receive_global_matrices(
        self,
        A_matrices: Dict[str, torch.Tensor],
        B_matrices: Optional[Dict[str, torch.Tensor]] = None,
        method: Optional[str] = None
    ):
        method = method or self.method
        method_l = str(method).lower()

        if method_l == "flora":
            if B_matrices is None:
                raise ValueError("flora requires B_matrices")
            self.set_A_matrices(A_matrices)
            self.set_B_matrices(B_matrices)
            return

        self.set_A_matrices(A_matrices)
        if method_l in {"fedhello", "fedra"}:
            if B_matrices is None:
                raise ValueError(f"{method_l} requires B_matrices")
            self.set_B_matrices(B_matrices)

    def flora_merge_and_reset(self, target_rank: int):
        target_rank = int(target_rank)
        for layer in self.lora_layers.values():
            layer.merge_lora_weights()
            layer.reset_lora(rank=target_rank)
    
    def calculate_B_sensitivity(self) -> Dict[str, float]:
        sensitivities = {}
        for name in self.lora_layers.keys():
            sensitivities[name] = 0.0
        return sensitivities

    def _setup_layer_trainability(self, allocated_layers: Set[str]):
        for name, layer in self.lora_layers.items():
            if name in allocated_layers:
                layer.unfreeze_all()
            else:
                layer.freeze_all()

    def _setup_layer_trainability_B_only(self, train_B_layers: Set[str]):
        for name, layer in self.lora_layers.items():
            if name in train_B_layers:
                layer.freeze_A()
                layer.unfreeze_B()
            else:
                layer.freeze_all()
    
    def _create_optimizer(self, warmup_mode: bool = False):
        trainable_params = []
        seen = set()
        for layer in self.lora_layers.values():
            for p in layer.get_trainable_params():
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                trainable_params.append(p)

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            trainable_params.append(p)
        
        if len(trainable_params) == 0:
            return None
        
        lr = self.lr
        return optim.Adam(trainable_params, lr=lr)

    def _train_epoch_collect_B_grad_norms(
        self, warmup_mode: bool = False, include_A: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}

        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)

        if self.optimizer is None:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        for _, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, layer in self.lora_layers.items():
                grad_sq = 0.0
                grad_b = layer.lora_B.grad
                if grad_b is not None:
                    n_b = torch.norm(grad_b.detach(), p=2).item()
                    grad_sq += float(n_b) * float(n_b)
                if include_A:
                    grad_a = layer.lora_A.grad
                    if grad_a is not None:
                        n_a = torch.norm(grad_a.detach(), p=2).item()
                        grad_sq += float(n_a) * float(n_a)
                if grad_sq <= 0.0:
                    continue
                n = float(grad_sq) ** 0.5
                grad_norm_sq_sum[name] += n * n

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return total_loss / num_batches, grad_sensitivity

    def _train_epoch(self, warmup_mode: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)
        
        if self.optimizer is None:
            return 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def local_train(
        self,
        allocated_layers: Set[str],
        num_epochs: int = 10,
        method: Optional[str] = None
    ):
        method = method or self.method
        upload_B = method in ("fedhello", "flora", "fedra")

        A_before = self.get_all_A_matrices()
        B_before = self.get_all_B_matrices() if upload_B else None
        
        self._setup_layer_trainability(allocated_layers)

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
        grad_norm_steps = 0
        
        for epoch in range(num_epochs):
            train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                warmup_mode=False, include_A=str(method).lower() == "fedhello"
            )
            print(f"  Client {self.client_id}: Training Epoch {epoch+1}/{num_epochs}, Loss = {train_loss:.4f}")

            for name, v in grad_sensitivity_epoch.items():
                grad_norm_sq_sum[name] += v * v
            grad_norm_steps += 1
        
        A_after = self.get_all_A_matrices()
        B_after = self.get_all_B_matrices() if upload_B else None

        updates_A: Dict[str, torch.Tensor] = {}
        updates_B: Optional[Dict[str, torch.Tensor]] = None

        if method == "flora":
            updates_B = {}
            for name in allocated_layers:
                if name in A_after and B_after is not None and name in B_after:
                    updates_A[name] = A_after[name]
                    updates_B[name] = B_after[name]
        else:
            for name in allocated_layers:
                if name in A_before and name in A_after:
                    updates_A[name] = A_after[name] - A_before[name]

            if upload_B and B_before is not None and B_after is not None:
                updates_B = {}
                for name in allocated_layers:
                    if name in B_before and name in B_after:
                        updates_B[name] = B_after[name] - B_before[name]
        
        all_layer_names = set(self.lora_layers.keys())
        self.previously_frozen_layers = all_layer_names - allocated_layers

        if grad_norm_steps > 0:
            b_grad_sensitivity = {name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5 for name in grad_norm_sq_sum.keys()}
        else:
            b_grad_sensitivity = {name: 0.0 for name in self.lora_layers.keys()}

        return updates_A, updates_B, b_grad_sensitivity

    def train_B_only(self, train_B_layers: Set[str], num_epochs: int = 1) -> Dict[str, float]:
        self._setup_layer_trainability_B_only(train_B_layers)
        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
        grad_norm_steps = 0
        for epoch in range(num_epochs):
            train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(warmup_mode=False)
            print(
                f"  Client {self.client_id}: B-only Training Epoch {epoch+1}/{num_epochs}, Loss = {train_loss:.4f}"
            )
            for name, v in grad_sensitivity_epoch.items():
                grad_norm_sq_sum[name] += v * v
            grad_norm_steps += 1

        if grad_norm_steps > 0:
            return {name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return {name: 0.0 for name in self.lora_layers.keys()}
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("No test DataLoader provided, and no local test_loader configured.")
            test_loader = self.test_loader
            
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


FedAnonClient = FedHarpClient
