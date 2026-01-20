"""
Fed-HARP Federated Learning Simulation Main Program.
Responsible for initializing the environment, loading data, building models, and executing the federated training loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Set


def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fed-HARP: Federated Asymmetric Non-stationary Optimization with Native Privacy")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="CIFAR-10 dataset directory")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                        help="Dataset name (cifar10/cifar100, default: cifar10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Non-IID Dirichlet parameter (default: 0.5)")
    parser.add_argument("--client_splits", type=str, default="0.9,0.1",
                        help="Local train/test split ratio per client, e.g., '0.9,0.1'")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data used per client (0-1, e.g. 0.01 means 1%%; default: 1.0)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT model name (default: vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to use pretrained weights")
    parser.add_argument("--pretrained_source", type=str, default="local", choices=["modelscope", "local"],
                        help="Source of pretrained weights (default: modelscope)")
    parser.add_argument("--pretrained_path", type=str, default="vit",
                        help="Local pretrained checkpoint path (pretrained_source=local)")
    parser.add_argument("--ms_model_id", type=str, default=None,
                        help="ModelScope model ID (pretrained_source=modelscope)")
    parser.add_argument("--ms_revision", type=str, default=None,
                        help="ModelScope revision/tag/commit (optional)")
    parser.add_argument("--ms_cache_dir", type=str, default=None,
                        help="ModelScope cache directory (optional)")
    parser.add_argument("--ms_checkpoint_file", type=str, default=None,
                        help="Relative path of checkpoint in ModelScope snapshot (optional)")
    parser.add_argument("--hf_endpoint", type=str, default=None,
                        help="HuggingFace Hub endpoint/mirror (e.g., https://hf-mirror.com)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor (default: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,  
                        help="LoRA dropout rate (default: 0.0)")
    
    # Federated Learning arguments
    parser.add_argument("--method", type=str, default="fedharp",
                        choices=[
                            "fedharp",
                            "fedharp_r",
                            "fedharp_g",
                            "fedharp_l",
                            "fedharp_a",
                            "fedanon",
                            "fedhello",
                            "fedra",
                            "flora",
                            "fedsalora",
                            "fedsa-lora",
                            "fedsa_lora",
                            "FedSA-LoRA",
                        ],
                        help="Comparison method (default: fedharp)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Total number of clients (default: 10)")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="Number of federated training rounds (default: 1)")
    parser.add_argument("--clients_per_round", type=int, default=None,
                        help="Number of clients selected per round (default: all)")
    parser.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"],
                        help="Client resource heterogeneity type: 'uniform' (all same) or '6-3-1' (simulating High/Mid/Low resource distribution)")
    parser.add_argument("--allocation_ratio", type=float, default=0.5,
                        help="Base allocation ratio (used in uniform mode or as low-resource baseline)")
    parser.add_argument("--aggregation_lr", type=float, default=1.0,
                        help="Server aggregation learning rate eta (default: 1.0)")
    parser.add_argument("--struct_to_data_rounds", type=int, default=20,
                        help="Number of rounds for transition from structure weight to data weight (default: 20)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--client_lr", type=float, default=0.01,
                        help="Client local learning rate (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of local epochs per round (default: 1)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Steps for B-matrix warmup alignment (default: 10)")
    parser.add_argument("--warmup_lr", type=float, default=0.0001,
                        help="Learning rate during warmup phase (default: 0.0001)")
    parser.add_argument("--b_train_every", type=int, default=5,
                        help="Trigger B-only training every N rounds (default: 5)")
    parser.add_argument("--b_num_epochs", type=int, default=1,
                        help="Number of local epochs during B-only phase (default: 1)")
    
    # Evaluation arguments
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate every N rounds (default: 1)")
    parser.add_argument("--test_batch_size", type=int, default=100,
                        help="Test batch size (default: 100)")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Whether to save model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    return parser.parse_args(args=argv)


def _strip_double_dash(argv):
    if argv is None:
        return None
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    """
    Generate resource capabilities (ratio of trainable layers) for each client.
    
    Args:
        num_clients: Total number of clients
        heterogeneity_type: 'uniform' or '6-3-1'
        base_ratio: Base ratio
        
    Returns:
        List[float]: List of allocation ratios for each client
    """
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    elif heterogeneity_type == "1-1-1":
        n_low = int(0.33 * num_clients)
        n_mid = int(0.33 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = []
        ratios.extend([base_ratio] * n_low)            # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    elif heterogeneity_type == "6-3-1":
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        
        ratios = []
        ratios.extend([base_ratio] * n_low)            # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)                  # High (Full Layers)
        
        np.random.shuffle(ratios)
        return ratios
    
    return [base_ratio] * num_clients
    

def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate global model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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

def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, v in d.items():
        total += _tensor_nbytes(v)
    return total


def _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_A)


def _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_B)


def _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(A_matrices)


def _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:
    n = 0
    for name in allocated_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_A.numel())
        n += int(layer.lora_B.numel())
    return n


def _b_only_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], train_B_layers: Set[str]) -> int:
    n = 0
    for name in train_B_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_B.numel())
    return n


def vision_main(argv=None):
    """Federated Learning Main Loop"""
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    if str(args.method).lower() == "fedanon":
        args.method = "fedharp"

    if str(args.method).lower() in {"fedsalora", "fedsa-lora", "fedsa_lora"}:
        if "--heterogeneity_type" not in raw_argv:
            args.heterogeneity_type = "uniform"
        if "--allocation_ratio" not in raw_argv:
            args.allocation_ratio = 1.0

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    
    global np
    global torch
    global nn
    global DataLoader
    global CIFAR10Dataset
    global CIFAR100Dataset
    global FedHarpClient
    global FedHarpServer
    global create_vit_model
    global create_client_model
    global set_seed
    global get_device
    global save_checkpoint
    global create_directories
    global print_model_info

    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'numpy'. Install it before running vision experiments.") from e

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'torch'. Install it before running vision experiments.") from e

    from client import FedHarpClient
    from dataset import CIFAR10Dataset, CIFAR100Dataset
    from models import create_client_model, create_vit_model
    from server import FedHarpServer
    from utils import (
        create_directories,
        get_device,
        print_model_info,
        save_checkpoint,
        set_seed,
    )

    set_seed(args.seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    if args.save_checkpoints:
        create_directories(args.checkpoint_dir)

    safe_method = str(args.method).replace("/", "_").replace(" ", "_")
    safe_dataset = str(args.dataset).replace("/", "_").replace(" ", "_")
    safe_model = str(args.model_name).replace("/", "_").replace(" ", "_")
    safe_alpha = str(args.alpha).replace("/", "_").replace(" ", "_")
    safe_heterogeneity_type = str(args.heterogeneity_type).replace("/", "_").replace(" ", "_")
    safe_b_train_every = str(args.b_train_every).replace("/", "_").replace(" ", "_")
    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_b_train_every}--{safe_heterogeneity_type}.txt"
    log_path = os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[WARNING] Cannot write log file {log_path}: {e}")
    
    print("\n" + "="*80)
    print("Fed-HARP: Federated Asymmetric Non-stationary Optimization with Native Privacy")
    print("="*80)
    print(f"Configuration Info:")
    print(f"  Num Clients: {args.num_clients}")
    print(f"  Num Rounds: {args.num_rounds}")
    print(f"  Method: {args.method}")
    print(f"  Resource Heterogeneity: {args.heterogeneity_type} (Base Ratio: {args.allocation_ratio})")
    print(f"  LoRA Params: Rank={args.lora_rank}, Alpha={args.lora_alpha}")
    print(f"  Non-IID Params (Alpha): {args.alpha}")
    print("="*80 + "\n")
    
    print(f"Loading {args.dataset.upper()} dataset...")
    try:
        splits = [float(x) for x in args.client_splits.split(",")]
    except:
        splits = [0.9, 0.1]
    
    if args.dataset == "cifar100":
        dataset = CIFAR100Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits
        )
        num_classes = 100
    else:
        dataset = CIFAR10Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits
        )
        num_classes = 10
    
    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)
    
    print("\nCreating Vision Transformer with LoRA...")
    global_model, global_lora_layers = create_vit_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        pretrained_source=args.pretrained_source,
        pretrained_checkpoint_path=args.pretrained_path,
        modelscope_model_id=args.ms_model_id,
        modelscope_revision=args.ms_revision,
        modelscope_cache_dir=args.ms_cache_dir,
        modelscope_checkpoint_file=args.ms_checkpoint_file,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    global_model.to(device)
    print_model_info(global_model, global_lora_layers)
    
    print(f"Generating client resource distribution ({args.heterogeneity_type})...")
    client_capacities = generate_client_resources(
        args.num_clients, 
        args.heterogeneity_type, 
        args.allocation_ratio
    )
    avg_cap = np.mean(client_capacities)
    print(f"  Client layer training ratio: Mean={avg_cap:.2f}, Min={min(client_capacities):.2f}, Max={max(client_capacities):.2f}")
    if args.num_clients <= 20:
        print(f"  Detailed distribution: {['{:.2f}'.format(c) for c in client_capacities]}")

    print("Initializing Fed-HARP Server...")
    server = FedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        method=args.method,
        allocation_ratio=args.allocation_ratio, 
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed
    )
    if hasattr(server, 'client_layer_counts'):
        num_total_layers = len(global_lora_layers)
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * cap))
        print("  Successfully injected heterogeneous resource config into server.")
    
    clients = []
    client_sample_counts = {}
    
    print(f"\nInitializing {args.num_clients} clients...")
    for client_id in range(args.num_clients):
        train_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.batch_size,
            shuffle=True,
            split="train",
            fraction=args.data_fraction
        )
        client_test_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.test_batch_size,
            shuffle=False,
            split="test",
            fraction=1.0
        )
        
        client_sample_counts[client_id] = len(train_loader.dataset)
        
        client_model, client_lora_layers = create_client_model(
            global_model,
            global_lora_layers
        )
        client_model.to(device)
        
        client = FedHarpClient(
            client_id=client_id,
            model=client_model,
            lora_layers=client_lora_layers,
            train_loader=train_loader,
            test_loader=client_test_loader, 
            device=device,
            lr=args.client_lr,
            method=args.method,
            warmup_steps=args.warmup_steps,
            warmup_lr=args.warmup_lr
        )
        clients.append(client)
        if client_id < 5:
            print(f"  Client {client_id}: {len(train_loader.dataset)} training samples")
    
    if args.num_clients > 5:
        print(f"  ... ({args.num_clients - 5} remaining clients initialized)")
    
    clients_per_round = args.clients_per_round or args.num_clients
    
    print("\n" + "="*80)
    print("Starting Fed-HARP Federated Learning Process")
    print("="*80)
    
    best_accuracy = 0.0
    training_history = []

    method_l = str(args.method).lower()
    is_fedharp_family = method_l in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a"}
    b_train_every = int(args.b_train_every) if (is_fedharp_family and method_l != "fedharp_l") else 0
    
    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Round {round_num}/{args.num_rounds}")
        print(f"{'='*80}")
        
        allocation_map = server.start_round()
        server.print_allocation_stats(allocation_map)
        
        selected_clients = np.random.choice(
            args.num_clients,
            size=min(clients_per_round, args.num_clients),
            replace=False
        )
        
        print(f"Active clients this round: {len(selected_clients)}")
        
        global_A = server.get_global_A_matrices()
        upload_b_method = str(args.method).lower() in {"fedhello", "flora", "fedra"}
        global_B = server.get_global_B_matrices() if upload_b_method else None
        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)
        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0
        client_updates = {}
        client_updates_B = {}
        client_metrics_ab = []
        
        for i, client_id in enumerate(selected_clients):
            client = clients[client_id]
            allocated_layers = allocation_map[client_id]
            
            client.receive_global_matrices(global_A, global_B, method=args.method)
            if args.method == "flora":
                client.flora_merge_and_reset(target_rank=args.lora_rank)

            if b_train_every > 0 and is_fedharp_family and method_l != "fedharp_l":
                stale_map = server.staleness.get(int(client_id), {})
                b_only_layers = {name for name, tau in stale_map.items() if int(tau) == int(b_train_every)}
                if len(b_only_layers) > 0:
                    new_b_sensitivity = client.train_B_only(b_only_layers, num_epochs=args.b_num_epochs)
                    server.update_client_sensitivities(int(client_id), new_b_sensitivity)
            
            if i < 3:
                print(f"  Client {client_id} starting training (Allocated layers: {len(allocated_layers)})...")

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            delta_A, delta_B, b_sensitivity = client.local_train(
                allocated_layers=allocated_layers,
                num_epochs=args.num_epochs,
                method=args.method
            )
            train_time_sec = time.time() - t0
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            else:
                peak_mem_gb = 0.0

            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)
            if upload_b_method and delta_B is None:
                raise ValueError(f"{args.method} requires both A and B upload, but delta_B is empty")
            uplink_B_mb = (_delta_B_nbytes(delta_B) / (1024.0 * 1024.0)) if (upload_b_method and delta_B is not None) else 0.0
            uplink_mb = uplink_A_mb + uplink_B_mb
            comm_mb = downlink_A_mb + downlink_B_mb + uplink_mb
            train_samples = len(client.train_loader.dataset)
            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)
            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12

            client_metrics_ab.append({
                "round": round_num,
                "client": int(client_id),
                "phase": "ab_train",
                "compute_tflops": float(compute_tflops),
                "peak_mem_gb": float(peak_mem_gb),
                "comm_mb": float(comm_mb),
                "train_time_sec": float(train_time_sec),
                "downlink_mb": float(downlink_A_mb + downlink_B_mb),
                "uplink_mb": float(uplink_mb)
            })
            
            client_updates[client_id] = delta_A

            if upload_b_method:
                client_updates_B[client_id] = delta_B
            
            server.update_client_sensitivities(client_id, b_sensitivity)

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        "round={round}, type=client_metrics, phase={phase}, client={client}, "
                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "
                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "
                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}\n".format(**client_metrics_ab[-1])
                    )
            except Exception as e:
                print(f"[WARNING] Cannot write log file {log_path}: {e}")

        if len(client_metrics_ab) > 0:
            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))
            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))
            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))
            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))
            print(f"Round Client Mean (A+B): Compute={avg_compute:.4f} TFLOPs, Peak Mem={avg_mem:.4f} GB, Comm={avg_comm:.2f} MB/R, Time={avg_time:.2f} s/R")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\n"
                    )
            except Exception as e:
                print(f"[WARNING] Cannot write log file {log_path}: {e}")
        
        print(f"\nServer aggregating LoRA updates from {len(client_updates)} clients...")
        server.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=(client_updates_B if upload_b_method else None),
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )

        updated_global_A = server.get_global_A_matrices()
        for name, layer in global_lora_layers.items():
            if name in updated_global_A:
                layer.set_A(updated_global_A[name])

        if upload_b_method:
            updated_global_B = server.get_global_B_matrices()
            for name, layer in global_lora_layers.items():
                if name in updated_global_B:
                    layer.set_B(updated_global_B[name])
        
        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            print(f"\nEvaluating on client local Test sets (Personalized Performance)...")
            
            total_weighted_correct = 0.0
            total_samples = 0
            total_weighted_loss = 0.0
            
            for cid in range(args.num_clients):
                client = clients[cid]
                eval_A = server.get_global_A_matrices()
                eval_B = server.get_global_B_matrices() if args.method in {"fedhello", "flora","fedra"} else None
                client.receive_global_matrices(eval_A, eval_B, method=args.method)
                
                if client.test_loader is None or len(client.test_loader.dataset) == 0:
                    continue
                    
                results = client.evaluate() 
                acc = results['accuracy']
                loss = results['loss']
                num_samples = len(client.test_loader.dataset)
                
                total_weighted_correct += (acc / 100.0) * num_samples
                total_weighted_loss += loss * num_samples
                total_samples += num_samples
                
                if cid < 3:
                    print(f"  Client {cid}: Acc={acc:.2f}%, Loss={loss:.4f}")

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            print(f"\n  [Round {round_num}] Global Weighted Accuracy (Personalized): {global_acc:.2f}%")
            print(f"  [Round {round_num}] Global Weighted Loss: {global_loss:.4f}")
            
            training_history.append({
                'round': round_num,
                'accuracy': global_acc,
                'loss': global_loss,
                'type': 'personalized'
            })

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=personalized, "
                        f"accuracy={global_acc:.4f}, loss={global_loss:.6f}\n"
                    )
            except Exception as e:
                print(f"[WARNING] Cannot write log file {log_path}: {e}")
            
            if global_acc > best_accuracy:
                best_accuracy = global_acc
                print(f"  ✓ New Record! Best Accuracy")
        
        if args.save_checkpoints and round_num % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_round_{round_num}.pt"
            )
            save_checkpoint(
                model_state=global_model.state_dict(),
                lora_layers_state={name: layer.get_A() for name, layer in global_lora_layers.items()},
                round_num=round_num,
                filepath=checkpoint_path
            )
        
        round_time = time.time() - round_start_time
        print(f"\nRound {round_num} time: {round_time:.2f} seconds")
    
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)
    print(f"Best Historical Accuracy (Personalized): {best_accuracy:.2f}%")
    
    personalized_history = [e for e in training_history if e.get('type') == 'personalized']
    
    if personalized_history:
        print(f"Final Round Personalized Accuracy: {personalized_history[-1]['accuracy']:.2f}%")
    
    if len(training_history) > 0:
        print("\nTraining History Snippet (Every 10 rounds):")
        if personalized_history:
            print("  Personalized Accuracy:")
            for i, entry in enumerate(personalized_history):
                if i % 10 == 0 or i == len(personalized_history) - 1:
                    print(f"    Round {entry['round']:3d}: Accuracy = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    def _has_flag(args: List[str], flag: str) -> bool:
        prefix = flag + "="
        return any(a == flag or a.startswith(prefix) for a in args)

    def _get_flag_value(args: List[str], flag: str) -> str | None:
        prefix = flag + "="
        for i, a in enumerate(args):
            if a == flag:
                if i + 1 < len(args):
                    return args[i + 1]
                return None
            if a.startswith(prefix):
                return a[len(prefix) :]
        return None

    def _remove_flag_with_value(args: List[str], flag: str) -> List[str]:
        out: List[str] = []
        prefix = flag + "="
        i = 0
        while i < len(args):
            a = args[i]
            if a == flag:
                i += 2
                continue
            if a.startswith(prefix):
                i += 1
                continue
            out.append(a)
            i += 1
        return out

    entry = argparse.ArgumentParser(description="Unified Fed-HARP Experiment Entry Point", add_help=True)
    entry.add_argument(
        "--task",
        type=str,
        default="vision",
        choices=["vision", "roberta", "flora", "fedra"],
    )
    known, rest = entry.parse_known_args(args=raw_argv)
    rest = _strip_double_dash(rest)

    task_explicit = _has_flag(raw_argv, "--task")
    method_raw = _get_flag_value(raw_argv, "--method")
    dataset_raw = _get_flag_value(raw_argv, "--dataset")
    model_path_raw = _get_flag_value(raw_argv, "--model_path")
    task_raw = _get_flag_value(raw_argv, "--task")

    method = (method_raw or "").strip().lower()
    dataset = (dataset_raw or "").strip().lower()
    task = (task_raw or str(known.task)).strip().lower()

    if method == "flora":
        if (not task_explicit) or task == "vision":
            task = "flora"

    if method == "fedra":
        is_nlp = bool(model_path_raw) or dataset in {"sst2", "qnli"} or _has_flag(raw_argv, "--lora_targets")
        is_vision = dataset in {"cifar10", "cifar100"} or _has_flag(raw_argv, "--model_name")
        if (not task_explicit) or task in {"vision", "fedra"}:
            if is_vision and not is_nlp:
                task = "fedra"
            else:
                task = "roberta"

    if task == "vision":
        return vision_main(rest)
    if task == "roberta":
        from robertamain import main as roberta_main

        return roberta_main(rest)
    if task == "flora":
        from floramain import main as flora_main

        return flora_main(rest)
    if task == "fedra":
        from fedramain import main as fedra_main

        if method == "fedra":
            rest = _remove_flag_with_value(rest, "--method")
        return fedra_main(rest)
    raise SystemExit(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
