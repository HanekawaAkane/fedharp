"""
Fed-HARP 联邦学习模拟主程序。
负责初始化环境、加载数据、构建模型以及执行联邦训练循环。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Set


def parse_args(argv=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Fed-HARP: 联邦非对称非平稳优化与原生隐私")
    
    # 数据集参数
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="CIFAR-10 数据集目录")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                        help="数据集名称 (cifar10/cifar100，默认: cifar10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Non-IID Dirichlet 分布参数 (默认: 0.5)")
    parser.add_argument("--client_splits", type=str, default="0.9,0.1",
                        help="每个客户端本地 train/test 划分比例，例如 '0.9,0.1'")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="每个客户端使用的训练数据比例 (0-1, 如 0.01 表示 1%% 数据；默认: 1.0)")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT 模型名称 (默认: vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true",
                        help="是否使用预训练权重")
    parser.add_argument("--pretrained_source", type=str, default="local", choices=["modelscope", "local"],
                        help="预训练权重来源 (默认: modelscope)")
    parser.add_argument("--pretrained_path", type=str, default="vit",
                        help="本地 pretrained checkpoint 路径 (pretrained_source=local)")
    parser.add_argument("--ms_model_id", type=str, default=None,
                        help="ModelScope 模型 ID (pretrained_source=modelscope)")
    parser.add_argument("--ms_revision", type=str, default=None,
                        help="ModelScope revision/tag/commit (可选)")
    parser.add_argument("--ms_cache_dir", type=str, default=None,
                        help="ModelScope 缓存目录 (可选)")
    parser.add_argument("--ms_checkpoint_file", type=str, default=None,
                        help="ModelScope snapshot 内 checkpoint 相对路径 (可选)")
    parser.add_argument("--hf_endpoint", type=str, default=None,
                        help="HuggingFace Hub 端点/镜像 (如 https://hf-mirror.com)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA 秩 (默认: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha 缩放因子 (默认: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,  
                        help="LoRA dropout 率 (默认: 0.0)")
    
    # 联邦学习参数
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
                        help="对比方法 (默认: fedharp)")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="客户端总数 (默认: 10)")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="联邦训练轮数 (默认: 1)")
    parser.add_argument("--clients_per_round", type=int, default=None,
                        help="每轮选中的客户端数量 (默认: 全部)")
    parser.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"],
                        help="客户端资源异构类型: 'uniform'(全部相同) 或 '6-3-1'(模拟高中低资源分布)")
    parser.add_argument("--allocation_ratio", type=float, default=0.5,
                        help="基础分配比例 (在 uniform 模式下使用，或作为低资源基准)")
    parser.add_argument("--aggregation_lr", type=float, default=1.0,
                        help="服务器聚合学习率 eta (默认: 1.0)")
    parser.add_argument("--struct_to_data_rounds", type=int, default=20,
                        help="结构权重到数据权重的过渡轮数 (默认: 20)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小 (默认: 128)")
    parser.add_argument("--client_lr", type=float, default=0.01,
                        help="客户端本地学习率 (默认: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="每轮本地训练的 Epoch 数 (默认: 1)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="B 矩阵预热对齐的步数 (默认: 10)")
    parser.add_argument("--warmup_lr", type=float, default=0.0001,
                        help="预热阶段的学习率 (默认: 0.0001)")
    parser.add_argument("--b_train_every", type=int, default=5,
                        help="每隔多少轮通信后触发一次 B-only 训练 (默认: 5)")
    parser.add_argument("--b_num_epochs", type=int, default=1,
                        help="B-only 阶段本地训练的 Epoch 数 (默认: 1)")
    
    # 评估参数
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮评估一次 (默认: 1)")
    parser.add_argument("--test_batch_size", type=int, default=100,
                        help="测试批次大小 (默认: 100)")
    
    # 系统参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="使用的设备 (cuda/cpu, 默认: 自动检测)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="是否保存模型检查点")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点保存目录")
    
    return parser.parse_args(args=argv)


def _strip_double_dash(argv):
    if argv is None:
        return None
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    """
    生成每个客户端的资源能力（即可以训练的层比例）。
    
    参数:
        num_clients: 客户端总数
        heterogeneity_type: 'uniform' 或 '6-3-1'
        base_ratio: 基础比例
        
    返回:
        List[float]: 每个客户端的分配比例列表
    """
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    elif heterogeneity_type == "1-1-1":
        n_low = int(0.33 * num_clients)
        n_mid = int(0.33 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    elif heterogeneity_type == "6-3-1":
        # 模拟 Fed-HeLLo 论文中的资源分布
        # 60% 低资源 (基准比例, 如 0.25 或 0.5)
        # 30% 中资源 (基准 * 1.5 或 0.75)
        # 10% 高资源 (全量训练 1.0)
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        
        # 定义不同等级的层比例
        # 注意：这里假设 base_ratio 是最低标准
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)                 # High (Full Layers)
        
        # 打乱分配，使得 ID 不直接对应资源等级
        np.random.shuffle(ratios)
        return ratios
    
    return [base_ratio] * num_clients
    

def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """在测试集上评估全局模型"""
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
    """联邦学习主循环"""
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
    
    # 设置随机种子
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
    
    # 获取计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # 创建目录
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

    # 将本次训练的配置写入日志文件（覆盖旧内容）
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[警告] 无法写入日志文件 {log_path}: {e}")
    
    print("\n" + "="*80)
    print("Fed-HARP: 联邦非对称非平稳优化与原生隐私 (Fed-HeLLo + FedSA 增强版)")
    print("="*80)
    print(f"配置信息:")
    print(f"  客户端数量: {args.num_clients}")
    print(f"  训练轮数: {args.num_rounds}")
    print(f"  方法: {args.method}")
    print(f"  资源异构模式: {args.heterogeneity_type} (基础比例: {args.allocation_ratio})")
    print(f"  LoRA 参数: Rank={args.lora_rank}, Alpha={args.lora_alpha}")
    print(f"  Non-IID 参数 (Alpha): {args.alpha}")
    print("="*80 + "\n")
    
    # 加载数据集
    print(f"正在加载 {args.dataset.upper()} 数据集...")
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
    
    # 获取全局测试集
    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)
    
    # 创建全局模型
    print("\n正在创建带有 LoRA 的 Vision Transformer...")
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
    
    # --- 关键修改：生成异构的客户端资源能力 ---
    print(f"正在生成客户端资源分布 ({args.heterogeneity_type})...")
    client_capacities = generate_client_resources(
        args.num_clients, 
        args.heterogeneity_type, 
        args.allocation_ratio
    )
    # 打印资源分布概览
    avg_cap = np.mean(client_capacities)
    print(f"  客户端层训练比例: 平均={avg_cap:.2f}, 最小={min(client_capacities):.2f}, 最大={max(client_capacities):.2f}")
    if args.num_clients <= 20:
        print(f"  详细分布: {['{:.2f}'.format(c) for c in client_capacities]}")

    # 初始化服务器
    print("正在初始化 Fed-HARP 服务器...")
    # 注意：这里我们假设 server.py 已经更新了 __init__ 来接收 client_capacities
    # 如果 server.py 尚未更新，它可能会忽略这个参数，但我们在 main 中显式生成它以体现逻辑。
    # 为了兼容旧版 server 代码，我们保留 allocation_ratio 参数，但期望 server 使用 client_capacities
    server = FedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        method=args.method,
        allocation_ratio=args.allocation_ratio, # 作为默认值
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed
    )
    # 手动将生成的异构能力注入到 server (模拟 Server 知晓客户端能力)
    if hasattr(server, 'client_layer_counts'):
        # 将比例转换为具体的层数
        num_total_layers = len(global_lora_layers)
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * cap))
        print("  成功将异构资源配置注入服务器。")
    
    # 初始化客户端
    clients = []
    client_sample_counts = {}
    
    print(f"\n正在初始化 {args.num_clients} 个客户端...")
    for client_id in range(args.num_clients):
        # 获取本地训练集
        train_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.batch_size,
            shuffle=True,
            split="train",
            fraction=args.data_fraction
        )
        # 获取本地测试集
        client_test_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.test_batch_size,
            shuffle=False,
            split="test",
            fraction=1.0
        )
        
        client_sample_counts[client_id] = len(train_loader.dataset)
        
        # 创建客户端模型副本
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
            print(f"  客户端 {client_id}: {len(train_loader.dataset)} 训练样本")
    
    if args.num_clients > 5:
        print(f"  ... (剩余 {args.num_clients - 5} 个客户端已初始化)")
    
    # 确定每轮参与的客户端
    clients_per_round = args.clients_per_round or args.num_clients
    
    # 开始联邦学习主循环
    print("\n" + "="*80)
    print("开始 Fed-HARP 联邦学习流程")
    print("="*80)
    
    best_accuracy = 0.0
    training_history = []

    method_l = str(args.method).lower()
    is_fedharp_family = method_l in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a"}
    b_train_every = int(args.b_train_every) if (is_fedharp_family and method_l != "fedharp_l") else 0
    
    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"第 {round_num}/{args.num_rounds} 轮")
        print(f"{'='*80}")
        
        # 步骤 1: 服务器生成分配映射
        allocation_map = server.start_round()
        server.print_allocation_stats(allocation_map)
        
        # 步骤 2: 选择客户端
        selected_clients = np.random.choice(
            args.num_clients,
            size=min(clients_per_round, args.num_clients),
            replace=False
        )
        
        print(f"本轮活跃客户端数: {len(selected_clients)}")
        
        # 步骤 3: 客户端接收全局 A 并执行本地训练
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
            
            # 本地训练
            if i < 3:
                print(f"  客户端 {client_id} 开始训练 (分配层数: {len(allocated_layers)})...")

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
                raise ValueError(f"{args.method} 需要同时上传 A+B，但 delta_B 为空")
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
            
            # 上传 B 矩阵敏感度
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
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")

        if len(client_metrics_ab) > 0:
            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))
            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))
            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))
            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))
            print(f"本轮客户端均值 (A+B): 计算={avg_compute:.4f} TFLOPs, 显存峰值={avg_mem:.4f} GB, 通信={avg_comm:.2f} MB/R, 时间={avg_time:.2f} s/R")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # 步骤 4: 服务器聚合更新
        print(f"\n服务器正在聚合来自 {len(client_updates)} 个客户端的 LoRA 更新...")
        server.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=(client_updates_B if upload_b_method else None),
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )

        # 步骤 5: 更新全局模型用于评估
        updated_global_A = server.get_global_A_matrices()
        for name, layer in global_lora_layers.items():
            if name in updated_global_A:
                layer.set_A(updated_global_A[name])

        if upload_b_method:
            updated_global_B = server.get_global_B_matrices()
            for name, layer in global_lora_layers.items():
                if name in updated_global_B:
                    layer.set_B(updated_global_B[name])
        
        # 步骤 5.5: 记录通信后的全局模型准确率 (已禁用，仅关注个性化)
        # print(f"\n正在评估通信后的全局模型准确率...")
        # global_model.eval()
        # comm_accuracy_results = evaluate_global_model(
        #     model=global_model,
        #     test_loader=global_test_loader,
        #     device=device
        # )
        # comm_accuracy = comm_accuracy_results['accuracy']
        # comm_loss = comm_accuracy_results['loss']
        # print(f"  [Round {round_num}] 通信后全局模型准确率: {comm_accuracy:.2f}%")
        # print(f"  [Round {round_num}] 通信后全局模型 Loss: {comm_loss:.4f}")

        # 追加写入通信后准确率到日志文件
        # try:
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(
        #             f"round={round_num}, type=post_communication, "
        #             f"accuracy={comm_accuracy:.4f}, loss={comm_loss:.6f}\n"
        #         )
        # except Exception as e:
        #     print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # 更新通信后的最佳准确率
        # if comm_accuracy > best_comm_accuracy:
        #     best_comm_accuracy = comm_accuracy
        #     print(f"  ✓ 通信后准确率创新高!")
        
        # 记录到训练历史中
        # training_history.append({
        #     'round': round_num,
        #     'accuracy': comm_accuracy,
        #     'loss': comm_loss,
        #     'type': 'post_communication'  # 标记为通信后准确率
        # })

        # 步骤 6: 评估 (仅关注个性化)
        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            print(f"\n正在各客户端本地 Test 集上评估 (个性化性能)...")
            
            total_weighted_correct = 0.0
            total_samples = 0
            total_weighted_loss = 0.0
            
            for cid in range(args.num_clients):
                client = clients[cid]
                # fedanon: Global A + Local B
                # 其他方法: Global A + Global B
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
                    print(f"  客户端 {cid}: Acc={acc:.2f}%, Loss={loss:.4f}")

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            print(f"\n  [Round {round_num}] 全局加权准确率 (个性化): {global_acc:.2f}%")
            print(f"  [Round {round_num}] 全局加权 Loss: {global_loss:.4f}")
            
            training_history.append({
                'round': round_num,
                'accuracy': global_acc,
                'loss': global_loss,
                'type': 'personalized'  # 标记为个性化评估
            })

            # 追加写入个性化评估准确率到日志文件
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=personalized, "
                        f"accuracy={global_acc:.4f}, loss={global_loss:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
            
            if global_acc > best_accuracy:
                best_accuracy = global_acc
                print(f"  ✓ 创新高! 新的最佳准确率")
        
        # 保存检查点
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
        print(f"\n第 {round_num} 轮耗时: {round_time:.2f}秒")
    
    # 最终总结
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    print(f"历史最佳准确率 (个性化): {best_accuracy:.2f}%")
    # print(f"历史最佳通信后准确率: {best_comm_accuracy:.2f}%")
    
    # 分别显示通信后和个性化的最终准确率
    # comm_history = [e for e in training_history if e.get('type') == 'post_communication']
    personalized_history = [e for e in training_history if e.get('type') == 'personalized']
    
    # if comm_history:
    #     print(f"最终轮次通信后准确率: {comm_history[-1]['accuracy']:.2f}%")
    if personalized_history:
        print(f"最终轮次个性化准确率: {personalized_history[-1]['accuracy']:.2f}%")
    
    if len(training_history) > 0:
        print("\n训练历史记录片段 (每10轮):")
        # print("  通信后准确率:")
        # for i, entry in enumerate(comm_history):
        #     if i % 10 == 0 or i == len(comm_history) - 1:
        #         print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")
        if personalized_history:
            print("  个性化准确率:")
            for i, entry in enumerate(personalized_history):
                if i % 10 == 0 or i == len(personalized_history) - 1:
                    print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")


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

    entry = argparse.ArgumentParser(description="统一 Fed-HARP 实验入口", add_help=True)
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
    raise SystemExit(f"未知 task: {task}")


if __name__ == "__main__":
    main()
