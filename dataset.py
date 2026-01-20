"""
CIFAR-10 dataset with Non-IID partitioning using Dirichlet distribution.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict
import os


class CIFAR10Dataset:
    """CIFAR-10 dataset handler with Non-IID partitioning"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        alpha: float = 0.5,
        num_clients: int = 10,
        seed: int = 42,
        splits: List[float] = None
    ):
        """
        Initialize CIFAR-10 dataset with Non-IID partitioning.
        
        Args:
            data_dir: Directory to store/download CIFAR-10
            alpha: Dirichlet distribution parameter (lower = more Non-IID)
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.alpha = alpha
        self.num_clients = num_clients
        self.seed = seed
        self.dataset_name = "cifar10"
        self.num_classes = 10
        # 本地 train / test 划分比例，例如 [0.9, 0.1]
        if splits is None:
            splits = [0.9, 0.1]
        if len(splits) < 2:
            raise ValueError("splits 必须至少包含两个值，例如 [0.9, 0.1]")
        splits = np.array(splits, dtype=float)
        if splits.sum() <= 0:
            raise ValueError("splits 之和必须大于 0")
        self.splits = (splits / splits.sum()).tolist()
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Download and load CIFAR-10
        self._load_datasets()
        
        # Partition data
        # 每个客户端内部再按 self.splits 划分出本地 train / test
        self.client_datasets = self._partition_non_iid()
    
    def _load_datasets(self):
        """Load CIFAR-10 train and test datasets"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            transforms.Resize(224)  # Resize for ViT
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            transforms.Resize(224)  # Resize for ViT
        ])
        
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        print(f"Loaded CIFAR-10: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    
    def _partition_non_iid(self) -> List[Dict[str, Subset]]:
        """
        Partition dataset using Dirichlet distribution for Non-IID data.
        先按 Dirichlet 做总体 Non-IID 划分，然后在每个客户端内部
        按 self.splits（如 [0.9, 0.1]）切分出本地 train / test。
        
        Returns:
            List of dicts, one per client: {'train': Subset, 'test': Subset}
        """
        num_classes = self.num_classes
        num_samples = len(self.train_dataset)
        
        if hasattr(self.train_dataset, "targets"):
            labels = np.asarray(self.train_dataset.targets)
        elif hasattr(self.train_dataset, "labels"):
            labels = np.asarray(self.train_dataset.labels)
        else:
            labels = np.array([self.train_dataset[i][1] for i in range(num_samples)])
        
        # Create Dirichlet distribution for each class
        # Each client gets a proportion of each class based on Dirichlet(alpha)
        client_indices = [[] for _ in range(self.num_clients)]
        
        for class_idx in range(num_classes):
            # Get indices for this class
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Allocate samples to clients based on proportions
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - sum(proportions[:-1])  # Ensure all samples allocated
            
            start_idx = 0
            for client_idx in range(self.num_clients):
                end_idx = start_idx + proportions[client_idx]
                client_indices[client_idx].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle each client's indices
        for client_idx in range(self.num_clients):
            np.random.shuffle(client_indices[client_idx])
        
        # 为每个客户端创建本地 train / test Subset
        client_datasets: List[Dict[str, Subset]] = []
        train_ratio = self.splits[0]
        for client_idx in range(self.num_clients):
            indices = client_indices[client_idx]
            num_client_samples = len(indices)
            split_point = int(num_client_samples * train_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            
            client_datasets.append({
                "train": Subset(self.train_dataset, train_indices),
                "test": Subset(self.train_dataset, test_indices)
            })
        
        # Print statistics（基于整体客户端数据分布）
        self._print_partition_stats(labels, client_indices)
        
        return client_datasets
    
    def _print_partition_stats(self, labels: np.ndarray, client_indices: List[List[int]]):
        """Print statistics about the data partition"""
        print("\n" + "="*60)
        print(f"Non-IID Partition Statistics ({self.dataset_name}, Dirichlet α={self.alpha})")
        print("="*60)
        
        for client_idx in range(self.num_clients):
            client_labels = labels[client_indices[client_idx]]
            unique, counts = np.unique(client_labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            
            total = len(client_labels)
            dist_str = ", ".join([f"Class {k}: {v} ({100*v/total:.1f}%)" 
                                 for k, v in sorted(label_dist.items())])
            
            print(f"Client {client_idx}: {total} samples")
            print(f"  Distribution: {dist_str}")
        
        print("="*60 + "\n")

    def get_client_dataset(self, client_idx: int, split: str = "train") -> Subset:
        """Get dataset for a specific client and split ('train' or 'test')"""
        if client_idx < 0 or client_idx >= self.num_clients:
            raise ValueError(f"Client index must be in [0, {self.num_clients})")
        if split not in self.client_datasets[client_idx]:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.client_datasets[client_idx].keys())}")
        return self.client_datasets[client_idx][split]

    def get_test_dataset(self) -> Dataset:
        """Get test dataset"""
        return self.test_dataset

    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = "train",
        fraction: float = 1.0
    ) -> DataLoader:
        """Get DataLoader for a specific client and split
        
        Args:
            client_idx: 客户端索引
            batch_size: 批大小
            shuffle: 是否打乱
            split: 使用的本地划分 ('train' 或 'test')
            fraction: 使用的数据比例 (0-1), 例如 0.01 表示使用 1% 的该客户端数据
        """
        base_dataset = self.get_client_dataset(client_idx, split=split)
        
        if 0.0 < fraction < 1.0:
            num_samples = len(base_dataset)
            subset_size = max(1, int(num_samples * fraction))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            subset_indices = all_indices[:subset_size]
            dataset = Subset(base_dataset, subset_indices)
        else:
            dataset = base_dataset
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    def get_test_dataloader(self, batch_size: int = 100) -> DataLoader:
        """Get DataLoader for test dataset"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


class CIFAR100Dataset(CIFAR10Dataset):
    """CIFAR-100 dataset handler with Non-IID partitioning (Dirichlet)"""

    def __init__(
        self,
        data_dir: str = "./data",
        alpha: float = 0.5,
        num_clients: int = 10,
        seed: int = 42,
        splits: List[float] = None
    ):
        super().__init__(
            data_dir=data_dir,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            splits=splits
        )

    def _load_datasets(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.Resize(224)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.Resize(224)
        ])

        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        self.dataset_name = "cifar100"
        self.num_classes = 100

        print(f"Loaded CIFAR-100: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    
    def get_client_dataset(self, client_idx: int, split: str = "train") -> Subset:
        """Get dataset for a specific client and split ('train' or 'test')"""
        if client_idx < 0 or client_idx >= self.num_clients:
            raise ValueError(f"Client index must be in [0, {self.num_clients})")
        if split not in self.client_datasets[client_idx]:
            raise ValueError(f"Unknown split '{split}', expected one of {list(self.client_datasets[client_idx].keys())}")
        return self.client_datasets[client_idx][split]
    
    def get_test_dataset(self) -> Dataset:
        """Get test dataset"""
        return self.test_dataset
    
    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 32,
        shuffle: bool = True,
        split: str = "train",
        fraction: float = 1.0
    ) -> DataLoader:
        """Get DataLoader for a specific client and split
        
        Args:
            client_idx: 客户端索引
            batch_size: 批大小
            shuffle: 是否打乱
            split: 使用的本地划分 ('train' 或 'test')
            fraction: 使用的数据比例 (0-1), 例如 0.01 表示使用 1% 的该客户端数据
        """
        base_dataset = self.get_client_dataset(client_idx, split=split)
        
        # 如果只使用部分数据，随机采样一个子集
        if 0.0 < fraction < 1.0:
            num_samples = len(base_dataset)
            subset_size = max(1, int(num_samples * fraction))
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            subset_indices = all_indices[:subset_size]
            dataset = Subset(base_dataset, subset_indices)
        else:
            dataset = base_dataset
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
    
    def get_test_dataloader(self, batch_size: int = 100) -> DataLoader:
        """Get DataLoader for test dataset"""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """Get class distribution in a dataset"""
    if isinstance(dataset, Subset):
        labels = [dataset.dataset[dataset.indices[i]][1] for i in range(len(dataset))]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
