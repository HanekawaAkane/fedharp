"""
Fed-HARP federated training entry point for RoBERTa + GLUE.
"""

import argparse
import csv
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency 'torch'. Run in fa_gpu or install requirements.") from e

from client import FedHarpClient
from server import FedHarpServer
from utils import create_directories, get_device, set_seed


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Fed-ANON on GLUE with RoBERTa + LoRA")

    p.add_argument(
        "--method",
        type=str,
        default="fedharp",
        choices=[
            "fedharp",
            "fedharp_r",
            "fedharp_g",
            "fedharp_l",
            "fedharp_a",
            "fedhello",
            "fedra",
            "flora",
            "fedsalora",
            "fedsa-lora",
            "fedsa_lora",
            "FedSA-LoRA",
        ],
    )
    p.add_argument("--model_path", type=str, default="./Roberta")
    p.add_argument("--hf_endpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "qnli"])
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--sst2_path", type=str, default=None)
    p.add_argument("--datasets_cache_dir", type=str, default=None)

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--client_splits", type=str, default="0.9,0.1")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--clients_per_round", type=int, default=None)
    p.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"])
    p.add_argument("--allocation_ratio", type=float, default=0.5)
    p.add_argument("--aggregation_lr", type=float, default=1.0)
    p.add_argument("--struct_to_data_rounds", type=int, default=20)

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default="query,value")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--b_train_every", type=int, default=25)
    p.add_argument("--b_num_epochs", type=int, default=1)
    p.add_argument("--client_lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_on", type=str, default="val", choices=["val", "client_holdout"])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    return p.parse_args(args=argv)


def _read_sst2_tsv(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = row.get("sentence")
            y = row.get("label")
            if s is None or y is None:
                continue
            try:
                labels.append(int(y))
            except Exception:
                continue
            texts.append(str(s))
    return texts, labels


def _read_sst2_parquet(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_parquet(path)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Parquet file {path} must contain 'sentence' and 'label' columns. Found: {df.columns}")
    
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


TextInput = Union[str, Tuple[str, str]]

def _read_qnli_tsv(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    texts: List[Tuple[str, str]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            q = row.get("question")
            s = row.get("sentence")
            y = row.get("label")
            if q is None or s is None or y is None:
                continue
            y_str = str(y).strip()
            if y_str.lower() == "entailment":
                yy = 0
            elif y_str.lower() == "not_entailment":
                yy = 1
            else:
                try:
                    yy = int(y_str)
                except Exception:
                    continue
            texts.append((str(q), str(s)))
            labels.append(int(yy))
    return texts, labels


def _read_qnli_parquet(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    df = pd.read_parquet(path)
    if "question" not in df.columns or "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Parquet file {path} must contain 'question', 'sentence' and 'label' columns. Found: {df.columns}"
        )
    texts = list(zip(df["question"].astype(str).tolist(), df["sentence"].astype(str).tolist()))
    labels = df["label"].astype(int).tolist()
    return texts, labels


def _read_qnli_jsonl(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:
    texts: List[Tuple[str, str]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue

            if "text1" in obj and "text2" in obj:
                q = obj.get("text1")
                sent = obj.get("text2")
            else:
                q = obj.get("question")
                sent = obj.get("sentence")

            y = obj.get("label")
            if q is None or sent is None or y is None:
                continue

            yy: Optional[int]
            if isinstance(y, str):
                y_str = y.strip().lower().replace(" ", "_")
                if y_str == "entailment":
                    yy = 0
                elif y_str in {"not_entailment", "not_entail"}:
                    yy = 1
                else:
                    try:
                        yy = int(y_str)
                    except Exception:
                        yy = None
            else:
                try:
                    yy = int(y)
                except Exception:
                    yy = None

            if yy is None or int(yy) < 0:
                continue

            texts.append((str(q), str(sent)))
            labels.append(int(yy))
    return texts, labels


def _candidate_dataset_dir(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


def _load_dataset(args) -> Tuple[List[TextInput], List[int], List[TextInput], List[int]]:
    dataset = str(args.dataset).lower()
    candidate_dir = _candidate_dataset_dir(
        args.data_path,
        args.sst2_path if dataset == "sst2" else None,
        "./data/sst-2" if dataset == "sst2" else None,
        "./data/sst2" if dataset == "sst2" else None,
        "./data/SST-2" if dataset == "sst2" else None,
        "./data/qnli" if dataset == "qnli" else None,
        "./data/QNLI" if dataset == "qnli" else None,
        "./QNLI" if dataset == "qnli" else None,
        "./qnli" if dataset == "qnli" else None,
    )

    if candidate_dir is not None:
        train_parquet = os.path.join(candidate_dir, "train-00000-of-00001.parquet")
        val_parquet = os.path.join(candidate_dir, "validation-00000-of-00001.parquet")
        if os.path.isfile(train_parquet) and os.path.isfile(val_parquet):
            if dataset == "sst2":
                print(f"Loading SST-2 from Parquet: {candidate_dir}")
                train_texts, train_labels = _read_sst2_parquet(train_parquet)
                val_texts, val_labels = _read_sst2_parquet(val_parquet)
                return train_texts, train_labels, val_texts, val_labels
            if dataset == "qnli":
                print(f"Loading QNLI from Parquet: {candidate_dir}")
                train_texts, train_labels = _read_qnli_parquet(train_parquet)
                val_texts, val_labels = _read_qnli_parquet(val_parquet)
                return train_texts, train_labels, val_texts, val_labels

        train_tsv = os.path.join(candidate_dir, "train.tsv")
        dev_tsv = os.path.join(candidate_dir, "dev.tsv")
        if not os.path.isfile(dev_tsv):
            dev_tsv = os.path.join(candidate_dir, "validation.tsv")
        if os.path.isfile(train_tsv) and os.path.isfile(dev_tsv):
            if dataset == "sst2":
                print(f"Loading SST-2 from TSV: {candidate_dir}")
                train_texts, train_labels = _read_sst2_tsv(train_tsv)
                val_texts, val_labels = _read_sst2_tsv(dev_tsv)
                return train_texts, train_labels, val_texts, val_labels
            if dataset == "qnli":
                print(f"Loading QNLI from TSV: {candidate_dir}")
                train_texts, train_labels = _read_qnli_tsv(train_tsv)
                val_texts, val_labels = _read_qnli_tsv(dev_tsv)
                return train_texts, train_labels, val_texts, val_labels

        train_jsonl = os.path.join(candidate_dir, "train.jsonl")
        dev_jsonl = os.path.join(candidate_dir, "dev.jsonl")
        if not os.path.isfile(dev_jsonl):
            dev_jsonl = os.path.join(candidate_dir, "validation.jsonl")
        if dataset == "qnli" and os.path.isfile(train_jsonl) and os.path.isfile(dev_jsonl):
            print(f"Loading QNLI from JSONL: {candidate_dir}")
            train_texts, train_labels = _read_qnli_jsonl(train_jsonl)
            val_texts, val_labels = _read_qnli_jsonl(dev_jsonl)
            return train_texts, train_labels, val_texts, val_labels

        if args.data_path is not None or (args.sst2_path is not None and dataset == "sst2"):
            raise SystemExit(f"Invalid dataset path: {candidate_dir}")

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("Missing dependency 'datasets'. Install with: pip install datasets") from e

    try:
        raw = load_dataset("glue", dataset, cache_dir=args.datasets_cache_dir)
    except Exception as e:
        raise SystemExit(
            f"Cannot load {dataset} from the Hub. Provide --data_path, or set --hf_endpoint / proxy."
        ) from e

    train_split = raw["train"]
    val_split = raw["validation"]
    if dataset == "sst2":
        train_texts = list(train_split["sentence"])
        train_labels = [int(x) for x in train_split["label"]]
        val_texts = list(val_split["sentence"])
        val_labels = [int(x) for x in val_split["label"]]
        return train_texts, train_labels, val_texts, val_labels
    if dataset == "qnli":
        train_texts = list(zip(train_split["question"], train_split["sentence"]))
        train_labels = [int(x) for x in train_split["label"]]
        val_texts = list(zip(val_split["question"], val_split["sentence"]))
        val_labels = [int(x) for x in val_split["label"]]
        return train_texts, train_labels, val_texts, val_labels
    raise SystemExit(f"Unsupported dataset: {dataset}")


def _safe_token(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _build_log_path(args) -> str:
    safe_method = _safe_token(args.method)
    safe_alpha = _safe_token(args.alpha)
    safe_dataset = _safe_token(args.dataset)
    safe_model = _safe_token(os.path.basename(os.path.abspath(args.model_path)) or "roberta")
    safe_heterogeneity = _safe_token(args.heterogeneity_type)
    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity}.txt"
    return os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    if heterogeneity_type == "6-3-1":
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios: List[float] = []
        ratios.extend([base_ratio] * n_low)
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid)
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    if heterogeneity_type == "1-1-1":
        n_low = int(num_clients // 3)
        n_mid = int(num_clients // 3)
        n_high = int(num_clients) - int(n_low) - int(n_mid)
        ratios: List[float] = []
        ratios.extend([base_ratio] * int(n_low))
        ratios.extend([min(1.0, base_ratio * 1.5)] * int(n_mid))
        ratios.extend([1.0] * int(n_high))
        np.random.shuffle(ratios)
        return ratios
    return [base_ratio] * num_clients


def _build_fedra_allocation_map(
    *,
    layer_names: List[str],
    selected_clients: np.ndarray,
    client_max_layers: Dict[int, int],
    seed: int,
    num_clients: int,
) -> Dict[int, Set[str]]:
    rng = np.random.RandomState(int(seed))
    suffixes = (".query", ".value", ".q_proj", ".v_proj")
    group_to_layers: Dict[str, List[str]] = {}
    for name in layer_names:
        group_id = name
        for suf in suffixes:
            if name.endswith(suf):
                group_id = name[: -len(suf)]
                break
        group_to_layers.setdefault(group_id, []).append(name)
    group_ids = sorted(group_to_layers.keys())

    allocation_map: Dict[int, Set[str]] = {cid: set() for cid in range(int(num_clients))}
    for cid in selected_clients:
        cid_int = int(cid)
        budget = int(client_max_layers.get(cid_int, max(1, len(layer_names))))
        budget = max(1, min(budget, len(layer_names)))
        target_budget = int(rng.randint(1, budget + 1))

        perm = rng.permutation(group_ids)
        chosen_groups: List[str] = []
        used = 0
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

        if len(chosen_groups) == 0:
            smallest = min(group_ids, key=lambda g: len(group_to_layers.get(str(g), [])))
            chosen_groups = [str(smallest)]

        allocated: Set[str] = set()
        for gid in chosen_groups:
            allocated.update(group_to_layers.get(str(gid), []))
        allocation_map[cid_int] = allocated
    return allocation_map


def _unfreeze_classifier_params(model: nn.Module):
    for attr in ("classifier", "score"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Module):
                for p in head.parameters():
                    p.requires_grad = True


def _freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default") -> Dict[str, nn.Module]:
    class _Adapter:
        def __init__(self, peft_linear_module: nn.Module, adapter: str):
            self._m = peft_linear_module
            self._adapter = adapter
            self.init_rank = int(self.lora_A.shape[0])

        @property
        def lora_A(self) -> torch.nn.Parameter:
            return self._m.lora_A[self._adapter].weight

        @property
        def lora_B(self) -> torch.nn.Parameter:
            return self._m.lora_B[self._adapter].weight

        def get_A(self) -> torch.Tensor:
            return self.lora_A.data.clone()

        def set_A(self, A: torch.Tensor):
            A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype).clone()
            if tuple(A.shape) != tuple(self.lora_A.shape):
                old_requires_grad = bool(self.lora_A.requires_grad)
                in_features = int(A.shape[1])
                out_features = int(A.shape[0])
                new_A = nn.Linear(in_features, out_features, bias=False).to(device=A.device, dtype=A.dtype)
                new_A.weight.data.copy_(A)
                new_A.weight.requires_grad = old_requires_grad
                self._m.lora_A[self._adapter] = new_A
                self._update_peft_rank_meta(rank=out_features)
            else:
                self.lora_A.data = A

        def get_B(self) -> torch.Tensor:
            return self.lora_B.data.clone()

        def set_B(self, B: torch.Tensor):
            B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype).clone()
            if tuple(B.shape) != tuple(self.lora_B.shape):
                old_requires_grad = bool(self.lora_B.requires_grad)
                in_features = int(B.shape[1])
                out_features = int(B.shape[0])
                new_B = nn.Linear(in_features, out_features, bias=False).to(device=B.device, dtype=B.dtype)
                new_B.weight.data.copy_(B)
                new_B.weight.requires_grad = old_requires_grad
                self._m.lora_B[self._adapter] = new_B
                self._update_peft_rank_meta(rank=in_features)
            else:
                self.lora_B.data = B

        def freeze_A(self):
            self.lora_A.requires_grad = False

        def unfreeze_A(self):
            self.lora_A.requires_grad = True

        def freeze_B(self):
            self.lora_B.requires_grad = False

        def unfreeze_B(self):
            self.lora_B.requires_grad = True

        def freeze_all(self):
            self.freeze_A()
            self.freeze_B()

        def unfreeze_all(self):
            self.unfreeze_A()
            self.unfreeze_B()

        def get_trainable_params(self) -> List[torch.nn.Parameter]:
            params: List[torch.nn.Parameter] = []
            if self.lora_A.requires_grad:
                params.append(self.lora_A)
            if self.lora_B.requires_grad:
                params.append(self.lora_B)
            return params

        def _get_base_weight(self) -> torch.nn.Parameter:
            base_layer = getattr(self._m, "base_layer", None)
            if base_layer is not None and hasattr(base_layer, "weight"):
                return base_layer.weight
            if hasattr(self._m, "weight"):
                return getattr(self._m, "weight")
            raise ValueError(f"Unsupported PEFT LoRA module type: {type(self._m)}")

        def _get_scaling(self) -> float:
            scaling = getattr(self._m, "scaling", None)
            if isinstance(scaling, dict) and self._adapter in scaling:
                try:
                    return float(scaling[self._adapter])
                except Exception:
                    pass
            lora_alpha = getattr(self._m, "lora_alpha", None)
            if isinstance(lora_alpha, dict) and self._adapter in lora_alpha:
                try:
                    alpha = float(lora_alpha[self._adapter])
                    r = int(self.lora_A.shape[0])
                    return alpha / float(r) if r > 0 else 0.0
                except Exception:
                    pass
            return 1.0

        def _update_peft_rank_meta(self, rank: int):
            rank = int(rank)
            r_map = getattr(self._m, "r", None)
            if isinstance(r_map, dict):
                r_map[self._adapter] = rank
            scaling_map = getattr(self._m, "scaling", None)
            lora_alpha = getattr(self._m, "lora_alpha", None)
            if isinstance(scaling_map, dict) and isinstance(lora_alpha, dict) and self._adapter in lora_alpha:
                try:
                    scaling_map[self._adapter] = float(lora_alpha[self._adapter]) / float(rank) if rank > 0 else 0.0
                except Exception:
                    pass

        def reset_lora(self, rank: Optional[int] = None):
            target_rank = int(rank if rank is not None else self.init_rank)
            A_old = self.lora_A
            B_old = self.lora_B
            A_requires_grad = bool(A_old.requires_grad)
            B_requires_grad = bool(B_old.requires_grad)
            device = A_old.device
            dtype = A_old.dtype

            in_features = int(A_old.shape[1])
            out_features = int(B_old.shape[0])

            new_A = nn.Linear(in_features, target_rank, bias=False).to(device=device, dtype=dtype)
            new_B = nn.Linear(target_rank, out_features, bias=False).to(device=device, dtype=dtype)
            new_A.weight.data.normal_(mean=0.0, std=0.02)
            new_B.weight.data.zero_()
            new_A.weight.requires_grad = A_requires_grad
            new_B.weight.requires_grad = B_requires_grad

            self._m.lora_A[self._adapter] = new_A
            self._m.lora_B[self._adapter] = new_B
            self._update_peft_rank_meta(rank=target_rank)

        def merge_lora_weights(self):
            r = int(self.lora_A.shape[0])
            if r <= 0:
                return
            scaling = self._get_scaling()
            delta_w = (self.lora_B @ self.lora_A) * float(scaling)
            base_w = self._get_base_weight()
            with torch.no_grad():
                base_w.data = base_w.data + delta_w.to(device=base_w.device, dtype=base_w.dtype)

        def merge_lora_into_base_and_reset(self, rank: Optional[int] = None):
            self.merge_lora_weights()
            self.reset_lora(rank=rank)

    lora_layers: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        try:
            _ = module.lora_A[adapter_name].weight
            _ = module.lora_B[adapter_name].weight
        except Exception:
            continue
        lora_layers[name] = _Adapter(module, adapter=adapter_name)
    return lora_layers


def create_roberta_lora_model(
    *,
    model_path: str,
    num_labels: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: Sequence[str],
):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import transformers.utils.import_utils
        import transformers.modeling_utils
        transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception as e:
        raise SystemExit("Missing dependency 'transformers'. Install requirements.") from e
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise SystemExit("Missing dependency 'peft'. Install requirements.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=int(num_labels), 
            weights_only=False,
            low_cpu_mem_usage=False
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=int(num_labels))

    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    lora_layers = _extract_peft_lora_layers(model, adapter_name="default")

    _freeze_all_params(model)
    _unfreeze_classifier_params(model)
    for layer in lora_layers.values():
        layer.unfreeze_all()

    return model, tokenizer, lora_layers


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
):
    rng = np.random.RandomState(int(seed))
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    labels = labels[indices]

    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]
    for cls in range(num_classes):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([float(alpha)] * int(num_clients))
        counts = (proportions * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - int(counts[:-1].sum())
        start = 0
        for cid in range(int(num_clients)):
            end = start + int(counts[cid])
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end
    for cid in range(int(num_clients)):
        rng.shuffle(client_indices[cid])
    return client_indices


@dataclass
class EncodedText:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def _encode_texts(tokenizer, texts: Sequence[TextInput], labels: Sequence[int], max_length: int) -> EncodedText:
    texts_list: List[TextInput] = list(texts)
    if len(texts_list) > 0 and isinstance(texts_list[0], tuple) and len(texts_list[0]) == 2:
        a: List[str] = []
        b: List[str] = []
        for t in texts_list:
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError("Mixed text inputs: expected (text_a, text_b) pairs")
            a.append(str(t[0]))
            b.append(str(t[1]))
        enc = tokenizer(
            a,
            b,
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            [str(t) for t in texts_list],
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    y = torch.tensor(list(labels), dtype=torch.long)
    return EncodedText(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=y)


class SST2ClientDataset(Dataset):
    def __init__(self, encoded: EncodedText, indices: Sequence[int]):
        idx = torch.tensor(list(indices), dtype=torch.long)
        self.input_ids = encoded.input_ids.index_select(0, idx)
        self.attention_mask = encoded.attention_mask.index_select(0, idx)
        self.labels = encoded.labels.index_select(0, idx)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}, self.labels[i]


class TextFedHarpClient(FedHarpClient):
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

        for _, (inputs, target) in enumerate(self.train_loader):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(**inputs)
            logits = getattr(out, "logits", out)
            loss = criterion(logits, target)
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

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return total_loss / num_batches, grad_sensitivity

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
            for inputs, target in test_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                target = target.to(self.device)
                out = self.model(**inputs)
                logits = getattr(out, "logits", out)
                test_loss += float(criterion(logits, target).item())
                pred = logits.argmax(dim=1)
                correct += int(pred.eq(target).sum().item())
                total += int(target.size(0))

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}


class RobertaFedHarpServer(FedHarpServer):
    _layer_re = re.compile(r"(?:encoder\\.)?layer\\.(\\d+)")

    def _get_layer_depth(self, layer_name: str) -> int:
        m = self._layer_re.search(layer_name)
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, v in d.items():
        total += _tensor_nbytes(v)
    return total


def _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(A_matrices)


def _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_A)

def _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_B)


def _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:
    n = 0
    for name in allocated_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_A.numel())
        n += int(layer.lora_B.numel())
    return n


def _clone_model_and_layers(template_model: nn.Module, template_lora_layers: Dict[str, nn.Module]):
    import copy

    m = copy.deepcopy(template_model)
    layers = _extract_peft_lora_layers(m, adapter_name="default")
    for name, layer in layers.items():
        src = template_lora_layers.get(name)
        if src is None:
            continue
        layer.set_A(src.get_A())
        layer.set_B(src.get_B())
        layer.unfreeze_all()
    _freeze_all_params(m)
    _unfreeze_classifier_params(m)
    return m, layers


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    set_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        dev = str(args.device)
        if (":" not in dev) or dev.endswith(":0") or dev == "cuda":
            print(f"Detected {torch.cuda.device_count()} GPUs. Use --device cuda:1 to select the second GPU.")

    if args.save_checkpoints:
        create_directories(args.checkpoint_dir)

    log_path = _build_log_path(args)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[WARNING] Cannot write log file {log_path}: {e}")

    train_texts_all, train_labels_all, val_texts_all, val_labels_all = _load_dataset(args)

    rng = np.random.RandomState(int(args.seed))
    train_perm = rng.permutation(len(train_texts_all))
    if args.max_train_samples is not None:
        train_perm = train_perm[: int(args.max_train_samples)]
    train_texts = [train_texts_all[int(i)] for i in train_perm.tolist()]
    train_labels = np.asarray([int(train_labels_all[int(i)]) for i in train_perm.tolist()], dtype=np.int64)

    val_perm = rng.permutation(len(val_texts_all))
    if args.max_eval_samples is not None:
        val_perm = val_perm[: int(args.max_eval_samples)]
    val_texts = [val_texts_all[int(i)] for i in val_perm.tolist()]
    val_labels = np.asarray([int(val_labels_all[int(i)]) for i in val_perm.tolist()], dtype=np.int64)

    lora_targets = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]
    global_model, tokenizer, global_lora_layers = create_roberta_lora_model(
        model_path=args.model_path,
        num_labels=2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
    )
    global_model.to(device)
    for layer in global_lora_layers.values():
        layer.unfreeze_all()

    encoded_train = _encode_texts(tokenizer, train_texts, train_labels.tolist(), max_length=args.max_length)
    encoded_val = _encode_texts(tokenizer, val_texts, val_labels.tolist(), max_length=args.max_length)

    client_indices = _dirichlet_partition(
        train_labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    if int(len(train_labels)) < int(args.num_clients):
        raise SystemExit(f"num_clients={args.num_clients} exceeds training samples={len(train_labels)}, cannot partition for every client.")

    empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]
    if empty_clients:
        donors = sorted(range(int(args.num_clients)), key=lambda c: len(client_indices[c]), reverse=True)
        for cid in empty_clients:
            for d in donors:
                if len(client_indices[d]) > 1:
                    client_indices[cid].append(client_indices[d].pop())
                    break

        empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]
        for cid in empty_clients:
            donor = max(range(int(args.num_clients)), key=lambda c: len(client_indices[c]))
            if len(client_indices[donor]) == 0:
                break
            client_indices[cid].append(client_indices[donor].pop())

    try:
        split_ratios = [float(x) for x in str(args.client_splits).split(",")]
    except Exception:
        split_ratios = [0.9, 0.1]
    train_ratio = float(split_ratios[0]) / max(1e-12, float(sum(split_ratios)))

    client_train_indices: List[List[int]] = []
    client_test_indices: List[List[int]] = []
    for cid in range(args.num_clients):
        idxs = client_indices[cid]
        if 0.0 < args.data_fraction < 1.0:
            k = max(1, int(len(idxs) * float(args.data_fraction)))
            idxs = idxs[:k]
        split_point = int(len(idxs) * train_ratio)
        if split_point <= 0:
            split_point = 1
        if split_point > len(idxs):
            split_point = len(idxs)
        client_train_indices.append(idxs[:split_point])
        client_test_indices.append(idxs[split_point:])

    client_capacities = generate_client_resources(args.num_clients, args.heterogeneity_type, args.allocation_ratio)
    num_total_layers = len(global_lora_layers)

    method = str(args.method).lower()
    server = RobertaFedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        allocation_ratio=args.allocation_ratio,
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed,
        method=args.method,
    )
    if hasattr(server, "client_layer_counts"):
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(cap)))

    clients: List[TextFedHarpClient] = []
    client_sample_counts: Dict[int, int] = {}
    eval_on = str(getattr(args, "eval_on", "val")).lower().strip()
    if eval_on == "client_holdout":
        print("Eval split: client_holdout (Internal split from training set, not GLUE validation)")
    else:
        print("Eval split: val (GLUE validation)")
    for cid in range(args.num_clients):
        train_ds = SST2ClientDataset(encoded_train, client_train_indices[cid])
        if eval_on == "client_holdout":
            test_ds = SST2ClientDataset(encoded_train, client_test_indices[cid])
            if len(test_ds) == 0:
                test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))
        else:
            test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)
        client_model.to(device)
        client_sample_counts[cid] = len(train_ds)

        clients.append(
            TextFedHarpClient(
                client_id=cid,
                model=client_model,
                lora_layers=client_lora_layers,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                lr=args.client_lr,
                method=args.method,
                warmup_steps=0,
                warmup_lr=args.client_lr,
            )
        )

    clients_per_round = args.clients_per_round or args.num_clients
    best_accuracy = 0.0
    layer_names = list(server.layer_names)
    if hasattr(server, "client_layer_counts"):
        client_max_layers = {cid: int(server.client_layer_counts.get(cid, len(layer_names))) for cid in range(args.num_clients)}
    else:
        client_max_layers = {cid: len(layer_names) for cid in range(args.num_clients)}

    is_fedharp_family = method in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a", "fedanon"}
    b_train_every = int(args.b_train_every) if (is_fedharp_family and method != "fedharp_l") else 0

    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()

        selected_clients = np.random.choice(
            args.num_clients, size=min(clients_per_round, args.num_clients), replace=False
        )
        needs_b = method in {"fedhello", "fedra", "flora"}
        if method == "fedra":
            server.current_round = int(round_num)
            allocation_map = _build_fedra_allocation_map(
                layer_names=layer_names,
                selected_clients=selected_clients,
                client_max_layers=client_max_layers,
                seed=int(args.seed) + int(round_num) * 10007,
                num_clients=args.num_clients,
            )
        elif method == "flora":
            allocation_map = {cid: set(layer_names) for cid in range(int(args.num_clients))}
        else:
            allocation_map = server.start_round()

        global_A = server.get_global_A_matrices()
        global_B = server.get_global_B_matrices() if needs_b else None
        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)
        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0

        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}
        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}
        client_metrics_ab = []

        for client_id in selected_clients:
            cid = int(client_id)
            client = clients[cid]
            allocated_layers = allocation_map[cid]

            client.receive_global_matrices(global_A, global_B, method=args.method)
            if method == "flora":
                client.flora_merge_and_reset(target_rank=args.lora_rank)

            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l":
                stale_map = server.staleness.get(int(cid), {})
                b_only_layers = {name for name, tau in stale_map.items() if int(tau) == int(b_train_every)}
                if len(b_only_layers) > 0:
                    new_b_sensitivity = client.train_B_only(b_only_layers, num_epochs=int(args.b_num_epochs))
                    server.update_client_sensitivities(int(cid), new_b_sensitivity)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            delta_A, delta_B, b_sensitivity = client.local_train(
                allocated_layers=allocated_layers,
                num_epochs=args.num_epochs,
                method=args.method,
            )
            train_time_sec = time.time() - t0
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            else:
                peak_mem_gb = 0.0

            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)
            uplink_B_mb = 0.0
            if needs_b:
                if delta_B is None:
                    raise ValueError(f"{method} requires uploading B")
                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)
            comm_mb = downlink_A_mb + downlink_B_mb + uplink_A_mb + uplink_B_mb

            train_samples = len(client.train_loader.dataset)
            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)
            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12

            client_metrics_ab.append(
                {
                    "round": round_num,
                    "client": int(cid),
                    "phase": "ab_train",
                    "compute_tflops": float(compute_tflops),
                    "peak_mem_gb": float(peak_mem_gb),
                    "comm_mb": float(comm_mb),
                    "train_time_sec": float(train_time_sec),
                    "downlink_mb": float(downlink_A_mb + downlink_B_mb),
                    "uplink_mb": float(uplink_A_mb + uplink_B_mb),
                }
            )

            client_updates[cid] = delta_A
            if needs_b:
                client_updates_B[cid] = delta_B
            server.update_client_sensitivities(cid, b_sensitivity)

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
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\n"
                    )
            except Exception as e:
                print(f"[WARNING] Cannot write log file {log_path}: {e}")

        if needs_b:
            server.aggregate_matrices(
                client_updates_A=client_updates,
                client_updates_B=client_updates_B,
                client_sample_counts=client_sample_counts,
                allocation_map=allocation_map,
            )
        else:
            server.aggregate_A_matrices(
                client_updates=client_updates,
                client_sample_counts=client_sample_counts,
                allocation_map=allocation_map,
            )

        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            total_weighted_correct = 0.0
            total_weighted_loss = 0.0
            total_samples = 0
            eval_A = server.get_global_A_matrices()
            eval_B = server.get_global_B_matrices() if needs_b else None
            for cid in range(args.num_clients):
                client = clients[cid]
                client.receive_global_matrices(eval_A, eval_B, method=args.method)
                if client.test_loader is None or len(client.test_loader.dataset) == 0:
                    continue
                results = client.evaluate()
                acc = float(results["accuracy"])
                loss = float(results["loss"])
                n = len(client.test_loader.dataset)
                total_weighted_correct += (acc / 100.0) * n
                total_weighted_loss += loss * n
                total_samples += n

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"round={round_num}, type=personalized, accuracy={global_acc:.4f}, loss={global_loss:.6f}\n")
            except Exception as e:
                print(f"[WARNING] Cannot write log file {log_path}: {e}")

            if global_acc > best_accuracy:
                best_accuracy = global_acc

        _ = time.time() - round_start_time

    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")
    print(f"Log saved to: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()

