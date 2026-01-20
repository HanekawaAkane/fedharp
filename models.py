"""
Custom LoRA implementation for Fed-Anon, adapted for Vision Transformer (ViT).
The core concept is the separate management of Matrix A (Shared) and Matrix B (Local/Private).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Optional, List, Dict

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not found, using torchvision's ViT implementation instead")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _unfreeze_classifier_params(model: nn.Module):
    classifier = None
    if hasattr(model, "get_classifier"):
        try:
            classifier = model.get_classifier()
        except Exception:
            classifier = None
    if classifier is None:
        for attr in ("head", "classifier", "fc"):
            if hasattr(model, attr):
                classifier = getattr(model, attr)
                break
    if classifier is None and hasattr(model, "heads"):
        heads = getattr(model, "heads")
        if hasattr(heads, "head"):
            classifier = heads.head
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True


def _select_checkpoint_file(snapshot_dir: str, checkpoint_file: Optional[str]) -> str:
    if checkpoint_file:
        ckpt = os.path.join(snapshot_dir, checkpoint_file)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
        return ckpt

    candidates: List[str] = []
    for root, _, files in os.walk(snapshot_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith((".safetensors", ".pth", ".pt", ".bin")):
                candidates.append(os.path.join(root, fn))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No available checkpoint found in ModelScope snapshot directory: {snapshot_dir}")

    candidates.sort()
    safetensors = [p for p in candidates if p.lower().endswith(".safetensors")]
    if len(safetensors) == 1:
        return safetensors[0]
    if len(candidates) == 1:
        return candidates[0]

    msg = "Multiple checkpoint candidates found in ModelScope snapshot directory, please specify one using --ms_checkpoint_file:\n"
    msg += "\n".join([os.path.relpath(p, snapshot_dir) for p in candidates[:50]])
    if len(candidates) > 50:
        msg += f"\n... (Total {len(candidates)} candidates)"
    raise ValueError(msg)


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    low = checkpoint_path.lower()
    if low.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(f"Loading .safetensors requires safetensors installed: {e}") from e
        state = load_file(checkpoint_path, device="cpu")
        return dict(state)

    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "params"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unable to parse state_dict from checkpoint: {checkpoint_path}")
    return obj


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if len(state_dict) == 0:
        return state_dict
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def _looks_like_transformers_vit_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    arch = cfg.get("architectures")
    if not isinstance(arch, list) or len(arch) == 0:
        return False
    if not any(isinstance(a, str) and a.startswith("ViT") for a in arch):
        return False
    for fn in ("pytorch_model.bin", "model.safetensors"):
        if os.path.isfile(os.path.join(path, fn)):
            return True
    return False


class _TransformersImageClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise ValueError("Transformers ViT output does not contain logits")
        return logits

    def get_classifier(self):
        return getattr(self.backbone, "classifier", None)


class LoRALinear(nn.Module):
    """
    Custom LoRA Linear Layer, formula: W = W_0 + B @ A.
    
    Fed-Anon Core Features:
    - Matrix A: Globally shared, aggregated by server (Initialized with Gaussian).
    - Matrix B: Native-privacy, kept locally, never uploaded (Initialized with Zeros).
    - Base Weight W_0: Frozen, not updated.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        output_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.init_rank = int(rank)
        self.alpha = alpha
        
        # Base weight (Frozen)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoRA Matrices
        # A: Shared matrix (Server aggregated) - Gaussian initialization
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        
        # B: Personalized matrix (Locally kept) - Zero initialization
        # This ensures LoRA output is 0 at the start of training, preserving original model behavior
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if output_mask is None:
            self.output_mask = None
        else:
            self.register_buffer("output_mask", output_mask.clone())
        
        # Track which matrices are currently trainable
        self.A_trainable = True
        self.B_trainable = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ W_0^T + x @ A^T @ B^T * scaling
        """
        # Base weight path
        base_output = F.linear(x, self.base_weight, getattr(self, "bias", None))
        
        # LoRA path
        current_rank = int(self.lora_A.shape[0])
        scaling = (self.alpha / float(current_rank)) if current_rank > 0 else 0.0
        lora_output = F.linear(
            F.linear(x, self.lora_A),
            self.lora_B
        ) * scaling

        if self.output_mask is not None:
            mask = self.output_mask
            view_shape = [1] * (lora_output.ndim - 1) + [-1]
            lora_output = lora_output * mask.view(*view_shape)

        return base_output + self.dropout(lora_output)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_a = prefix + "lora_A"
        key_b = prefix + "lora_B"
        if key_a in state_dict:
            A = state_dict[key_a]
            if tuple(A.shape) != tuple(self.lora_A.shape):
                self.lora_A = nn.Parameter(A.clone(), requires_grad=bool(self.lora_A.requires_grad))
        if key_b in state_dict:
            B = state_dict[key_b]
            if tuple(B.shape) != tuple(self.lora_B.shape):
                self.lora_B = nn.Parameter(B.clone(), requires_grad=bool(self.lora_B.requires_grad))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def get_A(self) -> torch.Tensor:
        """Get Shared Matrix A (For upload to server)"""
        return self.lora_A.data.clone()
    
    def set_A(self, A: torch.Tensor):
        """Set Shared Matrix A (Received from server)"""
        A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype).clone()
        if tuple(A.shape) != tuple(self.lora_A.shape):
            requires_grad = bool(self.lora_A.requires_grad)
            self.lora_A = nn.Parameter(A, requires_grad=requires_grad)
        else:
            self.lora_A.data = A
    
    def get_B(self) -> torch.Tensor:
        """Get Personalized Matrix B (For local persistence only)"""
        return self.lora_B.data.clone()
    
    def set_B(self, B: torch.Tensor):
        """Set Personalized Matrix B"""
        B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype).clone()
        if tuple(B.shape) != tuple(self.lora_B.shape):
            requires_grad = bool(self.lora_B.requires_grad)
            self.lora_B = nn.Parameter(B, requires_grad=requires_grad)
        else:
            self.lora_B.data = B

    def reset_lora(self, rank: Optional[int] = None):
        target_rank = int(rank if rank is not None else self.init_rank)
        device = self.base_weight.device
        dtype = self.base_weight.dtype

        A = torch.randn(target_rank, self.in_features, device=device, dtype=dtype) * 0.02
        B = torch.zeros(self.out_features, target_rank, device=device, dtype=dtype)

        A_requires_grad = bool(self.lora_A.requires_grad)
        B_requires_grad = bool(self.lora_B.requires_grad)
        self.lora_A = nn.Parameter(A, requires_grad=A_requires_grad)
        self.lora_B = nn.Parameter(B, requires_grad=B_requires_grad)

    def merge_lora_weights(self):
        current_rank = int(self.lora_A.shape[0])
        if current_rank <= 0:
            return
        scaling = self.alpha / float(current_rank)
        delta_w = (self.lora_B @ self.lora_A) * scaling
        self.base_weight.data = self.base_weight.data + delta_w.to(
            device=self.base_weight.device, dtype=self.base_weight.dtype
        )

    def merge_lora_into_base_and_reset(self, rank: Optional[int] = None):
        self.merge_lora_weights()
        self.reset_lora(rank=rank)
    
    def freeze_A(self):
        """Freeze Matrix A"""
        self.lora_A.requires_grad = False
        self.A_trainable = False
    
    def unfreeze_A(self):
        """Unfreeze Matrix A"""
        self.lora_A.requires_grad = True
        self.A_trainable = True
    
    def freeze_B(self):
        """Freeze Matrix B"""
        self.lora_B.requires_grad = False
        self.B_trainable = False
    
    def unfreeze_B(self):
        """Unfreeze Matrix B"""
        self.lora_B.requires_grad = True
        self.B_trainable = True
    
    def freeze_all(self):
        """Freeze both A and B"""
        self.freeze_A()
        self.freeze_B()
    
    def unfreeze_all(self):
        """Unfreeze both A and B"""
        self.unfreeze_A()
        self.unfreeze_B()
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get list of currently trainable parameters"""
        params = []
        if self.A_trainable:
            params.append(self.lora_A)
        if self.B_trainable:
            params.append(self.lora_B)
        return params


class PeftLoRALinearAdapter:
    def __init__(self, peft_linear_module: nn.Module, adapter_name: str = "default"):
        self._m = peft_linear_module
        self._adapter = adapter_name

    @property
    def lora_A(self) -> torch.nn.Parameter:
        return self._m.lora_A[self._adapter].weight

    @property
    def lora_B(self) -> torch.nn.Parameter:
        return self._m.lora_B[self._adapter].weight

    def get_A(self) -> torch.Tensor:
        return self.lora_A.data.clone()

    def set_A(self, A: torch.Tensor):
        self.lora_A.data = A.clone()

    def get_B(self) -> torch.Tensor:
        return self.lora_B.data.clone()

    def set_B(self, B: torch.Tensor):
        self.lora_B.data = B.clone()

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


def _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default") -> Dict[str, PeftLoRALinearAdapter]:
    lora_layers: Dict[str, PeftLoRALinearAdapter] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        try:
            _ = module.lora_A[adapter_name].weight
            _ = module.lora_B[adapter_name].weight
        except Exception:
            continue
        lora_layers[name] = PeftLoRALinearAdapter(module, adapter_name=adapter_name)
    return lora_layers


def _patch_peft_qkv_qv_only(peft_linear_module: nn.Module):
    if getattr(peft_linear_module, "_qv_only_patched", False):
        return
    base_layer = getattr(peft_linear_module, "base_layer", None)
    if base_layer is None or not hasattr(base_layer, "out_features"):
        return
    out_features = int(base_layer.out_features)
    if out_features % 3 != 0:
        return

    d = out_features // 3
    mask = torch.ones(out_features, dtype=torch.float32)
    mask[d:2 * d] = 0.0
    peft_linear_module.register_buffer("qv_output_mask", mask)
    orig_forward = peft_linear_module.forward

    def forward(x: torch.Tensor, *args, **kwargs):
        full_out = orig_forward(x, *args, **kwargs)
        base_out = base_layer(x)
        lora_out = full_out - base_out
        m = peft_linear_module.qv_output_mask.to(dtype=lora_out.dtype, device=lora_out.device)
        view_shape = [1] * (lora_out.ndim - 1) + [-1]
        return base_out + (lora_out * m.view(*view_shape))

    peft_linear_module.forward = forward
    peft_linear_module._qv_only_patched = True


def inject_lora_to_vit(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> Dict[str, LoRALinear]:
    """
    Inject LoRA layers into Vision Transformer.
    
    Args:
        model: ViT model instance
        target_modules: List of module names to replace. If None, uses default ViT modules.
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout rate
    
    Returns:
        Dictionary mapping layer names to LoRALinear modules
    """
    if target_modules is None:
        target_modules = ['qv']
    
    lora_layers = {}
    targets = [t.lower() for t in target_modules]
    
    def replace_module(parent, name, module):
        """Recursively replace target modules with LoRA layers"""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            child_name_lower = child_name.lower()
            
            qv_only = ("qv" in targets) and ("qkv" in child_name_lower)
            standard_match = any(t in child_name_lower for t in targets if t != "qv")

            if isinstance(child_module, nn.Linear) and (qv_only or standard_match):
                output_mask = None
                if qv_only:
                    out_features = int(child_module.out_features)
                    if out_features % 3 == 0:
                        d = out_features // 3
                        mask = torch.ones(out_features, dtype=torch.float32)
                        mask[d:2 * d] = 0.0
                        output_mask = mask

                lora_layer = LoRALinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    output_mask=output_mask
                )
                lora_layer.base_weight.data = child_module.weight.data.clone()
                if child_module.bias is not None:
                    lora_layer.register_buffer('bias', child_module.bias.data.clone())
                
                setattr(module, child_name, lora_layer)
                lora_layers[full_name] = lora_layer
            
            replace_module(module, full_name, child_module)
    
    replace_module(None, "", model)
    
    return lora_layers


def create_vit_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    pretrained: bool = False,
    pretrained_source: str = "modelscope",
    pretrained_checkpoint_path: Optional[str] = None,
    modelscope_model_id: Optional[str] = None,
    modelscope_revision: Optional[str] = None,
    modelscope_cache_dir: Optional[str] = None,
    modelscope_checkpoint_file: Optional[str] = None,
    use_peft: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0
) -> tuple:
    """
    Create ViT model with LoRA injection.
    
    Args:
        model_name: Model name (timm) or architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    Returns:
        Tuple of (Model instance, LoRA layers dictionary)
    """
    use_transformers_dir = (
        pretrained
        and pretrained_checkpoint_path is not None
        and _looks_like_transformers_vit_dir(pretrained_checkpoint_path)
    )

    if use_transformers_dir:
        try:
            from transformers import ViTConfig, ViTForImageClassification
        except Exception as e:
            raise ImportError(f"Loading local vit directory requires transformers: {e}") from e
        cfg = ViTConfig.from_pretrained(pretrained_checkpoint_path)
        cfg.num_labels = int(num_classes)
        backbone = ViTForImageClassification(cfg)
        weights_path = None
        for fn in ("model.safetensors", "pytorch_model.bin"):
            p = os.path.join(pretrained_checkpoint_path, fn)
            if os.path.isfile(p):
                weights_path = p
                break
        if not weights_path:
            raise FileNotFoundError(f"Weight file not found in directory: {pretrained_checkpoint_path}")
        state_dict = _load_state_dict_from_checkpoint(weights_path)
        state_dict = _normalize_state_dict_keys(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        backbone.load_state_dict(state_dict, strict=False)
        model = _TransformersImageClassifierWrapper(backbone)
    else:
        if TIMM_AVAILABLE:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=num_classes
                )
            except:
                # Fallback to torchvision
                from torchvision.models import vit_b_16
                model = vit_b_16(weights=None, num_classes=num_classes)
        else:
            from torchvision.models import vit_b_16
            model = vit_b_16(weights=None, num_classes=num_classes)

    if pretrained and not use_transformers_dir:
        ckpt_path = None
        if pretrained_checkpoint_path:
            if os.path.isfile(pretrained_checkpoint_path):
                ckpt_path = pretrained_checkpoint_path
            elif os.path.isdir(pretrained_checkpoint_path):
                ckpt_path = _select_checkpoint_file(pretrained_checkpoint_path, None)
        elif pretrained_source == "local":
            ckpt_path = pretrained_checkpoint_path
        elif pretrained_source == "modelscope":
            if not modelscope_model_id:
                raise ValueError("Must provide --ms_model_id when pretrained_source=modelscope")
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except Exception as e:
                raise ImportError(f"Using ModelScope requires modelscope installed: {e}") from e
            try:
                snapshot_dir = snapshot_download(
                    modelscope_model_id,
                    cache_dir=modelscope_cache_dir,
                    revision=modelscope_revision
                )
            except TypeError:
                snapshot_dir = snapshot_download(modelscope_model_id)
            ckpt_path = _select_checkpoint_file(snapshot_dir, modelscope_checkpoint_file)
        else:
            raise ValueError(f"Unsupported pretrained_source: {pretrained_source}")

        if not ckpt_path:
            raise ValueError("Must provide weight source and path when enabling --pretrained")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

        state_dict = _load_state_dict_from_checkpoint(ckpt_path)
        state_dict = _normalize_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
    
    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False

    _unfreeze_classifier_params(model)
    
    if use_peft and PEFT_AVAILABLE and not use_transformers_dir:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        for n, module in model.named_modules():
            if n.endswith("qkv") and hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                _patch_peft_qkv_qv_only(module)
        lora_layers = _extract_peft_lora_layers(model, adapter_name="default")
        _unfreeze_classifier_params(model)
    else:
        lora_layers = inject_lora_to_vit(
            model,
            target_modules=["query", "value"] if use_transformers_dir else ["qv"],
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
    
    if len(lora_layers) == 0:
        raise ValueError("No LoRA layers injected. Please check target_modules settings.")
    
    return model, lora_layers


def get_lora_layer_names(lora_layers: Dict[str, nn.Module]) -> List[str]:
    """Get sorted list of LoRA layer names"""
    return sorted(lora_layers.keys())


def create_client_model(
    template_model: nn.Module,
    template_lora_layers: Dict[str, nn.Module]
) -> tuple:
    """
    Create a model copy for the client with independent LoRA layer instances.
    This is critical for simulating Fed-Anon, as each client has an independent Matrix B.
    
    Args:
        template_model: Template model
        template_lora_layers: Template LoRA layers
    
    Returns:
        Tuple of (Client model copy, Client LoRA layers dictionary)
    """
    import copy
    
    # Deep copy model structure
    client_model = copy.deepcopy(template_model)
    
    if PEFT_AVAILABLE and hasattr(template_model, "peft_config"):
        client_lora_layers = _extract_peft_lora_layers(client_model, adapter_name="default")
        for name, layer in client_lora_layers.items():
            template_layer = template_lora_layers.get(name)
            if template_layer is None:
                continue
            layer.set_A(template_layer.get_A())
            layer.set_B(template_layer.get_B())
        return client_model, client_lora_layers
    
    client_lora_layers = {}

    # Replace LoRA layers in the copied model with new instances
    def replace_lora_layers(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                # This is a LoRA layer, we need to replace it
                template_layer = template_lora_layers.get(full_name)
                if template_layer is not None:
                    dropout_p = 0.0
                    if hasattr(template_layer, "dropout") and isinstance(template_layer.dropout, nn.Dropout):
                        dropout_p = float(template_layer.dropout.p)
                    output_mask = getattr(template_layer, "output_mask", None)
                    
                    new_layer = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=child.rank,
                        alpha=child.alpha,
                        dropout=dropout_p,
                        output_mask=output_mask
                    )
                    # Copy weights from template
                    new_layer.base_weight.data = template_layer.base_weight.data.clone()
                    if hasattr(template_layer, "bias"):
                        new_layer.register_buffer("bias", template_layer.bias.data.clone())
                    
                    new_layer.lora_A.data = template_layer.lora_A.data.clone()
                    new_layer.lora_B.data = template_layer.lora_B.data.clone()
                    
                    setattr(module, name, new_layer)
                    client_lora_layers[full_name] = new_layer
            
            replace_lora_layers(child, full_name)
    
    replace_lora_layers(client_model, "")
    
    return client_model, client_lora_layers
