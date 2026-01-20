# Fed-HARP


```bash
python main.py --task <vision|roberta|> [-- <task-args>]
```


```bash
python main.py --task vision -- --help
python main.py --task roberta -- --help
```



## Methodology

### Architecture Overview

Fed-HARP employs a Vision Transformer (ViT-Base) with custom LoRA layers injected into key components (attention and MLP layers). The LoRA decomposition follows:

$$W = W_0 + B \cdot A$$

where:
- $W_0$: Frozen base weight matrix
- $A \in \mathbb{R}^{r \times d}$: Shared matrix (aggregated by server, initialized with Gaussian)
- $B \in \mathbb{R}^{d' \times r}$: Personalized matrix (kept locally, initialized with zeros)
- $r$: LoRA rank (typically 8)

### Federated Learning Flow

```mermaid
graph TB
    subgraph "Round t"
        S[Server] -->|Allocation Map| C1[Client 1]
        S -->|Allocation Map| C2[Client 2]
        S -->|Allocation Map| CN[Client N]
        S -->|Global A| C1
        S -->|Global A| C2
        S -->|Global A| CN
    end
    
    subgraph "Client Processing"
        C1 -->|Check Staleness| W1{Warmup<br/>Needed?}
        W1 -->|Yes| WU1[Warmup Phase<br/>Freeze A, Train B]
        W1 -->|No| T1[Normal Training<br/>Train A & B]
        WU1 --> T1
        T1 -->|Delta A| S
    end
    
    subgraph "Server Aggregation"
        S -->|Collect Updates| AG[Aggregate with<br/>Staleness Weighting]
        AG -->|Update Global A| S
        S -->|Update Staleness| ST[Staleness Tracker]
    end
    
    style WU1 fill:#ffeb3b
    style AG fill:#4caf50
    style ST fill:#2196f3
```

### Mathematical Formulation

#### 1. LoRA Forward Pass

For a linear layer with input $x \in \mathbb{R}^{b \times d}$:

$$y = xW_0^T + \frac{\alpha}{r} \cdot xA^T B^T$$

where $\alpha$ is the LoRA scaling factor (typically 16.0).

#### 2. Client-Side Warmup

When client $i$ receives updated global $A_k$ for layer $k$ that was previously frozen:

**Step 1**: Freeze $A_k$, unfreeze $B_k$
$$A_k^{(i)} \leftarrow A_k^{global}, \quad \frac{\partial \mathcal{L}}{\partial A_k} = 0$$

**Step 2**: Train $B_k$ for $T_{warmup}$ steps:
$$B_k^{(i)} \leftarrow B_k^{(i)} - \eta_{warmup} \frac{\partial \mathcal{L}}{\partial B_k}$$

**Step 3**: Unfreeze both $A_k$ and $B_k$ for normal training.

#### 3. Server-Side Staleness-Weighted Aggregation

The server tracks staleness $\tau_{i,k}$ (number of rounds since client $i$ last updated layer $k$).

**Dampening Factor**:
$$\alpha_{i,k} = \frac{1}{\sqrt{1 + \tau_{i,k}}}$$

**Aggregation Rule**:
$$A_k^{global} \leftarrow A_k^{global} + \eta \sum_{i \in \mathcal{S}_k} \alpha_{i,k} \cdot \Delta A_{k}^{(i)}$$

where:
- $\mathcal{S}_k$: Set of clients that updated layer $k$ in this round
- $\Delta A_{k}^{(i)} = A_{k}^{(i,new)} - A_{k}^{(i,old)}$: Client $i$'s update for layer $k$
- $\eta$: Server aggregation learning rate (typically 0.1)

#### 4. Staleness Update

After each round:
$$\tau_{i,k} \leftarrow \begin{cases}
0 & \text{if client } i \text{ updated layer } k \\
\tau_{i,k} + 1 & \text{otherwise}
\end{cases}$$

### Non-IID Data Partitioning

The dataset is partitioned using a **Dirichlet distribution** with parameter $\alpha$:

$$p \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_N)$$

where $N$ is the number of clients. Lower values of $\alpha$ (e.g., 0.5) create more heterogeneous (Non-IID) distributions.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
cd /media/h3c/users/yanji/fed-a
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; import torchvision; import timm; print('All dependencies installed successfully!')"
```

## Usage

### Basic Usage

Run Fed-HARP with default hyperparameters:

```bash
python main.py --task vision -- --help
```

### Custom Configuration

```bash
python main.py --task vision -- \
    --num_clients 10 \
    --num_rounds 50 \
    --allocation_ratio 0.5 \
    --alpha 0.5 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --client_lr 0.001 \
    --batch_size 32 \
    --num_epochs 1 \
    --warmup_steps 10 \
    --aggregation_lr 0.1 \
    --eval_every 5 \
    --seed 42
```

### Command-Line Arguments

#### Dataset Arguments
- `--data_dir`: Directory for CIFAR-10 dataset (default: `./data`)
- `--alpha`: Dirichlet distribution parameter for Non-IID partitioning (default: `0.5`)

#### Model Arguments
- `--model_name`: ViT model name (default: `vit_base_patch16_224`)
- `--pretrained`: Use pretrained ImageNet weights
- `--lora_rank`: LoRA rank $r$ (default: `8`)
- `--lora_alpha`: LoRA alpha scaling factor (default: `16.0`)
- `--lora_dropout`: LoRA dropout rate (default: `0.0`)

#### Federated Learning Arguments
- `--num_clients`: Number of clients (default: `10`)
- `--num_rounds`: Number of federated rounds (default: `50`)
- `--clients_per_round`: Number of clients selected per round (default: all)
- `--allocation_ratio`: Fraction of layers each client updates (default: `0.5`)
- `--aggregation_lr`: Server aggregation learning rate $\eta$ (default: `0.1`)

#### Training Arguments
- `--batch_size`: Local batch size (default: `32`)
- `--client_lr`: Client learning rate (default: `0.001`)
- `--num_epochs`: Number of local epochs per round (default: `1`)
- `--warmup_steps`: Number of warmup steps for B-matrix alignment (default: `10`)
- `--warmup_lr`: Learning rate for warmup phase (default: `0.0001`)

#### Evaluation Arguments
- `--eval_every`: Evaluate every N rounds (default: `5`)
- `--test_batch_size`: Test batch size (default: `100`)

#### System Arguments
- `--seed`: Random seed for reproducibility (default: `42`)
- `--device`: Device to use (`cuda`/`cpu`, default: auto-detect)
- `--save_checkpoints`: Save model checkpoints
- `--checkpoint_dir`: Directory for checkpoints (default: `./checkpoints`)

### Example: Reproducing Paper Results

```bash
python main.py --task vision -- \
    --num_clients 10 \
    --num_rounds 100 \
    --allocation_ratio 0.5 \
    --alpha 0.5 \
    --lora_rank 8 \
    --client_lr 0.001 \
    --num_epochs 3 \
    --warmup_steps 10 \
    --aggregation_lr 0.1 \
    --eval_every 5 \
    --save_checkpoints \
    --seed 42


## License

This project is licensed under the MIT License.

## Contact
For questions or issues, please open an issue on the repository.

**Note**: This implementation is a research prototype. For production use, additional optimizations and security considerations should be implemented.


