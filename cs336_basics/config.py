# Dataclass config file (use .asdict() for arguments input)
# Serialize (via json) and save to disk
# Also can be a cfg.py python file that gets imported with import cfg
# type hint for @dataclass __init__() helper built-in function to work at instanciation

from dataclasses import dataclass
import json
import torch

@dataclass
class ModelConfig:
    d_model: int = 512
    d_ff: int = 1344
    context_length: int = 256
    theta: int = 10000
    vocab_size: int = 4096
    num_heads: int = 16
    num_blocks : int = 4
    device: str = 'mps'
    dtype: torch.dtype = torch.float32

@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

@dataclass
class TrainingConfig:
    lr_min: float = 1e-5
    lr_max: float = 1e-4
    T_warmup: int = 10
    T_c: int = 80
    max_grad: float = 1.0
    batch_size: int = 32
    steps: int = 100 # 10M training tokens
    val_and_log_every: int = 20
    train_file_path: str = './data/TinyStoriesV2-GPT4-valid.npy'
    val_file_path: str = './data/TinyStoriesV2-GPT4-train.npy'
    out_checkpoint_path: str = './data/checkpoints/checkpoint_'