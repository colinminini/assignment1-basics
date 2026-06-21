# Dataclass config file (use .asdict() for arguments input)
# Serialize (via json) and save to disk
# Also can be a cfg.py python file that gets imported with import cfg
# type hint for @dataclass __init__() helper built-in function to work at instanciation

from dataclasses import dataclass
import json

@dataclass
class Config:
    vocab_size: int = 1000
    d_model: int = 64
    d_ff: int = 256
    num_heads: int = 4
    num_blocks: int = 3
    context_length: int = 10
    device: str = "mps"

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

@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

@dataclass
class TrainingConfig:
    batch_size: int = 32
    steps: int = 20 # 10M training tokens
    max_grad: float = 1.0
    train_file_path: str = './data/TinyStoriesV2-GPT4-valid.npy'
    log_every: int = 10
    out_checkpoint_path: str = './data/checkpoints/checkpoint_'