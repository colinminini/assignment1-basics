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
    device: str = "cpu"