import torch
from einops import rearrange, einsum, reduce, repeat
from torch import nn
import math
import numpy as np

# ALl neural nets modules should inherent from nn.Module parent class -> inherits convenient methods such as: load_state_dict(), to(), get_parameters(), cpu(), cuda(), children(), bfloat16()...

# Implement a Linear Class (= "a Linear Module")
# y = xWT

class Linear(nn.Module): # Inherits nn.Module methods()
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        self.sigma = (2 / (in_features + out_features)) ** (1/2)

        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype=dtype, device=device), 
                                                                mean=0, std = self.sigma, 
                                                                a = -3 * self.sigma, b= 3 * self.sigma ))

    def forward(self, x: torch.tensor) -> torch.Tensor: # All nn.Module need to have a forward() method
        return einsum(x, self.weight, '... in_feature , out_feature in_feature -> ... out_feature')

# Create an embedding table Class
# Ounce again, every neural nets module should inherent nn.Module for convenient access to parent methods (.load_state_dict(), .Parameters(), .to()...)

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device), std=1, a=-3, b=3))

    def forward(self, x: torch.LongTensor) -> torch.Tensor: # (... T) -> (... T d_model)
        return self.weight[x] # shape: = x.shape + self.weights.shape[1:]

# Implement LayerNorm: RMSNorm
# Dtype: Prevent overflow of root mean square by using dtype = float32. 
# It is possible to change data type in this way: input_dtype -> float32 -> input_dtype

class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device)) # (d_model)
        self.eps = eps

    def forward(self, x): # (b T d_model -> b T d_model)
        in_dtype = x.dtype
        x = x.to(dtype = torch.float32)
        batched_rms = torch.sqrt(reduce(torch.square(x), '... d_model -> ... 1', reduction = 'mean') + self.eps)
        x_norm = torch.div(x, batched_rms)
        result = einsum(x_norm, self.weight.to(dtype=torch.float32), '... d_model, d_model -> ... d_model')
        return result.to(dtype=in_dtype)

# FFN Transformer Layer
# Swish activation function + Gated Linear Unit: W1 & W2, thus d_ffn becomes 2/3 what it would be without gated linear unit to keep same parameter count
# d_model -> d_ffn -> d_model. One single hidden layer. d_ffn = 8/3 * d_model (should be a multiplier of 64 for hardware efficiency)
# The module should - as always - inherent from PyTorch nn.Module parent class for convenient methods usage (.to(), .load_state_dict(), .parameters())
# Weights should be defined inside nn.Parameter() inside the child module for compatibility with Torch other parameter related methods (e.g. .parameters())

class SwiGLU_FFN(nn.Module):
    
    def __init__(self, d_model, d_ff=None, dtype=None, device=None):
        super().__init__()
        self.d_ff = d_ff
        if not self.d_ff:
            self.d_ff = int(((8/3) * d_model // 64) * 64)  # Kepping same parameter count with/without Gated Linear Unit
        self.w1 = Linear(in_features=d_model, out_features=self.d_ff, dtype=dtype, device=device)
        self.w2 = Linear(in_features=self.d_ff, out_features=d_model, dtype=dtype, device=device)
        self.w3 = Linear(in_features=d_model, out_features=self.d_ff, dtype=dtype, device=device)

    def forward(self, x): # x.shape == ([B T d_model])
        x_1 = self.w1(x) # ... d_model -> ... d_ff
        GLU = self.w3(x)
        SiLU = einsum(x_1, torch.sigmoid(x_1), '... d_ff, ... d_ff -> ... d_ff')
        SwiGLU = einsum(SiLU, GLU, '... d_ff, ... d_ff -> ... d_ff')
        return self.w2(SwiGLU)
    
# Relative Positional Embedding (RoPE)
# From (nn.Module). Uses buffer for the rotation angles with 'self.register_buffer()'
# Different dimensions of the latent space get different rotation speed. Rotation itself depends on the position of the token in the sequence.
# In the Latent Space, delta of angle between vectors is linearly proportional to the distance of their index in the sequence

class RoPE(nn.Module):
    # Create a ([T, d_k, d_k]) tensor of the rotation matrices is suboptimal
    # Create a ([T, d_k/2, 2, 2]) tensor of rotation matrices, rearrange x to become (... d_k/2 2) then MatMul then back to (... d_k)
    def __init__(self, theta: int, d_k: int, max_sequence_len: int, device=None, dtype=None):
        super().__init__()
        self.thetas_dim = theta ** (-torch.arange(0, d_k, step=2, device=device, dtype=dtype) / d_k)
        self.thetas_sequence = einsum(torch.arange(0, max_sequence_len, device=device, dtype=dtype), self.thetas_dim, 'maxT, d2 -> maxT d2')
        self.stack = torch.stack([torch.cos(self.thetas_sequence), -torch.sin(self.thetas_sequence), 
                                torch.sin(self.thetas_sequence), torch.cos(self.thetas_sequence)]) # (4, maxT, d_k/2)
        self.register_buffer('RoPE', rearrange(self.stack, ' (l c) maxT d2 -> maxT d2 l c', l=2, c=2), persistent=False)
    
    def forward(self, x, token_positions): # x size: (... T d_k) / token_positions size: (... T)
        sequence_rope = self.RoPE[token_positions] # (... T d2 l c)
        x_paired = rearrange(x, '... T (d1 d2) -> ... T d1 d2', d2=2)
        output_paired = einsum(sequence_rope, x_paired, '... T dk2 l c, ... T dk2 c -> ... T dk2 l')
        return rearrange(output_paired, '... T dk2 l -> ... T (dk2 l)')

def Softmax(x: torch.Tensor, dim: int):
    x_copy = torch.transpose(x, dim, -1) # (... i)
    x_minus_max = reduce(x_copy, '... i -> ... 1', reduction='max')
    x_copy = x_copy - x_minus_max # Stability trick to avoidd exp(vi) to become inf and then having inf/inf = NaN
    x_copy = torch.exp(x_copy)
    x_div = reduce(x_copy, '... i -> ... 1', reduction='sum')
    probs = torch.div(x_copy, x_div)
    return torch.transpose(probs, dim, -1)

def scaled_dot_product_attention(Q, K, V, mask=True):
    T1 = Q.shape[-2]
    T2 = K.shape[-2]
    true_mask = torch.ones(T1,T2, dtype=torch.bool, device=mask.device)
    mask_copy = mask * true_mask
    mask_matrix = torch.zeros_like(mask_copy, dtype=torch.float, device=mask.device)
    mask_matrix[~mask_copy] = float('-inf')
    d_k = K.shape[-1]
    logits_scores = torch.div(einsum(Q, K, '... T1 d_k, ... T2 d_k -> ... T1 T2'), ( d_k ** (1/2) ))
    mask_matrix = mask_matrix.to(device=Q.device)
    logits_scores += mask_matrix
    scores = Softmax(logits_scores, dim=-1)
    return einsum(scores, V, '... T1 T2, ... T2 d_v -> ... T1 d_v')

# Causal Multi Head Self Attention Module inherits from parent torch class nn.Module for methods like state_dict(), buffers(), to()...
# Three Linear Layer Matrices of learnable weights (no biases): WQ, WK, WV
# max_sequence_length is NOT necessarily input sequence length. T is the input size but it can be truncated and not be of the full context size of the LM!
# token_positions should be torch.arange(x.shape[1], device=x.device)

class causal_multihead_self_attention_with_rope(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_sequence_length:int, rope_theta=10000, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divided by num_heads'
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=rope_theta, d_k=self.d_k, max_sequence_len=max_sequence_length, device=device)
        mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool, device=device))
        self.register_buffer('mask', mask, persistent=False)
    
    def forward(self, x, token_positions=None): # (B T d_model)
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device)
        T = x.shape[1]
        Q, K, V = rearrange(self.q_proj(x), '... T (nh dk) -> ... nh T dk', nh=self.num_heads), rearrange(self.k_proj(x), '... T (nh dk) -> ... nh T dk', nh=self.num_heads), rearrange(self.v_proj(x), '... T (nh dk) -> ... nh T dk', nh=self.num_heads) # (B num_h T d_k)
        Q, K = self.rope(Q, token_positions=token_positions), self.rope(K, token_positions=token_positions)
        output = scaled_dot_product_attention(Q, K, V, mask=self.mask[:T,:T]) # (B n_h T d_k)
        return self.output_proj(rearrange(output, '... nh T dk -> ... T (nh dk)', nh=self.num_heads))

# Transformer Block Module

class Transformer_Block(nn.Module):

    def __init__(self, d_model:int, d_ff:int, num_heads:int, max_sequence_len:int, theta=10000, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, dtype=dtype, device=device)
        self.attn = causal_multihead_self_attention_with_rope(d_model=d_model, num_heads=num_heads, max_sequence_length=max_sequence_len, rope_theta=theta, device=device, dtype=dtype) # We need max_seq_len for RoPE buffer init when instantiating the Transformer_Block Module
        self.ln2 = RMSNorm(d_model=d_model, dtype=dtype, device=device)
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, dtype=dtype, device=device)
    
    def forward(self, x): # (B T d_model -> B T d_model)
        y = x + self.attn(self.ln1(x))
        output = y + self.ffn(self.ln2(y))
        return output

# Transformer LM Module

class transformer_lm(nn.Module):

    def __init__(self, vocab_size:int, d_model:int, d_ff:int, num_heads:int, num_blocks:int, context_length:int, theta=10000, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=dtype, device=device)
        self.layers = nn.Sequential(*[Transformer_Block(d_model=d_model, d_ff=d_ff, num_heads=num_heads, max_sequence_len=context_length, theta=theta, device=device, dtype=dtype) for _ in range(num_blocks)])
        self.ln_final = RMSNorm(d_model=d_model, dtype=dtype, device=device)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, dtype=dtype, device=device)
    
    def forward(self, x): # (B T -> B T vocab_size)
        x_embd = self.token_embeddings(x) # (B T d_model)
        for i in range(len(self.layers)):
            x_embd = self.layers[i](x_embd)
        x_embd = self.ln_final(x_embd)
        return self.lm_head(x_embd)
    
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
# Cross-entropy loss. Scalar value of the current cost. Approximated through batched sampling. Root node of backpropagaton pass -> non-convex optimization

def cross_entropy_loss(logits, targets) -> torch.float: # (... Vocab_size), (...)
    max_logit = reduce(logits, '... V -> ... 1', reduction='max')
    exp_norm_logits = torch.exp(logits - max_logit)
    sum_exps = reduce(exp_norm_logits, '... V -> ... 1', reduction='sum')
    gathered_logits = torch.gather(logits, dim=-1, index=repeat(targets, '... -> ... c', c=1)) # (...)
    log_probs = gathered_logits - max_logit - torch.log(sum_exps)
    return - ( reduce(log_probs, '... -> ', reduction='mean')) # averaged cost over batch and sequence

def cross_entropy_loss_logsumexp_fused_kernel(logits, targets):
    gathered_logits = torch.gather(input=logits, dim=-1, index=targets.unsqueeze(-1) )
    batched_cross_entropy = - gathered_logits + torch.logsumexp(logits, dim=-1, keepdim=True) # (B T 1)
    return reduce(batched_cross_entropy, '... -> ', reduction='mean')

# Implement AdamW: (lr, weight_decay, beta1, beta2)
# Adam optimizer tracks a running the gradients moment of order 1 (hyperparameter: beta 1) and order 2 (beta 2)
# AdamW adds a weight decay at the optimizer .step() (instead of at the gradient level for example)

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, 'weight_decay': weight_decay, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, weight_decay, (beta1, beta2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"] # Get the hyperparameters
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                if len(state) == 0:
                    state['moment_order_1'] = torch.zeros_like(p)
                    state['moment_order_2'] = torch.zeros_like(p)
                    state['t'] = 1

                t = state['t']
                grad = p.grad
                moment_order_1 = state['moment_order_1'] * beta1 + (1 - beta1) * grad
                moment_order_2 = state['moment_order_2'] * beta2 + (1 - beta2) * grad**2
                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                with torch.no_grad():
                    p -= p * weight_decay * lr + adjusted_lr * moment_order_1 / (torch.sqrt(moment_order_2) + eps) # AdamW weight update. -= makes the update in_place, very important!

                state['moment_order_1'] = moment_order_1 # stateful buffer of total size 4 * num_param (torch.float32)
                state['moment_order_2'] = moment_order_2 # stateful buffer of total size 4 * num_param (torch.float32)
                state["t"] = t + 1 

        return loss

# Learning Rate Scheduling 

def cosine_lr_schedule(t: int, lr_min: float, lr_max: float, T_warmup: int, T_c: int) -> float:
    '''
    Returns the learning rate at time t from the cosine_annealing scheduler
    '''
    if t < T_warmup:
        return t / T_warmup * lr_max
    if t <= T_c:
        return lr_min + 1/2 * (1 + math.cos((t-T_warmup) / (T_c - T_warmup) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

# Gradient Clipping Implementation

def gradient_clipping(parameters_list: list[torch.tensor], max_grad: torch.float):
    '''
    Modifies parameters gradients in_place to cap the global l2-norm at max_grad
    '''
    running_l2_grad = 0

    for p in parameters_list:

        if p.grad is None:
            continue

        running_l2_grad += reduce(torch.square(p.grad), '... -> ', reduction='sum')
    
    l2_grad = torch.sqrt(running_l2_grad)

    if l2_grad > max_grad:
        for p in parameters_list:
            if p.grad is not None:
                p.grad *= max_grad / (l2_grad + 1e-6)

# System training environment: load from disk (dataset) to system RAM + dataprocessing (CPU work) to GPU HBM (PCIe connection from system RAM to HBM)
# List of tensors (from numpy) into torch.stack

def get_batch(dataset, batch_size, context_len, device=None):
    batch_ids = np.random.randint(low=0, high=len(dataset) - context_len, size=(batch_size,))
    x_batch = []
    y_batch = []
    for i in batch_ids:
        x = dataset[i:i+context_len]
        y = dataset[i+1:i+context_len+1]
        x = np.copy(x)
        y = np.copy(y)
        x_batch.append(torch.from_numpy(x))
        y_batch.append(torch.from_numpy(y))
    x_batch, y_batch = torch.stack(x_batch), torch.stack(y_batch)
    x_batch, y_batch = x_batch.to(dtype=torch.int64), y_batch.to(dtype=torch.int64)
    return x_batch.to(device), y_batch.to(device)