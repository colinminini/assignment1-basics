import torch
from einops import rearrange, einsum, reduce
from torch import nn

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
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype)) # (d_model)
        self.eps = eps

    def forward(self, x): # (b T d_model -> b T d_model)
        in_dtype = x.dtype
        x = x.to(dtype = torch.float32)
        batched_rms = torch.sqrt(reduce(torch.square(x), '... d_model -> ... 1', reduction = 'mean') + self.eps)
        x_norm = torch.div(x, batched_rms)
        result = einsum(x_norm, self.weight, '... d_model, d_model -> ... d_model')
        return result.to(dtype=in_dtype)

# FFN Transformer Layer
# Swish activation function + Gated Linear Unit: W1 & W2, thus d_ffn becomes 2/3 what it would be without gated linear unit to keep same parameter count
# d_model -> d_ffn -> d_model. One single hidden layer. d_ffn = 8/3 * d_model (should be a multiplier of 64 for hardware efficiency)
# The module should - as always - inherent from PyTorch nn.Module parent class for convenient methods usage (.to(), .load_state_dict(), .parameters())
# Weights should be defined inside nn.Parameter() inside the child module for compatibility with Torch other parameter related methods (e.g. .parameters())

class SwiGLU_FFN(nn.Module):
    
    def __init__(self, d_model, dtype=None, device=None):
        super().__init__()
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
    def __init__(self, theta: int, d_k: int, max_sequence_len: int, device=None):
        super().__init__()
        self.thetas_dim = theta ** (-torch.arange(0, d_k, step=2, device=device) / d_k)
        self.thetas_sequence = einsum(torch.arange(0, max_sequence_len, device=device), self.thetas_dim, 'maxT, d2 -> maxT d2')
        self.stack = torch.stack([torch.cos(self.thetas_sequence), -torch.sin(self.thetas_sequence), 
                                torch.sin(self.thetas_sequence), torch.cos(self.thetas_sequence)]) # (4, maxT, d_k/2)
        self.register_buffer('RoPE', rearrange(self.stack, ' (l c) maxT d2 -> maxT d2 l c', l=2, c=2), persistent=False)
    
    def forward(self, x, token_positions): # x size: (... T d_k) / token_positions size: (... T)
        sequence_rope = self.RoPE[token_positions] # (... T d2 l c)
        x_paired = rearrange(x, '... T (d1 d2) -> ... T d1 d2', d2=2)
        output_paired = einsum(sequence_rope, x_paired, '... T dk2 l c, ... T dk2 c -> ... T dk2 l')
        return rearrange(output_paired, '... T dk2 l -> ... T (dk2 l)')