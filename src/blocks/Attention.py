import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from src.blocks.patchify import patchify, unpatchify
from src.blocks.rotary_embedding import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, causal=False, emb_dim=None, positional_encoding="absolute", layer_idx=None, legacy_norm=False):
        super().__init__()

        self.layer_idx = layer_idx
        self.positional_encoding = positional_encoding
        self.legacy_norm = legacy_norm
        
        # Projections
        self.query_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.key_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads

        self.scale = self.head_dim ** -0.5

        # Softmax attention also needs q k norms
        if legacy_norm:
            # I messed this up here
            self.q_norm = nn.RMSNorm(dim, dim)
            self.k_norm = nn.RMSNorm(dim, dim)
        else:
            # I messed this up here
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.causal = causal


        # Rotary embeddings
        if positional_encoding == "RoPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        
        
        
    def forward(self, x):
        N, C, d = x.shape




        # RMSNorm and Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.legacy_norm:
            queries = self.q_norm(self.query_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.k_norm(self.key_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            queries = self.q_norm(self.query_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3))
            keys = self.k_norm(self.key_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3))
        values = self.value_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)





        # Apply rotary embeddings
        if self.positional_encoding == "RoPE":
            queries = self.rotary_emb.rotate_queries_or_keys(queries)
            keys = self.rotary_emb.rotate_queries_or_keys(keys)



        # # Create mask
        # if self.causal:
        #     mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device)
                
        # # Flash attention
        #### NOTE: For some reason flash attenetion expects (batch, seqlen, num heads, dim) instead of what you usually see (batch, num heads, seqlen, dim)
        ####       Transposing before and after works for me.
        # attn_ = flash_attn_func(queries.transpose(1, 2).to(torch.float16), keys.transpose(1, 2).to(torch.float16), values.transpose(1, 2).to(torch.float16), causal=self.causal, softmax_scale=self.scale).transpose(1, 2).to(queries.dtype)

        # # Manual
        # attn = (queries @ keys.mT) * self.scale        
        # if self.causal:
        #     attn = attn.masked_fill(mask, float('-inf')).softmax(dim=-1)
        # else:
        #     attn = attn.softmax(dim=-1)
        # attn = attn @ values

        # PyTorch
        attn = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=self.causal, scale=self.scale)


        # Output projection
        return self.out_proj(attn.permute(0, 2, 1, 3).reshape(N, C, -1))
