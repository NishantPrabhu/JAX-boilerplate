
import functools
import jax.numpy as jnp 
import flax.linen as nn 
from typing import Any, Callable, Tuple, Optional

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class MultiHeadSelfAttention(nn.Module):
    hidden_dim: int = 512 
    num_heads: int = 8
    kernel_init: Optional[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
    bias_init: Optional[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-06)
    
    @nn.compact 
    def __call__(self, x, key=None, value=None):
        assert x.ndim == 3, f'Expected input to have ndim == 3, got shape {x.shape}'
        bs, sl, fdim = x.shape
        x = nn.LayerNorm()(x)
        
        query = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        query = query.reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        if key is None:
            key = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            key = key.reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        if value is None:
            value = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            value = value.reshape(bs, sl, self.num_heads, -1).transpose((0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhid,bhjd->bhij', query, key)
        attn_probs = nn.softmax(x / jnp.sqrt(self.hidden_dim), axis=-1)
        out = jnp.einsum('bhij,bhjd->bhid', attn_probs, value)
        out = out.transpose((0, 2, 1, 3)).reshape(bs, sl, -1) + x
        return out, key, value
    
    
class EncoderLayer(nn.Module):
    hidden_dim: int = 512
    mlp_dim: int = 2048 
    num_heads: int = 8
    kernel_init: Optional[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
    bias_init: Optional[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-06)
    
    @nn.compact 
    def __call__(self, x, key=None, value=None):
        x, key, value = MultiHeadSelfAttention(
            self.hidden_dim, self.num_heads, self.kernel_init, self.bias_init
        )(x, key, value)
        
        # Feedforward
        out = nn.LayerNorm(x)
        out = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(out)
        out = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(out)
        out = out + x 
        return out, key, value 