import haiku as hk
import jax.numpy as jnp
import numpy as np

class AttentionModel(hk.Module):
    
  def __init__(self, hidden_dim=16, num_heads=1, vocab_size=0, name=None):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim * num_heads
    self.embeddings = hk.Embed(embed_dim=self.hidden_dim, vocab_size=self.vocab_size)
    w_init = hk.initializers.RandomNormal()
    self.attention = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.hidden_dim, w_init=w_init)

  def __call__(self, x, y):
    value_keys = self.embeddings(x)
    queries = self.embeddings(y)
    encoding = self.attention(queries, value_keys, value_keys)
    return jnp.einsum("bli, bli -> bl", encoding, queries).reshape(-1,1)
  
  @classmethod
  def forward_function(cls, **kwargs):
    """Instanciate the forward function to be used with JAX Haiku.

    Returns:
        _type_: Forward function.
    """
    return lambda x, y: cls(**kwargs)(x, y)