import flax.linen as nn
import jax.numpy as jnp


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AttentionBlock(nn.Module):
    embed_dim: int  # Dim of input and attention feature vectors
    dtype: jnp.dtype = jnp.float32  # Datatype of input
    hidden_dim: int  # Dim of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in Multi-Head Atten
    dropout_rate: float = 0.0  # Amount of dropout

    @nn.compact
    def __call__(self, inputs, train=True):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=x  # k and v will be copied from x
        )
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + inputs

        linear_out = nn.LayerNorm(dtype=self.dtype)(x)
        linear_out = nn.Sequential(
            [
                nn.Dense(features=self.hidden_dim),
                nn.gelu,
                nn.Dropout(rate=self.dropout_rate, deterministic=not train),
                nn.Dense(features=self.embed_dim),
            ]
        )(x)
        x = x + nn.Dropout(rate=self.dropout_rate)(linear_out, deterministic=not train)
        return x

    # TODO: Test usign code here: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html
