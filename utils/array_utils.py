import jax.numpy as jnp 
from jax import jit 

@jit
def add_two_leading_dims(arr: jnp.ndarray): 

    expanded_arr = jnp.expand_dims(
        jnp.expand_dims(arr, axis=0), 
        axis=0
    )

    return expanded_arr