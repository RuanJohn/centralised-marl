import chex 
import jax.numpy as jnp

from typing import Tuple

# TODO merge all states into 1. 
# PPO states 

@chex.dataclass
class BufferState: 
    states: jnp.ndarray
    actions: jnp.ndarray 
    rewards: jnp.ndarray 
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    entropy: jnp.ndarray
    counter: jnp.int32 
    key: chex.PRNGKey

@chex.dataclass
class BufferData: 
    state: jnp.ndarray
    action: jnp.ndarray 
    reward: jnp.ndarray 
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    entropy: jnp.ndarray

@chex.dataclass
class NetworkParams: 
    policy_params: dict
    target_policy_params : dict = None
    critic_params: dict = None

@chex.dataclass
class OptimiserStates: 
    # TODO: more detailed types here. 
    policy_state: Tuple
    critic_state: Tuple = None

# Could always make a system config 

@chex.dataclass
class PPOSystemState: 
    buffer: BufferState
    actors_key: chex.PRNGKey
    networks_key: chex.PRNGKey
    network_params: NetworkParams
    optimiser_states: OptimiserStates

# DQN states

@chex.dataclass
class DQNBufferState: 
    states: jnp.ndarray
    actions: jnp.ndarray 
    rewards: jnp.ndarray 
    dones: jnp.ndarray
    next_states: jnp.ndarray
    buffer_size: jnp.int32
    min_buffer_size: jnp.int32
    batch_size: jnp.int32
    counter: jnp.int32 
    key: chex.PRNGKey

@chex.dataclass
class DQNBufferData: 
    state: jnp.ndarray
    action: jnp.ndarray 
    reward: jnp.ndarray 
    done: jnp.ndarray
    next_state: jnp.ndarray

@chex.dataclass
class DQNSystemState: 
    buffer: BufferState
    actors_key: chex.PRNGKey
    networks_key: chex.PRNGKey
    network_params: NetworkParams
    optimiser_states: OptimiserStates