import chex
import jax.numpy as jnp

from typing import Tuple, Any, Optional

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
    policy_hidden_states: jnp.ndarray = None
    critic_hidden_states: jnp.ndarray = None
    joint_observations: jnp.ndarray = None


@chex.dataclass
class BufferData:
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    entropy: jnp.ndarray
    policy_hidden_state: jnp.ndarray = None
    critic_hidden_state: jnp.ndarray = None
    joint_observation: jnp.ndarray = None


@chex.dataclass
class NetworkParams:
    policy_params: dict
    target_policy_params: dict = None
    critic_params: dict = None
    target_critic_params: dict = None
    policy_hidden_state: Optional[Any] = None
    critic_hidden_state: Optional[Any] = None
    policy_init_state: Optional[Any] = None
    critic_init_state: Optional[Any] = None
    w: Optional[float] = None


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
    train_buffer: BufferState = None


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
    t: jnp.int32 = None
    sequence_length: jnp.int32 = None
    masks: jnp.ndarray = None
    policy_hidden_states: jnp.ndarray = None


@chex.dataclass
class DQNBufferData:
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_state: jnp.ndarray
    policy_hidden_state: jnp.ndarray = None
    mask: jnp.ndarray = None


@chex.dataclass
class DQNSystemState:
    buffer: BufferState
    actors_key: chex.PRNGKey
    networks_key: chex.PRNGKey
    network_params: NetworkParams
    optimiser_states: OptimiserStates
    training_iterations: jnp.int32
