import jax.numpy as jnp 
import jax.random as random 
import chex 
from utils.types import DQNBufferState, DQNBufferData
import jax
from typing import Tuple

def create_buffer(
    observation_dim: int, 
    buffer_size: int = 200_000,
    min_buffer_size: int = 1_000, 
    batch_size: int = 64, 
    num_agents: int = 1, 
    num_envs: int = 1,  
    action_dim: int = 1, 
    buffer_key: chex.PRNGKey = random.PRNGKey(0),
) -> DQNBufferState: 

    """A simple trajectory buffer. 
    
    Args: 
        buffer_size: the size of the experience horizon 
        num_agents: number of agents in an environment 
        num_envs: number of environments run in parallel
        observation_dim: dimension of the observations being stored 
        action_dim: this will default to 1 but could be more if agents have 
            MultiDiscrete action spaces for example. 
        buffer_key: PRNGkey for sampling from the buffer if need be. 
    
    """

    # Always store as buffer_size x env x agent x observation_dim

    buffer_state = DQNBufferState(
        states = jnp.empty((buffer_size, num_envs, num_agents, observation_dim), dtype=jnp.float32), 
        actions = jnp.empty((buffer_size, num_envs, num_agents, action_dim), dtype=jnp.int32),
        rewards = jnp.empty((buffer_size, num_envs, num_agents), dtype=jnp.float32),  
        dones = jnp.empty((buffer_size, num_envs, num_agents), dtype=bool), 
        next_states = jnp.empty((buffer_size, num_envs, num_agents, observation_dim), dtype=jnp.float32),
        min_buffer_size = jnp.int32(min_buffer_size), 
        buffer_size = jnp.int32(buffer_size), 
        batch_size = jnp.int32(batch_size), 
        counter = jnp.int32(0), 
        key = buffer_key, 

    ) 

    return buffer_state

def add(
    buffer_state: DQNBufferState, 
    data: DQNBufferData, 
) -> DQNBufferState:

    idx = buffer_state.counter % buffer_state.buffer_size

    buffer_state.states = buffer_state.states.at[idx].set(data.state)
    buffer_state.actions = buffer_state.actions.at[idx].set(data.action) 
    buffer_state.rewards = buffer_state.rewards.at[idx].set(data.reward)
    buffer_state.dones = buffer_state.dones.at[idx].set(data.done)
    buffer_state.next_states = buffer_state.next_states.at[idx].set(data.next_state)

    buffer_state.counter += 1

    return buffer_state

def sample_batch(
    buffer_state: DQNBufferState, 
) -> Tuple[DQNBufferState, DQNBufferData]: 

    buffer_key = buffer_state.key
    buffer_key, sample_key = jax.random.split(buffer_key)
    buffer_state.key = buffer_key

    # Make sure that maxval is inclusive. 
    batch_idxs = jax.random.randint(
        key=sample_key, 
        shape=(buffer_state.batch_size,), 
        minval=0, 
        maxval=jnp.minimum(buffer_state.counter, buffer_state.buffer_size))

    # Double check this slicing. 
    sampled_data = DQNBufferData(
        state = buffer_state.states[batch_idxs], 
        action = buffer_state.actions[batch_idxs],
        reward = buffer_state.rewards[batch_idxs], 
        done = buffer_state.dones[batch_idxs], 
        next_state = buffer_state.next_states[batch_idxs], 
    )

    return buffer_state, sampled_data


def can_sample(
    buffer_state: DQNBufferState 
) -> bool:
        
    return jnp.greater_equal(buffer_state.counter, buffer_state.min_buffer_size)
