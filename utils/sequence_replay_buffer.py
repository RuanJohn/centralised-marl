import jax.numpy as jnp 
import jax.random as random 
import chex 
from utils.types import DQNBufferState, DQNBufferData
import jax
from typing import Tuple 

def create_buffer(
    buffer_size: int,
    min_buffer_size: int, 
    batch_size: int, 
    num_agents: int, 
    num_envs: int,  
    observation_dim: int, 
    hidden_state_dims: tuple, 
    sequence_length: int = 10,
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

    obs_shape = (buffer_size, sequence_length, num_envs, num_agents, observation_dim)
    action_shape = (buffer_size, sequence_length, num_envs, num_agents, action_dim)
    reward_shape = (buffer_size, sequence_length, num_envs, num_agents)
    hidden_state_shape = (buffer_size, sequence_length, num_envs, num_agents, *hidden_state_dims)

    buffer_state = DQNBufferState(
        states = jnp.zeros(obs_shape, dtype=jnp.float32), 
        actions = jnp.zeros(action_shape, dtype=jnp.int32),
        rewards = jnp.zeros(reward_shape, dtype=jnp.float32),  
        dones = jnp.zeros(reward_shape, dtype=bool), 
        next_states = jnp.zeros(obs_shape, dtype=jnp.float32), 
        masks = jnp.zeros(reward_shape, dtype=jnp.float32),
        policy_hidden_states = jnp.zeros(hidden_state_shape, dtype=jnp.float32), 
        # Index for where to place sequences 
        counter = jnp.int32(0), 
        # Index for where in sequence
        t = jnp.int32(0),
        key = buffer_key, 
        sequence_length=sequence_length, 
        min_buffer_size=min_buffer_size, 
        buffer_size=buffer_size, 
        batch_size=batch_size
    ) 

    return buffer_state

def add(
    buffer_state: DQNBufferState, 
    data: DQNBufferData, 
) -> DQNBufferState:

    counter = buffer_state.counter
    idx = counter % buffer_state.buffer_size
    t = buffer_state.t 

    buffer_state.states = buffer_state.states.at[idx, t].set(data.state)
    buffer_state.actions = buffer_state.actions.at[idx, t].set(data.action) 
    buffer_state.rewards = buffer_state.rewards.at[idx, t].set(data.reward)
    buffer_state.dones = buffer_state.dones.at[idx, t].set(data.done)
    buffer_state.next_states = buffer_state.next_states.at[idx, t].set(data.next_state)
    buffer_state.masks = buffer_state.masks.at[idx, t].set(jnp.float32(1.0))
    buffer_state.policy_hidden_states = buffer_state.policy_hidden_states.at[idx, t].set(data.policy_hidden_state)
    
    t += 1

    counter, t = jax.lax.cond(
        (buffer_state.t == buffer_state.sequence_length) | (jnp.all(jnp.squeeze(data.done)) == True), 
        lambda counter, t: (counter + 1, jnp.int32(0)), 
        lambda counter, t: (counter, t),
        counter, 
        t, 
    )

    buffer_state.counter = counter
    buffer_state.t = t 

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
        policy_hidden_state = buffer_state.policy_hidden_states[batch_idxs],
        mask = buffer_state.masks[batch_idxs],
    )

    return buffer_state, sampled_data

def should_train(
    buffer_state 
) -> bool:
    return jnp.greater_equal(buffer_state.counter, buffer_state.min_buffer_size)