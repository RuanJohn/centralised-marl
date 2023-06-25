import jax.numpy as jnp
import jax.random as random
import chex
from utils.types import BufferState, BufferData
import jax


def create_buffer(
    buffer_size: int,
    num_agents: int,
    num_envs: int,
    observation_dim: int,
    # TODO: Handle multiple hidden states.
    # For now only one.
    policy_hidden_state_dim: tuple,
    critic_hidden_state_dim: tuple,
    action_dim: int = 1,
    buffer_key: chex.PRNGKey = random.PRNGKey(0),
    joint_observation_dim: int = None,
) -> BufferState:
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

    buffer_state = BufferState(
        states=jnp.empty(
            (buffer_size + 1, num_envs, num_agents, observation_dim), dtype=jnp.float32
        ),
        actions=jnp.empty(
            (buffer_size + 1, num_envs, num_agents, action_dim), dtype=jnp.int32
        ),
        rewards=jnp.empty((buffer_size + 1, num_envs, num_agents), dtype=jnp.float32),
        dones=jnp.empty((buffer_size + 1, num_envs, num_agents), dtype=bool),
        log_probs=jnp.empty((buffer_size + 1, num_envs, num_agents), dtype=jnp.float32),
        values=jnp.empty((buffer_size + 1, num_envs, num_agents), dtype=jnp.float32),
        entropy=jnp.empty((buffer_size + 1, num_envs, num_agents), dtype=jnp.float32),
        policy_hidden_states=jnp.empty(
            (buffer_size + 1, num_envs, num_agents, *policy_hidden_state_dim),
            dtype=jnp.float32,
        ),
        critic_hidden_states=jnp.empty(
            (buffer_size + 1, num_envs, num_agents, *critic_hidden_state_dim),
            dtype=jnp.float32,
        ),
        joint_observations=jnp.empty(
            (buffer_size + 1, num_envs, joint_observation_dim), dtype=jnp.float32
        ),
        counter=jnp.int32(0),
        key=buffer_key,
    )

    return buffer_state


def add(
    buffer_state: BufferState,
    data: BufferData,
) -> BufferState:
    buffer_state.states = buffer_state.states.at[buffer_state.counter].set(data.state)
    buffer_state.actions = buffer_state.actions.at[buffer_state.counter].set(
        data.action
    )
    buffer_state.rewards = buffer_state.rewards.at[buffer_state.counter].set(
        data.reward
    )
    buffer_state.dones = buffer_state.dones.at[buffer_state.counter].set(data.done)
    buffer_state.log_probs = buffer_state.log_probs.at[buffer_state.counter].set(
        data.log_prob
    )
    buffer_state.values = buffer_state.values.at[buffer_state.counter].set(data.value)
    buffer_state.entropy = buffer_state.entropy.at[buffer_state.counter].set(
        data.entropy
    )
    buffer_state.policy_hidden_states = buffer_state.policy_hidden_states.at[
        buffer_state.counter
    ].set(data.policy_hidden_state)
    buffer_state.critic_hidden_states = buffer_state.critic_hidden_states.at[
        buffer_state.counter
    ].set(data.critic_hidden_state)
    buffer_state.joint_observations = buffer_state.joint_observations.at[
        buffer_state.counter
    ].set(data.joint_observation)

    buffer_state.counter += 1

    return buffer_state


def reset_buffer(buffer_state) -> BufferState:
    """Reset buffer while keeping key."""
    current_buffer_state = buffer_state

    new_buffer_state = BufferState(
        states=jnp.empty_like(current_buffer_state.states),
        actions=jnp.empty_like(current_buffer_state.actions),
        rewards=jnp.empty_like(current_buffer_state.rewards),
        dones=jnp.empty_like(current_buffer_state.dones),
        log_probs=jnp.empty_like(current_buffer_state.log_probs),
        values=jnp.empty_like(current_buffer_state.values),
        entropy=jnp.empty_like(current_buffer_state.entropy),
        policy_hidden_states=jnp.empty_like(current_buffer_state.policy_hidden_states),
        critic_hidden_states=jnp.empty_like(current_buffer_state.critic_hidden_states),
        joint_observations=jnp.empty_like(current_buffer_state.joint_observations),
        counter=jnp.int32(0),
        key=current_buffer_state.key,
    )

    return new_buffer_state


def should_train(buffer_state) -> bool:
    return jnp.equal(buffer_state.counter, buffer_state.buffer_size + 1)


def split_buffer_into_chunks(buffer_state: BufferState, num_chunks: int) -> BufferState:
    """Slice off last value used for bootstrapping and split into recurrent
    chunk length."""
    current_buffer_state = buffer_state

    split_buffer_state = BufferState(
        states=jnp.array(
            jnp.split(current_buffer_state.states[:-1, :, :, :], num_chunks),
        ),
        actions=jnp.array(
            jnp.split(current_buffer_state.actions[:-1, :, :, :], num_chunks),
        ),
        rewards=jnp.array(
            jnp.split(current_buffer_state.rewards[:-1, :, :], num_chunks),
        ),
        dones=jnp.array(
            jnp.split(current_buffer_state.dones[:-1, :, :], num_chunks),
        ),
        log_probs=jnp.array(
            jnp.split(current_buffer_state.log_probs[:-1, :, :], num_chunks),
        ),
        values=jnp.array(
            jnp.split(current_buffer_state.values[:-1, :, :], num_chunks),
        ),
        entropy=jnp.array(
            jnp.split(current_buffer_state.entropy[:-1, :, :], num_chunks),
        ),
        policy_hidden_states=jnp.array(
            jnp.split(
                current_buffer_state.policy_hidden_states[:-1, :, :, :, :], num_chunks
            ),
        ),
        critic_hidden_states=jnp.array(
            jnp.split(
                current_buffer_state.critic_hidden_states[:-1, :, :, :, :], num_chunks
            ),
        ),
        joint_observations=jnp.array(
            jnp.split(current_buffer_state.joint_observations[:-1, :, :], num_chunks),
        ),
        counter=current_buffer_state.counter,
        key=current_buffer_state.key,
    )

    return split_buffer_state
