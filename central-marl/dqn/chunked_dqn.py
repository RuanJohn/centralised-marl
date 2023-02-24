"""Multi-agent 'chunked' JAX DQN. This is somewhere between 
    value decomposition and fully centralised MARL. 
"""

import jax.numpy as jnp 
import numpy as np
import jax 
import haiku as hk
import optax
import rlax
import chex

from utils.types import (
    DQNBufferData,
    DQNSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.dqn_replay_buffer import (
    create_buffer, 
    add, 
    sample_batch,
    can_sample,
)

from utils.array_utils import (
    add_two_leading_dims,
)

from wrappers.ma_gym_wrapper import CentralChunkedControllerWrapper

import gym

# Constants: 
MAX_REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 1_000
BATCH_SIZE = 128
TARGET_UPDATE_PERIOD = 500
TRAIN_EVERY = 20
POLICY_LR = 0.005
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
EPSILON = 1.0 
MIN_EPSILON = 0.05 
EPSILON_DECAY_STEPS = 10_000
EPSILON_DECAY_RATE = 0.9999
ENV_NAME = "ma_gym:Switch4-v0"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(ENV_NAME)
env = CentralChunkedControllerWrapper(env)

num_agents = env.num_agents
observation_dim = env.observation_space.shape[0]
# Num actions must now be the sum over all agent 
# obs dims. 
agent_actions = env.action_space.nvec
num_actions = np.sum(agent_actions)

# This will be used to split the network output q values for 
# action selection.
action_chunk_dims = env.action_map

# TODO: Make simple linear epsilon scheduler. 

# Make networks 

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = [64, 64],):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x):

        return hk.nets.MLP(policy_layer_sizes + [num_actions])(x) 

    return policy_network

policy_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)
networks_key, policy_init_key = jax.random.split(networks_key, 2)

policy_params = policy_network.init(policy_init_key, dummy_obs_data)

network_params = NetworkParams(
    policy_params=policy_params, 
    target_policy_params=policy_params, 
)

# Create optimisers and states
policy_optimiser = optax.adam(POLICY_LR)

policy_optimiser_state = policy_optimiser.init(policy_params)

# Better idea is probably a high level Policy and Critic state. 

optimiser_states = OptimiserStates(
    policy_state=policy_optimiser_state, 
)

# Initialise buffer 
buffer_state = create_buffer(
    buffer_size=MAX_REPLAY_SIZE,
    min_buffer_size=MIN_REPLAY_SIZE, 
    batch_size=BATCH_SIZE, 
    num_agents=1, 
    num_envs=1, 
    observation_dim=observation_dim, 
    action_dim=num_agents,
)

system_state = DQNSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

# NB must sample randomly like this. 
def select_random_action(key): 
    
    action = jax.random.randint(
            key, 
            shape=(num_agents,), 
            minval=jnp.zeros(num_agents), 
            maxval=jnp.array(agent_actions),
        )

    return action

def greedy_action(q_vals):

    action = jnp.argmax(q_vals, axis=1) 

    return action 

def reshape_qvalues(qvals): 

    reshaped_qvals = jnp.array(
        jnp.split(
        qvals, 
        indices_or_sections=action_chunk_dims[1:], 
        axis=0)
    )

    return reshaped_qvals

@jax.jit
@chex.assert_max_traces(n=1)
def choose_action(
    actors_key,
    q_values,
    epsilon, 
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)
    sample_randomly = jax.random.uniform(sample_key) < epsilon

    actors_key, action_key = jax.random.split(actors_key)

    # Reshape the network q values. 
    q_values = reshape_qvalues(q_values)

    action = jax.lax.cond(
        sample_randomly, 
        lambda: select_random_action(action_key), 
        lambda: greedy_action(q_values), 
    ) 

    return actors_key, action

@jax.jit
@chex.assert_max_traces(n=1)
def dqn_loss(
        
    # TODO: High level idea again
    # 1. Given each agent the same reward and dones. 
    # 2. Compute mean over q_vals and target q_vals 
    # 3. Get per agent rewards and dones. 

    policy_params, 
    states, 
    actions, 
    rewards, 
    dones, 
    next_states, 
    target_policy_params, ):

    q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(policy_params, states)

    # Map actions to relevant actions 
    actions = actions + action_chunk_dims
    
    target_q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(target_policy_params, next_states)
    # Reshape the target q_values 
    target_q_values = jax.vmap(reshape_qvalues)(target_q_values)
    selected_target_q_values = jnp.max(target_q_values, axis=-1)
    
    selected_q_values = q_values[jnp.arange(q_values.shape[0])[:, jnp.newaxis], actions]
    selected_q_values = jnp.squeeze(selected_q_values)

    # Give each agent the same done and reward 
    dones = jnp.stack([dones] * num_agents, axis=1)
    rewards = jnp.stack([rewards] * num_agents, axis=1)
    
    bellman_target = rewards + DISCOUNT_GAMMA * (1 - dones) * selected_target_q_values
    bellman_target = jax.lax.stop_gradient(bellman_target)
    td_error = (bellman_target - selected_q_values) 

    # Can also just use rlax here. 

    # td_error = jax.vmap(rlax.q_learning)(
    #     q_tm1=q_values, 
    #     a_tm1=actions, 
    #     r_t=rewards, 
    #     discount_t=(1 - dones) * DISCOUNT_GAMMA,
    #     q_t=target_q_values
    # )

    loss = jnp.mean(rlax.l2_loss(td_error))
    
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: DQNSystemState, sampled_batch: DQNBufferData, global_step: int): 

    states = jnp.squeeze(sampled_batch.state)
    actions = jnp.squeeze(sampled_batch.action)
    rewards = jnp.squeeze(sampled_batch.reward)
    dones = jnp.squeeze(sampled_batch.done)
    next_states = jnp.squeeze(sampled_batch.next_state)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params

    # NB here. TARGET_UPDATE_PERIOD must be divisble by TRAIN_EVERY. 
    target_policy_params = optax.periodic_update(
            policy_params, target_policy_params, global_step, TARGET_UPDATE_PERIOD
        )
    
    grads = jax.grad(dqn_loss)(
        policy_params, 
        states, 
        actions, 
        rewards, 
        dones, 
        next_states, 
        target_policy_params, 
    )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params
    system_state.network_params.target_policy_params = target_policy_params

    return system_state

global_step = 0
episode = 0 
while global_step < 100_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
    while not done: 

        q_values = policy_network.apply(system_state.network_params.policy_params, obs)
        
        if can_sample(system_state.buffer): 
            EPSILON = jnp.maximum(EPSILON * EPSILON_DECAY_RATE, 0.05)

        actors_key = system_state.actors_key
        actors_key, action = choose_action(actors_key, q_values, EPSILON)
        system_state.actors_key = actors_key

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(action.tolist())
        global_step += 1 # TODO: With vec envs this should be more. 

        # NB: Correct shapes here. 
        data = DQNBufferData(
            state = add_two_leading_dims(obs), 
            action = add_two_leading_dims(action), 
            reward = add_two_leading_dims(reward), 
            done = add_two_leading_dims(done), 
            next_state = add_two_leading_dims(obs_), 
        )

        obs = obs_ 

        buffer_state = system_state.buffer 
        buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        episode_return += reward
        
        if can_sample(system_state.buffer) and (global_step % TRAIN_EVERY == 0): 
            
            buffer_state = system_state.buffer
            buffer_state, sampled_data = sample_batch(buffer_state)
            system_state.buffer = buffer_state
            system_state = update_policy(system_state, sampled_data, global_step)

    
    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}, EPSILON: {jnp.round(EPSILON, 2)}")   
