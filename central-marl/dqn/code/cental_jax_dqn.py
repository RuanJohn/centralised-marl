"""Multi-agent JAX PPO with enumerated centralised controller.
   Essentially centralised training with centralised execution. 
"""

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import distrax
import rlax
import chex
import copy

from utils.types import (
    DQNBufferData, 
    DQNBufferState, 
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

from wrappers.ma_gym_wrapper import CentralControllerWrapper

import gym

# Constants: 
MAX_REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 1_000
BATCH_SIZE = 64
TARGET_UPDATE_PERIOD = 500
TRAIN_EVERY = 20
POLICY_LR = 0.005
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
EPSILON = 1.0 
MIN_EPSILON = 0.05 
EPSILON_DECAY_STEPS = 10_000
EPSILON_DECAY_RATE = 0.9995
# ENV_NAME = "ma_gym:Switch4-v0"
ENV_NAME = "CartPole-v0"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(ENV_NAME)
# env = CentralControllerWrapper(env)

observation_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def decay_epsilon(epsilon): 

    slope = (MIN_EPSILON - EPSILON) / EPSILON_DECAY_STEPS
    decayed_epsilon = epsilon
    epsilon = jnp.maximum(epsilon, decayed_epsilon)

    return epsilon

# Make networks 

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = [64],):

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
)

system_state = DQNSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

# TODO: Cannot select like this when jitting. 
def random_action(): 

    action = env.action_space.sample()

    return action

# NB must sample randomly like this. 
def select_random_action(key, num_actions): 
    
    action = jax.random.randint(
            key, 
            shape=(), 
            minval=0, 
            maxval=num_actions
        )

    return action

def greedy_action(q_vals):

    action = jnp.argmax(q_vals) 

    return action 

@jax.jit
@chex.assert_max_traces(n=1)
def choose_action(
    actors_key,
    q_values,
    epsilon, 
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)
    # sample_randomly = jax.random.uniform(sample_key, (1,))[0] < epsilon
    sample_randomly = jax.random.uniform(sample_key) < epsilon

    actors_key, action_key = jax.random.split(actors_key)

    action = jax.lax.cond(
        sample_randomly, 
        # TODO: Add num actions here 
        lambda: select_random_action(action_key, 2), 
        lambda: greedy_action(q_values), 
    ) 

    return actors_key, action

# @jax.jit
# @chex.assert_max_traces(n=1)
def dqn_loss(
    policy_params, 
    states, 
    actions, 
    rewards, 
    dones, 
    next_states, 
    target_policy_params, ):

    q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(policy_params, states)
    # TODO: infer num classes from q_values
    # selected_q_values = jnp.sum(
    #     jax.nn.one_hot(actions, num_classes = num_actions) * q_values, 
    #     axis=-1, 
    #     keepdims=True)
    
    # selected_q_values = jnp.squeeze(selected_q_values)

    target_q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(target_policy_params, next_states)
    # selected_target_q_values = jnp.max(target_q_values, axis=-1)

    # bellman_target = rewards + DISCOUNT_GAMMA * (1 - dones) * selected_target_q_values
    
    # bellman_target = jax.lax.stop_gradient(bellman_target)

    # td_error = (bellman_target - selected_q_values) ** 2 

    td_error = jax.vmap(rlax.q_learning)(
        q_tm1=q_values, 
        a_tm1=actions, 
        r_t=rewards, 
        discount_t=(1 - dones) * DISCOUNT_GAMMA,
        q_t=target_q_values
    )

    loss = jnp.mean(rlax.l2_loss(td_error))
    
    return loss

# @jax.jit
# @chex.assert_max_traces(n=1)
def update_policy(system_state: DQNSystemState, sampled_batch: DQNBufferData,): 

    states = jnp.squeeze(sampled_batch.state)
    actions = jnp.squeeze(sampled_batch.action)
    rewards = jnp.squeeze(sampled_batch.reward)
    dones = jnp.squeeze(sampled_batch.done)
    next_states = jnp.squeeze(sampled_batch.next_state)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params
    
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

    return system_state

global_step = 0
episode = 0 
while global_step < 50_000: 

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
        
        # TODO: set TRAIN_EVERY
        if can_sample(system_state.buffer) and (global_step % TRAIN_EVERY == 0): 
            
            buffer_state = system_state.buffer
            buffer_state, sampled_data = sample_batch(buffer_state)
            system_state.buffer = buffer_state
            system_state = update_policy(system_state, sampled_data)

    
    episode += 1
    if episode % 1 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}, EPSILON: {jnp.round(EPSILON, 2)}")   
