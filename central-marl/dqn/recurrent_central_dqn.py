"""Multi-agent JAX DQN with enumerated centralised controller.
   Essentially centralised training with centralised execution. 
"""

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import rlax
import chex
import time 

from utils.types import (
    DQNBufferData,
    DQNSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.sequence_replay_buffer import (
    create_buffer, 
    add, 
    should_train,
    sample_batch, 
)

add = jax.jit(chex.assert_max_traces(add, n=1), donate_argnums=(0,))
# add = jax.jit(chex.assert_max_traces(add, n=1))
# add = jax.jit(add, donate_argnums=(0,))

from utils.array_utils import (
    add_two_leading_dims,
)

from wrappers.ma_gym_wrapper import CentralControllerWrapper

import gym

# Constants: 
MAX_REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 200
BATCH_SIZE = 32
SEQUENCE_LENGTH = 20
TARGET_UPDATE_PERIOD = 50
TRAIN_EVERY = 50
POLICY_LR = 0.005
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
EPSILON = 1.0 
MIN_EPSILON = 0.05 
EPSILON_DECAY_STEPS = 10_00
EPSILON_DECAY_RATE = 0.99995
POLICY_RECURRENT_LAYER_SIZES = [32]
POLICY_LAYER_SIZES = [32]
# ENV_NAME = "ma_gym:Switch2-v0"
ENV_NAME = "CartPole-v1"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(ENV_NAME)
# env = CentralControllerWrapper(env)

observation_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

def create_epsilon_schedule(
        start=1.0, 
        stop=MIN_EPSILON, 
        warm_up=MIN_REPLAY_SIZE * SEQUENCE_LENGTH, 
        decay_steps=EPSILON_DECAY_STEPS):

    def linear_epsilon_schedule(step):
        

        slope = (stop - start) / decay_steps
        epsilon = (step - warm_up) * slope + start

        jax.lax.cond(
            step >= warm_up,
            lambda: epsilon, 
            lambda: 1.0
        )

        return jnp.maximum(epsilon, stop)
    
    return linear_epsilon_schedule

epsilon_schedule = create_epsilon_schedule()

# Make networks 

# TODO: Only single recurrent layer at the moment. 
def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = POLICY_LAYER_SIZES,
    policy_recurrent_layer_sizes: list = POLICY_RECURRENT_LAYER_SIZES):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x, y): 

        return hk.DeepRNN(
            [
            hk.nets.MLP(policy_layer_sizes, activate_final=True), 
            hk.GRU(policy_recurrent_layer_sizes[0]), 
            # hk.GRU(policy_recurrent_layer_sizes[1]), 
            # hk.nets.MLP(policy_layer_sizes, activate_final=True), 
            hk.Linear(num_actions),
            ]
            )(x, y) 

    return policy_network

policy_network = make_networks(num_actions=num_actions)

def create_hidden_states(policy_recurrent_layer_sizes:list = POLICY_RECURRENT_LAYER_SIZES): 

    hidden_states = jnp.array(
        [jnp.zeros(layer, dtype=jnp.float32)
         for layer in policy_recurrent_layer_sizes] 
        )

    return hidden_states
# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)
policy_hidden_state = create_hidden_states()
networks_key, policy_init_key = jax.random.split(networks_key, 2)

policy_params = policy_network.init(policy_init_key, dummy_obs_data, policy_hidden_state)

network_params = NetworkParams(
    policy_params=policy_params, 
    target_policy_params=policy_params, 
    policy_hidden_state=policy_hidden_state,
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
    hidden_state_dims=policy_hidden_state.shape, 
    sequence_length=SEQUENCE_LENGTH, 
)

system_state = DQNSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
    training_iterations=jnp.int32(0)
) 

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
    sample_randomly = jax.random.uniform(sample_key) < epsilon

    actors_key, action_key = jax.random.split(actors_key)

    action = jax.lax.cond(
        sample_randomly, 
        lambda: select_random_action(action_key, num_actions), 
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
    policy_hidden_states, 
    masks, 
    target_policy_params, ):

    def core_online(state, hidden_state): 

        return policy_network.apply(policy_params, state, hidden_state)
    
    def core_target(state, hidden_state): 

        return policy_network.apply(target_policy_params, state, hidden_state)

    policy_hidden_states = policy_hidden_states[:, 0, :, :]
    policy_hidden_states = jnp.zeros_like(policy_hidden_states)
    # zero_hidden_states = jnp.zeros_like(policy_hidden_states)

    q_values, _ = hk.static_unroll(
        core_online, 
        states, 
        policy_hidden_states, 
        time_major=False, 
    )

    target_q_values, _ = hk.static_unroll(
        core_target, 
        next_states, 
        policy_hidden_states, 
        time_major=False, 
    )

    selector_q_values, _ = hk.static_unroll(
        core_online, 
        next_states, 
        policy_hidden_states, 
        time_major=False, 
    )
    
    # q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(policy_params, states)
    # target_q_values = jax.vmap(policy_network.apply, in_axes=(None, 0))(target_policy_params, next_states)
    
    # TODO: infer num classes from q_values
    # selected_q_values = jnp.sum(
    #     jax.nn.one_hot(actions, num_classes = num_actions) * q_values, 
    #     axis=-1, 
    #     keepdims=True)
    # selected_q_values = jnp.squeeze(selected_q_values)

    
    # selected_target_q_values = jnp.max(target_q_values, axis=-1)
    # bellman_target = rewards + DISCOUNT_GAMMA * (1 - dones) * selected_target_q_values
    # bellman_target = jax.lax.stop_gradient(bellman_target)
    # td_error = (bellman_target - selected_q_values) 

    # Can also just use rlax here. 
    batched_loss = jax.vmap( jax.vmap( rlax.q_learning) )

    td_error = batched_loss(
        q_tm1=q_values, 
        a_tm1=actions, 
        r_t=rewards, 
        discount_t=(1 - dones) * DISCOUNT_GAMMA,
        q_t=target_q_values,
        # q_t_selector=selector_q_values, 
    )
    
    # Mask the td error 
    td_error = td_error * masks 

    loss = jnp.sum(rlax.l2_loss(td_error)) / jnp.sum(masks)
    # loss = jnp.mean(rlax.l2_loss(td_error))
    jax.debug.print("LOSS {x}", x=loss)
    
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: DQNSystemState, sampled_batch: DQNBufferData, global_step: int): 

    states = jnp.squeeze(sampled_batch.state)
    actions = jnp.squeeze(sampled_batch.action)
    rewards = jnp.squeeze(sampled_batch.reward)
    dones = jnp.squeeze(sampled_batch.done)
    next_states = jnp.squeeze(sampled_batch.next_state)
    # This squeeze could be wrong. 
    policy_hidden_states = jnp.squeeze(sampled_batch.policy_hidden_state, axis=(2, 3))
    masks = jnp.squeeze(sampled_batch.mask)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params

    # NB here. TARGET_UPDATE_PERIOD must be divisble by TRAIN_EVERY. 
    target_policy_params = optax.periodic_update(
            policy_params, target_policy_params, system_state.training_iterations, TARGET_UPDATE_PERIOD
        )
    
    grads = jax.grad(dqn_loss)(
        policy_params, 
        states, 
        actions, 
        rewards, 
        dones, 
        next_states, 
        policy_hidden_states, 
        masks, 
        target_policy_params, 
    )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params
    system_state.network_params.target_policy_params = target_policy_params

    system_state.training_iterations += 1

    return system_state

global_step = 0.0
episode = 0 
while global_step < 200_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
    episode_steps = 0 
    start_time = time.time()
    policy_hidden_state = create_hidden_states()
    while not done: 

        q_values, new_policy_hidden_state = policy_network.apply(system_state.network_params.policy_params, obs, policy_hidden_state)

        new_policy_hidden_state = jnp.expand_dims(new_policy_hidden_state[0], axis=0)
        if should_train(system_state.buffer): 
            EPSILON = jnp.maximum(EPSILON * EPSILON_DECAY_RATE, 0.05)
            # EPSILON = epsilon_schedule(global_step)

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
            policy_hidden_state = add_two_leading_dims(policy_hidden_state)
        )

        obs = obs_ 
        policy_hidden_state = new_policy_hidden_state

        buffer_state = system_state.buffer 
        buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        episode_return += reward
        episode_steps += 1
         
        if should_train(system_state.buffer) and (global_step % TRAIN_EVERY == 0): 

            buffer_state = system_state.buffer
            buffer_state, sampled_data = sample_batch(buffer_state)
            system_state.buffer = buffer_state
            system_state = update_policy(system_state, sampled_data, global_step)

    
    sps = episode_steps / (time.time() - start_time)
    episode += 1
    if episode % 1 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}, EPSILON: {jnp.round(EPSILON, 2)}, SPS: {sps}")   
