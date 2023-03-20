"""Independent multi-agent JAX DQN."""

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import rlax
import chex

from utils.types import (
    DQNBufferData, 
    DQNBufferState, 
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

from utils.array_utils import (
    add_two_leading_dims,
)

add = jax.jit(add, donate_argnums=(0))

from wrappers.ma_gym_wrapper import CentralControllerWrapper

import gym

# Constants: 
MAX_REPLAY_SIZE = 500_0
MIN_REPLAY_SIZE = 100 # 1000
BATCH_SIZE = 64
SEQUENCE_LENGTH = 20
TARGET_UPDATE_PERIOD = 100
TRAIN_EVERY = 50
POLICY_LR = 0.005
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
EPSILON = 1.0 
MIN_EPSILON = 0.05 
EPSILON_DECAY_STEPS = 10_000
EPSILON_DECAY_RATE = 0.9999
POLICY_LAYER_SIZES = [32]
POLICY_RECURRENT_LAYER_SIZES = [64]
ENV_NAME = "ma_gym:Checkers-v0"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(ENV_NAME)

# TODO: Assuming fully homogeneous agents here. 
# Handle this later on to be per agent. 
observation_dim = env.observation_space[0].shape[0]
num_actions = env.action_space[0].n
num_agents = env.n_agents

# NOTE: Should each agent have a unique hidden state? 

# Make networks 

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

policy_params = policy_network.init(
    policy_init_key, 
    dummy_obs_data, 
    policy_hidden_state)

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
    num_agents=num_agents, 
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
    training_iterations=jnp.int32(0), 
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

def select_q_values(q_value, action, target_q_value): 

    chex.assert_rank([q_value, action, target_q_value], [1, 0, 1])
    chex.assert_type([q_value, action, target_q_value],
                    [float, int, float])
    
    return q_value[action], jnp.max(target_q_value)

def select_double_q_values(q_value, action, target_q_value, selector_q_value): 

    chex.assert_rank([q_value, action, target_q_value, selector_q_value], [1, 0, 1, 1])
    chex.assert_type([q_value, action, target_q_value],
                    [float, int, float])
    
    return q_value[action], target_q_value[jnp.argmax(selector_q_value)]

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

    q_out = jnp.empty((BATCH_SIZE, SEQUENCE_LENGTH, num_agents, num_actions), dtype=jnp.float32)
    q_next_out =  jnp.empty((BATCH_SIZE, SEQUENCE_LENGTH, num_agents, num_actions), dtype=jnp.float32)
    selector_q_out =  jnp.empty((BATCH_SIZE, SEQUENCE_LENGTH, num_agents, num_actions), dtype=jnp.float32)

    def core_online(state, hidden_state): 

        return policy_network.apply(policy_params, state, hidden_state)
    
    def core_target(state, hidden_state): 

        return policy_network.apply(target_policy_params, state, hidden_state)


    for agent_idx in range(num_agents):

        agent_policy_hidden_states = jnp.zeros_like(policy_hidden_states[:, 0, agent_idx, :, :])
        agent_states = states[:, :, agent_idx, :]
        agent_next_states = next_states[:, :, agent_idx, :]

        q_values, _ = hk.static_unroll(
        core_online, 
        agent_states, 
        agent_policy_hidden_states, 
        time_major=False, 
        )

        target_q_values, _ = hk.static_unroll(
            core_target, 
            agent_next_states, 
            agent_policy_hidden_states, 
            time_major=False, 
        )

        selector_q_values, _ = hk.static_unroll(
            core_online, 
            agent_next_states, 
            agent_policy_hidden_states, 
            time_major=False, 
        )

        q_out = q_out.at[:, :, agent_idx].set(q_values)
        q_next_out = q_next_out.at[:, :, agent_idx].set(target_q_values)
        selector_q_out = selector_q_out.at[:, :, agent_idx].set(selector_q_values)

    # Can also just use rlax here. 
    # batched_select_q_values = jax.vmap(jax.vmap(jax.vmap(select_q_values)))
    # q_out, q_next_out = batched_select_q_values(q_out, actions, q_next_out)

    batched_select_q_values = jax.vmap(jax.vmap(jax.vmap(select_double_q_values)))
    q_out, q_next_out = batched_select_q_values(q_out, actions, q_next_out, selector_q_out)


    # Doing VDN mixing here. 
    q_out = jnp.sum(q_out, axis=-1)
    rewards = jnp.sum(rewards, axis=-1)
    q_next_out = jnp.sum(q_next_out, axis=-1)
    # dones = jnp.all(dones, axis=-1)

    # TODO: fix this. 
    # Just selecting the first agent's done and masks. 
    # There is definitely a better way to do this. 
    dones = dones[:, :, 0]
    masks = masks[:, :, 0]

    target = rewards + (1 - dones) * q_next_out * DISCOUNT_GAMMA
    target = jax.lax.stop_gradient(target)

    td_error = (q_out - target) ** 2

    # Mask the td error 
    td_error = td_error * masks 
    loss = jnp.sum(rlax.l2_loss(td_error)) / jnp.sum(masks)
    
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(
    system_state: DQNSystemState, 
    sampled_batch: DQNBufferData, ): 

    
    states = jnp.squeeze(sampled_batch.state[:,:,:,:])
    actions = jnp.squeeze(sampled_batch.action[:,:,:,:])
    rewards = jnp.squeeze(sampled_batch.reward[:,:,:])
    dones = jnp.squeeze(sampled_batch.done[:,:,:])
    next_states = jnp.squeeze(sampled_batch.next_state[:,:,:,:])
    policy_hidden_states = jnp.squeeze(sampled_batch.policy_hidden_state, axis=2)
    masks = jnp.squeeze(sampled_batch.mask)

    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params

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

global_step = 0
episode = 0 
while global_step < 500_000: 

    team_done = False 
    obs = env.reset()
    episode_return = 0
    policy_hidden_state = create_hidden_states()
    while not team_done: 

        if should_train(system_state.buffer): 
            EPSILON = jnp.maximum(EPSILON * EPSILON_DECAY_RATE, 0.05)

        # For stepping the environment
        step_joint_action = jnp.empty(num_agents, dtype=jnp.int32)

        # Data to append to buffer
        act_joint_action = jnp.empty((num_agents,1), dtype=jnp.int32) 

        for agent in range(num_agents):
            q_values, new_policy_hidden_state = policy_network.apply(
                system_state.network_params.policy_params, 
                jnp.array(obs[agent], dtype=jnp.float32), 
                policy_hidden_state)

            new_policy_hidden_state = jnp.expand_dims(new_policy_hidden_state[0], axis=0)
            actors_key = system_state.actors_key
            actors_key, action = choose_action(actors_key, q_values, EPSILON)
            system_state.actors_key = actors_key

            step_joint_action = step_joint_action.at[agent].set(action)

            act_joint_action = act_joint_action.at[agent, 0].set(action)

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_joint_action.tolist())
        team_done = all(done)
        global_step += 1 # TODO: With vec envs this should be more. 

        # NB: Correct shapes here. 
        data = DQNBufferData(
            state = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0),  
            action = jnp.expand_dims(act_joint_action, axis=0), 
            reward = jnp.expand_dims(jnp.array(reward, dtype=jnp.float32), axis=0), 
            done = jnp.expand_dims(jnp.array(done, dtype=bool), axis=0), 
            next_state = jnp.expand_dims(jnp.array(obs_, dtype=jnp.float32), axis=0), 
            policy_hidden_state = jnp.expand_dims(jnp.broadcast_to(policy_hidden_state, (num_agents, *policy_hidden_state.shape)), axis=0)
        )

        obs = obs_ 
        policy_hidden_state = new_policy_hidden_state

        buffer_state = system_state.buffer 
        buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        episode_return += jnp.sum(jnp.array(reward, dtype=jnp.float32))
        
        if should_train(system_state.buffer) and (global_step % TRAIN_EVERY == 0): 
            
            buffer_state = system_state.buffer
            buffer_state, sampled_data = sample_batch(buffer_state)
            system_state.buffer = buffer_state
            system_state = update_policy(system_state, sampled_data)

    
    episode += 1
    if episode % 1 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}, EPSILON: {jnp.round(EPSILON, 2)}")   
