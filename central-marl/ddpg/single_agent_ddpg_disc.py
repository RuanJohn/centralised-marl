"""Single-agent JAX DDPG with continuous actions. 
"""

import jax.numpy as jnp 
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

from wrappers.ma_gym_wrapper import CentralControllerWrapper

import gym

# Constants: 
MAX_REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 1_000
BATCH_SIZE = 100
TRAIN_EVERY = 50
POLYAK_UPDATE_VALUE = 0.005
POLICY_LR = 0.001
CRITIC_LR = 0.001
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
SOFTMAX_TEMP = 1.0
# ENV_NAME = "ma_gym:Switch2-v0"
ENV_NAME = "CartPole-v0"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(
    ENV_NAME,     
)
# env = CentralControllerWrapper(env)

observation_dim = env.observation_space.shape[0]

# Discrete action spaces 
num_actions = env.action_space.n

# Make networks 

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = [64, 64],
    critic_layer_sizes: list = [64, 64]):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x):

        return hk.nets.MLP(policy_layer_sizes + [num_actions])(x) 

    @hk.without_apply_rng
    @hk.transform
    def critic_network(x):
        
        # NOTE: Might be a better way to do this. 
        # But concatenating outside the network for now. 
        return hk.nets.MLP(critic_layer_sizes + [1])(x) 

    return policy_network, critic_network 

policy_network, critic_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)

# TODO: Double check datatype here. 
dummy_action_data = jnp.ones(num_actions, dtype=jnp.int32)

networks_key, policy_init_key, critic_init_key = jax.random.split(networks_key, 3)

policy_params = policy_network.init(policy_init_key, dummy_obs_data)
critic_params = critic_network.init(
    critic_init_key, jnp.concatenate((dummy_obs_data, dummy_action_data)))

network_params = NetworkParams(
    policy_params=policy_params, 
    target_policy_params=policy_params,
    critic_params=critic_params, 
    target_critic_params=critic_params,  
)

# Create optimisers and states
policy_optimiser = optax.adam(POLICY_LR)
policy_optimiser_state = policy_optimiser.init(policy_params)

critic_optimiser = optax.adam(CRITIC_LR)
critic_optimiser_state = critic_optimiser.init(critic_params)

# Better idea is probably a high level Policy and Critic state. 

optimiser_states = OptimiserStates(
    policy_state=policy_optimiser_state, 
    critic_state=critic_optimiser_state, 
)

# Initialise buffer 
buffer_state = create_buffer(
    buffer_size=MAX_REPLAY_SIZE,
    min_buffer_size=MIN_REPLAY_SIZE, 
    batch_size=BATCH_SIZE, 
    num_agents=1, 
    num_envs=1, 
    observation_dim=observation_dim, 
    action_dim=num_actions, 
)

system_state = DQNSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
)

# @jax.jit
# @chex.assert_max_traces(n=1)
def select_action(
    logits, 
    actors_key,
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)
    gumbel_noise = jax.random.gumbel(sample_key, shape=(num_actions,))

    shifted_logits = logits + gumbel_noise

    hard_action = jax.nn.one_hot(
        jnp.argmax(shifted_logits), 
        num_classes=num_actions, 
    )

    soft_action = jax.nn.softmax(shifted_logits / SOFTMAX_TEMP)

    return actors_key, hard_action, soft_action

def select_action_train(
    logits, 
    ):

    shifted_logits = logits 

    hard_action = jax.nn.one_hot(
        jnp.argmax(shifted_logits), 
        num_classes=num_actions, 
    )

    soft_action = jax.nn.softmax(shifted_logits / SOFTMAX_TEMP)

    return hard_action, soft_action


# @jax.jit
# @chex.assert_max_traces(n=1)
def critic_loss(
    critic_params, 
    states, 
    actions, 
    rewards, 
    dones, 
    next_states, 
    target_critic_params, 
    target_policy_params, 
    networks_key, ):
    
    # Infer the batch size: 
    batch_size = states.shape[0]
    # Doesn't look like it should be clipped? 
    # NOTE: Look into this. 
    logits = jax.vmap(policy_network.apply, in_axes=(None, 0))(target_policy_params, next_states)
    train_keys = jax.random.split(networks_key, batch_size+1)
    networks_key = train_keys[0]
    train_keys = train_keys[1:]

    _, target_actions, _ = jax.vmap(select_action, in_axes=(0, 0))(logits, train_keys)
    target_state_actions = jnp.concatenate((next_states, target_actions), axis=1)
    target_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0))(target_critic_params, target_state_actions)
    target_action_values = jnp.squeeze(target_action_values)

    online_state_actions = jnp.concatenate((states, actions), axis=1)
    online_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, online_state_actions)
    online_action_values = jnp.squeeze(online_action_values)
    
    bellman_target = rewards + DISCOUNT_GAMMA * (1 - dones) * target_action_values
    bellman_target = jax.lax.stop_gradient(bellman_target)
    td_error = (online_action_values - bellman_target) 


    loss = jnp.mean(rlax.l2_loss(td_error))
    jax.debug.print("CRITIC LOSS: {x}", x=loss)

    return loss, networks_key

def policy_loss(
    policy_params, 
    states, 
    critic_params, 
    networks_key,):

    # Infer the batch size: 
    batch_size = states.shape[0]

    logits = jax.vmap(policy_network.apply, in_axes=(None, 0))(policy_params, states)
    train_keys = jax.random.split(networks_key, batch_size+1)
    networks_key = train_keys[0]
    train_keys = train_keys[1:]

    # Should there be gumbel noise here? 
    _, online_hard_actions, online_soft_actions = jax.vmap(select_action, in_axes=(0, 0))(logits, train_keys)
    
    train_actions = online_hard_actions -jax.lax.stop_gradient(online_soft_actions) + online_soft_actions
    online_state_actions = jnp.concatenate((states, train_actions), axis=1) 

    online_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, online_state_actions)
    online_action_values = jnp.squeeze(online_action_values)

    loss = -jnp.mean(online_action_values)

    jax.debug.print("POLICY LOSS: {x}", x=loss)
    return loss, networks_key


# @jax.jit
# @chex.assert_max_traces(n=1)
def update_critic(system_state: DQNSystemState, sampled_batch: DQNBufferData): 

    states = jnp.squeeze(sampled_batch.state)
    actions = jnp.squeeze(sampled_batch.action)
    rewards = jnp.squeeze(sampled_batch.reward)
    dones = jnp.squeeze(sampled_batch.done)
    next_states = jnp.squeeze(sampled_batch.next_state)
    
    critic_optimiser_state = system_state.optimiser_states.critic_state
    critic_params = system_state.network_params.critic_params
    target_critic_params = system_state.network_params.target_critic_params
    target_policy_params = system_state.network_params.target_policy_params
    networks_key = system_state.networks_key
    
    grads, networks_key = jax.grad(critic_loss, has_aux=True)(
        critic_params, 
        states, 
        actions, 
        rewards, 
        dones, 
        next_states, 
        target_critic_params,
        target_policy_params, 
        networks_key,  
    )
 
    target_critic_params = optax.incremental_update(
            critic_params, target_critic_params, POLYAK_UPDATE_VALUE, 
        )

    updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, updates)

    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    system_state.network_params.critic_params = new_critic_params
    system_state.network_params.target_critic_params = target_critic_params
    system_state.networks_key = networks_key

    return system_state

# @jax.jit
# @chex.assert_max_traces(n=1)
def update_policy(system_state: DQNSystemState, sampled_batch: DQNBufferData): 

    states = jnp.squeeze(sampled_batch.state)

    policy_optimiser_state = system_state.optimiser_states.policy_state
    critic_params = system_state.network_params.critic_params
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params
    networks_key = system_state.networks_key
    
    grads, networks_key = jax.grad(policy_loss, has_aux=True)(
        policy_params, 
        states, 
        critic_params,   
        networks_key, 
    )

    target_policy_params = optax.incremental_update(
            policy_params, target_policy_params, POLYAK_UPDATE_VALUE, 
        )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params
    system_state.network_params.target_policy_params = target_policy_params
    system_state.networks_key = networks_key

    return system_state

global_step = 0
episode = 0 
while global_step < 200_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
    while not done: 

        logits = policy_network.apply(system_state.network_params.policy_params, obs)

        actors_key = system_state.actors_key
        actors_key, action, _ = select_action(logits, actors_key)
        system_state.actors_key = actors_key
        
        # Get action from one_hot actions. 
        step_action = jnp.argmax(action)
        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_action.tolist())
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
            
            # Can do multiple updates here. 
            buffer_state = system_state.buffer
            buffer_state, sampled_data = sample_batch(buffer_state)
            system_state.buffer = buffer_state
            system_state = update_critic(system_state, sampled_data)
            system_state = update_policy(system_state, sampled_data)   
    
    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}")   