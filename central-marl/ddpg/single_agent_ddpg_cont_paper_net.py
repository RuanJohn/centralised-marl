"""Single-agent JAX DDPG with continuous actions. 
   This implementation mirrors the network in the orginal DDPG 
   paper. 
"""

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import rlax
import chex
import time
from typing import Optional

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

from utils.loggers import WandbLogger

# Constants: 
MAX_REPLAY_SIZE = 1_000_000
MIN_REPLAY_SIZE = 1_000
BATCH_SIZE = 64
TRAIN_EVERY = 1
POLYAK_UPDATE_VALUE = 0.001
POLICY_LR = 0.000025
CRITIC_LR = 0.00025
ADAM_EPS=1e-5
DISCOUNT_GAMMA = 0.99 
MAX_GLOBAL_NORM = 0.5
POLICY_LAYER_SIZES = [100, 75]
CRITIC_LAYER_SIZES = [100, 75]
# ENV_NAME = "ma_gym:Switch2-v0"
ENV_NAME = "LunarLanderContinuous-v2"
ALGORITHM = "ddpg-continuous"

MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

LOG = False

env = gym.make(
    ENV_NAME,     
)

if LOG: 
    logger = WandbLogger(
        exp_config={
            "algorithm": ALGORITHM,
            "env_name": ENV_NAME,
            "max_replay_size": MAX_REPLAY_SIZE,
            "min_replay_size": MIN_REPLAY_SIZE, 
            "policy_lr": POLICY_LR, 
            "critic_lr": CRITIC_LR, 
            "gamma": DISCOUNT_GAMMA, 
            "batch_size": BATCH_SIZE, 
            "train_every": TRAIN_EVERY, 
            "polyak_value": POLYAK_UPDATE_VALUE,
            "add_noise_train": False,  
            "policy_layer_sizes": POLICY_LAYER_SIZES, 
            "critic_layer_sizes": CRITIC_LAYER_SIZES, 
            },  
    )
# env = CentralControllerWrapper(env)

observation_dim = env.observation_space.shape[0]

# Discrete action spaces 
# num_actions = env.action_space.n

# Continuous action space 
num_actions = env.action_space.shape[0]
action_space_low = env.action_space.low 
action_space_high = env.action_space.high


def polyak_update(target_params, online_params, tau=0.005): 

    new_target_params = jax.tree_util.tree_map(
      lambda target, online: tau * online + (1.0 - tau) * target,
      target_params, online_params)
    
    return new_target_params

# Make networks 

class DDPGCriticNetwork(hk.Module):

    def __init__(self, num_actions: int, layer_sizes: list = [200, 300], name: Optional[str]= None,):
        
        super().__init__(name=name)
        self.layer_sizes = layer_sizes

        # TODO: Do layer inits later. 
        self._fc1 = hk.Linear(
            layer_sizes[0], )
        
        self._bn1 = hk.LayerNorm(axis=0, create_offset=False, create_scale=False)
        self._bn2 = hk.LayerNorm(axis=0, create_offset=False, create_scale=False)

        self._fc2 = hk.Linear(
            layer_sizes[1], )
        
        self._action_value = hk.nets.MLP([num_actions, layer_sizes[1]], activate_final=True, )
        self.q_value = hk.nets.MLP([layer_sizes[1], 1])

    def __call__(self, state, action): 

        state_value = self._fc1(state)
        state_value = self._bn1(state_value)
        state_value = jax.nn.relu(state_value)
        state_value = self._fc2(state_value)
        state_value = self._bn2(state_value)

        action_value = self._action_value(action)
        state_action_value = jax.nn.relu(jnp.add(state_value, action_value))
        state_action_value = self.q_value(state_action_value)

        return state_action_value

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = POLICY_LAYER_SIZES,
    critic_layer_sizes: list = CRITIC_LAYER_SIZES):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x):

        return hk.Sequential([
            hk.nets.MLP(policy_layer_sizes + [num_actions]), jax.nn.tanh
            ])(x) 

    @hk.without_apply_rng
    @hk.transform
    def critic_network(state, action):
        
        # NOTE: Might be a better way to do this. 
        # But concatenating outside the network for now. 
        return DDPGCriticNetwork(num_actions=num_actions, layer_sizes=critic_layer_sizes)(state, action) 

    return policy_network, critic_network 

policy_network, critic_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)

# NOTE: For now actions are continuos. 
dummy_action_data = jnp.ones(num_actions, dtype=jnp.float32)

networks_key, policy_init_key, critic_init_key = jax.random.split(networks_key, 3)

policy_params = policy_network.init(policy_init_key, dummy_obs_data)
critic_params = critic_network.init(
    critic_init_key, dummy_obs_data, dummy_action_data)

network_params = NetworkParams(
    policy_params=policy_params, 
    target_policy_params=policy_params,
    critic_params=critic_params, 
    target_critic_params=critic_params,  
)

policy_optimiser = optax.adam(POLICY_LR)
critic_optimiser = optax.adam(CRITIC_LR)

policy_optimiser_state = policy_optimiser.init(policy_params)
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
    action_dtype=jnp.float32,
)

system_state = DQNSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
)

def add_action_noise(
    action, 
    actors_key,
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)
    exploration_noise = jax.random.normal(sample_key, shape=(num_actions,))

    action = jnp.clip(
        action + exploration_noise, 
        a_min=action_space_low, 
        a_max=action_space_high,)

    return actors_key, action

def critic_loss(
    critic_params, 
    states, 
    actions, 
    rewards, 
    dones, 
    next_states, 
    target_critic_params, 
    target_policy_params, ):
    
    # Add noise during training?

    target_actions = jax.vmap(policy_network.apply, in_axes=(None, 0))(target_policy_params, next_states)

    target_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0, 0))(target_critic_params, states, target_actions)
    target_action_values = jnp.squeeze(target_action_values)

    # TODO: Make this an if statement 
    # or handle in a better way. 
    # Optional action expand for action dim = 1: 
    # actions = jnp.expand_dims(actions, axis=1)

    online_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0, 0))(critic_params, states, actions)
    online_action_values = jnp.squeeze(online_action_values)
    
    bellman_target = rewards + DISCOUNT_GAMMA * (1 - dones) * target_action_values
    bellman_target = jax.lax.stop_gradient(bellman_target)
    td_error = (online_action_values - bellman_target) 

    loss = jnp.mean(rlax.l2_loss(td_error))
    
    return loss

def policy_loss(
    policy_params, 
    states, 
    critic_params, ):

    online_actions = jax.vmap(policy_network.apply, in_axes=(None, 0))(policy_params, states) 

    online_action_values = jax.vmap(critic_network.apply, in_axes=(None, 0, 0))(critic_params, states, online_actions)
    online_action_values = jnp.squeeze(online_action_values)

    loss = -jnp.mean(online_action_values)

    return loss


@jax.jit
@chex.assert_max_traces(n=1)
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
    
    grads = jax.grad(critic_loss)(
        critic_params, 
        states, 
        actions, 
        rewards, 
        dones, 
        next_states, 
        target_critic_params,
        target_policy_params,  
    )

    updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, updates)

    # Update params, then polyak average. 
    # target_critic_params = optax.incremental_update(
    #         new_critic_params, target_critic_params, POLYAK_UPDATE_VALUE, 
    #     )

    target_critic_params = polyak_update(
        target_params=target_critic_params, 
        online_params=new_critic_params, 
        tau=POLYAK_UPDATE_VALUE)

    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    system_state.network_params.critic_params = new_critic_params
    system_state.network_params.target_critic_params = target_critic_params

    return system_state

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: DQNSystemState, sampled_batch: DQNBufferData): 

    states = jnp.squeeze(sampled_batch.state)

    policy_optimiser_state = system_state.optimiser_states.policy_state
    critic_params = system_state.network_params.critic_params
    policy_params = system_state.network_params.policy_params
    target_policy_params = system_state.network_params.target_policy_params
    
    grads = jax.grad(policy_loss)(
        policy_params, 
        states, 
        critic_params,   
    )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    # target_policy_params = optax.incremental_update(
    #         new_policy_params, target_policy_params, POLYAK_UPDATE_VALUE, 
    #     )

    target_policy_params = polyak_update(
        target_params=target_policy_params, 
        online_params=new_policy_params, 
        tau=POLYAK_UPDATE_VALUE)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params
    system_state.network_params.target_policy_params = target_policy_params

    return system_state

@jax.jit
@chex.assert_max_traces(n=1)
def update(system_state: DQNSystemState, sampled_batch: DQNBufferData): 

    # Data
    states = jnp.squeeze(sampled_batch.state)
    actions = jnp.squeeze(sampled_batch.action)
    rewards = jnp.squeeze(sampled_batch.reward)
    dones = jnp.squeeze(sampled_batch.done)
    next_states = jnp.squeeze(sampled_batch.next_state)

    # Current params
    critic_params = system_state.network_params.critic_params
    policy_params = system_state.network_params.policy_params
    target_critic_params = system_state.network_params.target_critic_params
    target_policy_params = system_state.network_params.target_policy_params

    policy_optimiser_state = system_state.optimiser_states.policy_state
    critic_optimiser_state = system_state.optimiser_states.critic_state

    # Update the critic
    critic_grads = jax.grad(critic_loss)(
        critic_params, 
        states, 
        actions, 
        rewards, 
        dones, 
        next_states, 
        target_critic_params,
        target_policy_params,
    )

    critic_updates, new_critic_optimiser_state = critic_optimiser.update(critic_grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, critic_updates)

    # Update the policy
    policy_grads = jax.grad(policy_loss)(
        policy_params, 
        states, 
        critic_params,   
    )

    policy_updates, new_policy_optimiser_state = policy_optimiser.update(policy_grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, policy_updates)

    # Update target nets
    target_critic_params = polyak_update(
        target_params=target_critic_params, 
        online_params=new_critic_params, 
        tau=POLYAK_UPDATE_VALUE)
    
    target_policy_params = polyak_update(
        target_params=target_policy_params, 
        online_params=new_policy_params, 
        tau=POLYAK_UPDATE_VALUE)
    
    # Set new parameters in system state 
    system_state.network_params.critic_params = new_critic_params
    system_state.network_params.policy_params = new_policy_params
    system_state.network_params.target_critic_params = target_critic_params
    system_state.network_params.target_policy_params = target_policy_params

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    
    return system_state

@jax.jit
@chex.assert_max_traces(n=1)
def actor_step(obs, policy_params, actors_key): 

    action = policy_network.apply(policy_params, obs)
    actors_key, action = add_action_noise(action, actors_key)

    return action, actors_key


global_step = 0
episode = 0 
while global_step < 500_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
    episode_steps = 0 
    start_time = time.time()
    while not done: 

        action, actors_key = actor_step(
            obs=obs,
            policy_params=system_state.network_params.policy_params, 
            actors_key=system_state.actors_key 
        )

        system_state.actors_key = actors_key

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(action.tolist())
        global_step += 1 # TODO: With vec envs this should be more. 
        episode_steps += 1

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

            # TODO: Set an if statement to handle separate updates or one update 
            # function. 
            # system_state = update_critic(system_state, sampled_data)
            # system_state = update_policy(system_state, sampled_data)   
            system_state = update(system_state, sampled_data)
    
    steps_per_second = episode_steps / (time.time() - start_time)

    episode_results = {
        "episode": episode, 
        "episode_return": episode_return,
        "global_step": global_step,
        "steps_per_second": steps_per_second, 
    }

    if LOG: 
        logger.write(logging_details=episode_results, step=global_step)
    episode += 1
    print(f"STEPS PER SECOND: {steps_per_second}")
    if episode % 1 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}")   

logger.close()