"""Independent multi-agent JAX PPO. NOte that this implementation 
uses shared network weights between all agetns."""

import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import distrax
import rlax
import chex
import time 

from utils.types import (
    BufferData, 
    PPOSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.replay_buffer import (
    create_buffer, 
    add, 
    reset_buffer,
)

from utils.loggers import WandbLogger
from wrappers.agent_id_wrapper import AgentIDWrapper

import gym

jit_add = jax.jit(add)

# Constants: 
HORIZON = 104 * 8 
CLIP_EPSILON = 0.2 
POLICY_LR = 0.005
CRITIC_LR = 0.005
DISCOUNT_GAMMA = 0.99 
GAE_LAMBDA = 0.95
NUM_EPOCHS = 3
NUM_MINIBATCHES = 8 * 8 
MAX_GLOBAL_NORM = 0.5
ADAM_EPS = 1e-5
POLICY_LAYER_SIZES = [64, 64]
CRITIC_LAYER_SIZES = [64, 64]

# TODO: Add agent IDS. 
ADD_ONE_HOT_IDS = True
ENV_NAME = "ma_gym:Switch2-v0"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

NORMALISE_ADVANTAGE = True
ADD_ENTROPY_LOSS = True

ALGORITHM = "ff_ippo_batched"
LOG = False

if LOG: 
    logger = WandbLogger(
        exp_config={
        "algorithm": ALGORITHM,
        "env_name": ENV_NAME,
        "horizon": HORIZON, 
        "clip_epsilon": CLIP_EPSILON, 
        "policy_lr": POLICY_LR, 
        "critic_lr": CRITIC_LR, 
        "gamma": DISCOUNT_GAMMA, 
        "gae_lambda": GAE_LAMBDA, 
        "num_epochs": NUM_EPOCHS, 
        "num_minibatches": NUM_MINIBATCHES,
        "max_global_norm": MAX_GLOBAL_NORM,
        "adam_epsilon": ADAM_EPS, 
        "policy_layer_sizes": POLICY_LAYER_SIZES, 
        "critic_layer_sizes": CRITIC_LAYER_SIZES, 
        "normalise_advantage": NORMALISE_ADVANTAGE,
        "add_entropy_loss": ADD_ENTROPY_LOSS, 
        },  
    )

env = gym.make(ENV_NAME)
if ADD_ONE_HOT_IDS: 
    env = AgentIDWrapper(env)

# TODO: Assuming fully homogeneous agents here. 
# Handle this later on to be per agent. 

num_actions = env.action_space[0].n
num_agents = env.n_agents
observation_dim = env.observation_space[0].shape[0]

if ADD_ONE_HOT_IDS: 
    observation_dim += num_agents

# Make networks 
def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = POLICY_LAYER_SIZES, 
    critic_layer_sizes: list = CRITIC_LAYER_SIZES, ):

    @hk.without_apply_rng
    @hk.transform
    def policy_network(x):

        return hk.nets.MLP(policy_layer_sizes + [num_actions])(x)

    @hk.without_apply_rng
    @hk.transform
    def critic_network(x):

        return hk.nets.MLP(critic_layer_sizes + [1])(x) 

    return policy_network, critic_network

policy_network, critic_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)
networks_key, policy_init_key, critic_init_key = jax.random.split(networks_key, 3)

policy_params = policy_network.init(policy_init_key, dummy_obs_data)
critic_params = critic_network.init(critic_init_key, dummy_obs_data)

network_params = NetworkParams(
    policy_params=policy_params, 
    critic_params=critic_params,
)

# Create optimisers and states
policy_optimiser = optax.chain(
      optax.clip_by_global_norm(MAX_GLOBAL_NORM),
      optax.adam(learning_rate = POLICY_LR, eps = ADAM_EPS),
    )
critic_optimiser = optax.chain(
    optax.clip_by_global_norm(MAX_GLOBAL_NORM),
    optax.adam(learning_rate = CRITIC_LR, eps = ADAM_EPS),
    )


policy_optimiser_state = policy_optimiser.init(policy_params)
critic_optimiser_state = critic_optimiser.init(critic_params)

# Better idea is probably a high level Policy and Critic state. 

optimiser_states = OptimiserStates(
    policy_state=policy_optimiser_state, 
    critic_state=critic_optimiser_state, 
)

# Initialise buffer 
buffer_state = create_buffer(
    buffer_size=HORIZON, 
    num_agents=num_agents, 
    num_envs=1, 
    observation_dim=observation_dim, 
)

system_state = PPOSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

@jax.jit
@chex.assert_max_traces(n=1)
def choose_action(
    logits,  
    actors_key,
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)

    dist = distrax.Categorical(logits=logits)

    action, logprob = dist.sample_and_log_prob(
        seed = sample_key, 
    )
    entropy = dist.entropy()

    return actors_key, action, logprob, entropy

def policy_loss(
    policy_params, 
    states, 
    actions, 
    old_log_probs, 
    advantages, 
    entropies_):

    logits = policy_network.apply(policy_params, states)
    dist = distrax.Categorical(logits=logits)

    new_log_probs = dist.log_prob(value=actions)

    logratio = new_log_probs - old_log_probs
    ratio = jnp.exp(logratio)

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean() 
    if ADD_ENTROPY_LOSS: 
        loss -= 0.01 * jnp.mean(entropies_)

    return loss

def critic_loss(
    critic_params, 
    states, 
    returns
    ):

    new_values = jnp.squeeze(critic_network.apply(critic_params, states))
    
    loss = 0.5 * ((new_values - returns) ** 2).mean()
    return loss

batched_policy_loss = jax.vmap(policy_loss, in_axes=(None, 1, 1, 1, 1, 1))
def full_policy_loss(policy_params, 
    states, 
    actions, 
    old_log_probs, 
    advantages, 
    entropies_):

    return jnp.mean(batched_policy_loss(policy_params, 
    states, 
    actions, 
    old_log_probs, 
    advantages, 
    entropies_))

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: PPOSystemState, advantages, mb_idx): 

    states_ = jnp.squeeze(system_state.buffer.states[:,:,:,:])[jnp.array(mb_idx)]
    old_log_probs_ = jnp.squeeze(system_state.buffer.log_probs[:,:,:])[jnp.array(mb_idx)]
    actions_ = jnp.squeeze(system_state.buffer.actions[:,:,:,:])[jnp.array(mb_idx)]
    entropies_ = jnp.squeeze(system_state.buffer.entropy[:,:,:])[jnp.array(mb_idx)]
    advantages_ = advantages[:, :][jnp.array(mb_idx)]

    if NORMALISE_ADVANTAGE: 
        advantages_ = (advantages_ - jnp.mean(advantages_)) / (jnp.std(advantages_) + 1e-5)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params

    grads = jax.grad(full_policy_loss)(
        policy_params, 
        states_, 
        actions_, 
        old_log_probs_, 
        advantages_,
        entropies_,
    )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params

    return system_state

batched_critic_loss = jax.vmap(critic_loss, in_axes=(None, 1, 1))
def full_critic_loss(critic_params, states, returns):

    return jnp.mean(batched_critic_loss(critic_params, states, returns))

@jax.jit
@chex.assert_max_traces(n=1)
def update_critic(
    system_state: PPOSystemState, 
    returns,
    mb_idx,
): 

    states_ = jnp.squeeze(system_state.buffer.states[:,:,:,:])[jnp.array(mb_idx)]
    returns_ = returns[:, :][jnp.array(mb_idx)]
    
    critic_optimiser_state = system_state.optimiser_states.critic_state
    critic_params = system_state.network_params.critic_params

    # Mean over agents for loss
    grads = jax.grad(full_critic_loss)(
        critic_params, 
        states_, 
        returns_,
    )

    updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, updates)

    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    system_state.network_params.critic_params = new_critic_params

    return system_state

def minibatch_update(carry, mb_idx):

    system_state, advantages, returns = carry

    system_state = update_policy(system_state, advantages, mb_idx)
    system_state = update_critic(system_state, returns, mb_idx)

    return (system_state, advantages, returns), mb_idx

@jax.jit
@chex.assert_max_traces(n=1)
def epoch_update(carry, none_in): 

    system_state, advantages, returns = carry

    system_state.networks_key, sample_idx_key = jax.random.split(system_state.networks_key)

    idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
    mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)
    
    # Minibatch update 
    (system_state, _, _), _ = jax.lax.scan(
        f=minibatch_update, 
        init=(system_state, advantages, returns), 
        xs=mb_idxs,
    )

    return (system_state, advantages, returns), none_in

global_step = 0
episode = 0 
log_data = {}
batched_policy_apply = jax.vmap(policy_network.apply, in_axes=(None, 0))
batched_choose_action = jax.vmap(choose_action, in_axes=(0, 0))
batched_critic_apply = jax.vmap(critic_network.apply, in_axes=(None, 0))
batched_gae = jax.vmap(rlax.truncated_generalized_advantage_estimation, in_axes=(1, 1, None, 1, None), out_axes=1)
while global_step < 200_000: 

    team_done = False 
    obs = env.reset()
    obs = jnp.array(obs, dtype=jnp.float32) 
    episode_return = 0
    episode_step = 0 
    start_time = time.time()
    while not team_done: 
        
        logits = batched_policy_apply(system_state.network_params.policy_params, obs)
        keys = jax.random.split(system_state.actors_key, num_agents+1)
        system_state.actors_key = keys[0]
        actor_keys = keys[1:]
        _, step_joint_action, act_log_probs, act_entropies = batched_choose_action(logits, actor_keys)
        act_joint_action = jnp.expand_dims(step_joint_action, axis=1)
        act_values = jnp.squeeze(batched_critic_apply(system_state.network_params.critic_params, obs))

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_joint_action.tolist())  
        obs_ = jnp.array(obs_, dtype=jnp.float32)     

        team_done = all(done)
        global_step += 1 # TODO: With vec envs this should be more. 
        
        # NB: Correct shapes here. 
        data = BufferData(
            state = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0), 
            action = jnp.expand_dims(act_joint_action, axis=0), 
            reward = jnp.expand_dims(jnp.array(reward, dtype=jnp.float32), axis=0), 
            done = jnp.expand_dims(jnp.array(done, dtype=bool), axis=0), 
            log_prob = jnp.expand_dims(act_log_probs, axis=0), 
            value = jnp.expand_dims(act_values, axis=0), 
            entropy = jnp.expand_dims(act_entropies, axis=0)
        )
        system_state.buffer = jit_add(system_state.buffer, data)

        obs = obs_ 
        episode_return += jnp.sum(jnp.array(reward, dtype=jnp.float32))
        episode_step += 1 
        
        if global_step % (HORIZON + 1) == 0: 
            
            rewards_batch = jnp.squeeze(system_state.buffer.rewards)[:-1]
            dones_batch = jnp.squeeze(system_state.buffer.dones)[:-1]
            values_batch = jnp.squeeze(system_state.buffer.values)

            advantages = batched_gae(
                rewards_batch, 
                (1 - dones_batch) * DISCOUNT_GAMMA,
                GAE_LAMBDA, 
                values_batch,
                True, 
            )
            returns = advantages + values_batch[:-1]

            advantages = jax.lax.stop_gradient(advantages)
            returns = jax.lax.stop_gradient(returns)

            (system_state, _, _), _ = jax.lax.scan(
                f=epoch_update, 
                init=(system_state, advantages, returns),
                xs=None,  
                length=NUM_EPOCHS, 
            )   

            system_state.buffer = reset_buffer(buffer_state) 

    sps = episode_step / (time.time() - start_time)

    if LOG: 
        log_data["episode"] = episode
        log_data["episode_return"] = episode_return
        log_data["global_step"] = global_step
        log_data["sps"] = sps
        logger.write(logging_details=log_data)

    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {jnp.round(episode_return, 3)}, SPS: {int(sps)}")   

if LOG: 
    logger.close()  