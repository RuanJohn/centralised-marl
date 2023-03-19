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
import time
import gymnax

from utils.types import (
    BufferData, 
    BufferState, 
    PPOSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.loggers import WandbLogger

from wrappers.ma_gym_wrapper import CentralControllerWrapper

import gym

# Constants: 
HORIZON = 200 
CLIP_EPSILON = 0.2 
POLICY_LR = 0.005
CRITIC_LR = 0.005
DISCOUNT_GAMMA = 0.99 
GAE_LAMBDA = 0.95
NUM_EPOCHS = 3
NUM_MINIBATCHES = 8 
MAX_GLOBAL_NORM = 0.5
ADAM_EPS = 1e-5
POLICY_LAYER_SIZES = [64, 64]
CRITIC_LAYER_SIZES = [64, 64]
NORMALISE_ADVANTAGE = True
# ENV_NAME = "ma_gym:PredatorPrey5x5-v0"
ENV_NAME = "CartPole-v1"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key, env_create_key, env_reset_key = jax.random.split(MASTER_PRNGKEY, 6)

ALGORITHM = "ff_central_ppo"
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
        },  
    )

# Instantiate the environment & its settings.
env, env_params = gymnax.make(ENV_NAME)
eval_env, eval_env_params = gymnax.make(ENV_NAME)

# Reset the environment.
obs, state = env.reset(env_reset_key, env_params)

observation_dim = env.observation_space(env_params).shape[0]
num_actions = env.num_actions

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
    def critic_nerwork(x):

        return hk.nets.MLP(critic_layer_sizes + [1])(x) 

    return policy_network, critic_nerwork

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

system_state = PPOSystemState(
    buffer=BufferState(
        states = None,
        actions = None,
        rewards = None,
        dones = None,
        log_probs = None,
        values = None,
        entropy = None,
        counter = None,
        key = None,
    ), 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

# @jax.jit
# @chex.assert_max_traces(n=1)
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
    entropies, 
    ):

    logits = policy_network.apply(policy_params, states)
    dist = distrax.Categorical(logits=logits)

    new_log_probs = dist.log_prob(value=actions)

    logratio = new_log_probs - old_log_probs
    ratio = jnp.exp(logratio)

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean() - 0.01 * jnp.mean(entropies)

    return loss

def critic_loss(
    critic_params, 
    states, 
    returns
    ):

    new_values = jnp.squeeze(critic_network.apply(critic_params, states))
    
    loss = 0.5 * ((new_values - returns) ** 2).mean()

    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: PPOSystemState, advantages, mb_idx): 

    states = jnp.squeeze(system_state.buffer.states)[jnp.array(mb_idx)]
    old_log_probs = jnp.squeeze(system_state.buffer.log_probs)[jnp.array(mb_idx)]
    actions = jnp.squeeze(system_state.buffer.actions)[jnp.array(mb_idx)]
    entropies = jnp.squeeze(system_state.buffer.entropy)[jnp.array(mb_idx)]
    advantages = advantages[jnp.array(mb_idx)]

    if NORMALISE_ADVANTAGE: 
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-5)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params

    grads = jax.grad(policy_loss)(
        policy_params, 
        states, 
        actions, 
        old_log_probs, 
        advantages,
        entropies, 
    )

    updates, new_policy_optimiser_state = policy_optimiser.update(grads, policy_optimiser_state)
    new_policy_params = optax.apply_updates(policy_params, updates)

    system_state.optimiser_states.policy_state = new_policy_optimiser_state
    system_state.network_params.policy_params = new_policy_params

    return system_state

@jax.jit
@chex.assert_max_traces(n=1)
def update_critic(
    system_state: PPOSystemState, 
    returns,
    mb_idx
): 

    states = jnp.squeeze(system_state.buffer.states)[jnp.array(mb_idx)]
    returns = returns[jnp.array(mb_idx)]
    
    critic_optimiser_state = system_state.optimiser_states.critic_state
    critic_params = system_state.network_params.critic_params

    grads = jax.grad(critic_loss)(
        critic_params, 
        states, 
        returns,
    )

    updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, updates)

    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    system_state.network_params.critic_params = new_critic_params

    return system_state


def minibatch_update(carry, mb_idx):

    system_state = carry[0]
    advantages = carry[1]
    returns = carry[2]

    system_state = update_policy(system_state, advantages, mb_idx)
    system_state = update_critic(system_state, returns, mb_idx)

    return (system_state, advantages, returns), mb_idx

def epoch_update(carry, none_in): 

    system_state = carry[0]
    advantages = carry[1]
    returns = carry[2]

    networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
    system_state.networks_key = networks_key

    idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
    mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)
    
    # Minibatch update 
    update_scan_out, _ = jax.lax.scan(
        f=minibatch_update, 
        init=(system_state, advantages, returns), 
        xs=mb_idxs, 
    )
    system_state = update_scan_out[0]

    return (system_state, advantages, returns), none_in

def rollout(rng_input, policy_params, critic_params, env_params, steps_in_episode, obs, state):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, policy_params, critic_params, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        logits = policy_network.apply(policy_params, obs)
        _, action, logprob, entropy = choose_action(logits, rng_net)
        value = jnp.squeeze(critic_network.apply(critic_params, obs))
        next_obs, next_state, reward, done, _ = env.step(
            rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, policy_params, critic_params, rng]
        return carry, [obs, action, reward, next_obs, done, logprob, entropy, value]

    # Scan over episode step loop
    carry, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, policy_params, critic_params, rng_episode],
        (),
        steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done, logprob, entropy, value = scan_out

    out_obs, out_state, rng = carry[0], carry[1], carry[4]
    return obs, action, reward, next_obs, done, logprob, entropy, value, out_obs, out_state, rng


global_step = 0
rollouts = 0
episode = 0 
log_data = {}
eval_key = jax.random.PRNGKey(100)
while rollouts < 1000: 

    episode_return = 0
    start_time = time.time()
    
    states, actions, rewards, _, dones, logprobs, entropies, values, obs, state, buffer_key = rollout(
        buffer_key, system_state.network_params.policy_params, system_state.network_params.critic_params, 
        env_params, HORIZON+1, obs, state, 
    )
    
    system_state.buffer.states = states
    system_state.buffer.actions = actions
    system_state.buffer.rewards = rewards
    system_state.buffer.dones = dones
    system_state.buffer.log_probs = logprobs
    system_state.buffer.entropy = entropies
    system_state.buffer.values = values
    
        
    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t = jnp.squeeze(system_state.buffer.rewards)[1:],
        discount_t = (1 - jnp.squeeze(system_state.buffer.dones))[1:] * DISCOUNT_GAMMA,
        lambda_ = GAE_LAMBDA, 
        values = jnp.squeeze(system_state.buffer.values),
        stop_target_gradients=True
    )

    advantages = jax.lax.stop_gradient(advantages)
    # Just not sure how to index the values here. 
    returns = advantages + jnp.squeeze(system_state.buffer.values)[1:]
    returns = jax.lax.stop_gradient(returns)

    epoch_scan_out, _ = jax.lax.scan(
        f=epoch_update, 
        init=(system_state, advantages, returns),
        xs=None,  
        length=NUM_EPOCHS, 
    )
    system_state = epoch_scan_out[0]
    
    sps = (HORIZON + 1) / (time.time() - start_time)

    rollouts += 1
    if rollouts % 10 == 0: 
        print(f"ROLLOUT: {rollouts}, SPS: {int(sps)},", end = " ")
        eval_key, eval_env_reset_key, eval_act_key = jax.random.split(eval_key, 3)
        eval_obs, eval_state = env.reset(eval_env_reset_key, eval_env_params)
        done = False
        returns = 0
        
        while not done: 
            logits = policy_network.apply(system_state.network_params.policy_params, eval_obs)
            eval_act_key, eval_sample_key, eval_step_key = jax.random.split(eval_act_key, 3)
            _, action, logprob, entropy = choose_action(logits, eval_sample_key)
            n_eval_obs, n_eval_state, reward, done, _ = env.step(eval_step_key, eval_state, action, eval_env_params)
            returns += reward

            eval_obs = n_eval_obs
            eval_state = n_eval_state

        print(f"EVAL RETURNS: {returns}")