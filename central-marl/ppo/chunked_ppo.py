"""Multi-agent 'chunked' JAX PPO. This is somewhere between 
    value decomposition and fully centralised MARL. 
"""

import jax.numpy as jnp 
import numpy as np 
import jax 
import haiku as hk
import optax
import distrax
import rlax
import chex
import time 

from utils.types import (
    BufferData, 
    BufferState, 
    PPOSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.chunked_replay_buffer import (
    create_buffer, 
    add, 
    reset_buffer,
    should_train,
)

# add = jax.jit(chex.assert_max_traces(add, n=1), donate_argnums=(0, ))
# add = jax.jit(chex.assert_max_traces(add, n=1))
add = jax.jit(add)

from utils.array_utils import (
    add_two_leading_dims,
)

from utils.loggers import WandbLogger

from wrappers.ma_gym_wrapper import CentralChunkedControllerWrapper

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
ENV_NAME = "ma_gym:Switch4-v0"
# ENV_NAME = "CartPole-v0"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

ALGORITHM = "ff_chunked_ppo"
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
        },  
    )

env = gym.make(ENV_NAME)

# Uncomment for centralised marl envs. 
env = CentralChunkedControllerWrapper(env)

num_agents = env.num_agents
observation_dim = env.observation_space.shape[0]
# Num actions must now be the sum over all agent 
# obs dims. 
num_actions = np.sum(env.action_space.nvec)

# This will be used to split the network output logits for 
# action selection.
action_chunk_dims = env.action_map

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

# Initialise buffer 
buffer_state = create_buffer(
    buffer_size=HORIZON, 
    num_agents=1, 
    num_envs=1, 
    action_dim=num_agents, 
    observation_dim=observation_dim, 
)

system_state = PPOSystemState(
    buffer=buffer_state, 
    actors_key=actors_key, 
    networks_key=networks_key, 
    network_params=network_params, 
    optimiser_states=optimiser_states, 
) 

def reshape_logits(logits): 

    reshaped_logits = jnp.array(
        jnp.split(
        logits, 
        indices_or_sections=action_chunk_dims[1:], 
        axis=0)
    )

    return reshaped_logits

@jax.jit
@chex.assert_max_traces(n=1)
def choose_action(
    logits,  
    actors_key,
    ):
    
    actors_key, sample_key = jax.random.split(actors_key)

    # Need to split the logits here. 
    logits = reshape_logits(logits)

    dist = distrax.Categorical(logits=logits)

    # TODO: shapes are correct here. 
    # worth checking the distribution is correctly sampled. 
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
    advantages, ):

    logits = policy_network.apply(policy_params, states)

    # Reshape the logits. 
    logits = jax.vmap(reshape_logits)(logits)

    dist = distrax.Categorical(logits=logits)

    new_log_probs = dist.log_prob(value=actions) 

    logratio = new_log_probs - old_log_probs

    # There is a problem that the logits and actions are 
    # now computed per agent while the value, reward and advantage are 
    # computed for the team. There are 3 ways around this. 
    # 1. Mean the log ratios over the agents to make it 
    #   the same dim as the advantages.  
    # 2. Keep the global advantages but give each agent the same 
    #   advantage. 
    # 3. Store per agent values and rewards and then compute 
    #   per agent advantages.

    # Option 1. 
    # logratio = jnp.mean(logratio, axis = 1)

    ratio = jnp.exp(logratio)

    # Option 2.
    # TODO: Make this better.  
    advantages = jnp.stack([advantages] * num_agents).T

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean()

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
    advantages = advantages[jnp.array(mb_idx)]
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params

    grads = jax.grad(policy_loss)(
        policy_params, 
        states, 
        actions, 
        old_log_probs, 
        advantages,
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

global_step = 0
episode = 0
log_data = {}
while global_step < 250_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
    episode_step = 0 
    start_time = time.time()
    while not done: 
        
        logits = policy_network.apply(system_state.network_params.policy_params, obs)
        
        actors_key = system_state.actors_key
        actors_key, action, logprob, entropy = choose_action(logits, actors_key)
        system_state.actors_key = actors_key

        value = jnp.squeeze(critic_network.apply(system_state.network_params.critic_params, obs))
        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(action.tolist())
        global_step += 1 # TODO: With vec envs this should be more. 
        
        # NB: Correct shapes here. 
        data = BufferData(
            state = add_two_leading_dims(obs), 
            action = add_two_leading_dims(action), 
            reward = add_two_leading_dims(reward), 
            done = add_two_leading_dims(done), 
            log_prob = add_two_leading_dims(logprob), 
            value = add_two_leading_dims(value), 
            entropy = add_two_leading_dims(entropy)
        )

        buffer_state = system_state.buffer 
        buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        obs = obs_ 
        episode_return += reward
        episode_step += 1
        
        if global_step % (HORIZON + 1) == 0: 
            
            # Can also compute this inside the scan. 
            advantages = rlax.truncated_generalized_advantage_estimation(
                r_t = jnp.squeeze(system_state.buffer.rewards)[:-1],
                discount_t = (1 - jnp.squeeze(system_state.buffer.dones))[:-1] * DISCOUNT_GAMMA,
                lambda_ = GAE_LAMBDA, 
                values = jnp.squeeze(system_state.buffer.values),
                stop_target_gradients=True
            )

            advantages = jax.lax.stop_gradient(advantages)
            # Just not sure how to index the values here. 
            returns = advantages + jnp.squeeze(system_state.buffer.values)[:-1]
            returns = jax.lax.stop_gradient(returns)

            epoch_scan_out, _ = jax.lax.scan(
                f=epoch_update, 
                init=(system_state, advantages, returns),
                xs=None,  
                length=NUM_EPOCHS, 
            )
            system_state = epoch_scan_out[0]
            # for _ in range(NUM_EPOCHS):
                
            #     # Create data minibatches 
            #     # Generate random numbers 
            #     networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
            #     system_state.networks_key = networks_key

            #     idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
            #     mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)
                
            #     # # TODO: Update in a scan. 
            #     # update_scan_out, _ = jax.lax.scan(
            #     #     f=update, 
            #     #     init=(system_state, advantages, returns), 
            #     #     xs=mb_idxs, 
            #     # )
            #     # system_state = update_scan_out[0]
            #     for mb_idx in mb_idxs:
            #         system_state = update_policy(system_state, advantages, mb_idx)
            #         system_state = update_critic(system_state, returns, mb_idx)
            
            
            buffer_state = reset_buffer(buffer_state) 
            system_state.buffer = buffer_state

    sps = episode_step / (time.time() - start_time)

    if LOG: 
        log_data["episode"] = episode
        log_data["episode_return"] = episode_return
        log_data["global_step"] = global_step
        log_data["steps_per_second"] = sps
        logger.write(logging_details=log_data)
    
    episode += 1

    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}, SPS: {sps}")   

logger.close()