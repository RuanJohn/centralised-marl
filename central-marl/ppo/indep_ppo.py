"""Independent multi-agent JAX PPO. NOte that this implementation 
uses shared network weights between all agetns."""

import jax.numpy as jnp 
import jax 
import haiku as hk
import optax
import distrax
import rlax
import chex

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

import gym

jit_add = jax.jit(add)

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
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

ALGORITHM = "ff_indep_ppo"
LOG = True 

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

# TODO: Assuming fully homogeneous agents here. 
# Handle this later on to be per agent. 
observation_dim = env.observation_space[0].shape[0]
num_actions = env.action_space[0].n
num_agents = env.n_agents

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
    advantages, ):

    logits = policy_network.apply(policy_params, states)
    dist = distrax.Categorical(logits=logits)

    new_log_probs = dist.log_prob(value=actions)

    logratio = new_log_probs - old_log_probs
    ratio = jnp.exp(logratio)

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean()

    # jax.debug.print("policy loss {x}", x= loss)

    return loss

def critic_loss(
    critic_params, 
    states, 
    returns
    ):

    new_values = jnp.squeeze(critic_network.apply(critic_params, states))
    
    loss = 0.5 * ((new_values - returns) ** 2).mean()
    # jax.debug.print("critic loss {x}", x= loss)
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: PPOSystemState, advantages, mb_idx, agent_idx): 

    states = jnp.squeeze(system_state.buffer.states[:,:,agent_idx,:])[mb_idx]
    old_log_probs = jnp.squeeze(system_state.buffer.log_probs[:,:,agent_idx])[mb_idx]
    actions = jnp.squeeze(system_state.buffer.actions[:,:,agent_idx,:])[mb_idx]
    advantages = advantages[mb_idx]
    
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
    mb_idx,
    agent_idx,
): 

    states = jnp.squeeze(system_state.buffer.states[:,:,agent_idx,:])[mb_idx]
    returns = returns[mb_idx]
    
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

global_step = 0
episode = 0 
log_data = {}
while global_step < 100_000: 

    team_done = False 
    obs = env.reset()
    episode_return = 0
    while not team_done: 
        
        # For stepping the environment
        step_joint_action = jnp.empty(num_agents, dtype=jnp.int32)

        # Data to append to buffer
        act_joint_action = jnp.empty((num_agents,1), dtype=jnp.int32)
        act_values = jnp.empty((num_agents), dtype=jnp.float32)
        act_log_probs = jnp.empty((num_agents), dtype=jnp.float32)
        act_entropies = jnp.empty((num_agents), dtype=jnp.float32)

        for agent in range(num_agents):
            logits = policy_network.apply(system_state.network_params.policy_params, jnp.array(obs[agent], dtype=jnp.float32))
            actors_key = system_state.actors_key
            actors_key, action, logprob, entropy = choose_action(logits, actors_key)
            system_state.actors_key = actors_key

            value = jnp.squeeze(critic_network.apply(system_state.network_params.critic_params, jnp.array(obs[agent], dtype=jnp.float32)))

            step_joint_action = step_joint_action.at[agent].set(action)
            
            act_joint_action = act_joint_action.at[agent, 0].set(action)
            act_values = act_values.at[agent].set(value)
            act_log_probs = act_log_probs.at[agent].set(logprob)
            act_entropies = act_entropies.at[agent].set(entropy)

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_joint_action.tolist())
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

        buffer_state = system_state.buffer 
        buffer_state = jit_add(buffer_state, data)
        # buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        obs = obs_ 
        episode_return += jnp.sum(jnp.array(reward, dtype=jnp.float32))
        
        if global_step % (HORIZON + 1) == 0: 
            
            for agent in range(num_agents): 
                
                advantages = rlax.truncated_generalized_advantage_estimation(
                    r_t = jnp.squeeze(system_state.buffer.rewards[:,:,agent])[:-1],
                    discount_t = (1 - jnp.squeeze(system_state.buffer.dones[:,:,agent]))[:-1] * DISCOUNT_GAMMA,
                    lambda_ = GAE_LAMBDA, 
                    values = jnp.squeeze(system_state.buffer.values[:,:,agent]),
                    stop_target_gradients=True
                )

                advantages = jax.lax.stop_gradient(advantages)
                # Just not sure how to index the values here. 
                returns = advantages + jnp.squeeze(system_state.buffer.values[:,:,agent])[:-1]
                returns = jax.lax.stop_gradient(returns)


                for _ in range(NUM_EPOCHS):
                    
                    # Create data minibatches 
                    # Generate random numbers 
                    networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
                    system_state.actors_key = networks_key

                    idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
                    mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)
                    
                    for mb_idx in mb_idxs:
                        system_state = update_policy(system_state, advantages, mb_idx, agent)
                        system_state = update_critic(system_state, returns, mb_idx, agent)
                
                
            buffer_state = reset_buffer(buffer_state) 
            system_state.buffer = buffer_state

    if LOG: 
        log_data["episode"] = episode
        log_data["episode_return"] = episode_return
        log_data["global_step"] = global_step
        logger.write(logging_details=log_data)

    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}")   

logger.close()  