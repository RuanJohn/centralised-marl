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

from utils.recurrent_replay_buffer import (
    create_buffer, 
    add, 
    reset_buffer,
    split_buffer_into_chunks,
)

from utils.loggers import WandbLogger
from wrappers.agent_id_wrapper import AgentIDWrapper

import gym

jit_add = jax.jit(add, donate_argnums=(0))

# Constants: 
HORIZON = 200 
RECURRENT_CHUNK_LENGTH = 10
CLIP_EPSILON = 0.2 
POLICY_LR = 0.005
CRITIC_LR = 0.005
DISCOUNT_GAMMA = 0.99 
GAE_LAMBDA = 0.95
NUM_EPOCHS = 1
NUM_MINIBATCHES = 4 
MAX_GLOBAL_NORM = 0.5
ADAM_EPS = 1e-5
POLICY_LAYER_SIZES = [64]
CRITIC_LAYER_SIZES = [64]
POLICY_RECURRENT_LAYER_SIZES = [64]
CRITIC_RECURRENT_LAYER_SIZES = [64]

# TODO: Add agent IDS. 
ADD_ONE_HOT_IDS = False
ENV_NAME = "ma_gym:Switch4-v0"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

NORMALISE_ADVANTAGE = True
ADD_ENTROPY_LOSS = False

ALGORITHM = "rec_ippo"
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
    critic_layer_sizes: list = CRITIC_LAYER_SIZES, 
    policy_recurrent_layer_sizes: list = POLICY_RECURRENT_LAYER_SIZES, 
    critic_recurrent_layer_sizes: list = CRITIC_RECURRENT_LAYER_SIZES,):

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

    @hk.without_apply_rng
    @hk.transform
    def critic_network(x, y): 

        return hk.DeepRNN(
            [
            hk.nets.MLP(critic_layer_sizes, activate_final=True), 
            hk.GRU(critic_recurrent_layer_sizes[0]),  
            hk.Linear(1),
            ]
            )(x, y) 

    return policy_network, critic_network

def create_policy_hidden_states(
        policy_recurrent_layer_sizes:list = POLICY_RECURRENT_LAYER_SIZES, ): 

    hidden_states = jnp.array(
        [jnp.zeros(layer, dtype=jnp.float32)
         for layer in policy_recurrent_layer_sizes] 
        )

    return hidden_states

def create_critic_hidden_states(
        critic_recurrent_layer_sizes:list = CRITIC_RECURRENT_LAYER_SIZES, ): 

    hidden_states = jnp.array(
        [jnp.zeros(layer, dtype=jnp.float32)
         for layer in critic_recurrent_layer_sizes] 
        )

    return hidden_states

policy_network, critic_network = make_networks(num_actions=num_actions)

# Create network params 

dummy_obs_data = jnp.zeros(observation_dim, dtype=jnp.float32)
policy_hidden_state = create_policy_hidden_states()
critic_hidden_state = create_critic_hidden_states()
networks_key, policy_init_key, critic_init_key = jax.random.split(networks_key, 3)

policy_params = policy_network.init(policy_init_key, dummy_obs_data, policy_hidden_state)
critic_params = critic_network.init(critic_init_key, dummy_obs_data, critic_hidden_state)

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
    # TODO: Handle multiple hidden states. 
    # For now only one. 
    policy_hidden_state_dim=policy_hidden_state.shape, 
    critic_hidden_state_dim=critic_hidden_state.shape, 
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
    entropies_,
    policy_hidden_states_):

    new_log_probs = jnp.empty_like(old_log_probs)

    def policy_core(input: jnp.ndarray, hidden_state: jnp.ndarray) -> jnp.ndarray:
        
        return policy_network.apply(policy_params, input, hidden_state)

    policy_hidden_states_ = policy_hidden_states_[:, 0, :, :]

    for agent in range(num_agents):

        logits, _ = hk.static_unroll(
            policy_core, 
            states[:,:,agent,:], 
            policy_hidden_states_[:, agent, :, :], 
            time_major=False, 
        )
        dist = distrax.Categorical(logits=logits)

        new_log_probs_ = dist.log_prob(value=actions[:, :, agent])
        new_log_probs = new_log_probs.at[:, :, agent].set(new_log_probs_)


    logratio = new_log_probs - old_log_probs
    ratio = jnp.exp(logratio)

    # Policy loss
    loss_term_1 = -advantages * ratio
    loss_term_2 = -advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
    loss = jnp.maximum(loss_term_1, loss_term_2).mean() 
    
    if ADD_ENTROPY_LOSS: 
        loss -= 0.01 * jnp.mean(entropies_)

    # jax.debug.print("policy loss {x}", x= loss)

    return loss

def critic_loss(
    critic_params, 
    states, 
    returns, 
    critic_hidden_states_,
    ):

    new_values = jnp.empty_like(returns)

    def critic_core(input: jnp.ndarray, hidden_state: jnp.ndarray) -> jnp.ndarray:
        
        return critic_network.apply(critic_params, input, hidden_state)

    critic_hidden_states_ = critic_hidden_states_[:, 0, :, :]
    
    for agent in range(num_agents): 
        new_values_, _ = hk.static_unroll(
            critic_core, 
            states[:,:,agent,:], 
            critic_hidden_states_[:, agent, :, :], 
            time_major=False, 
        )
        
        new_values_ = jnp.squeeze(new_values_)
        new_values = new_values.at[:, :, agent].set(new_values_)
    
    loss = 0.5 * ((new_values - returns) ** 2).mean()
    # jax.debug.print("critic loss {x}", x= loss)
    return loss

@jax.jit
@chex.assert_max_traces(n=1)
def update_policy(system_state: PPOSystemState, advantages, mb_idx): 

    states_ = jnp.squeeze(system_state.train_buffer.states)[mb_idx]
    old_log_probs_ = jnp.squeeze(system_state.train_buffer.log_probs)[mb_idx]
    actions_ = jnp.squeeze(system_state.train_buffer.actions)[mb_idx]
    entropies_ = jnp.squeeze(system_state.train_buffer.entropy)[mb_idx]
    policy_hidden_states_ = jnp.squeeze(system_state.train_buffer.policy_hidden_states, axis=2)[mb_idx]
    advantages_ = advantages[mb_idx]

    if NORMALISE_ADVANTAGE: 
        advantages_ = (advantages_ - jnp.mean(advantages_)) / (jnp.std(advantages_) + 1e-5)
    
    policy_optimiser_state = system_state.optimiser_states.policy_state
    policy_params = system_state.network_params.policy_params

    grads = jax.grad(policy_loss)(
        policy_params, 
        states_, 
        actions_, 
        old_log_probs_, 
        advantages_,
        entropies_,
        policy_hidden_states_, 
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
): 

    states_ = jnp.squeeze(system_state.train_buffer.states)[mb_idx]
    critic_hidden_states_ = jnp.squeeze(system_state.train_buffer.critic_hidden_states, axis=2)[mb_idx]
    returns_ = returns[mb_idx]
    
    critic_optimiser_state = system_state.optimiser_states.critic_state
    critic_params = system_state.network_params.critic_params

    grads = jax.grad(critic_loss)(
        critic_params, 
        states_, 
        returns_,
        critic_hidden_states_,
    )

    updates, new_critic_optimiser_state = critic_optimiser.update(grads, critic_optimiser_state)
    new_critic_params = optax.apply_updates(critic_params, updates)

    system_state.optimiser_states.critic_state = new_critic_optimiser_state
    system_state.network_params.critic_params = new_critic_params

    return system_state

# NOTE: Can terminate episode if one agent is done. Doesn't have to be all agents. 

global_step = 0
episode = 0 
log_data = {}
while global_step < 200_000: 

    team_done = False 
    obs = env.reset()
    obs = jnp.array(obs, dtype=jnp.float32) 
    episode_return = 0
    episode_step = 0 
    start_time = time.time()
    # Each agent has its own hidden state.
    policy_hidden_state = create_policy_hidden_states() 
    policy_hidden_state = jnp.broadcast_to(policy_hidden_state, (num_agents, *policy_hidden_state.shape))
    new_policy_hidden_state = jnp.empty_like(policy_hidden_state)
    
    critic_hidden_state = create_critic_hidden_states() 
    critic_hidden_state = jnp.broadcast_to(critic_hidden_state, (num_agents, *critic_hidden_state.shape))
    new_critic_hidden_state = jnp.empty_like(critic_hidden_state)
    while not team_done: 
        
        # For stepping the environment
        step_joint_action = jnp.empty(num_agents, dtype=jnp.int32)

        # Data to append to buffer
        act_joint_action = jnp.empty((num_agents,1), dtype=jnp.int32)
        act_values = jnp.empty((num_agents), dtype=jnp.float32)
        act_log_probs = jnp.empty((num_agents), dtype=jnp.float32)
        act_entropies = jnp.empty((num_agents), dtype=jnp.float32)

        for agent in range(num_agents):
            
            logits, new_policy_hidden_state_ = policy_network.apply(
                system_state.network_params.policy_params, 
                obs[agent], 
                policy_hidden_state[agent])
            new_policy_hidden_state_ = jnp.expand_dims(new_policy_hidden_state_[0], axis=0)

            actors_key = system_state.actors_key
            actors_key, action, logprob, entropy = choose_action(logits, actors_key)
            system_state.actors_key = actors_key

            value, new_critic_hidden_state_ = critic_network.apply(
                system_state.network_params.critic_params, 
                obs[agent], 
                critic_hidden_state[agent])
            
            value = jnp.squeeze(value)
            new_critic_hidden_state_ = jnp.expand_dims(new_critic_hidden_state_[0], axis=0)

            step_joint_action = step_joint_action.at[agent].set(action)
            
            act_joint_action = act_joint_action.at[agent, 0].set(action)
            act_values = act_values.at[agent].set(value)
            act_log_probs = act_log_probs.at[agent].set(logprob)
            act_entropies = act_entropies.at[agent].set(entropy)

            new_policy_hidden_state = new_policy_hidden_state.at[agent].set(new_policy_hidden_state_)
            new_critic_hidden_state = new_critic_hidden_state.at[agent].set(new_critic_hidden_state_)

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
            entropy = jnp.expand_dims(act_entropies, axis=0), 
            critic_hidden_state = jnp.expand_dims(critic_hidden_state, axis=0), 
            policy_hidden_state = jnp.expand_dims(policy_hidden_state, axis=0)
        )

        buffer_state = system_state.buffer 
        buffer_state = jit_add(buffer_state, data)
        # buffer_state = add(buffer_state, data)
        system_state.buffer = buffer_state

        obs = obs_ 
        critic_hidden_state = new_critic_hidden_state
        policy_hidden_state = new_policy_hidden_state
        episode_return += jnp.sum(jnp.array(reward, dtype=jnp.float32))
        episode_step += 1 
        
        if global_step % (HORIZON + 1) == 0: 
            
            advantages = jnp.empty_like(jnp.squeeze(system_state.buffer.rewards)[:-1], dtype=jnp.float32)
            returns = jnp.empty_like(jnp.squeeze(system_state.buffer.rewards)[:-1], dtype=jnp.float32)

            for agent in range(num_agents): 
                
                advantage = rlax.truncated_generalized_advantage_estimation(
                    r_t = jnp.squeeze(system_state.buffer.rewards[:,:,agent])[:-1],
                    discount_t = (1 - jnp.squeeze(system_state.buffer.dones[:,:,agent]))[:-1] * DISCOUNT_GAMMA,
                    lambda_ = GAE_LAMBDA, 
                    values = jnp.squeeze(system_state.buffer.values[:,:,agent]),
                    stop_target_gradients=True
                )

                advantage = jax.lax.stop_gradient(advantage)
                # Just not sure how to index the values here. 
                return_ = advantage + jnp.squeeze(system_state.buffer.values[:,:,agent])[:-1]
                return_ = jax.lax.stop_gradient(return_)

                advantages = advantages.at[:, agent].set(advantage)
                returns = returns.at[:, agent].set(return_)

            num_chunks = int(HORIZON // RECURRENT_CHUNK_LENGTH)

            # Need to reshape advantages and returns as well
            # NOTE make sure the gradients are stopped after reshaping. 
            advantages = jnp.array(
                jnp.split(advantages, num_chunks)
            )
            returns = jnp.array(
                jnp.split(returns, num_chunks)
            )

            # TODO: 
            # 1. VMAP over advantage and return calculations
            # 2. Scan the epoch update
            # 3. Scan / vmap over agents in the loss. 

            system_state.train_buffer = split_buffer_into_chunks(system_state.buffer, num_chunks=num_chunks)

            for _ in range(NUM_EPOCHS):
                
                # Create data minibatches 
                # Generate random numbers 
                networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
                system_state.actors_key = networks_key

                idxs = jax.random.permutation(key = sample_idx_key, x=num_chunks)
                mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)

                for mb_idx in mb_idxs:
                    system_state = update_policy(system_state, advantages, mb_idx)
                    system_state = update_critic(system_state, returns, mb_idx)
                
                
            buffer_state = reset_buffer(buffer_state) 
            system_state.buffer = buffer_state

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

logger.close()  