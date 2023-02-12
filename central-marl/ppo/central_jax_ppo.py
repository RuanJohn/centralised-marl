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

from utils.types import (
    BufferData, 
    BufferState, 
    PPOSystemState, 
    NetworkParams,
    OptimiserStates, 
)

from utils.replay_buffer import (
    create_buffer, 
    add, 
    reset_buffer,
    should_train,
)

from utils.array_utils import (
    add_two_leading_dims,
)

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
# ENV_NAME = "ma_gym:PredatorPrey7x7-v0"
ENV_NAME = "CartPole-v0"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, networks_key, actors_key, buffer_key = jax.random.split(MASTER_PRNGKEY, 4)

env = gym.make(ENV_NAME)

# Uncomment for centralised marl envs. 
# env = CentralControllerWrapper(env)

observation_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Make networks 

def make_networks(
    num_actions: int, 
    policy_layer_sizes: list = [64, 64], 
    critic_layer_sizes: list = [64, 64], ):

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

    states = jnp.squeeze(system_state.buffer.states)[mb_idx]
    old_log_probs = jnp.squeeze(system_state.buffer.log_probs)[mb_idx]
    actions = jnp.squeeze(system_state.buffer.actions)[mb_idx]
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
    mb_idx
): 

    states = jnp.squeeze(system_state.buffer.states)[mb_idx]
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
while global_step < 50_000: 

    done = False 
    obs = env.reset()
    episode_return = 0
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
        
        if global_step % (HORIZON + 1) == 0: 
            
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


            for _ in range(NUM_EPOCHS):
                
                # Create data minibatches 
                # Generate random numbers 
                networks_key, sample_idx_key = jax.random.split(system_state.networks_key)
                system_state.actors_key = networks_key

                idxs = jax.random.permutation(key = sample_idx_key, x=HORIZON)
                mb_idxs = jnp.split(idxs, NUM_MINIBATCHES)
                
                for mb_idx in mb_idxs:
                    system_state = update_policy(system_state, advantages, mb_idx)
                    system_state = update_critic(system_state, returns, mb_idx)
            
            
            buffer_state = reset_buffer(buffer_state) 
            system_state.buffer = buffer_state
    
    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}")   
