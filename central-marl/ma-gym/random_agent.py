"""A random agent to be used as a baseline for performance on the 
ma-gym environment."""

import jax.numpy as jnp 
import jax 
import gym 
from utils.loggers import WandbLogger

ENV_NAME = "ma_gym:Lumberjacks-v0"
MASTER_PRNGKEY = jax.random.PRNGKey(2022)
MASTER_PRNGKEY, actors_key = jax.random.split(MASTER_PRNGKEY)

LOG = False

if LOG: 
    logger = WandbLogger(
        exp_config={
        "agent_type": "random",
        "env_name": ENV_NAME}, 
        run_name="test_logger2", 
    )

env = gym.make(ENV_NAME)

# TODO: Assuming fully homogeneous agents here. 
# Handle this later on to be per agent. 
observation_dim = env.observation_space[0].shape[0]
num_actions = env.action_space[0].n
num_agents = env.n_agents

def choose_action(actors_key): 

    actors_key, step_key = jax.random.split(actors_key)
    action = jax.random.randint(
            step_key, 
            shape=(), 
            minval=0, 
            maxval=num_actions
        )

    return actors_key, action

global_step = 0
episode = 0 
while global_step < 50_00: 

    team_done = False 
    obs = env.reset()
    episode_return = 0
    while not team_done: 
        
        log_details = {}
        step_joint_action = jnp.empty(num_agents, dtype=jnp.int32)

        for agent in range(num_agents):
            
            actors_key, action = choose_action(actors_key)

            step_joint_action = step_joint_action.at[agent].set(action)

        # Covert action to int in order to step the env. 
        # Can also handle in the wrapper
        obs_, reward, done, _ = env.step(step_joint_action.tolist())
        team_done = all(done)
        global_step += 1 # TODO: With vec envs this should be more. 
        
        obs = obs_ 
        episode_return += jnp.sum(jnp.array(reward, dtype=jnp.float32))

    log_details = {}
    log_details["global_step"] = global_step
    log_details["episode_return"] = episode_return
    log_details["episode"] = episode
    logger.write(logging_details=log_details, step=global_step)

    episode += 1
    if episode % 10 == 0: 
        print(f"EPISODE: {episode}, GLOBAL_STEP: {global_step}, EPISODE_RETURN: {episode_return}") 

logger.close()  