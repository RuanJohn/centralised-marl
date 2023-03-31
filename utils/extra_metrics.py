import numpy as np 
from typing import Tuple
import copy 
import jax.numpy as jnp 
import jax 

def create_visitation_grids(grid_size: Tuple[int, int], num_agents: int): 
    """Create a dictionary of visitation grids for each agent."""

    visitionation_grids = {i: np.zeros(grid_size) for i in range(num_agents)}

    return visitionation_grids

def diff_reward_step(env, team_action: jnp.ndarray, agent_no_op_idxs: list): 
    """Calculate the difference reward for each agent.
     
       This computes the difference in team reward for each agent 
       in the case that that agent no-ops."""

    team_action = team_action.tolist()
    num_agents = len(team_action)
    diff_envs = [copy.copy(env) for _ in range(num_agents)]
    diff_rewards = []

    for agent in range(num_agents):
        agent_diff_step = team_action
        agent_diff_step[agent] = agent_no_op_idxs[agent]
        obs, reward, done, info = diff_envs[agent].step(agent_diff_step)
        diff_rewards.append(np.sum(reward[agent]))

    return diff_rewards

# TODO: Complete this and make it correct. 
def compute_stiffness(gradients_s, gradients_s_next, eps=1e-8):
    """Compute stiffness between gradients of two states."""
    

    # Get values of all leaves in the gradient tree.
    gradient_s_values = jax.tree_util.tree_leaves(gradients_s)
    gradient_s_next_values = jax.tree_util.tree_leaves(gradients_s_next)

    # Transpose the values in gradient_s_next_values.
    gradient_s_values_T = [jnp.transpose(g_s) for g_s in gradient_s_values]

    # Compute the dot product of the gradients.
    dot_products = jnp.array([jnp.dot(g_s.flatten(), g_s_next.flatten()) for g_s, g_s_next in zip(gradient_s_values_T, gradient_s_next_values)])

    # Compute the norm of the gradients.
    norm_s = [jnp.linalg.norm(g_s) for g_s in gradient_s_values]
    norm_s_next = [jnp.linalg.norm(g_s_next) for g_s_next in gradient_s_next_values]

    # Compute the stiffness.
    stiffness = jnp.sum(dot_products) / (norm_s * norm_s_next + eps)

    return stiffness