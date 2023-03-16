import numpy as np 

class AgentIDWrapper:
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.agent_id_encoding = np.eye(self.n_agents)

    def _add_agent_id_to_observation(self, observations):
        return np.array([np.concatenate((self.agent_id_encoding[i], obs)) for i, obs in enumerate(observations)])

    def reset(self):
        observations = self.env.reset()
        return self._add_agent_id_to_observation(observations)

    def step(self, actions):
        next_observations, rewards, dones, infos = self.env.step(actions)
        next_observations = self._add_agent_id_to_observation(next_observations)
        return next_observations, rewards, dones, infos

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()