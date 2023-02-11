import numpy as np 
import gym 

class CentralControllerWrapper(gym.Env): 
    
    def __init__(self, ma_env):
        super().__init__()
        self.env = ma_env 
        self.num_agents = ma_env.n_agents 
        self.action_mapping = self.enumerate_agent_actions()
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))
        full_obs_size = sum([len(i) for i in ma_env.reset()])
        self.observation_space = gym.spaces.Box(np.zeros(full_obs_size), np.ones(full_obs_size), (full_obs_size,), np.float32)
        self.metadata = {'render.modes': ['human', 'rgb_array']}

    def reset(self, ):
        
        obs_n = self.env.reset()
        joint_obs = self.create_joint_obs(obs_n)
        
        return joint_obs
    
    def step(self, joint_action): 
        
        action = self.action_mapping[joint_action]
        obs_n, reward_n, done_n, info = self.env.step(action)
        
        joint_obs = self.create_joint_obs(obs_n)
        team_reward = np.sum(np.array(reward_n))
        team_done = all(done_n)
        
        return joint_obs, team_reward, team_done, info
    
    def random_action(self,): 
        
        action = np.random.randint(low = 0, high = self.action_space)
        return action 
    
    def enumerate_agent_actions(self, ):
        
        agent_actions = [np.arange(self.env.action_space[i].n) for i in range(len(self.env.action_space))]
        enumerated_actions = np.array(np.meshgrid(*agent_actions)).T.reshape(-1,self.num_agents)
        action_mapping = {int(i): list(action) for i, action in enumerate(enumerated_actions)}
        return action_mapping
    
    def create_joint_obs(self, env_obs):
        
        array_obs = np.array(env_obs)
        joint_obs = np.concatenate(array_obs, axis = -1)
        
        return joint_obs
    
    def unwrapped_env(self):
        return self