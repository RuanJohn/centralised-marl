import numpy as np 
import gym 

class CentralControllerWrapper(gym.Env): 
    
    def __init__(self, ma_env, one_hot_ids: bool = False,):
        super().__init__()
        self.env = ma_env 
        self.num_agents = ma_env.n_agents 
        self.action_mapping = self.enumerate_agent_actions()
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))
        full_obs_size = sum([len(i) for i in ma_env.reset()])
        self.observation_space = gym.spaces.Box(np.zeros(full_obs_size), np.ones(full_obs_size), (full_obs_size,), np.float32)
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.zero_agent_mask = np.zeros(ma_env.n_agents)
        self._one_hot_ids = one_hot_ids 
        self._grid_shape = ma_env._grid_shape

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
        
        # TODO: Vectorise this. 
        if self._one_hot_ids: 
            one_hot_ids = []
            
            for agent in range(self.num_agents): 
                one_hot_mask = self.zero_agent_mask
                one_hot_mask[agent] = 1.0
                one_hot_ids.append(one_hot_mask)
            
            one_hot_ids = np.array(one_hot_ids)

        joint_obs = np.concatenate(array_obs, axis = -1)

        if self._one_hot_ids:
            joint_obs = np.concatenate((joint_obs, one_hot_ids), axis=0)
        
        return joint_obs
    
    def unwrapped_env(self):
        return self

class CentralChunkedControllerWrapper(gym.Env): 
    
    def __init__(self, ma_env):
        super().__init__()
        self.env = ma_env 
        self.num_agents = ma_env.n_agents 

        agent_action_space_dims = np.array(
            [ma_env.action_space[i].n for i in range(ma_env.n_agents)]
        )

        self.action_space = gym.spaces.MultiDiscrete(agent_action_space_dims)
        
        full_obs_size = sum([len(i) for i in ma_env.reset()])
        self.observation_space = gym.spaces.Box(np.zeros(full_obs_size), np.ones(full_obs_size), (full_obs_size,), np.float32)
        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # This will be used to shift joint actions to the full action space later on. 
        # Need to be shift up by 1 since the first agent's actions don't have to be 
        # shifted. 
        self.action_map = np.concatenate(([0], np.cumsum(agent_action_space_dims[1:])))
    def reset(self, ):
        
        obs_n = self.env.reset()
        joint_obs = self.create_joint_obs(obs_n)
        
        return joint_obs
    
    def step(self, joint_action): 
        
        # The joint action here is an array where each index corresponds to each agents 
        # action. 
        obs_n, reward_n, done_n, info = self.env.step(joint_action)
        
        joint_obs = self.create_joint_obs(obs_n)
        team_reward = np.sum(np.array(reward_n))
        team_done = all(done_n)
        
        return joint_obs, team_reward, team_done, info
    
    def create_joint_obs(self, env_obs):
        
        array_obs = np.array(env_obs)
        joint_obs = np.concatenate(array_obs, axis = -1)
        
        return joint_obs
    
    def unwrapped_env(self):
        return self