import torch
import numpy as np
import math

class ReplayMemory:
    def __init__(self, observation_space, action_space, device = False, num_state = None, memory_size = 100000, batch_size = 128, obs_size = 4, alpha = 0.6, n_step = 1, is_image = False):
        assert is_image or not num_state is None, 'the dimension of state is none'
        self.index = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        #場合によってはint型にしてメモリ使用量を削減してることもある
        self.obs = np.zeros((self.memory_size, observation_space.shape[0], 84, 84), dtype = np.float)
        if not is_image:
            self.obs = np.zeros((self.memory_size, num_state), dtype = np.float)
        self.actions = np.zeros((self.memory_size, action_space.shape[0]), dtype = np.float)
        self.rewards = np.zeros((self.memory_size, 1), dtype = np.float)
        #メモリ使用量削減のためにfloatではなくintで保存する
        self.priorities = np.zeros((self.memory_size), dtype = np.float)
        self.terminates = np.zeros((self.memory_size, 1), dtype = np.int)
        self.alpha = alpha
        self.N_STEP = n_step
        self.state_mean = 0
        self.state_halfwidth = 1
        if abs(observation_space.high[0]) != math.inf:
            self.state_mean = 0.5*(observation_space.high + observation_space.low)
            self.state_halfwidth = 0.5*(observation_space.high - observation_space.low)
        self.is_image = is_image
        self.device = device
        
    def add(self, obs, action, reward, terminate):
        self.obs[self.index % self.memory_size] = obs
        self.actions[self.index % self.memory_size] = action
        #報酬と終端に関してはサイズ1の要素を格納している
        self.rewards[self.index % self.memory_size][0] = reward
        self.terminates[self.index % self.memory_size][0] = terminate
        self.index += 1
        
    def random_sample(self):
        index = min(self.memory_size, self.index)
        probability_distribution = np.ones(index) / index
        size_indices = np.arange(index)
        indices = np.random.choice(a = size_indices, 
                                   size = self.batch_size, 
                                   p = probability_distribution.squeeze())
        batch = dict()
        if not self.is_image:
            batch['obs'] = torch.FloatTensor((self.obs[indices] - self.state_mean) / self.state_halfwidth).to(self.device)
        else:
            batch['obs'] = torch.FloatTensor(self.obs[indices] / 255.).to(self.device)
        #intだとNNで読み込めないのでラベルやtensorのindexとして使えないのでlongを使う必要がある
        batch['actions'] = torch.FloatTensor(self.actions[indices]).to(self.device)
        batch['rewards'] = torch.FloatTensor(self.rewards[indices]).to(self.device)
        if not self.is_image:
            batch['next_obs'] = torch.FloatTensor((self.obs[(indices + self.N_STEP) % self.memory_size] - self.state_mean) / self.state_halfwidth).to(self.device)
        else:
            batch['next_obs'] = torch.FloatTensor(self.obs[(indices + self.N_STEP) % self.memory_size] / 255.).to(self.device)
        batch['terminates'] = torch.FloatTensor(self.terminates[indices]).to(self.device)
        return batch, indices, probability_distribution
    
    def update_priority(self, indices, priorities):
        self.priorities[indices] = np.minimum(np.power(priorities, self.alpha), 1.0)


