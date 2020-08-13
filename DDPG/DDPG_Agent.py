import copy
from DDPG import CriticNetwork
from DDPG import ActorNetwork
from Memory import ReplayMemory
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import math

class DdpgAgent:
    def __init__(self, observation_space, action_space, device, gamma = 0.99,
                 actor_lr = 1e-4, critic_lr = 1e-3, batch_size = 64, memory_size = 50000, tau = 1e-3, weight_decay = 1e-2, writer = None, is_image = False):
        super(DdpgAgent, self).__init__()
        self.num_state = observation_space.shape[0]
        self.num_action = action_space.shape[0]
        self.state_mean = None
        self.state_halfwidth = None
        if abs(observation_space.high[0]) != math.inf:
            self.state_mean = 0.5 * (observation_space.high + observation_space.low)
            self.state_halfwidth = 0.5 * (observation_space.high - observation_space.low)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.actor = ActorNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.actor_target = ActorNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = CriticNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.critic_target = CriticNetwork(self.num_state, action_space, device, is_image = is_image).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr, weight_decay=weight_decay)
        self.memory = ReplayMemory(observation_space, action_space, device, num_state = self.num_state, memory_size = memory_size, is_image = is_image)
        self.criterion = nn.SmoothL1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tau = tau
        self.writer = writer
        self.update_step = 0
        self.is_image =is_image


        
    def normalize_state(self, state):
        if self.state_mean is None:
            return state
        state = (state - self.state_mean) / self.state_halfwidth
        return state
    
    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def update(self):
        self.update_step += 1
        with torch.no_grad():
            batch, indices, probability_distribution = self.memory.random_sample()
            #各サンプルにおける状態行動の値を取ってくる
            action_batch = batch['actions'].to(self.device)
            state_batch = batch['obs'].to(self.device)
            next_obs_batch = batch['next_obs'].clone().to(self.device)
            reward_batch = batch['rewards'].to(self.device)
            terminate_batch = batch['terminates'].to(self.device)
            next_q_value_index = self.actor_target(next_obs_batch)
            #target-Q-network内の、対応する行動のインデックスにおける価値関数の値を取ってくる
            next_q_value = self.critic_target(next_obs_batch, next_q_value_index)
            #目的とする値の導出
            target_q_values = reward_batch + self.gamma * next_q_value * (1 - terminate_batch)
        self.actor.train()
        self.critic.train()
        q_values = self.critic(state_batch, action_batch)
        #誤差の計算
        critic_loss = self.criterion(q_values, target_q_values)
        #勾配を0にリセットする
        self.critic_optimizer.zero_grad()
        #逆誤差伝搬を計算する
        critic_loss.backward()
        #勾配を更新する
        self.critic_optimizer.step()
        if self.writer and self.update_step % 1000 == 0:
            self.writer.add_scalar("loss/critic", critic_loss.item(), self.update_step / 1000)
            #print("loss/critic", critic_loss.item())
        actor_loss = - self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.writer and self.update_step % 1000 == 0:
            self.writer.add_scalar("loss/actor", actor_loss.item(), self.update_step / 1000)
            #print("loss/actor", actor_loss.item())
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.actor.eval()
        self.critic.eval()
        
        
    # Q値が最大の行動を選択
    def get_action(self, state, noise = None, timestep = 0):
        if not self.is_image:
            state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float).view(-1, self.num_state).to(self.device)
        else:
            state_tensor = torch.tensor(state.copy() / 255., dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).view(self.num_action)
            if noise is not None:
                noise = noise(timestep)
                action = np.clip(action.to('cpu').detach().numpy().copy() + noise, -1, 1)
            else:
                action = np.clip(action.to('cpu').detach().numpy().copy(), -1, 1)
        return action