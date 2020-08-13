import torch
import torch.nn as nn
import torch.nn.functional as F
import util

class ActorNetwork(nn.Module):
    def __init__(self, num_state, action_space, device, hidden_size = 200, is_image = False):
        super(ActorNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        
        self.conv1 = nn.Conv2d(num_state, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 32
        if not is_image:
            linear_input_size = num_state
            self.fc1 = nn.Linear(linear_input_size, 256)
            self.fc2 = nn.Linear(256, 256)
            util.layer_init(self.fc1)
            util.layer_init(self.fc2)
            self.fc_last = nn.Linear(256, action_space.shape[0])
        else:
            self.fc1 = nn.Linear(linear_input_size, 200)
            util.layer_init(self.fc1)
            self.fc_last = nn.Linear(200, action_space.shape[0])
        self.is_image = is_image
        util.actor_last_layer_init(self.fc_last)
        
    def forward(self, state):
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        action = self.action_mean + self.action_halfwidth * torch.tanh(self.fc_last(x))
        return action
    
class CriticNetwork(nn.Module):
    def __init__(self, num_state, action_space, device, is_image = False):
        super(CriticNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype = torch.float).to(device)
        self.action_halfwidth = torch.tensor(0.5 * (action_space.high - action_space.low), dtype = torch.float).to(device)
        self.conv1 = nn.Conv2d(num_state, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 32
        if not is_image:
            linear_input_size = num_state
            self.fc1 = nn.Linear(linear_input_size + action_space.shape[0], 256)
            self.fc2 = nn.Linear(256, 256)
            util.layer_init(self.fc1)
            util.layer_init(self.fc2)
            self.fc_last = nn.Linear(256, action_space.shape[0])
        else:
            self.fc1 = nn.Linear(linear_input_size + action_space.shape[0], 200)
            util.layer_init(self.fc1)
            self.fc_last = nn.Linear(200, action_space.shape[0])
        self.is_image = is_image
        util.critic_last_layer_init(self.fc_last)
        
        self.conv2_1 = nn.Conv2d(num_state, 32, kernel_size=8, stride=4)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        linear_input_size = 7 * 7 * 32
        if not is_image:
            linear_input_size = num_state
            self.fc2_1 = nn.Linear(linear_input_size + action_space.shape[0], 256)
            self.fc2_2 = nn.Linear(256, 256)
            util.layer_init(self.fc2_1)
            util.layer_init(self.fc2_2)
            self.fc2_last = nn.Linear(256, action_space.shape[0])
        else:
            self.fc2_1 = nn.Linear(linear_input_size + action_space.shape[0], 200)
            util.layer_init(self.fc2_1)
            self.fc2_last = nn.Linear(200, action_space.shape[0])
        self.is_image = is_image
        util.critic_last_layer_init(self.fc2_last)
        
    def forward(self, state, action):
        a = (action - self.action_mean) / self.action_halfwidth
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(torch.cat([x.view(x.size(0), -1), a],1)))
        else:
            x = F.relu(self.fc1(torch.cat([state, action],1)))
            x = F.relu(self.fc2(x))
        q1 = self.fc_last(x)
        
        if self.is_image:
            x = F.relu(self.conv2_1(state))
            x = F.relu(self.conv2_2(x))
            x = F.relu(self.conv2_3(x))
            x = F.relu(self.fc2_1(torch.cat([x.view(x.size(0), -1), a],1)))
        else:
            x = F.relu(self.fc2_1(torch.cat([state, action],1)))
            x = F.relu(self.fc2_2(x))
        q2 = self.fc2_last(x)
        return q1, q2
    
    def q1_forward(self, state, action):
        a = (action - self.action_mean) / self.action_halfwidth
        if self.is_image:
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc1(torch.cat([x.view(x.size(0), -1), a],1)))
        else:
            x = F.relu(self.fc1(torch.cat([state, action],1)))
            x = F.relu(self.fc2(x))
        q1 = self.fc_last(x)
        return q1