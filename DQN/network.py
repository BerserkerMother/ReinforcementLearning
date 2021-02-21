import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Agent(nn.Module):
    def __init__(self, num_state_features, num_actions, memory_size):
        super(Agent, self).__init__()

        self.num_state_features = num_state_features
        self.num_actions = num_actions
        self.memory_size = memory_size

        # NN mapping state to Q values
        self.net = nn.Sequential(
            nn.Linear(self.num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # define memory
        self.state_memory = np.zeros((self.memory_size, self.num_state_features))
        self.next_state_memory = np.zeros((self.memory_size, self.num_state_features))
        self.action_memory = torch.zeros((self.memory_size, 1), dtype=torch.int64).cuda()
        self.reward_memory = torch.zeros((self.memory_size, 1), dtype=torch.float).cuda()
        self.is_done_memory = torch.zeros((self.memory_size, 1), dtype=torch.bool)
        self.capacity = 0
        self.pointer = -1

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float).cuda()
        return self.net(inputs)

    # is full property to check if we can start training
    @property
    def is_full(self):
        if self.capacity == self.memory_size:
            return True
        self.capacity += 1
        return False

    def add_replay(self, state, action, reward, next_state, is_done):
        self.pointer = (self.pointer + 1) % self.memory_size
        self.state_memory[self.pointer] = state
        self.next_state_memory[self.pointer] = next_state
        self.action_memory[self.pointer] = torch.tensor(action, dtype=torch.int64).cuda()
        self.reward_memory[self.pointer] = torch.tensor(reward, dtype=torch.float).cuda()
        self.is_done_memory[self.pointer] = torch.tensor([is_done], dtype=torch.bool)

    def get_batch(self, num_samples):
        indices = np.random.choice(self.capacity, num_samples, replace=False)
        return self.state_memory[indices], self.next_state_memory[indices], self.action_memory[indices], \
               self.reward_memory[indices], self.is_done_memory[indices]

    def optimize(self, optimizer, target_network, batch_size):
        states, next_states, actions, rewards, is_done = self.get_batch(batch_size)
        state_actions_values = self.forward(states)
        next_states_max_actions = self.forward(next_states).detach().max(1)[1].view(-1, 1)
        next_state_actions_values = target_network.forward(next_states).detach()

        state_actions_values = torch.gather(state_actions_values, dim=1, index=actions)
        next_state_actions_values = torch.gather(next_state_actions_values, dim=1, index=next_states_max_actions)

        # uncomment for DQN or else is DDPG
        # next_state_actions_values = next_state_actions_values.max(1)[0].view(-1, 1)
        next_state_actions_values[is_done] = 0.0
        loss = F.smooth_l1_loss(state_actions_values, (rewards + .99 * next_state_actions_values), reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
