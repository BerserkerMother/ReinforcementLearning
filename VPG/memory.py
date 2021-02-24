import torch

import numpy as np


class ReplyBuffer:
    def __init__(self, capacity, num_state_features):
        self.capacity = capacity
        self.num_state_features = num_state_features

        self.pointer = -1
        self.n = 0

        self.state_memory = np.zeros((self.capacity, self.num_state_features))
        self.next_state_memory = np.zeros((self.capacity, self.num_state_features))
        self.entropy_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.action_prob_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.reward_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.is_done_memory = torch.zeros((self.capacity, 1), dtype=torch.bool)

    def add_replay(self, state, entropy, reward, next_state, log_prob, is_done):
        self.pointer = (self.pointer + 1) % self.capacity
        self.state_memory[self.pointer] = state
        self.next_state_memory[self.pointer] = next_state
        self.entropy_memory[self.pointer] = entropy.cuda()
        self.action_prob_memory[self.pointer] = log_prob
        self.reward_memory[self.pointer] = torch.tensor(reward, dtype=torch.float).cuda()
        self.is_done_memory[self.pointer] = torch.tensor([is_done], dtype=torch.bool)

        if self.n != self.capacity:
            self.n += 1

    def get_batch(self):
        indices = np.arange(self.pointer + 1)
        return self.state_memory[indices], self.next_state_memory[indices], self.entropy_memory[indices], \
               self.reward_memory[indices], self.is_done_memory[indices], self.action_prob_memory[indices]

    def reset(self):
        self.pointer = -1
        self.state_memory = np.zeros((self.capacity, self.num_state_features))
        self.next_state_memory = np.zeros((self.capacity, self.num_state_features))
        self.entropy_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.action_prob_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.reward_memory = torch.zeros((self.capacity, 1), dtype=torch.float).cuda()
        self.is_done_memory = torch.zeros((self.capacity, 1), dtype=torch.bool)
