import torch
import gym

import math
import random
import numpy as np
from network import Agent

env = gym.make('CartPole-v1')

# hyper parameters need to be tuned for different environments
# hyper Parameters
num_state_features = env.observation_space.shape[0]
num_actions = env.action_space.n
learning_rate = 5e-4
num_iterations = 10000
memory_size = 10000
batch_size = 1024
update_target_net_every = 5

# hyper parameters for e-greedy policy
min_e = .01
max_e = .6
e_decay = 5000.0

# define brain
brain = Agent(num_state_features, num_actions, memory_size).cuda()
brain_prime = Agent(num_state_features, num_actions, memory_size).cuda()
brain_prime.net.load_state_dict(brain.net.state_dict())

optimizer = torch.optim.RMSprop(brain.net.parameters(), lr=learning_rate)


def main():
    eps_step_tracker = 0
    loss_tracker = AverageMeter()
    for i in range(num_iterations):
        state = env.reset()
        ep_length = 0
        ep_reward = 0
        while True:
            # env.render()
            actions_values = brain(state)
            action = get_action(actions_values, get_epsilon(eps_step_tracker))
            next_state, reward, is_done, info = env.step(action)

            brain.add_replay(state, action, reward, next_state, is_done)
            state = next_state
            ep_length += 1
            ep_reward += reward

            _ = brain.is_full
            if brain.capacity >= batch_size:
                loss = brain.optimize(optimizer, brain_prime, batch_size)
                loss_tracker.update(loss)
                eps_step_tracker += 1

            if is_done:
                break

        print('episode %d, length:%d, reward:%.1f loss:%f, eps:%.3f' % (
            i, ep_length, ep_reward, loss_tracker.value, get_epsilon(eps_step_tracker)))
        loss_tracker.reset()

        if i % update_target_net_every == 0:
            brain_prime.net.load_state_dict(brain.net.state_dict())


def get_epsilon(step):
    return min_e + (max_e - min_e) * math.exp(-1. * step / e_decay)


def get_action(actions_values, eps):
    thresh_hold = random.random()
    if thresh_hold > eps:
        return torch.argmax(actions_values).item()
    return np.random.choice(np.arange(num_actions))


class AverageMeter:
    def __init__(self):
        self.number = 0
        self.sum = 0.0

    def reset(self):
        self.number = 0
        self.sum = 0.0

    def update(self, value):
        self.sum += value
        self.number += 1

    @property
    def value(self):
        if self.number:
            return self.sum / self.number
        return 0


if __name__ == '__main__':
    main()
