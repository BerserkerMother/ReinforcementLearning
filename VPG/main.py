import gym
import torch
from torch.distributions import Categorical

import numpy as np

from network import VPG

env = gym.make('CartPole-v1')

# Env Parameters
num_state_features = env.observation_space.shape[0]
num_actions = env.action_space.n

# Hyper Parameters
# alpha is lr for actor net and beta for value function estimator
alpha = 1e-3
beta = 2e-3
gamma = .99
num_episodes = 1000
memory_size = 1000

# define VPG network
network = VPG(num_state_features=num_state_features, num_actions=num_actions, memory_size=memory_size,
              gamma=gamma, alpha=alpha, beta=beta).cuda()


def main():
    for episode in range(num_episodes):
        state = env.reset()
        loss_tracker = AverageMeter()
        ep_length = 0
        ep_reward = 0

        while True:
            # env.render(mode='rgb_array')
            out = network.actor(state)

            m = Categorical(out)
            entropy = m.entropy()
            action = m.sample(sample_shape=torch.Size([1]))
            log_prob = m.log_prob(action)

            next_state, reward, is_done, info = env.step(np.array(action.item()))

            network.reply_buffer.add_replay(state=state, entropy=entropy, reward=reward, next_state=next_state,
                                            log_prob=log_prob, is_done=is_done)

            state = next_state
            ep_length += 1
            ep_reward += reward

            # if network.reply_buffer.enough_reply:
            #    loss_actor, loss_critic = network.optimize()
            #    loss_tracker.update(loss_actor + loss_critic)
            if is_done:
                loss_actor, loss_critic = network.optimize()
                loss_tracker.update(loss_actor + loss_critic)
                network.reply_buffer.reset()
                break
        if episode % 10 == 0:
            network.update_target_network()
        print('episode: %d, reward: %f, length: %d, loss: %f' % (episode, ep_reward, ep_length, loss_tracker.value))


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
