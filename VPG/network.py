import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import ReplyBuffer


class VPG(nn.Module):
    def __init__(self, num_state_features, num_actions, memory_size, alpha, beta, gamma):
        super(VPG, self).__init__()
        self.num_state_features = num_state_features
        self.num_actions = num_actions
        self.a = alpha
        self.b = beta
        self.gamma = gamma

        # define Actor and Critic, Critic uses target network
        self.actor = Actor(self.num_state_features, self.num_actions)
        self.critic = Critic(self.num_state_features, self.num_actions)
        self.critic_target = Critic(self.num_state_features, self.num_actions)

        # define optimizer
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.a)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.b)

        # define reply buffer
        self.reply_buffer = ReplyBuffer(memory_size, self.num_state_features)

    def update_target_network(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def optimize(self):
        states, next_states, entropys, rewards, is_done, log_probs = self.reply_buffer.get_batch()
        state_actions_values = self.critic(states)

        R = 0
        returns = []

        # calculate Gt for actions
        for i, r in enumerate(rewards):
            R = R * self.gamma + rewards[-i].item()
            returns.insert(0, R)

        returns = torch.tensor([returns]).view(-1, 1).cuda()

        # remove state_actions_values from td error to have REINFORCE otherwise is VPG
        td_error = returns - state_actions_values

        # critic update
        critic_loss = F.mse_loss(state_actions_values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        # entropy term is suppose to force exploration, you can drop it if you want
        actor_loss = (-log_probs * td_error.detach()).mean() + -entropys.mean() * 1  # this is entropy weight
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), critic_loss.item()


class Actor(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(Actor, self).__init__()
        self.num_state_features = num_state_features
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(self.num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float).cuda()
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(Critic, self).__init__()
        self.num_state_features = num_state_features
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(self.num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float).cuda()
        return self.net(state)
