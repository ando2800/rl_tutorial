import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import pandas as pd
import os
from collections import deque, namedtuple

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCETrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])

        if isinstance(self.env.observation_space, gym.spaces.Box):
            state_size = self.env.observation_space.shape[0]
        else:
            raise TypeError("Unsupported observation space type")

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_size = self.env.action_space.n
        else:
            raise TypeError("Unsupported action space type")

        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config['learning_rate'])
        self.rewards_history = []
        self.log_probs_history = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs_history.append(m.log_prob(action))
        return action.item()

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)

        for i_episode in range(1, self.config['episodes'] + 1):
            state, _ = self.env.reset()
            episode_rewards = []
            self.log_probs_history = []
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _, _ = self.env.step(action)
                episode_rewards.append(reward)

            scores.append(sum(episode_rewards))
            scores_window.append(sum(episode_rewards))

            # Calculate discounted rewards
            discounted_rewards = []
            R = 0
            for r in episode_rewards[::-1]:
                R = r + self.config['gamma'] * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Calculate loss and update policy
            policy_loss = []
            for log_prob, G in zip(self.log_probs_history, discounted_rewards):
                policy_loss.append(-log_prob * G)
            self.optimizer.zero_grad()
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()

            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)
