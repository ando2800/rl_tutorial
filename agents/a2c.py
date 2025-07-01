import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import os
import pandas as pd
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

class A2CTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.log_probs = []
        self.rewards = []
        self.values = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, state_value = self.model(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.values.append(state_value)
        return action.item()

    def step(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

    def learn(self, next_state_value):
        returns = self._compute_returns(next_state_value)

        actor_loss = []
        critic_loss = []
        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(F.mse_loss(value, torch.tensor([R])))

        self.optimizer.zero_grad()
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self.values = []

    def _compute_returns(self, next_state_value):
        returns = []
        R = next_state_value
        for reward in reversed(self.rewards):
            R = reward + self.config['gamma'] * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)

        print("Starting A2C training...")
        for i_episode in range(1, self.config['episodes'] + 1):
            state, _ = self.env.reset()
            episode_rewards = []
            self.log_probs = []
            self.rewards = []
            self.values = []
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_rewards.append(reward)
                self.rewards.append(reward) # Add reward to self.rewards for return calculation
                state = next_state

            scores.append(sum(episode_rewards))
            scores_window.append(sum(episode_rewards))

            # Calculate next state value for the last state of the episode
            next_state_value = 0.0
            if not done: # If episode did not end by reaching max steps
                with torch.no_grad():
                    next_state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    _, next_state_value_tensor = self.model(next_state_tensor)
                    next_state_value = next_state_value_tensor.item()

            self.learn(next_state_value)

            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)
