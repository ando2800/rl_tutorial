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

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, clip_epsilon, ppo_epochs, gae_lambda):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, state_value = self.model(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(m.log_prob(action))
        self.values.append(state_value)

        return action.item()

    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self):
        # Convert lists to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()
        values = torch.cat(self.values).detach()

        # Calculate advantages and returns using GAE
        advantages = []
        returns = []
        last_advantage = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0 # Last state, no next value
            else:
                next_value = values[i+1]

            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - values[i]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * last_advantage
            advantages.insert(0, last_advantage)
            returns.insert(0, last_advantage + values[i])
        
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO update with mini-batching
        batch_size = len(states) # For now, use full batch
        num_batches = 1 # For now, one batch

        for _ in range(self.ppo_epochs):
            # Shuffle and create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start_idx in range(0, len(states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                action_probs, state_values = self.model(batch_states)
                m = torch.distributions.Categorical(action_probs)
                new_log_probs = m.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values, batch_returns.unsqueeze(1))

                loss = actor_loss + 0.5 * critic_loss # Hyperparameter for critic loss weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = PPOAgent(state_size, action_size, 
                              self.config['learning_rate'], 
                              self.config['gamma'], 
                              self.config['clip_epsilon'],
                              self.config['ppo_epochs'],
                              self.config['gae_lambda'])

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)

        print("Starting PPO training...")
        for i_episode in range(1, self.config['episodes'] + 1):
            state, _ = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.store_transition(reward, done)
                state = next_state
                score += reward

            self.agent.learn()

            scores.append(score)
            scores_window.append(score)

            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)
