import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import pandas as pd
import os

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])
        
        # Ensure state_size and action_size are integers
        if isinstance(self.env.observation_space, gym.spaces.Box):
            state_size = self.env.observation_space.shape[0]
        else:
            raise TypeError("Unsupported observation space type")

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_size = self.env.action_space.n
        else:
            raise TypeError("Unsupported action space type")

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['learning_rate'])
        self.memory = ReplayBuffer(self.config['replay_buffer_size'], self.config['batch_size'], self.config['seed'])
        self.t_step = 0

    def select_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.env.action_space.n))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1)
        if len(self.memory) > self.config['batch_size']:
            experiences = self.memory.sample()
            self.learn(experiences, self.config['gamma'])
        
        if self.t_step % self.config['target_update_freq'] == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from local model
        q_targets_next_local = self.qnetwork_local(next_states).detach()
        best_actions = q_targets_next_local.argmax(1).unsqueeze(1)

        # Get Q values for next states from target model
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        epsilon = self.config['epsilon_start']
        for i_episode in range(1, self.config['episodes']+1):
            state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
            scores_window.append(score)
            scores.append(score)
            epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * epsilon)
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        
        os.makedirs(os.path.dirname(self.config['save_path']), exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), self.config['save_path'])
        
        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)
