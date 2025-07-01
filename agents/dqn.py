import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import namedtuple, deque
import os
import pandas as pd

## ===== DQN Agent Implementation ===== ##
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
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.float32))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
## ===== DQN Trainer Implementation ===== ##


class DQNTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

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

        if np.random.rand() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.env.action_space.n))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1)
        if len(self.memory) >= self.config['batch_size']:
            experiences = self.memory.sample()
            loss = self.learn(experiences, self.config['gamma'])
            return loss
        return None

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next_local = self.qnetwork_local(next_states).detach()
        best_actions = q_targets_next_local.argmax(1).unsqueeze(1)
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.smooth_l1_loss(q_expected, q_targets)  # ← Huber loss に変更
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        epsilon = self.config['epsilon_start']
        best_avg_score = -float('inf')

        print("Filling replay buffer with random experience...")
        while len(self.memory) < max(self.config['batch_size'], 1000):  # ← より多めに蓄積
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

        print("Starting training...")
        for i_episode in range(1, self.config['episodes'] + 1):
            reset_output = self.env.reset()
            state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            score = 0
            done = False
            loss = None
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                loss = self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if self.t_step > 0 and self.t_step % self.config['target_update_freq'] == 0:  # ← 更新条件を厳密に
                    self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)
            epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * epsilon)

            if avg_score > best_avg_score:
                best_avg_score = avg_score
                os.makedirs(os.path.dirname(self.config['save_path']), exist_ok=True)
                torch.save(self.qnetwork_local.state_dict(), self.config['save_path'])
                print(f"New best model saved with average score: {avg_score:.2f}")

            loss_value = loss if loss is not None else 0.0
            print(f"Episode {i_episode} | Avg Score: {np.mean(scores_window):.2f} | Eps: {epsilon:.3f} | Loss: {loss_value:.4f}")
        
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)

