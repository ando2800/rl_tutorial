import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
import os
import pandas as pd
from collections import deque, namedtuple

# --- SumTree Implementation ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.ptr = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha, beta_start, beta_frames):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 0.01 # Small value to ensure non-zero priority
        self.max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []
        
        self.beta = self.beta_by_frame(self.frame)
        self.frame += 1

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data is not None:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones), idxs, torch.from_numpy(is_weight).float()

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.tree.n_entries

# --- QNetwork (same as in dqn.py) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        return self.fc3(x)

# --- PER DQN Trainer ---
class PERDQNTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config['env_name'])

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config['learning_rate'])
        
        self.memory = PrioritizedReplayBuffer(
            self.config['replay_buffer_size'], 
            self.config['batch_size'],
            self.config['per_alpha'],
            self.config['per_beta_start'],
            self.config['per_beta_frames']
        )
        
        self.epsilon = self.config['epsilon_start']
        self.current_loss = 0.0

    def _normalize_state(self, state):
        # Optional: Normalize state if needed
        return state

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

    def _learn(self):
        (states, actions, rewards, next_states, dones), idxs, is_weights = self.memory.sample()

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        q_targets = rewards + (self.config['gamma'] * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        td_errors = torch.abs(q_targets - q_expected).detach().numpy()
        loss = (F.mse_loss(q_expected, q_targets, reduction='none') * is_weights.unsqueeze(1)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(idxs, td_errors.flatten())

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.config['tau'])
        
        return loss.item()

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        
        print("Starting PER-DQN training...")
        for i_episode in range(1, self.config['episodes'] + 1):
            state, _ = self.env.reset()
            state = self._normalize_state(state)
            score = 0
            done = False
            while not done:
                action = self.select_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self._normalize_state(next_state)
                
                self.memory.add(state, action, reward, next_state, done)
                
                if len(self.memory) > self.config['batch_size']:
                    self.current_loss = self._learn()

                state = next_state
                score += reward

            scores_window.append(score)
            scores.append(score)
            self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)
            
            avg_score = np.mean(scores_window)
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {self.epsilon:.3f}\tLoss: {self.current_loss:.4f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {self.epsilon:.3f}\tLoss: {self.current_loss:.4f}')
                # Save model checkpoint
                os.makedirs(os.path.dirname(self.config['save_path']), exist_ok=True)
                torch.save(self.qnetwork_local.state_dict(), self.config['save_path'])

        # Save final model and logs
        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)
        print(f"\nTraining complete. Logs saved to {self.config['log_path']}")
