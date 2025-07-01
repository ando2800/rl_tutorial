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

## ===== DDQN Agent Implementation ===== ##
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

class DDQNTrainer:
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
        self.best_avg_score = -float('inf')
        self.epsilon = self.config['epsilon_start']
        self.current_loss = 0.0

    def _normalize_state(self, state):
        state[0] /= 2.5
        state[1] /= 2.5
        state[2] /= 0.3
        state[3] /= 0.3
        return state

    def _state_reward(self, state, env_reward):
        return env_reward - (abs(state[0]) + abs(state[2])) / 2.5

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

    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # DDQN: Use local network to select actions, target network to evaluate them
        q_local_next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        q_target_next = self.qnetwork_target(next_states).detach().gather(1, q_local_next_actions)

        q_targets = rewards + (gamma * q_target_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.smooth_l1_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        best_test_reward = -float('inf')

        print("Filling replay buffer with random experience...")
        state, _ = self.env.reset()
        state = self._normalize_state(state)
        for _ in range(max(self.config['batch_size'], 1000)):
            action = self.env.action_space.sample()
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_state = self._normalize_state(next_state)
            self.memory.add(state, action, env_reward, next_state, done)
            state = next_state
            if done:
                state, _ = self.env.reset()
                state = self._normalize_state(state)
        print("Replay buffer filled.")

        print("Starting training...")
        for i_episode in range(1, self.config['episodes'] + 1):
            state, _ = self.env.reset()
            state = self._normalize_state(state)
            score = 0
            reward_sum = 0
            done = False
            while not done:
                action = self.select_action(state, self.epsilon)
                next_state, env_reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self._normalize_state(next_state)
                
                self.memory.add(state, action, env_reward, next_state, done)
                self.current_loss = self._learn(self.memory.sample(), self.config['gamma'])

                state = next_state
                score += env_reward
                reward_sum += self._state_reward(next_state, env_reward)

            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)
            self.epsilon = max(self.config['epsilon_end'], self.config['epsilon_decay'] * self.epsilon)

            if i_episode % self.config['target_update_freq'] == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
                self.qnetwork_target.eval()

            print(f"Episode {i_episode} | Avg Score: {avg_score:.2f} | Eps: {self.epsilon:.3f} | Loss: {self.current_loss:.4f}")

            if (i_episode % self.config['test_delay'] == 0) or (i_episode == self.config['episodes']):
                test_score, test_reward = self._test_agent()
                print(f'Test Episode {i_episode}: test score: {test_score} - test reward: {test_reward}')
                if test_reward > best_test_reward:
                    print('New best test reward. Saving model')
                    best_test_reward = test_reward
                    os.makedirs(os.path.dirname(self.config['save_path']), exist_ok=True)
                    torch.save(self.qnetwork_local.state_dict(), self.config['save_path'])

        pd.DataFrame(scores, columns=['reward']).to_csv(self.config['log_path'], index=False)

    def _test_agent(self):
        state, _ = self.env.reset()
        state = self._normalize_state(state)
        done = False
        score = 0
        reward_sum = 0
        while not done:
            action = self.select_action(state, 0.0) # Use 0.0 epsilon for testing
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_state = self._normalize_state(next_state)
            state = next_state
            score += env_reward
            reward_sum += self._state_reward(next_state, env_reward)
        return score, reward_sum