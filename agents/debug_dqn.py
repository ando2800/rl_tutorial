import torch
import numpy as np
from dqn import DQNTrainer
import gymnasium as gym
import pprint

# コンフィグ設定（必要に応じて修正）
config = {
    'env_name': 'CartPole-v1',
    'episodes': 8,  # エピソード数を3に設定してReplayBufferを充足させる
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.99,
    'replay_buffer_size': 10000,
    'batch_size': 64,
    'target_update_freq': 10,
    'seed': 42,
    'save_path': 'models/debug_dqn.pt',
    'log_path': 'logs/debug_dqn.csv'
}

# インスタンス化
agent = DQNTrainer(config)
env = agent.env

# ✅ Qネットワークの初期出力確認
obs, _ = env.reset()
obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
with torch.no_grad():
    q_output = agent.qnetwork_local(obs_tensor)
print("\n[初期Qネット出力]")
print(q_output)

# ✅ select_action の挙動確認
print("\n[select_action 出力（epsilon=0.0 強制）]")
for i in range(5):
    action = agent.select_action(obs, epsilon=0.0)
    print(f"Action {i}: {action}")

# ✅ 複数エピソード実行してReplayBuffer充足と学習確認
for episode in range(config['episodes']):
    print(f"\n[エピソード {episode+1} の環境挙動ログ]")
    state, _ = env.reset()
    score = 0
    for t in range(1, 100):
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"t={t}, action={action}, reward={reward}, done={done}")
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    print(f"→ 終了時スコア: {score}")

# ✅ ReplayBufferの中身チェック
print(f"\nReplayBufferのサイズ: {len(agent.memory)}")
if len(agent.memory) >= agent.config['batch_size']:
    batch = agent.memory.sample()
    states, actions, rewards, next_states, dones = batch
    print("ReplayBufferからサンプリングした状態のshape:", states.shape)

# ✅ 学習1回分だけ実行
if len(agent.memory) >= agent.config['batch_size']:
    loss = agent.learn(batch, config['gamma'])
    print(f"\n学習1回目のloss: {loss:.4f}")
else:
    print("\nReplayBufferが未充足のため学習スキップ")

print("\n✅ デバッグ完了。各出力を確認してください。")