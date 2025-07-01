import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_rewards(log_paths, output_dir='logs'):
    """Plots the rewards from multiple CSV files and saves the plot."""
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found at {log_path}. Skipping.")
            continue

        df = pd.read_csv(log_path)
        algo_name = os.path.basename(log_path).replace('_cartpole_rewards.csv', '').replace('_', ' ').upper()
        plt.plot(df['reward'], label=f'{algo_name} Reward per Episode', alpha=0.3)
        plt.plot(df['reward'].rolling(100).mean(), label=f'{algo_name} 100-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('RL Algorithm Performance Comparison on CartPole-v1')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_rewards_comparison.png')
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == '__main__':
    dqn_log = 'logs/dqn_cartpole_rewards.csv'
    a2c_log = 'logs/a2c_cartpole_rewards.csv'
    ppo_log = 'logs/ppo_cartpole_rewards.csv'
    reinforce_log = 'logs/reinforce_cartpole_rewards.csv'
    ddqn_log = 'logs/ddqn_cartpole_rewards.csv'
    
    plot_rewards([dqn_log, a2c_log, ppo_log, reinforce_log, ddqn_log])
