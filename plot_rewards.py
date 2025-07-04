import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison(log_paths, output_dir='results/plots', window=100):
    """Plots the moving average of rewards from multiple CSV files for comparison."""
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))

    for log_path in log_paths:
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found at {log_path}. Skipping.")
            continue

        df = pd.read_csv(log_path)
        algo_name = os.path.basename(log_path).replace('_cartpole_rewards.csv', '').replace('_', ' ').upper()
        plt.plot(df['reward'].rolling(window).mean(), label=f'{algo_name} {window}-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('RL Algorithm Performance Comparison on CartPole-v1 (Moving Average)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_rewards_comparison.png')
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")
    plt.close()

def plot_individual_rewards(log_path, output_dir='results/plots', window=100):
    """Plots the reward and moving average for a single algorithm."""
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}. Skipping.")
        return

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 8))

    df = pd.read_csv(log_path)
    algo_name = os.path.basename(log_path).replace('_cartpole_rewards.csv', '').replace('_', ' ').upper()
    
    plt.plot(df['reward'], label=f'{algo_name} Reward per Episode', alpha=0.3)
    plt.plot(df['reward'].rolling(window).mean(), label=f'{algo_name} {window}-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{algo_name} Performance on CartPole-v1')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{algo_name.lower().replace(" ", "_")}_rewards.png')
    plt.savefig(output_path)
    print(f"Individual plot for {algo_name} saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    log_dir = 'experiments/cartpole/logs'
    output_dir = 'results/plots'
    
    log_files = [
        os.path.join(log_dir, 'dqn_cartpole_rewards.csv'),
        os.path.join(log_dir, 'a2c_cartpole_rewards.csv'),
        os.path.join(log_dir, 'ppo_cartpole_rewards.csv'),
        os.path.join(log_dir, 'reinforce_cartpole_rewards.csv'),
        os.path.join(log_dir, 'ddqn_cartpole_rewards.csv'),
        os.path.join(log_dir, 'per_dqn_cartpole_rewards.csv')
    ]

    # Generate and save the comparison plot
    plot_comparison(log_files, output_dir=output_dir)

    # Generate and save individual plots
    for log_file in log_files:
        plot_individual_rewards(log_file, output_dir=output_dir)