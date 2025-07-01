
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(log_path='logs/dqn_cartpole_rewards.csv', output_path='logs/rewards_plot.png'):
    """Plots the rewards from a CSV file and saves the plot."""
    df = pd.read_csv(log_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(df['reward'], label='Reward per Episode')
    plt.plot(df['reward'].rolling(100).mean(), label='100-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN CartPole Rewards')
    plt.legend()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_rewards()
