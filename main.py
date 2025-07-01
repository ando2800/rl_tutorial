import yaml
import argparse
from agents.dqn import DQNTrainer
from agents.reinforce import REINFORCETrainer
from agents.a2c import A2CTrainer
from agents.ppo import PPOTrainer
from agents.ddqn import DDQNTrainer

def main():
    parser = argparse.ArgumentParser(description='Run RL training.')
    parser.add_argument('--algorithm', type=str, default='DQN', help='RL algorithm to use (DQN or REINFORCE)')
    args = parser.parse_args()

    if args.algorithm == 'DQN':
        config_path = "configs/dqn_cartpole.yaml"
        trainer_class = DQNTrainer
    elif args.algorithm == 'REINFORCE':
        config_path = "configs/reinforce_cartpole.yaml"
        trainer_class = REINFORCETrainer
    elif args.algorithm == 'A2C':
        config_path = "configs/a2c_cartpole.yaml"
        trainer_class = A2CTrainer
    elif args.algorithm == 'PPO':
        config_path = "configs/ppo_cartpole.yaml"
        trainer_class = PPOTrainer
    elif args.algorithm == 'DDQN':
        config_path = "configs/ddqn_cartpole.yaml"
        trainer_class = DDQNTrainer
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = trainer_class(config)
    trainer.train()

if __name__ == '__main__':
    main()