import yaml
import argparse
from agents.dqn import DQNTrainer
from agents.reinforce import REINFORCETrainer

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
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = trainer_class(config)
    trainer.train()

if __name__ == '__main__':
    main()