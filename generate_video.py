
import torch
import gymnasium as gym
from agents.dqn import DQNTrainer, QNetwork
import yaml

def generate_video(config_path='configs/dqn_cartpole.yaml', video_folder='logs/videos'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    qnetwork = QNetwork(state_size, action_size)
    qnetwork.load_state_dict(torch.load(config['save_path']))
    qnetwork.eval()

    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = qnetwork(state_tensor)
        action = action_values.argmax().item()
        state, reward, done, _, _ = env.step(action)

    env.close()
    print(f"Video saved in {video_folder}")

if __name__ == '__main__':
    generate_video()
