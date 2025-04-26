import os
import random
import numpy as np
import cv2
from collections import deque
from ale_py import ALEInterface
import argparse
import pickle

class AtariEnv:
    def __init__(self, game, seed=123, history_length=4, max_episode_length=108e3):
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', seed)
        self.ale.setInt('max_num_frames_per_episode', int(max_episode_length))
        self.ale.setFloat('repeat_action_probability', 0)   # disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        
        # load game ROM
        if game == 'Pong':
            from ale_py import roms
            self.ale.loadROM(roms.Pong)
        elif game == 'Breakout':
            from ale_py import roms
            self.ale.loadROM(roms.Breakout)
        elif game == 'Seaquest':
            from ale_py import roms
            self.ale.loadROM(roms.Seaquest)
        elif game == 'Qbert':
            from ale_py import roms
            self.ale.loadROM(roms.Qbert)
        else:
            raise ValueError("Unsupported game: choose from 'Breakout', 'Pong', 'Seaquest', 'Qbert'")
            
        self.actions = self.ale.getMinimalActionSet()
        self.history_length = history_length
        self.state_buffer = deque(maxlen=history_length)
        self._reset_buffer()

    def _reset_buffer(self):
        for _ in range(self.history_length):
            self.state_buffer.append(np.zeros((84,84), dtype=np.uint8))
    
    def reset(self):
        self.ale.reset_game()
        self._reset_buffer()
        frame = self._get_state()
        self.state_buffer.append(frame)
        return np.stack(self.state_buffer, axis=0)
    
    def step(self, action):
        reward = 0
        done = False
        frame_buffer = np.zeros((2, 84, 84), dtype=np.uint8)
        # repeat action 4 times with max pooling over the last 2 frames
        for t in range(4):
            reward += self.ale.act(self.actions[action])
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            if self.ale.game_over():
                done = True
                break
        observation = np.maximum(frame_buffer[0], frame_buffer[1])
        self.state_buffer.append(observation)
        return np.stack(self.state_buffer, axis=0), reward, done

    def _get_state(self):
        screen = self.ale.getScreenGrayscale()
        # resize to 84x84
        state = cv2.resize(screen, (84,84), interpolation=cv2.INTER_LINEAR)
        return state.astype(np.uint8)

def generate_dataset(game, num_episodes, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    env = AtariEnv(game, seed=seed)
    dataset = []
    for episode in range(num_episodes):
        episode_data = {'observations': [], 'actions': [], 'rewards': []}
        state = env.reset()
        done = False
        while not done:
            # using a random agent; you can later replace this with a learned or scripted policy
            action = random.randrange(len(env.actions))
            next_state, reward, done = env.step(action)
            episode_data['observations'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            state = next_state
        dataset.append(episode_data)
        total_reward = sum(episode_data['rewards'])
        print(f"Episode {episode+1}/{num_episodes} finished. Total reward: {total_reward}")
    output_file = os.path.join(output_dir, f"{game}_dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset for {game} saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, required=True,
                        choices=['Breakout','Pong','Seaquest','Qbert'],
                        help="Atari game to generate data for")
    parser.add_argument('--episodes', type=int, default=50,
                        help="Number of episodes to generate")
    parser.add_argument('--output_dir', type=str, default='./dqn_replay',
                        help="Directory in which to save the dataset")
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    generate_dataset(args.game, args.episodes, args.output_dir, args.seed)