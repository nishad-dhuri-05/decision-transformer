import gymnasium as gym
import numpy as np

import collections
import pickle

import minari

datasets = []

for env_name in ['halfcheetah', 'hopper', 'walker2d']:
    # for dataset_type in ['medium', 'simple', 'expert']:
    for dataset_type in ['medium','expert']:
        name = f'mujoco/{env_name}/{dataset_type}-v0'
        dataset = minari.load_dataset(name,download=True)

        data_ = collections.defaultdict(list)
        paths = []
        # for episode in dataset.iterate_episodes():
        #     for transition in episode:
        #         data_['observations'].append(transition.observation)
        #         data_['next_observations'].append(transition.next_observation)
        #         data_['actions'].append(transition.action)
        #         data_['rewards'].append(transition.reward)
        #         data_['terminals'].append(transition.terminal)
        #     # Create episode data after finishing current episode
        #     episode_data = {k: np.array(v) for k, v in data_.items()}
        #     paths.append(episode_data)
        #     data_ = collections.defaultdict(list)
        
        for episode in dataset.iterate_episodes():
            length = len(episode.rewards)
            for i in range(length):
                data_['observations'].append(episode.observations[i])

                # Infer next observation manually
                if i + 1 < length:
                    next_obs = episode.observations[i + 1]
                else:
                    next_obs = episode.observations[i]  # or skip, depending on context
            
                data_['next_observations'].append(next_obs)
                data_['actions'].append(episode.actions[i])
                data_['rewards'].append(episode.rewards[i])
                done = episode.terminations[i] or episode.truncations[i]
                data_['terminals'].append(done)
            episode_data = {k: np.array(v) for k, v in data_.items()}
            paths.append(episode_data)
            data_ = collections.defaultdict(list)



        returns = np.array([np.sum(p['rewards']) for p in paths])
        num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        print(f'Number of samples collected: {num_samples}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
        file_name=name.split('/')[1]+'-'+name.split('/')[2]
        with open(f'{file_name}.pkl', 'wb') as f:
            pickle.dump(paths, f)
