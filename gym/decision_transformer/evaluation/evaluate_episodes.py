import numpy as np
import torch
import time

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state,_ = env.reset()
    # state=env.reset()
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    eval_episode_start = time.time()
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # state, reward, done, _ = env.step(action)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break
    episode_eval_time = time.time() - eval_episode_start
    return episode_return, episode_length, episode_eval_time


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        max_context_len=20,
    ):

    model.eval()
    model.set_eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state,_ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []
    past_key_values = None
    episode_return, episode_length = 0, 0
    eval_episode_start = time.time()
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        #if past_key_values have more than max_context_len, remove the first one
        if past_key_values is not None:
            for i in range(len(past_key_values)):
                if past_key_values[i] is not None:
                    #past_key_values[i] has shape (batch_size,sequence_length,hidden_size)
                    #remove the first sequence_length-max_context_len elements from past_key_values[i], if present
                    if past_key_values[i].shape[1] >= max_context_len-1:
                        past_key_values[i] = past_key_values[i][:, -(max_context_len-2):,:]

        # Get the current inputs for this timestep
        if model.use_cache:
            # # With caching: only process the latest observation
            current_state = ((states[-1:].to(dtype=torch.float32) - state_mean) / state_std).reshape(1, 1, state_dim)  # Latest state (padding)
            current_action = actions[-1:].to(dtype=torch.float32).reshape(1,1,act_dim)  # Latest action (padding)
            current_reward = rewards[-1:].to(dtype=torch.float32).reshape(1,1,1)  # Latest reward (padding)
            current_return = target_return[:, -1:].to(dtype=torch.float32).reshape(1,1,1)  # Latest target return
            current_timestep = timesteps[:, -1:].to(dtype=torch.long).reshape(1,1)  # Latest timestep
            
            action, past_key_values = model.get_action(
                current_state,
                current_action, 
                current_reward,
                current_return,
                current_timestep,
                past_key_values=past_key_values,
            )
            # current_state = ((state.to(dtype=torch.float32) - state_mean) / state_std).reshape(1, 1, state_dim)
            # current_action = actions[-1:].reshape(1, 1, act_dim)
            # current_reward = rewards[-1:].reshape(1, 1, 1)
            # current_timestep = torch.as_tensor([t]).reshape(1, 1).to(device=device, dtype=torch.long)
            # current_return = target_return.reshape(1, 1, 1)

            # action, past_key_values = model.get_action(
            #     current_state,
            #     current_action, 
            #     current_reward,
            #     current_return,
            #     current_timestep,
            #     past_key_values=past_key_values,
            # )
        else:
            # First step or no caching: process the entire sequence
            action, past_key_values = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                past_key_values=None,  # First call has no past key values
            )
        # action,past_key_values = model.get_action(
        #         (states.to(dtype=torch.float32) - state_mean) / state_std,
        #         actions.to(dtype=torch.float32),
        #         rewards.to(dtype=torch.float32),
        #         target_return.to(dtype=torch.float32),
        #         timesteps.to(dtype=torch.long),
        #         past_key_values=past_key_values,
        #     )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # state, reward, done, _ = env.step(action)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        # cache_size=model.get_cache_size()
        if done:
            break
    cache_size = 0
    for layer_cache in past_key_values:
        if layer_cache is not None:
            cache_size+=layer_cache.element_size() * layer_cache.nelement()
    episode_eval_time = time.time() - eval_episode_start
    return episode_return, episode_length, cache_size, episode_eval_time
