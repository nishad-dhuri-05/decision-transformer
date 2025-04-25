import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import random
from collections import deque

# Actor network for continuous actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Critic network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6):
        self.capacity = capacity
        self.alpha = alpha        # Priority exponent
        self.beta = beta          # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Convert priorities to sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# CQL-SAC with Per-DDQN implementation
class CQLSAC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        
        # Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Critics (Double Q-learning)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005  # target network update rate
        self.alpha = 0.2  # entropy coefficient
        self.cql_weight = 1.0  # conservative Q-learning weight
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100):
        # Sample from replay buffer with priorities
        state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)
        weights = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)
        
        # Get next action from current policy
        next_action = self.actor(next_state)
        
        # Get target Q values
        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Get current Q values
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        
        # Calculate TD errors for priority updates
        td_error1 = torch.abs(current_Q1 - target_Q).detach().cpu().numpy()
        td_error2 = torch.abs(current_Q2 - target_Q).detach().cpu().numpy()
        td_errors = np.mean([td_error1, td_error2], axis=0)
        
        # Update priorities
        replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Critic loss with importance sampling weights
        critic1_loss = (weights * F.mse_loss(current_Q1, target_Q, reduction='none')).mean()
        critic2_loss = (weights * F.mse_loss(current_Q2, target_Q, reduction='none')).mean()
        
        # Conservative Q-Learning regularization
        # Sample random actions for regularization
        batch_size = state.shape[0]
        random_actions = torch.FloatTensor(batch_size, action.shape[1]).uniform_(-self.max_action, self.max_action).to(self.device)
        
        # Q-values for random actions
        random_q1 = self.critic1(state, random_actions)
        random_q2 = self.critic2(state, random_actions)
        
        # Q-values for actions from the current policy
        current_actions = self.actor(state)
        policy_q1 = self.critic1(state, current_actions)
        policy_q2 = self.critic2(state, current_actions)
        
        # CQL regularization terms (expectation over random actions - Q-value of dataset actions)
        cql_reg1 = torch.logsumexp(random_q1, dim=0).mean() - current_Q1.mean()
        cql_reg2 = torch.logsumexp(random_q2, dim=0).mean() - current_Q2.mean()
        
        # Add CQL regularization to critic loss
        critic1_loss += self.cql_weight * cql_reg1
        critic2_loss += self.cql_weight * cql_reg2
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic1(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'actor_loss': actor_loss.item(),
            'cql_reg': (cql_reg1.item() + cql_reg2.item()) / 2
        }

# Training function
def train_offline_rl(env_name, dataset_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_name = f"{env_name}-{dataset_type}-v0"
    with open(f'{file_name}.pkl', 'rb') as f:
        paths = pickle.load(f)
    
    # Get environment dimensions
    if env_name == 'halfcheetah':
        state_dim = 17
        action_dim = 6
    elif env_name == 'hopper':
        state_dim = 11
        action_dim = 3
    elif env_name == 'walker2d':
        state_dim = 17
        action_dim = 6
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    max_action = 1.0  # MuJoCo environments typically have actions in [-1, 1]
    
    # Initialize algorithm
    policy = CQLSAC(state_dim, action_dim, max_action, device)
    
    # Create replay buffer from offline dataset
    buffer_size = sum([len(p['rewards']) for p in paths])
    replay_buffer = PrioritizedReplayBuffer(buffer_size)
    
    # Populate replay buffer
    for path in paths:
        obs = path['observations']
        actions = path['actions']
        rewards = path['rewards']
        next_obs = path['next_observations']
        dones = path['terminals']
        
        for i in range(len(rewards)):
            replay_buffer.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
    
    print(f"Buffer size: {len(replay_buffer)}")
    
    # Training loop
    iterations = 5000
    batch_size = 256
    eval_freq = 500
    
    for t in range(iterations):
        train_info = policy.train(replay_buffer, batch_size)
        
        # Print training information
        if (t + 1) % 1000 == 0:
            print(f"Iteration: {t+1}, Critic Loss: {train_info['critic_loss']:.3f}, "
                  f"Actor Loss: {train_info['actor_loss']:.3f}, CQL Reg: {train_info['cql_reg']:.3f}")
    
    # Save trained policy
    torch.save(policy.actor.state_dict(), f"{file_name}_actor.pth")
    torch.save(policy.critic1.state_dict(), f"{file_name}_critic1.pth")
    torch.save(policy.critic2.state_dict(), f"{file_name}_critic2.pth")
    print(f"Training complete. Model saved as {file_name}_actor.pth")
    return policy

def update_dataset_with_q_lower_bound(paths, policy, discount=0.99):
    """
    For each trajectory in paths, update the return-to-go (RTG) using the Q-value lower bound.
    For each time step (backwards), compute the standard RTG; if this value is less than the Q-value
    lower bound (min(Q1, Q2) for that state and action), replace it with the Q-value.
    Returns the updated paths and the count of total state updates.
    """
    update_count = 0
    updated_paths = []
    
    policy.critic1.eval()
    policy.critic2.eval()
    print("Total trajectories to process:", len(paths))
    ind=0
    with torch.no_grad():
        for traj in paths:
            print(f"Processing trajectory {ind+1}")
            ind+=1
            obs = traj['observations']
            actions = traj['actions']
            rewards = traj['rewards']
            rtg_list = [0] * len(rewards)
            rtg = 0
            # Process trajectory backwards:
            for t in reversed(range(len(rewards))):
                rtg = rewards[t] + discount * rtg
                # Compute Q-value lower bound for state-action pair at time t
                state = torch.FloatTensor(obs[t]).unsqueeze(0).to(policy.device)
                action = torch.FloatTensor(actions[t]).unsqueeze(0).to(policy.device)
                q1 = policy.critic1(state, action)
                q2 = policy.critic2(state, action)
                q_val = torch.min(q1, q2).item()
                # If computed return is less than q_val, update rtg
                if rtg < q_val:
                    rtg = q_val
                    update_count += 1
                rtg_list[t] = rtg
            # Store the new rtg values in the trajectory dictionary
            traj['returns_to_go'] = rtg_list
            updated_paths.append(traj)
            
    print(f"Total states updated with Q lower bound: {update_count}")
    return updated_paths, update_count

def load_policy(env_name, dataset_type, state_dim, action_dim, max_action, device):
    """
    Loads the trained actor and critic networks from disk.
    """
    file_name = f"{env_name}-{dataset_type}-v0"
    policy = CQLSAC(state_dim, action_dim, max_action, device)
    policy.actor.load_state_dict(torch.load(f"{file_name}_actor.pth", map_location=device))
    policy.critic1.load_state_dict(torch.load(f"{file_name}_critic1.pth", map_location=device))
    policy.critic2.load_state_dict(torch.load(f"{file_name}_critic2.pth", map_location=device))
    
    policy.actor.eval()
    policy.critic1.eval()
    policy.critic2.eval()
    print("Loaded trained policy from disk.")
    return policy

def evaluate_state_action(policy, state, action):
    """
    Evaluates the Q-value for a given state and action using the trained critics.
    The actor is used to output an action for a given state, and the critics compute
    the corresponding Q-values for any state-action pair.
    
    Returns the minimum Q-value from the two critics.
    """
    state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(policy.device)
    action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(policy.device)
    q1 = policy.critic1(state_tensor, action_tensor)
    q2 = policy.critic2(state_tensor, action_tensor)
    q_val = torch.min(q1, q2).item()
    return q_val

# Example usage after training:
if __name__ == "__main__":
    policy=train_offline_rl("walker2d", "medium")
    
    # Load dataset (assuming same file_name used in training)
    file_name = f"walker2d-medium-v0"
    with open(f'{file_name}.pkl', 'rb') as f:
        paths = pickle.load(f)
    
    # Note: policy is your trained CQLSAC instance from train_offline_rl.
    # If you want to use the trained policy here, you might need to load it from disk.
    # For demonstration, assume `policy` is available.
    # Update dataset with returns-to-go using Q lower bound:
    policy =load_policy("walker2d", "medium", 17, 6, 1.0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    updated_paths, count = update_dataset_with_q_lower_bound(paths, policy, discount=1)
    
    # Optionally, save the updated dataset for later use with Decision Transformer:
    with open(f'{file_name}_updated.pkl', 'wb') as f_out:
        pickle.dump(updated_paths, f_out)
    print(f"Updated dataset saved as {file_name}_updated.pkl")
    print("Total states updated with Q lower bound:", count)    


# Example usage
# if __name__ == "__main__":
#     train_offline_rl("halfcheetah", "medium")