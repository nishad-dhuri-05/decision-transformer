import pickle
import numpy as np

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

with open("halfcheetah-medium-v0.pkl", "rb") as f:
    data = pickle.load(f)

for traj in data:
    rewards = traj['rewards']
    returns_to_go = discount_cumsum(x=rewards, gamma=1)
    traj['returns_to_go'] = returns_to_go

with open("halfcheetah-medium-v0_with_rtg.pkl", "wb") as f:
    pickle.dump(data, f)