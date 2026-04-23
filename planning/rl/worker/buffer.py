import numpy as np
import torch
import joblib
from collections import defaultdict
from planning.rl.util.torch_util import combined_shape, discount_cumsum, to_tensor


class OnPolicyBuffer:
    r"""
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.obs_dim, self.act_dim = obs_dim, act_dim
        #I MANUALLY CHANGE THE OBS DIM BC I WAS TOO LAZY TO FIX IT
        #self.obs_dim = (3, 64, 64)
        self.max_size = size
        self.clear()

    def clear(self):
        self.obs_buf = np.zeros(combined_shape(self.max_size, self.obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, self.act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp, done, cost=0, cost_val=0):
        r"""
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_rew_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0):
        r"""
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        cost_rews = np.append(self.cost_rew_buf[path_slice], last_cost_val)
        cost_vals = np.append(self.cost_val_buf[path_slice], last_cost_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # the next two lines implement GAE-Lambda advantage calculation
        cost_deltas = cost_rews[:-1] + self.gamma * cost_vals[1:] - cost_vals[:-1]
        self.cost_adv_buf[path_slice] = discount_cumsum(cost_deltas,
                                                        self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.cost_ret_buf[path_slice] = discount_cumsum(cost_rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        r"""
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr > 0  # buffer has to have something before you can get
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf[:self.ptr]), np.std(
            self.adv_buf[:self.ptr])
        self.adv_buf[:self.ptr] = (self.adv_buf[:self.ptr] - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf[:self.ptr],
            act=self.act_buf[:self.ptr],
            ret=self.ret_buf[:self.ptr],
            adv=self.adv_buf[:self.ptr],
            cost_ret=self.cost_ret_buf[:self.ptr],
            cost_adv=self.cost_adv_buf[:self.ptr],
            logp=self.logp_buf[:self.ptr],
            done=self.done_buf[:self.ptr],
        )
        return to_tensor(data, dtype=torch.float32, device=torch.device("cuda"))



class OnPolicySequentialBuffer:
    r"""
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs with recurrence.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.episodes = []

        self.gamma, self.lam = gamma, lam
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.max_size = size
        self.T = 32
        self.B = 8
        self.clear()

    def clear(self):
        self.episodes = [defaultdict(list)]


    def store(self, obs, act, rew, val, logp, done, cost=0, cost_val=0):
        r"""
        Append one timestep of agent-environment interaction to the buffer. Append to most recent episode
        """
        current_episode = self.episodes[-1]

        current_episode['obs'].append(obs)
        current_episode['act'].append(act)
        current_episode['rew'].append(rew)
        current_episode['val'].append(val)
        current_episode['cost_rew'].append(cost)
        current_episode['cost_val'].append(cost_val)
        current_episode['logp'].append(logp)
        current_episode['done'].append(done)

    def finish_path(self, last_val=0, last_cost_val=0):
        r"""
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        #path_slice = slice(self.path_start_idx, self.ptr) #Episodes should all be in one
        current_episode = self.episodes[-1]

        rews = np.append(current_episode['rew'], last_val)
        vals = np.append(current_episode['val'], last_val)

        cost_rews = np.append(current_episode['cost_rew'], last_cost_val)
        cost_vals = np.append(current_episode['cost_val'], last_cost_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        current_episode['adv'] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        current_episode['ret'] = discount_cumsum(rews, self.gamma)[:-1]

        # the next two lines implement GAE-Lambda advantage calculation
        cost_deltas = cost_rews[:-1] + self.gamma * cost_vals[1:] - cost_vals[:-1]
        current_episode['cost_adv']= discount_cumsum(cost_deltas,
                                                        self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        current_episode['cost_ret'] = discount_cumsum(cost_rews, self.gamma)[:-1]

        self.episodes.append(defaultdict(list)) #Start a new episode dictionary

    def get(self):
        r"""
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        T = self.T
        B = self.B
        assert len(self.episodes) > 1  # buffer has to have something before you can get. Will always have the last episode be empty or partially filled so ignore it
        # the next two lines implement the advantage normalization trick
        assert T > B #burn in must be shorter
        data = defaultdict(list)
        
        all_adv = np.concatenate([ep['adv'] for ep in self.episodes[:-1]])
        mean, std = all_adv.mean(), all_adv.std()

        for ep in self.episodes[:-1]:
            ep['adv'] = (ep['adv'] - mean) / (std + 1e-8)

        for ep in self.episodes[:-1]:
            ep_len = len(ep['obs'])
            start = 0
            while start < ep_len:
                
                #Handle indexing first with burn in
                burn_len = min(B, start)
                ind_start = start - burn_len
                end = start + T
                
                actual_len = min(end, len(ep['obs'])) - ind_start
                pad_size = (T+B) - (end - ind_start)
                for k in ep:
                    chunk = self.pad_to_T(ep[k][ind_start:end], T+B)
                    data[k].append(chunk)

                valid = np.zeros( T + B, dtype=np.float32)
                valid[:actual_len] = 1
                valid[0:burn_len] = 0
                data['mask'].append(valid)
                start += T
        for k in data:
            data[k] = np.stack(data[k], axis = 0)
        tensor_dict = to_tensor(data, dtype=torch.float32)
        return tensor_dict
    

    def pad_to_T(self, x, T, pad_value=0):
        x = np.array(x)  # ensure it's an ndarray
        pad_len = T - len(x)
        if pad_len <= 0:
            return x
        # handle both 1D and higher-D arrays
        pad_shape = (pad_len, *x.shape[1:]) if x.ndim > 1 else (pad_len,)
        return np.concatenate([x, np.full(pad_shape, pad_value)])

