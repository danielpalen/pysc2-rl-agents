import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

from pysc2.lib.actions import FunctionCall, FUNCTIONS

from rl.agents.runner import BaseRunner
from rl.common.pre_processing import Preprocessor
from rl.common.pre_processing import is_spatial_action, stack_ndarray_dicts
from rl.common.util import mask_unused_argument_samples, flatten_first_dims, flatten_first_dims_dict

class FeudalRunner(BaseRunner):

    def __init__(self, agent, envs, summary_writer, args):
        """
        Args:
             agent: A2CAgent instance.
             envs: SubprocVecEnv instance.
             summary_writer: summary writer to log episode scores.
             args: {

             }
        """

        self.agent = agent
        self.envs = envs
        self.summary_writer = summary_writer

        self.train = args.train
        self.c = args.c
        self.d = args.d
        self.T = args.steps_per_batch #T is the length of the actualy trajectory
        self.n_steps = 2 * self.c + self.T
        self.discount = args.discount

        self.preproc = Preprocessor()
        self.last_obs = self.preproc.preprocess_obs(self.envs.reset())

        print('\n### Feudal Runner #######')
        print(f'# agent = {self.agent}')
        print(f'# train = {self.train}')
        print(f'# n_steps = {self.n_steps}')
        print(f'# discount = {self.discount}')
        print('######################\n')

        self.states = agent.initial_state # Holds the managers and workers hidden states
        self.last_c_goals = np.zeros((self.envs.n_envs,self.c,self.d))

        self.episode_counter = 1
        self.max_score = 0.0
        self.cumulative_score = 0.0


    def run_batch(self, train_summary):

        last_obs = self.last_obs
        shapes   = (self.n_steps, self.envs.n_envs)
        values   = np.zeros(np.concatenate([[2], shapes]), dtype=np.float32) #first dim: manager values, second dim: worker values
        rewards  = np.zeros(shapes, dtype=np.float32)
        dones    = np.zeros(shapes, dtype=np.float32)
        all_obs, all_actions = [], []
        mb_states = self.states #first dim: manager values, second dim: worker values
        s = np.zeros((self.n_steps, self.envs.n_envs, self.d), dtype=np.float32)
        mb_last_c_goals = np.zeros((self.n_steps, self.envs.n_envs, self.c, self.d), dtype=np.float32)

        for n in range(self.n_steps):
            actions, values[:,n,:], states, s[n,:,:], self.last_c_goals = self.agent.step(last_obs, self.states, self.last_c_goals)
            actions = mask_unused_argument_samples(actions)

            all_obs.append(last_obs)
            all_actions.append(actions)
            mb_last_c_goals[n,:,:] = self.last_c_goals

            pysc2_actions = actions_to_pysc2(actions, size=last_obs['screen'].shape[1:3])
            obs_raw  = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw)
            rewards[n,:], dones[n,:] = zip(*[(t.reward,t.last()) for t in obs_raw])
            self.states = states

            for t in obs_raw:
                if t.last():
                    self.cumulative_score += self._summarize_episode(t)

        #next_values = self.agent.get_value(last_obs, states, self.last_c_goals)

        returns, returns_intr, adv_m, adv_w = compute_returns_and_advantages(
            rewards, dones, values, s, mb_last_c_goals[:,:,-1,:], self.discount, self.T, self.envs.n_envs, self.c
        )
        s_diff = compute_sdiff(s, self.c, self.T, self.envs.n_envs, self.d)
        # last_c_goals = compute_last_c_goals(goals, self.envs.n_envs, self.T, self.c, self.d)
        actions = stack_and_flatten_actions(all_actions[self.c:self.c+self.T])
        obs = stack_ndarray_dicts(all_obs)
        obs = { k:obs[k][self.c:self.c+self.T] for k in obs }
        obs = flatten_first_dims_dict(obs)
        returns = flatten_first_dims(returns)
        returns_intr = flatten_first_dims(returns_intr)
        adv_m = flatten_first_dims(adv_m)
        adv_w = flatten_first_dims(adv_w)
        s_diff = flatten_first_dims(s_diff)
        mb_last_c_goals = flatten_first_dims(mb_last_c_goals[self.c:self.c+self.T])
        self.last_obs = last_obs

        if self.train:
            return self.agent.train(
                obs,
                mb_states,
                actions,
                returns, returns_intr,
                adv_m, adv_w,
                s_diff,
                mb_last_c_goals,
                summary=train_summary
            )
        else:
            return None


    def get_mean_score(self):
        return cumulative_score / episode_counter


    def get_max_score(self):
        return max_score


def compute_sdiff(s, c, T, nenvs, d):
    s_diff = np.zeros((T,nenvs,d))
    for t in range(T):
        s_diff[t,:,:] = s[t+2*c,:,:] - s[t+c,:,:]
    return s_diff


# def compute_last_c_goals(goals, nenvs, T, c, d):
#     last_c_g = np.zeros((T,nenvs,c,d))
#     # goal (nsteps, nenvs, d)
#     for t in range(c,c+T):
#         last_c_g[t-c] = np.transpose(goals[t-c:t], (1,0,2))
#     return last_c_g


def compute_returns_and_advantages(rewards, dones, values, s, goals, discount, T, nenvs, c):
    alpha = 0.5

    # print('s', s.shape)
    # print('goals', goals.shape)

    # Intrinsic rewards
    r_i = np.zeros((T+1,nenvs))
    for t in range(c,c+T+1):
        sum_cos_dists = 0
        for env in range(nenvs):
            for i in range(1,c):
                _s,_g = s[t,env]-s[t-i,env], goals[t-i,env]
                den = np.expand_dims(_s,axis=0)@np.expand_dims(_g,axis=1)
                num = np.linalg.norm(_s)*np.linalg.norm(_g)+1e-8
                sum_cos_dists += den/num
            r_i[t-c,env] = 1/c * sum_cos_dists
    # print('r_i', r_i.shape)

    # Returns
    returns = np.zeros((T+1, nenvs))
    returns_intr = np.zeros((T+1, nenvs))
    returns[-1,:] = values[0,c+T]
    returns_intr[-1,:] = r_i[-1]
    for t in reversed(range(T)):
        returns[t,:] = rewards[t+c,:] + discount * returns[t+1,:] * (1-dones[t+c,:])
        returns_intr[t,:] = r_i[t,:] + discount * returns_intr[t+1,:] * (1-dones[t+c,:])
    returns = returns[:-1,:]
    returns_intr = returns_intr[:-1,:]
    adv_m = returns - values[0,c:c+T,:]
    adv_w = returns + alpha * returns_intr - values[1,c:c+T,:]
    return returns, returns_intr, adv_m, adv_w


def actions_to_pysc2(actions, size):
    """Convert agent action representation to FunctionCall representation."""
    height, width = size
    fn_id, arg_ids = actions
    actions_list = []
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        a_l = []
        for arg_type in FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type][n]
            if is_spatial_action[arg_type]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = FunctionCall(a_0, a_l)
        actions_list.append(action)
    return actions_list


def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return (fn_id, arg_ids)
