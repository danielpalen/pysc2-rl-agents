import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

from rl.agents.runner import BaseRunner

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
        self.c = args.worker_steps
        self.n_steps = 2*self.c + args.steps_per_batch
        self.discount = args.discount

        print('\n### Feudal Runner #######')
        print(f'# agent = {self.agent}')
        print(f'# train = {self.train}')
        print(f'# n_steps = {self.n_steps}')
        print(f'# discount = {self.discount}')
        print('######################\n')

        self.states = (None, None) # Holds the managers and workers hidden states

        self.episode_counter = 1
        self.max_score = 0.0
        self.cummulative_score = 0.0
        self.d #TODO: read from args

        #TODO: set t % c

    def run_batch(self, train_summary):

        last_obs = self.last_obs
        shapes   = (self.n_steps, self.envs.n_envs)
        values   = np.zeros(np.concatenate([[2], shapes]), dtype=np.float32) #first dim: manager values, second dim: worker values
        rewards  = np.zeros(shapes, dtype=np.float32)
        dones    = np.zeros(shapes, dtype=np.float32)
        all_obs, all_actions = [], []
        mb_states = self.states
        s     = np.zeros((self.nsteps, self.envs.n_envs, d), dtype=np.float32)
        goals = np.zeros((self.nsteps, self.envs.n_envs, d), dtype=np.float32) #TODO: check for dx1

        for n in range(self.n_steps):
            actions, values[:,n,:], states, s[n,:,:], goals[n,:,:] = self.agent.step(last_obs, self.states, goals)
            actions = mask_unused_argument_samples(actions)

            all_obs.append(last_obs)
            all_actions.append(actions)

            pysc2_actions = actions_to_pysc2(actions, size=last_obs['screen'].shape[1:3])
            obs_raw  = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw)
            rewards[n,:], dones[n,:] = zip(*[(t.reward,r.last()) for t in obs_raw()])
            self.states = states

            for t in obs_raw:
                if t.last():
                    self.cumulative_score += self._summarize_episode(t)

        next_values = self.agent.get_value(last_obs, states) #TODO: do we need this?

        #TODO: rest of runner
        returns, returns_intr, adv_m, adv_w = compute_returns_and_advantages(rewards, dones, values, next_values, s, goals, self.discount, self.c)

        # TODO: maybe throw away first and lst c observations here.
        actions = stack_and_flatten_actions(all_actions)
        obs     = flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
        returns = flatten_first_dims(returns)
        returns_intr = flatten_first_dims(returns_intr)
        adv_m    = flatten_first_dims(adv_m)
        adv_w    = flatten_first_dims(adv_w)

        self.last_obs = last_obs

        if self.train:
            return self.agent.train(
                obs,
                mb_states,
                actions,
                returns, returns_intr,
                adv_m, adv_w,
                s,
                goals,
                summary=train_summary
            )
        else:
            return None


    def get_mean_score(self):
        return cummulative_score / episode_counter


    def get_max_score(self):
        return max_score


def compute_returns_and_advantages(rewards, dones, values, next_values, states, goals, discount, c):
    # REVIEW: this function
    alpha = 0.5

    # Intrinsic rewards
    T = rewards.shape[0]-2*c
    nenvs = rewards.shape[1]
    r_i = np.zeros((T,nenvs))
    for t in range(c,c+T):
        sum_cos_dists = 0
        for i in range(1,c):
            sum_cos_dists += cosine_similarity(states[t]-states[t-i] , goals[t-i])
        r_i[i] = 1/c * sum_cos_dists

    # Returns
    returns = np.zeros((c+1, nenvs))
    returns_intr = np.zeros((c+1, nenvs))
    returns[-1,:] = next_values
    for t in range(c,c+T):
        returns[t,:] = rewards[t,:]  + discount * returns[t+1,:] * (1-dones[t,:])
    for t in range(c,c+T-1):
        returns_intr[t,:] = r_i[t,:] + discount * returns_intr[t+1,:] * (1-dones[t,:])
    returns = returns[:-1,:]
    adv_m = returns - values[0]
    adv_w = returns + alpha * returns_intr - values[1]
    return returns, returns_intr, r_i, adv_m, adv_w


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
