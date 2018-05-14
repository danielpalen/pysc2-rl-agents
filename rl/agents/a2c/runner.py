import numpy as np
import tensorflow as tf

from pysc2.lib.actions import FunctionCall, FUNCTIONS

from rl.agents.runner import BaseRunner
from rl.common.pre_processing import Preprocessor
from rl.common.pre_processing import is_spatial_action, stack_ndarray_dicts

from rl.common.util import mask_unused_argument_samples, flatten_first_dims, flatten_first_dims_dict


class A2CRunner(BaseRunner):
    def __init__(self, agent, envs, summary_writer=None,
                 train=True, n_steps=8, discount=0.99):
        """
        Args:
          agent: A2CAgent instance.
          envs: SubprocVecEnv instance.
          summary_writer: summary writer to log episode scores.
          train: whether to train the agent.
          n_steps: number of agent steps for collecting rollouts.
          discount: future reward discount.
        """
        self.agent = agent
        self.envs = envs
        self.summary_writer = summary_writer
        self.train = train
        self.n_steps = n_steps
        self.discount = discount
        self.preproc = Preprocessor()
        self.last_obs = self.preproc.preprocess_obs(self.envs.reset())

        # TODO: we probably need to save this state during checkpoing
        self.states = agent.initial_state
        self.episode_counter = 1
        self.cumulative_score = 0.0

    def run_batch(self, train_summary=False):
        """Collect trajectories for a single batch and train (if self.train).

        Args:
          train_summary: return a Summary of the training step (losses, etc.).

        Returns:
          result: None (if not self.train) or the return value of agent.train.
        """
        last_obs = self.last_obs
        shapes   = (self.n_steps, self.envs.n_envs)
        values   = np.zeros(shapes, dtype=np.float32)
        rewards  = np.zeros(shapes, dtype=np.float32)
        dones    = np.zeros(shapes, dtype=np.float32)
        all_obs, all_actions = [], []
        mb_states = self.states # save the initial states at the beginning of each mb for later training.

        for n in range(self.n_steps):
            actions, values[n,:], states = self.agent.step(last_obs, self.states)
            actions = mask_unused_argument_samples(actions)

            all_obs.append(last_obs)
            all_actions.append(actions)

            pysc2_actions = actions_to_pysc2(actions, size=last_obs['screen'].shape[1:3])
            obs_raw  = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw)
            rewards[n,:], dones[n,:] = zip(*[(t.reward,t.last()) for t in obs_raw])
            self.states = states

            for t in obs_raw:
                if t.last():
                    self.cumulative_score += self._summarize_episode(t)

        next_values = self.agent.get_value(last_obs, states)

        returns, advs = compute_returns_and_advs(rewards, dones, values, next_values, self.discount)

        actions = stack_and_flatten_actions(all_actions)
        obs     = flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
        returns = flatten_first_dims(returns)
        advs    = flatten_first_dims(advs)

        self.last_obs = last_obs

        if self.train:
            return self.agent.train(obs, mb_states, actions, returns, advs, summary=train_summary)
        else:
            return None


    def get_mean_score(self):
        return self.cumulative_score / self.episode_counter


    def _summarize_episode(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        episode = (self.agent.get_global_step() // self.n_steps) + 1 # because global_step is zero based
        if self.summary_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag='sc2/episode_score', simple_value=score)
            self.summary_writer.add_summary(summary, episode)

        print("episode %d: score = %f" % (episode, score))
        self.episode_counter += 1
        return score


def compute_returns_and_advs(rewards, dones, values, next_values, discount):
    """Compute returns and advantages from received rewards and value estimates.

    Args:
      rewards: array of shape [n_steps, n_env] containing received rewards.
      dones: array of shape [n_steps, n_env] indicating whether an episode is
        finished after a time step.
      values: array of shape [n_steps, n_env] containing estimated values.
      next_values: array of shape [n_env] containing estimated values after the
        last step for each environment.
      discount: scalar discount for future rewards.

    Returns:
      returns: array of shape [n_steps, n_env]
      advs: array of shape [n_steps, n_env]
    """
    returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])

    returns[-1, :] = next_values
    for t in reversed(range(rewards.shape[0])):
        future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
        returns[t, :] = rewards[t, :] + future_rewards

    returns = returns[:-1, :]
    advs = returns - values

    return returns, advs


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
