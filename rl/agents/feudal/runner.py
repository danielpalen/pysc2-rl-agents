import numpy as np
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
        self.n_steps = args.steps_per_batch

        self.c = args.worker_steps

        print('\n### Feudal Runner ######')

        self.states = None # Holds the managers and workers hidden states

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
        goals = np.zeros(d, dtype=np.float32) #TODO: check for dx1

        for n in range(self.n_steps):
            actions, values[:,n,:], states, goals = self.agent.step(last_obs, self.states, goals)
            actions = mask_unused_argument_samples(actions)

            all_obs.append(last_obs)
            all_actions.append(actions)
            pysc2_actions = actions_to_pysc2(actions, size=last_obs['screen'].shape[1:3])
            obs_raw  = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw)

            for t in obs_raw:
                if t.last():
                    self.cumulative_score += self._summarize_episode(t)

        next_values = self.agent.get_value(last_obs, states, goals)

        #TODO: rest of runner


    def get_mean_score(self):
        return cummulative_score / episode_counter


    def get_max_score(self):
        return max_score
