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


    def run_batch(self, train_summary):


        worker = None
        manager = None

        

        for n in range(self.nsteps):
            actions, values, states = self.worker.step


    def get_mean_score(self):
        return cummulative_score / episode_counter


    def get_max_score(self):
        return max_score
