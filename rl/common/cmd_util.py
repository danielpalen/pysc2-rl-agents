import os
import argparse

class SC2ArgumentParser():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')

        parser.add_argument('experiment_id', type=str,
                            help='identifier to store experiment results')



        # Starcraft Args


        # Other Args




        parser.add_argument('--agent', type=str, default='a2c',
                            help='which agent to use')
        parser.add_argument('--eval', action='store_true',
                            help='if false, episode scores are evaluated')
        parser.add_argument('--ow', action='store_true',
                            help='overwrite existing experiments (if --train=True)')
        parser.add_argument('--map', type=str, default='MoveToBeacon',
                            help='name of SC2 map')
        parser.add_argument('--vis', action='store_true',
                            help='render with pygame')
        parser.add_argument('--max_windows', type=int, default=1,
                            help='maximum number of visualization windows to open')
        parser.add_argument('--res', type=int, default=32,
                            help='screen and minimap resolution')
        parser.add_argument('--envs', type=int, default=32,
                            help='number of environments simulated in parallel')
        parser.add_argument('--step_mul', type=int, default=8,
                            help='number of game steps per agent step')
        parser.add_argument('--steps_per_batch', type=int, default=16,
                            help='number of agent steps when collecting trajectories for a single batch')
        parser.add_argument('--discount', type=float, default=0.99,
                            help='discount for future rewards')
        parser.add_argument('--iters', type=int, default=-1,
                            help='number of iterations to run (-1 to run forever)')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed')
        parser.add_argument('--gpu', type=str, default='0',
                            help='gpu device id')
        parser.add_argument('--nhwc', action='store_true',
                            help='train fullyConv in NCHW mode')
        parser.add_argument('--summary_iters', type=int, default=10,
                            help='record training summary after this many iterations')
        parser.add_argument('--save_iters', type=int, default=5000,
                            help='store checkpoint after this many iterations')
        parser.add_argument('--max_to_keep', type=int, default=5,
                            help='maximum number of checkpoints to keep before discarding older ones')
        parser.add_argument('--entropy_weight', type=float, default=1e-3,
                            help='weight of entropy loss')
        parser.add_argument('--value_loss_weight', type=float, default=0.5,
                            help='weight of value function loss')
        parser.add_argument('--lr', type=float, default=7e-4,
                            help='initial learning rate')
        parser.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                            help='root directory for checkpoint storage')
        parser.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                            help='root directory for summary storage')


        def parse_args():
            # TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
            args = parser.parse_args()
            args.train = not args.eval
            return args


        self.parse_args = parse_args