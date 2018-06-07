import os
import argparse
import json

class Namespace(object):
    """Helper class for restoring command line args that have been saved to json."""
    def __init__(self, adict):
        self.__dict__.update(adict)

class SC2ArgumentParser():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')

        parser.add_argument('experiment_id', type=str,
                            help='identifier to store experiment results')

        # General Args
        parser.add_argument('--agent', type=str, default='a2c',
                            help='which agent to use')
        parser.add_argument('--policy', type=str, default='default',
                            help='which policy the agent shoul use.'),
        parser.add_argument('--eval', action='store_true',
                            help='if false, episode scores are evaluated')
        parser.add_argument('--gpu', type=str, default='0',
                            help='gpu device id')
        parser.add_argument('--nhwc', action='store_true',
                            help='train fullyConv in NCHW mode')
        parser.add_argument('--resume', action='store_true',
                            help='continue experiment with given name.')
        parser.add_argument('--ow', action='store_true',
                            help='overwrite existing experiments (if --train=True)')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed')
        parser.add_argument('--res', type=int, default=32,
                            help='screen and minimap resolution')


        # Environment Execution Args
        parser.add_argument('--iters', type=int, default=-1,
                            help='number of iterations to run (-1 to run forever)')
        parser.add_argument('--step_mul', type=int, default=8,
                            help='number of game steps per agent step')
        parser.add_argument('--steps_per_batch', type=int, default=16,
                            help='number of agent steps when collecting trajectories for a single batch')

        # Debug Args
        ## documentation: https://www.tensorflow.org/programmers_guide/debugger
        ## taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/debug_mnist.py
        parser.add_argument("--debug", type=bool, nargs="?", const=True, default=False,
                            help="Use debugger to track down bad values during training. "
                            "Mutually exclusive with the --tensorboard_debug_address flag.")
        parser.add_argument("--tensorboard_debug_address", type=str, default=None,
                            help="Connect to the TensorBoard Debugger Plugin backend specified by "
                            "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
                            "--debug flag.")

        # Summary and Checkpoint Args
        parser.add_argument('--max_to_keep', type=int, default=5,
                            help='maximum number of checkpoints to keep before discarding older ones')
        parser.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                            help='root directory for checkpoint storage')
        parser.add_argument('--save_iters', type=int, default=5000,
                            help='store checkpoint after this many iterations')
        parser.add_argument('--summary_iters', type=int, default=10,
                            help='record training summary after this many iterations')
        parser.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                            help='root directory for summary storage')


        # Starcraft Args
        parser.add_argument('--map', type=str, default='MoveToBeacon',
                            help='name of SC2 map')
        parser.add_argument('--max_windows', type=int, default=1,
                            help='maximum number of visualization windows to open')
        parser.add_argument('--vis', action='store_true',
                            help='render with pygame')


        # Neural Net Args
        parser.add_argument('--lr', type=float, default=7e-4,
                            help='initial learning rate')


        #A2C Args
        parser.add_argument('--discount', type=float, default=0.99,
                            help='discount for future rewards')
        parser.add_argument('--entropy_weight', type=float, default=1e-3,
                            help='weight of entropy loss')
        parser.add_argument('--envs', type=int, default=32,
                            help='number of environments simulated in parallel')
        parser.add_argument('--value_loss_weight', type=float, default=0.5,
                            help='weight of value function loss')


        #Feudal Args
        parser.add_argument('--d', type=int, default=512,
                            help='manager dimension')
        parser.add_argument('--k', type=int, default=32,
                            help='size of goal-embedding space')
        parser.add_argument('--c', type=int, default=10,
                            help='number of cores')


        def parse_args():
            # TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
            args = parser.parse_args()
            args.train = not args.eval
            return args


        def save(args, path):
            with open(os.path.join(path,'args.json'), 'w') as fp:
                print(f'Saved Args to {os.path.join(path,"args.json")}')
                json.dump(vars(args), fp, sort_keys=True, indent=4)


        def restore(path):
            with open(os.path.join(path,'args.json'), 'r') as fp:
                print('Restored Args')
                return Namespace(json.load(fp))


        self.parse_args = parse_args
        self.restore = restore
        self.save = save
