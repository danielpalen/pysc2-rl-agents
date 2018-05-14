import sys
import os
import shutil
import sys
import argparse
from functools import partial

import tensorflow as tf

from rl.agents.a2c.runner import A2CRunner
from rl.agents.a2c.agent import A2CAgent
from rl.agents.ppo.runner import PPORunner
from rl.agents.ppo.agent import PPOAgent
from rl.networks.fully_conv import FullyConv
from rl.networks.conv_lstm import ConvLSTM
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv
from rl.common.cmd_util import SC2ArgumentParser

# Just disables warnings for mussing AVX/FMA instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])

agents = {
    'a2c': {
        'agent' : A2CAgent,
        'runner': A2CRunner,
        'policies' : {
            'fully_conv' : FullyConv,
            'conv_lstm' : ConvLSTM
        }
    },
    # 'feudal' : {}
    # 'ppo': {},
}

args = SC2ArgumentParser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)


def main():

    if not args.agent in agents:
        print("Error '{}' agent does not exist!".format(args.agent))
        sys.exit(1)

    if not args.policy in agents[args.agent]['policies']:
        print("Error: '{}' policy does not exist for '{}' agent!".format(args.policy, args.agent))
        sys.exit(1)

    if args.train and args.ow and (os.path.isdir(summary_path) or os.path.isdir(ckpt_path)):
        yes,no = {'yes','y'},{'no','n', ''}
        choice = input(
            "\nWARNING! An experiment with the name '{}' already exists.\nAre you sure you want to overwrite it? [y/N]: "
            .format(args.experiment_id)
        ).lower()
        if choice in yes:
            shutil.rmtree(ckpt_path, ignore_errors=True)
            shutil.rmtree(summary_path, ignore_errors=True)
        else:
            print('Quitting program.')
            sys.exit(0)

    size_px = (args.res, args.res)
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px
    )
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis
    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
        env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

    envs = SubprocVecEnv(env_fns)

    summary_writer = tf.summary.FileWriter(summary_path)

    network_data_format = 'NHWC' if args.nhwc else 'NCHW'

    # TODO: We should actually do individual setup and argument parser methods
    # for each agent since they require different parameters etc.
    print('################################\n#')
    print('#  Running {} Agent with {} policy'.format(args.agent, args.policy))
    print('#\n################################')

    # TODO: pass args directly so each agent and runner can pick theirs
    agent = agents[args.agent]['agent'](
        network=agents[args.agent]['policies'][args.policy],
        network_data_format=network_data_format,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr,
        max_to_keep=args.max_to_keep,
        nenvs=args.envs,
        nsteps=args.steps_per_batch,
        res=args.res,
        checkpoint_path=ckpt_path,
        debug=args.debug,
        debug_tb_adress=args.tensorboard_debug_address
    )

    runner = agents[args.agent]['runner'](
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch
    )

    i = agent.get_global_step()
    try:
        while args.iters==-1 or i<args.iters:

            write_summary = args.train and i % args.summary_iters == 0

            if i > 0 and i % args.save_iters == 0:
                _save_if_training(agent, summary_writer)

            result = runner.run_batch(train_summary=write_summary)

            if write_summary:
                agent_step, loss, summary = result
                summary_writer.add_summary(summary, global_step=agent_step)
                print('iter %d: loss = %f' % (agent_step, loss))

            i+=1

    except KeyboardInterrupt:
        pass

    _save_if_training(agent, summary_writer)

    envs.close()
    summary_writer.close()

    print('mean score: %f' % runner.get_mean_score())


def _save_if_training(agent, summary_writer):
    if args.train:
        agent.save(ckpt_path)
        summary_writer.flush()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
