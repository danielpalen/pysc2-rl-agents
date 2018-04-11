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
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv
from rl.common.cmd_util import SC2ArgumentParser

# Workaround for pysc2 flags
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])

agents = {
    'a2c': [A2CAgent, A2CRunner],
    'ppo': [PPOAgent, PPORunner]
}

args = SC2ArgumentParser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)


def main():
    if args.train and args.ow:
        shutil.rmtree(ckpt_path, ignore_errors=True)
        shutil.rmtree(summary_path, ignore_errors=True)

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
    print('Running', args.agent, 'Agent #TODO: implement!!')

    agent = A2CAgent(
        network_data_format=network_data_format,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr,
        max_to_keep=args.max_to_keep,
        nsteps=args.steps_per_batch,
        res=args.res,
        checkpoint_path=ckpt_path
    )

    runner = A2CRunner(
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch
    )

    i = 0
    try:
        while True:
            write_summary = args.train and i % args.summary_iters == 0

            if i > 0 and i % args.save_iters == 0:
                _save_if_training(agent, summary_writer)

            result = runner.run_batch(train_summary=write_summary)

            if write_summary:
                agent_step, loss, summary = result
                summary_writer.add_summary(summary, global_step=agent_step)
                print('iter %d: loss = %f' % (agent_step, loss))

            i += 1

            if 0 <= args.iters <= i:
                break

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
