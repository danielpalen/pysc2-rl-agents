import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Categorical
from tensorflow.python import debug as tf_debug

from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.networks.fully_conv import FullyConv
from rl.networks.conv_lstm import ConvLSTM
from rl.common.pre_processing import Preprocessor, get_input_channels
from rl.common.util import compute_entropy, safe_log, safe_div, mask_unavailable_actions

class PPOAgent():

    def __init__(self, policy, args):

        network_data_format = 'NHWC' if args.nhwc else 'NCHW'
        value_loss_weight = args.value_loss_weight
        entropy_weight = args.entropy_weight
        learning_rate = args.lr
        max_to_keep = args.max_to_keep
        nenvs = args.envs
        nsteps = args.steps_per_batch
        res = args.res
        checkpoint_path = args.ckpt_path
        summary_writer = args.summary_writer
        clip = args.clip
        debug = args.debug
        clip_value_loss = args.clip_value_loss

        print('\n### PPO Agent #######')
        print(f'# policy = {policy}')
        print(f'# network_data_format = {network_data_format}')
        print(f'# value_loss_weight = {value_loss_weight}')
        print(f'# entropy_weight = {entropy_weight}')
        print(f'# learning_rate = {learning_rate}')
        print(f'# max_to_keep = {max_to_keep}')
        print(f'# nenvs = {nenvs}')
        print(f'# nsteps = {nsteps}')
        print(f'# res = {res}')
        print(f'# checkpoint_path = {checkpoint_path}')
        print(f'# clip = {clip}')
        print('######################\n')

        max_gradient_norm = 1.0

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        #if debug and debug_tb_adress:
        #    raise ValueError(
        #"The --debug and --tensorboard_debug_address flags are mutually "
        #"exclusive.")
        if  debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #elif  debug_tb_adress:
        #    sess = tf_debug.TensorBoardDebugWrapperSession(sess, debug_tb_adress)

        nbatch = nenvs*nsteps
        # assert nbatch % nminibatches == 0
        # nbatch_train = nbatch // nminibatches
        ch = get_input_channels()
        ob_space = {
            'screen'  : [None, res, res, ch['screen']],
            'minimap' : [None, res, res, ch['minimap']],
            'flat'    : [None, ch['flat']],
            'available_actions' : [None, ch['available_actions']]
        }

        step_model  = policy(sess, ob_space=ob_space, nbatch=nenvs,  nsteps=1,      reuse=None, data_format=network_data_format)
        train_model = policy(sess, ob_space=ob_space, nbatch=nbatch, nsteps=nsteps, reuse=True, data_format=network_data_format)

        # Define placeholders
        fn_id = tf.placeholder(tf.int32, [None], name='fn_id')
        arg_ids = {
            k: tf.placeholder(tf.int32, [None], name=f'arg_{k.id}')
            for k in train_model.policy[1].keys()
        }
        ACTIONS = (fn_id, arg_ids)
        ADVS = tf.placeholder(tf.float32, [None], name='advs')
        RETURNS = tf.placeholder(tf.float32, [None], name='returns')
        OLD_LOGPROBS = tf.placeholder(tf.float32, [None], name='old_log_probs') # TODO:
        OLD_VALUES = tf.placeholder(tf.float32, [None], name='old_values') # TODO:

        # Define Loss
        log_probs = compute_policy_log_probs(train_model.AV_ACTS, train_model.policy, ACTIONS)
        ratio = tf.exp(log_probs-OLD_LOGPROBS)
        # Policy Loss
        p_loss_1 = -ADVS * ratio
        p_loss_2 = -ADVS * tf.clip_by_value(ratio, 1.0-clip, 1.0+clip)
        policy_loss = tf.reduce_mean(tf.maximum(p_loss_1,p_loss_2))
        # Value Loss
        if clip_value_loss:
            value_pred_clipped = OLD_VALUES + tf.clip_by_value((train_model.value - OLD_VALUES), -clip, clip) 
            value_losses = tf.square(RETURNS - train_model.value)
            value_losses_clipped = tf.square(RETURNS - value_pred_clipped) 
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = tf.reduce_mean(tf.square(RETURNS - train_model.value)) # TODO: make PPO style value loss.
        #Entropy Loss
        entropy = compute_policy_entropy(train_model.AV_ACTS, train_model.policy, ACTIONS)
        
        loss = policy_loss  + value_loss * value_loss_weight - entropy * entropy_weight

        # Define Optimizer
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.94)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=1e-5)
        train_op = layers.optimize_loss(loss=loss, global_step=global_step,
            optimizer=optimizer, clip_gradients=max_gradient_norm, learning_rate=None, name="train_op")

        tf.summary.scalar('entropy', entropy)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss/policy', policy_loss)
        tf.summary.scalar('loss/value', value_loss)
        tf.summary.scalar('rl/value', tf.reduce_mean(train_model.value))
        tf.summary.scalar('rl/returns', tf.reduce_mean(RETURNS))
        tf.summary.scalar('rl/advs', tf.reduce_mean(ADVS))
        summary_writer.add_graph(sess.graph)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(variables, max_to_keep=max_to_keep)
        train_summaries  = tf.get_collection(tf.GraphKeys.SUMMARIES)
        train_summary_op = tf.summary.merge(train_summaries)

        # Load checkpoints if exist
        if os.path.exists(checkpoint_path):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded agent at episode {} (step {})".format(self.train_step//nsteps, self.train_step))
        else:
            self.train_step = 0
            sess.run(tf.variables_initializer(variables))


        def train(obs, states, actions, returns, advs, old_log_probs, old_values, summary=False):
            """
            Args:
              obs: dict of preprocessed observation arrays, with num_batch elements
                in the first dimensions.
              actions: see `compute_total_log_probs`.
              returns: array of shape [num_batch].
              advs: array of shape [num_batch].
              summary: Whether to return a summary.

            Returns:
              summary: (agent_step, loss, Summary) or None.
            """
            feed_dict = {
                train_model.SCREEN : obs['screen'],
                train_model.MINIMAP: obs['minimap'],
                train_model.FLAT   : obs['flat'],
                train_model.AV_ACTS: obs['available_actions'],
                RETURNS   : returns,
                ADVS      : advs,
                ACTIONS[0]   : actions[0],
                OLD_LOGPROBS : old_log_probs,
                OLD_VALUES : old_values
            }
            feed_dict.update({ v: actions[1][k] for k, v in ACTIONS[1].items() })
            if states is not None: # For recurrent polices
                feed_dict.update({train_model.STATES : states})

            agent_step = self.train_step
            self.train_step += 1

            if summary:
                _,_step,_loss,_summary = sess.run([train_op, global_step, loss, train_summary_op], feed_dict=feed_dict)
                return _step, _loss, _summary
            else:
                _train_op,_loss = sess.run([train_op, loss], feed_dict=feed_dict)
                return _train_op, _loss, None


        def save(path, step=None):
            os.makedirs(path, exist_ok=True)
            print("Saving agent to %s, step %d" % (path, sess.run(global_step)))
            ckpt_path = os.path.join(path, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=global_step)


        def get_global_step():
            return sess.run(global_step)


        def get_log_probs(obs, states, actions):
            feed_dict = {
                train_model.SCREEN : obs['screen'],
                train_model.MINIMAP: obs['minimap'],
                train_model.FLAT   : obs['flat'],
                train_model.AV_ACTS: obs['available_actions'],
                ACTIONS[0]   : actions[0],
            }
            feed_dict.update({ v: actions[1][k] for k, v in ACTIONS[1].items() })
            #if states is not None: # For recurrent polices
            #    feed_dict.update({train_model.STATES : states})
            return sess.run(log_probs, feed_dict)


        self.train = train
        self.step = step_model.step
        self.get_value = step_model.get_value
        self.save = save
        self.initial_state = step_model.initial_state
        self.get_global_step = get_global_step
        self.get_log_probs = get_log_probs


def compute_policy_entropy(available_actions, policy, actions):
    """Compute total policy entropy.

    Args: (same as compute_policy_log_probs)

    Returns:
      entropy: a scalar float tensor.
    """
    _,arg_ids = actions

    fn_pi, arg_pis = policy
    fn_pi = mask_unavailable_actions(available_actions, fn_pi)
    entropy = tf.reduce_mean(compute_entropy(fn_pi))
    tf.summary.scalar('entropy/fn', entropy)

    for arg_type in arg_ids.keys():
        arg_id = arg_ids[arg_type]
        arg_pi = arg_pis[arg_type]
        batch_mask = tf.to_float(tf.not_equal(arg_id, -1))
        arg_entropy = safe_div(
            tf.reduce_sum(compute_entropy(arg_pi) * batch_mask),
            tf.reduce_sum(batch_mask))
        entropy += arg_entropy
        tf.summary.scalar('used/arg/%s' % arg_type.name,
                          tf.reduce_mean(batch_mask))
        tf.summary.scalar('entropy/arg/%s' % arg_type.name, arg_entropy)

    return entropy


def compute_policy_log_probs(available_actions, policy, actions):
    """Compute action log probabilities given predicted policies and selected
    actions.

    Args:
      available_actions: one-hot (in last dimenson) tensor of shape
        [num_batch, NUM_FUNCTIONS].
      policy: [fn_pi, {arg_0: arg_0_pi, ..., arg_n: arg_n_pi}]], where
        each value is a tensor of shape [num_batch, num_params] representing
        probability distributions over the function ids or over discrete
        argument values.
      actions: [fn_ids, {arg_0: arg_0_ids, ..., arg_n: arg_n_ids}], where
        each value is a tensor of shape [num_batch] representing the selected
        argument or actions ids. The argument id will be -1 if the argument is
        not available for a specific (state, action) pair.

    Returns:
      log_prob: a tensor of shape [num_batch]
    """
    def compute_log_probs(probs, labels):
        # Select arbitrary element for unused arguments (log probs will be masked)
        labels = tf.maximum(labels, 0)
        indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
        # TODO tf.log should suffice
        return safe_log(tf.gather_nd(probs, indices))


    fn_id, arg_ids = actions
    fn_pi, arg_pis = policy
    # TODO: this should be unneccessary
    fn_pi = mask_unavailable_actions(available_actions, fn_pi)
    fn_log_prob = compute_log_probs(fn_pi, fn_id)
    tf.summary.scalar('log_prob/fn', tf.reduce_mean(fn_log_prob))

    log_prob = fn_log_prob
    for arg_type in arg_ids.keys():
        arg_id = arg_ids[arg_type]
        arg_pi = arg_pis[arg_type]
        arg_log_prob = compute_log_probs(arg_pi, arg_id)
        arg_log_prob *= tf.to_float(tf.not_equal(arg_id, -1))
        log_prob += arg_log_prob
        tf.summary.scalar('log_prob/arg/%s' % arg_type.name,
                          tf.reduce_mean(arg_log_prob))

    return log_prob
