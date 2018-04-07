import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Categorical

from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.networks.fully_conv import FullyConv
from rl.pre_processing import Preprocessor, get_input_channels
from rl.util import safe_log, safe_div


class A2CAgent():
    """A2C agent.

    Run build(...) first, then init() or load(...).
    """

    def __init__(self,
                 sess,
                 network_cls=FullyConv,
                 network_data_format='NCHW',
                 value_loss_weight=0.5,
                 entropy_weight=1e-3,
                 learning_rate=7e-4,
                 max_gradient_norm=1.0,
                 max_to_keep=5,
                 res=32,
                 checkpoint_path=None):

        self.sess = sess

        ch = get_input_channels()

        # Create placeholder
        screen  = tf.placeholder(tf.float32, [None, res, res, ch['screen']], name='input_screen')
        minimap = tf.placeholder(tf.float32, [None, res, res, ch['minimap']], name='input_minimap')
        flat    = tf.placeholder(tf.float32, [None, ch['flat']], name='input_flat')
        available_actions = tf.placeholder(tf.float32, [None, ch['available_actions']], name='input_available_actions')
        advs    = tf.placeholder(tf.float32, [None], name='advs')
        returns = tf.placeholder(tf.float32, [None], name='returns')

        policy, value = network_cls(data_format=network_data_format).build(screen, minimap, flat)

        fn_id = tf.placeholder(tf.int32, [None], name='fn_id')
        arg_ids = {
            k: tf.placeholder(tf.int32, [None], name='arg_{}_id'.format(k.id))
            for k in policy[1].keys()
        }
        actions = (fn_id, arg_ids)

        log_probs = compute_policy_log_probs(available_actions, policy, actions)
        policy_loss = -tf.reduce_mean(advs * log_probs)
        value_loss  = tf.reduce_mean(tf.square(returns - value) / 2.)
        entropy     = compute_policy_entropy(available_actions, policy, actions)
        loss = (policy_loss
                + value_loss * value_loss_weight
                - entropy * entropy_weight)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.94)
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        decay=0.99,
                                        epsilon=1e-5)

        train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=opt,
            clip_gradients=max_gradient_norm,
            learning_rate=None,
            name="train_op"
        )

        samples = sample_actions(available_actions, policy)

        # Summary writer
        tf.summary.scalar('entropy', entropy)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss/policy', policy_loss)
        tf.summary.scalar('loss/value', value_loss)
        tf.summary.scalar('rl/value', tf.reduce_mean(value))
        tf.summary.scalar('rl/returns', tf.reduce_mean(returns))
        tf.summary.scalar('rl/advs', tf.reduce_mean(advs))

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver   = tf.train.Saver(variables, max_to_keep=max_to_keep)
        train_summaries  = tf.get_collection(tf.GraphKeys.SUMMARIES)
        train_summary_op = tf.summary.merge(train_summaries)

        if os.path.exists(checkpoint_path):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded agent at train_step %d" % train_step)
        else:
            train_step = 0
            sess.run(tf.variables_initializer(variables))


        def get_obs_feed(obs):
            return {
                screen: obs['screen'],
                minimap: obs['minimap'],
                flat: obs['flat'],
                available_actions: obs['available_actions']
            }


        def get_actions_feed(_actions):
            feed_dict = {actions[0]: _actions[0]}
            feed_dict.update(
                { v: _actions[1][k] for k, v in actions[1].items()}
            )
            return feed_dict


        self.get_obs_feed = get_obs_feed
        self.get_actions_feed = get_actions_feed


        self.train_step = train_step
        self.screen = screen
        self.minimap = minimap
        self.flat = flat
        self.advs = advs
        self.returns = returns
        self.available_actions = available_actions
        self.policy = policy
        self.value = value
        self.actions = actions
        self.loss = loss
        self.train_op = train_op
        self.samples = samples
        self.saver = saver
        self.train_summary_op = train_summary_op


    def train(self, obs, actions, returns, advs, summary=False):
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
        feed_dict = self.get_obs_feed(obs)
        feed_dict.update(self.get_actions_feed(actions))
        feed_dict.update({
            self.returns: returns,
            self.advs: advs
        })

        ops = [self.train_op, self.loss]

        if summary:
            ops.append(self.train_summary_op)

        res = self.sess.run(ops, feed_dict=feed_dict)
        agent_step = self.train_step
        self.train_step += 1

        if summary:
            return (agent_step, res[1], res[-1])


    def step(self, obs):
        """
        Args:
          obs: dict of preprocessed observation arrays, with num_batch elements
            in the first dimensions.

        Returns:
          actions: arrays (see `compute_total_log_probs`)
          values: array of shape [num_batch] containing value estimates.
        """
        feed_dict = self.get_obs_feed(obs)
        return self.sess.run([self.samples, self.value], feed_dict=feed_dict)


    def get_value(self, obs):
        return self.sess.run(
            self.value,
            feed_dict=self.get_obs_feed(obs))


    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("Saving agent to %s, step %d" % (path, step))
        ckpt_path = os.path.join(path, 'model.ckpt')
        self.saver.save(self.sess, ckpt_path, global_step=step)


def mask_unavailable_actions(available_actions, fn_pi):
    fn_pi *= available_actions
    fn_pi /= tf.reduce_sum(fn_pi, axis=1, keepdims=True)
    return fn_pi


def compute_policy_entropy(available_actions, policy, actions):
    """Compute total policy entropy.

    Args: (same as compute_policy_log_probs)

    Returns:
      entropy: a scalar float tensor.
    """

    def compute_entropy(probs):
        return -tf.reduce_sum(safe_log(probs) * probs, axis=-1)

    _, arg_ids = actions

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


def sample_actions(available_actions, policy):
    """Sample function ids and arguments from a predicted policy."""

    def sample(probs):
        dist = Categorical(probs=probs)
        return dist.sample()

    fn_pi, arg_pis = policy
    fn_pi = mask_unavailable_actions(available_actions, fn_pi)
    fn_samples = sample(fn_pi)

    arg_samples = dict()
    for arg_type, arg_pi in arg_pis.items():
        arg_samples[arg_type] = sample(arg_pi)

    return fn_samples, arg_samples


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
