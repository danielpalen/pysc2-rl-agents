import numpy as np
import tensorflow as tf
#from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected, flatten, conv2d
from tensorflow.contrib.distributions import Categorical

from pysc2.lib import actions
from pysc2.lib import features

from rl.common.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
from rl.common.util import mask_unavailable_actions


class FullyConv():
    """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

    Both, NHWC and NCHW data formats are supported for the network
    computations. Inputs and outputs are always in NHWC.
    """

    def __init__(self, sess, ob_space, nbatch, nsteps, reuse=False, data_format='NCHW'):

        def embed_obs(x, spec, embed_fn, name):
            feats = tf.split(x, len(spec), -1)
            out_list = []
            for s in spec:
                f = feats[s.index]
                if s.type == features.FeatureType.CATEGORICAL:
                    dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                    dims = max(dims, 1)
                    indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
                    out = embed_fn(indices, dims, "{}/{}".format(name,s.name))
                elif s.type == features.FeatureType.SCALAR:
                    out = tf.log(f + 1.)
                else:
                    raise NotImplementedError
                out_list.append(out)
            return tf.concat(out_list, -1)

        def embed_spatial(x, dims, name):
            x = from_nhwc(x)
            out = conv2d(
                x, dims,
                kernel_size=1,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv_embSpatial" % name)
            return to_nhwc(out)

        def embed_flat(x, dims, name):
            return fully_connected(
                x, dims,
                activation_fn=tf.nn.relu,
                #scope="%s/emb_flat" % name)
                scope="%s/conv_embFlat" % name)

        def input_conv(x, name):
            conv1 = conv2d(
                x, 16,
                kernel_size=5,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv1" % name)
            conv2 = conv2d(
                conv1, 32,
                kernel_size=3,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv2" % name)
            return conv2

        def non_spatial_output(x, channels, name):
            logits = fully_connected(x, channels, activation_fn=None, scope="non_spatial_output/flat/{}".format(name))
            return tf.nn.softmax(logits)

        def spatial_output(x, name):
            logits = conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None, data_format=data_format, scope="spatial_output/conv/{}".format(name))
            logits = flatten(to_nhwc(logits), scope="spatial_output/flat/{}".format(name))
            return tf.nn.softmax(logits)

        def concat2DAlongChannel(lst):
            """Concat along the channel axis"""
            axis = 1 if data_format == 'NCHW' else 3
            return tf.concat(lst, axis=axis)

        def broadcast_along_channels(flat, size2d):
            if data_format == 'NCHW':
                return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                               tf.stack([1, 1, size2d[0], size2d[1]]))
            return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                           tf.stack([1, size2d[0], size2d[1], 1]))

        def to_nhwc(map2d):
            if data_format == 'NCHW':
                return tf.transpose(map2d, [0, 2, 3, 1])
            return map2d

        def from_nhwc(map2d):
            if data_format == 'NCHW':
                return tf.transpose(map2d, [0, 3, 1, 2])
            return map2d


        SCREEN  = tf.placeholder(tf.float32, shape=ob_space['screen'],  name='input_screen')
        MINIMAP = tf.placeholder(tf.float32, shape=ob_space['minimap'], name='input_minimap')
        FLAT    = tf.placeholder(tf.float32, shape=ob_space['flat'],    name='input_flat')
        AV_ACTS = tf.placeholder(tf.float32, shape=ob_space['available_actions'], name='available_actions')


        with tf.variable_scope('model', reuse=reuse):

            screen_emb  = embed_obs(SCREEN,  features.SCREEN_FEATURES,  embed_spatial, 'screen')
            minimap_emb = embed_obs(MINIMAP, features.MINIMAP_FEATURES, embed_spatial, 'minimap')
            flat_emb    = embed_obs(FLAT, FLAT_FEATURES, embed_flat, 'flat')

            screen_out    = input_conv(from_nhwc(screen_emb), 'screen')
            minimap_out   = input_conv(from_nhwc(minimap_emb), 'minimap')

            broadcast_out = broadcast_along_channels(flat_emb, ob_space['screen'][1:3])
            state_out     = concat2DAlongChannel([screen_out, minimap_out, broadcast_out])

            flat_out = flatten(to_nhwc(state_out), scope="flat_out")
            fc = fully_connected(flat_out, 256, activation_fn=tf.nn.relu, scope="fully_con")

            value = fully_connected(fc, 1, activation_fn=None, scope="value")
            value = tf.reshape(value, [-1])

            fn_out = non_spatial_output(fc, NUM_FUNCTIONS, name='fn_out')

            args_out = dict()
            for arg_type in actions.TYPES:
                if is_spatial_action[arg_type]:
                    arg_out = spatial_output(state_out, name=arg_type.name)
                else:
                    arg_out = non_spatial_output(fc, arg_type.sizes[0], name=arg_type.name)
                args_out[arg_type] = arg_out

            policy = (fn_out, args_out)


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

        action = sample_actions(AV_ACTS, policy)

        def step(obs, state, mask=None):
            feed_dict = {
                SCREEN : obs['screen'],
                MINIMAP: obs['minimap'],
                FLAT   : obs['flat'],
                AV_ACTS: obs['available_actions']
            }
            a, v = sess.run([action, value], feed_dict=feed_dict)
            return a, v, self.initial_state


        def get_value(obs, state, mask=None):
            feed_dict = {
                SCREEN : obs['screen'],
                MINIMAP: obs['minimap'],
                FLAT   : obs['flat']
            }
            return sess.run(value, feed_dict=feed_dict)


        self.SCREEN  = SCREEN
        self.MINIMAP = MINIMAP
        self.FLAT    = FLAT
        self.AV_ACTS = AV_ACTS
        self.policy = policy
        self.value  = value
        self.step = step
        self.get_value = get_value
        self.initial_state = None
