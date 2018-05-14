import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn # TODO: remove
from tensorflow.contrib.layers import fully_connected, flatten, conv2d
from tensorflow.contrib.distributions import Categorical

from pysc2.lib import actions
from pysc2.lib import features

from rl.common.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
from rl.common.util import mask_unavailable_actions


class ConvLSTM():
    """LSTM network.

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


        nenvs = nbatch//nsteps
        res = ob_space['screen'][1]
        filters = 75
        state_shape = (2, nenvs, res, res, filters)

        SCREEN  = tf.placeholder(tf.float32, shape=ob_space['screen'],  name='input_screen')
        MINIMAP = tf.placeholder(tf.float32, shape=ob_space['minimap'], name='input_minimap')
        FLAT    = tf.placeholder(tf.float32, shape=ob_space['flat'],    name='input_flat')
        AV_ACTS = tf.placeholder(tf.float32, shape=ob_space['available_actions'], name='available_actions')
        STATES  = tf.placeholder(tf.float32, shape=state_shape, name='initial_state')

        with tf.variable_scope('model', reuse=reuse):

            screen_emb  = embed_obs(SCREEN,  features.SCREEN_FEATURES,  embed_spatial, 'screen')
            minimap_emb = embed_obs(MINIMAP, features.MINIMAP_FEATURES, embed_spatial, 'minimap')
            flat_emb    = embed_obs(FLAT, FLAT_FEATURES, embed_flat, 'flat')

            screen_out    = input_conv(from_nhwc(screen_emb), 'screen')
            minimap_out   = input_conv(from_nhwc(minimap_emb), 'minimap')

            broadcast_out = broadcast_along_channels(flat_emb, ob_space['screen'][1:3])
            state_out     = concat2DAlongChannel([screen_out, minimap_out, broadcast_out])

            # [batch_size, traj_len, res, res, features]
            convLSTM_shape = tf.concat([[nenvs, nsteps],tf.shape(state_out)[1:]], axis=0)
            convLSTM_inputs = tf.reshape(state_out, convLSTM_shape)

            convLSTMCell = ConvLSTMCell(shape=ob_space['screen'][1:3], filters=filters, kernel=[3, 3], reuse=reuse) # TODO: padding: same?
            convLSTM_outputs, convLSTM_state = tf.nn.dynamic_rnn(
                convLSTMCell,
                convLSTM_inputs,
                initial_state=tf.nn.rnn_cell.LSTMStateTuple(STATES[0],STATES[1]),
                time_major=False,
                dtype=tf.float32,
                scope="dynamic_rnn"
            )
            outputs = tf.reshape(convLSTM_outputs, tf.concat([[nenvs*nsteps],tf.shape(convLSTM_outputs)[2:]], axis=0))
            
            flat_out = flatten(outputs, scope='flat_out')
            fc = fully_connected(flat_out, 256, activation_fn=tf.nn.relu, scope='fully_con')

            value = fully_connected(fc, 1, activation_fn=None, scope='value')
            value = tf.reshape(value, [-1])

            fn_out = non_spatial_output(fc, channels=NUM_FUNCTIONS, name='fn_name')

            args_out = dict()
            for arg_type in actions.TYPES:
                if is_spatial_action[arg_type]:
                    arg_out = spatial_output(outputs, name=arg_type.name) # TODO: use something like convLSTM_outputs here
                else:
                    arg_out = non_spatial_output(fc, channels=arg_type.sizes[0], name=arg_type.name)
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
                AV_ACTS: obs['available_actions'],
                STATES : state
            }
            return sess.run([action, value, convLSTM_state], feed_dict=feed_dict)


        def get_value(obs, state, mask=None):
            feed_dict = {
                SCREEN : obs['screen'],
                MINIMAP: obs['minimap'],
                FLAT   : obs['flat'],
                STATES : state
            }
            return sess.run(value, feed_dict=feed_dict)


        self.SCREEN  = SCREEN
        self.MINIMAP = MINIMAP
        self.FLAT    = FLAT
        self.AV_ACTS = AV_ACTS
        self.STATES  = STATES
        self.policy = policy
        self.value  = value
        self.step = step
        self.get_value = get_value
        self.initial_state = np.zeros(state_shape, dtype=np.float32) # TODO: figure out dimensions


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    """

    From: https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py

    A LSTM cell with convolutions instead of multiplications.
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
        super(ConvLSTMCell, self).__init__(_reuse=reuse)
        self._kernel = kernel
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole
        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + shape)
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size

    def call(self, x, state):
        c, h = state

        x = tf.concat([x, h], axis=self._feature_axis)
        n = x.shape[-1].value
        m = 4 * self._filters if self._filters > 1 else 4
        W = tf.get_variable('kernel', self._kernel + [n, m])
        y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
        if not self._normalize:
            y += tf.get_variable('bias', [m],
                                 initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:]) * c
            f += tf.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
            j = tf.contrib.layers.layer_norm(j)
            i = tf.contrib.layers.layer_norm(i)
            f = tf.contrib.layers.layer_norm(f)

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
            o = tf.contrib.layers.layer_norm(o)
            c = tf.contrib.layers.layer_norm(c)

        o = tf.sigmoid(o)
        h = o * self._activation(c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state
