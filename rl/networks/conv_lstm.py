import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn

from pysc2.lib import actions
from pysc2.lib import features

from rl.common.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES


class ConvLSTM():
    """LSTM network.

    Both, NHWC and NCHW data formats are supported for the network
    computations. Inputs and outputs are always in NHWC.
    """

    def __init__(self, screen_input, minimap_input, flat_input, reuse=False, data_format='NCHW'):
        """
        Args:
            screen_input  : tensor of shape (None x res x res x 17)
            minimap_input : tensor of shape (None x res x res x 7)
            flat_input    : tensor of shape (None x 11)
        """

        def embed_obs(x, spec, embed_fn):
            feats = tf.split(x, len(spec), -1)
            out_list = []
            for s in spec:
                f = feats[s.index]
                if s.type == features.FeatureType.CATEGORICAL:
                    dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                    dims = max(dims, 1)
                    indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
                    out = embed_fn(indices, dims)
                elif s.type == features.FeatureType.SCALAR:
                    out = tf.log(f + 1.)
                else:
                    raise NotImplementedError
                out_list.append(out)
            return tf.concat(out_list, -1)

        def embed_spatial(x, dims):
            x = from_nhwc(x)
            out = layers.conv2d(
                x, dims,
                kernel_size=1,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format)
            return to_nhwc(out)

        def embed_flat(x, dims):
            return layers.fully_connected(
                x, dims,
                activation_fn=tf.nn.relu)

        def input_conv(x, name):
            conv1 = layers.conv2d(
                x, 16,
                kernel_size=5,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv1" % name)
            conv2 = layers.conv2d(
                conv1, 32,
                kernel_size=3,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv2" % name)
            return conv2

        def non_spatial_output(x, channels):
            logits = layers.fully_connected(x, channels, activation_fn=None)
            return tf.nn.softmax(logits)

        def spatial_output(x):
            logits = layers.conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None,
                                   data_format=data_format)
            logits = layers.flatten(to_nhwc(logits))
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

        with tf.variable_scope('model', reuse=reuse):
            screen_emb = embed_obs(    # None x res x res x None
                screen_input,               # None x res x res x 17
                features.SCREEN_FEATURES,   # 17
                embed_spatial
            )

            minimap_emb = embed_obs(   # None x res x res x None
                minimap_input,              # None x res x res x 17
                features.MINIMAP_FEATURES,  # 7
                embed_spatial
            )

            flat_emb = embed_obs(  # None x 11
                flat_input,    # None x 11
                FLAT_FEATURES,  # 11
                embed_flat
            )

            screen_out = input_conv(from_nhwc(screen_emb), 'screen')
            minimap_out = input_conv(from_nhwc(minimap_emb), 'minimap')

            size2d = tf.unstack(tf.shape(screen_input)[1:3])
            broadcast_out = broadcast_along_channels(flat_emb, size2d)
            state_out = to_nhwc(concat2DAlongChannel([screen_out, minimap_out, broadcast_out]))

            # recurrent trajectory length?
            n_steps = 16  # TODO: pass this here!

            #print(state_out)

            # [batch_size, traj_len, res, res, features]
            convLSTM_shape = tf.concat([[-1, n_steps],tf.shape(state_out)[1:]], axis=0)

            # [traj_len, batch_size, res, res, features]
            convLSTM_inputs = tf.transpose(tf.reshape(state_out, convLSTM_shape), [1, 0, 2, 3, 4])

            #print(tf.Print(convLSTM_shape, [convLSTM_shape]))

            convLSTMCell = ConvLSTMCell(
                shape=[32, 32],  # TODO: infer these!
                filters=16,
                kernel=[3, 3]
            )

            convLSTM_outputs, convLSTM_state = tf.nn.dynamic_rnn(
                convLSTMCell,
                convLSTM_inputs,
                time_major=False,
                dtype=tf.float32
            )

            flat_out = layers.flatten(convLSTM_state)
            fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)

            value = layers.fully_connected(fc, 1, activation_fn=None)
            value = tf.reshape(value, [-1])

            fn_out = non_spatial_output(fc, NUM_FUNCTIONS)

            args_out = dict()
            for arg_type in actions.TYPES:
                if is_spatial_action[arg_type]:
                    arg_out = spatial_output(state_out)
                else:
                    arg_out = non_spatial_output(fc, arg_type.sizes[0])
                args_out[arg_type] = arg_out

            policy = (fn_out, args_out)

            for n in tf.get_default_graph().as_graph_def().node:
                print(n.name)

        self.policy = policy
        self.value = value


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
