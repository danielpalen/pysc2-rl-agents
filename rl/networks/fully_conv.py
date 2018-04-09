import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers import fully_connected, conv2d # TODO: use these

from pysc2.lib import actions
from pysc2.lib import features

from rl.common.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES


class FullyConv():
    """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

    Both, NHWC and NCHW data formats are supported for the network
    computations. Inputs and outputs are always in NHWC.
    """

    def __init__(self, screen_input, minimap_input, flat_input, reuse=False, data_format='NCHW'):

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
            out = layers.conv2d(
                x, dims,
                kernel_size=1,
                stride=1,
                padding='SAME',
                activation_fn=tf.nn.relu,
                data_format=data_format,
                scope="%s/conv_embSpatial" % name)
            return to_nhwc(out)

        def embed_flat(x, dims, name):
            return layers.fully_connected(
                x, dims,
                activation_fn=tf.nn.relu,
                #scope="%s/emb_flat" % name)
                scope="%s/conv_embFlat" % name)

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

        def non_spatial_output(x, channels, name):
            logits = layers.fully_connected(x, channels, activation_fn=None, scope="non_spatial_output/flat/{}".format(name))
            return tf.nn.softmax(logits)

        def spatial_output(x, name):
            logits = layers.conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None, data_format=data_format, scope="spatial_output/conv/{}".format(name))
            logits = layers.flatten(to_nhwc(logits), scope="spatial_output/flat/{}".format(name))
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
            size2d = tf.unstack(tf.shape(screen_input)[1:3])
            screen_emb  = embed_obs(screen_input,  features.SCREEN_FEATURES,  embed_spatial, 'screen')
            minimap_emb = embed_obs(minimap_input, features.MINIMAP_FEATURES, embed_spatial, 'minimap')
            flat_emb    = embed_obs(flat_input, FLAT_FEATURES, embed_flat, 'flat')

            screen_out    = input_conv(from_nhwc(screen_emb), 'screen')
            minimap_out   = input_conv(from_nhwc(minimap_emb), 'minimap')
            broadcast_out = broadcast_along_channels(flat_emb, size2d)
            state_out     = concat2DAlongChannel([screen_out, minimap_out, broadcast_out])

            flat_out = layers.flatten(to_nhwc(state_out), scope="flat_out")
            fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu, scope="fully_conf")

            value = layers.fully_connected(fc, 1, activation_fn=None, scope="value")
            value = tf.reshape(value, [-1])

            fn_out = non_spatial_output(fc, NUM_FUNCTIONS, name='fn_out')

            args_out = dict()
            for arg_type in actions.TYPES:
                if is_spatial_action[arg_type]:
                    print(arg_type)
                    arg_out = spatial_output(state_out, name=arg_type.name)
                else:
                    print(arg_type)
                    arg_out = non_spatial_output(fc, arg_type.sizes[0], name=arg_type.name)
                args_out[arg_type] = arg_out

            policy = (fn_out, args_out)

            for n in tf.get_default_graph().as_graph_def().node:
                print('###' if reuse else '---', n.name)

        #def step():
        #    pass

        # def value():
        #     pass

        self.policy = policy
        self.value  = value
