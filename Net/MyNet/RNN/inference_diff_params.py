# -*- coding: utf-8 -*-
from Config import Config
from Net.tools import do_conv, FC_layer, pool, convert_two_dim, batch_norm
import tensorflow as tf


# input_tensor [batch_size, w, h, 3]
def inference(input_tensor, regularizer):
    input_shape = input_tensor.get_shape().as_list()
    CONV_OUT = None
    with tf.variable_scope('rnn-part'):
        for i in range(input_shape[3]):
            if i == 0:
                reuse = False
            else:
                reuse = True
            cur_input = input_tensor[:, :, :, i]
            cur_input = tf.reshape(
                cur_input,
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    1
                ]
            )
            layer_keys = list(Config.CONV_LAYERS_CONFIG)
            layer_keys.sort()
            for key in layer_keys:
                layer_config = Config.CONV_LAYERS_CONFIG[key]
                conv_res = do_conv(
                    key,
                    cur_input,
                    layer_config['deep'],
                    [layer_config['size'], layer_config['size']],
                    dropout=layer_config['dropout'],
                    reuse=reuse

                )
                cur_input = conv_res
                if layer_config['pooling']['exists']:
                    pooling = pool(
                        layer_config['pooling']['name'],
                        cur_input
                    )
                    cur_input = pooling
            # 完成了卷积操作
            cur_input = convert_two_dim(cur_input)  # 展成二维的变量
            if CONV_OUT is None:
                CONV_OUT = cur_input
            else:
                CONV_OUT = tf.concat([CONV_OUT, cur_input], axis=1)
        input_tensor = CONV_OUT
        layer_keys = list(Config.FC_LAYERS_CONFIG)
        layer_keys.sort()
        for key in layer_keys:
            layer_config = Config.FC_LAYERS_CONFIG[key]
            if not layer_config['regularizer']:
                cur_regularizer = None
            else:
                cur_regularizer = regularizer
            input_tensor = FC_layer(
                key,
                input_tensor,
                layer_config['size'],
                cur_regularizer
            )
            if layer_config['batch_norm']:
                input_tensor = batch_norm(input_tensor)
        return input_tensor
