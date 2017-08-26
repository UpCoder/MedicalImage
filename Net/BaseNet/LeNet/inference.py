# -*- coding: utf-8 -*-
import tensorflow as tf
from Net.tools import do_conv, pool, FC_layer, batch_norm
from Config import Config


# 实现ＬｅＮｅｔ网络结构
def inference(input_tensor, regularizer=None):
    # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, dropout=False, regularizer=None):
    for key in Config.CONV_LAYERS_CONFIG:
        layer_config = Config.CONV_LAYERS_CONFIG[key]
        conv_res = do_conv(
            key,
            input_tensor,
            layer_config['deep'],
            [layer_config['size'], layer_config['size']],
            dropout=layer_config['dropout']

        )
        input_tensor = conv_res
        if layer_config['pooling']['exists']:
            pooling = pool(
                layer_config['pooling']['name'],
                input_tensor
            )
            input_tensor = pooling

    # FC_layer(layer_name, x, out_nodes, regularizer=None):
    fc1 = FC_layer(
        'fc1',
        input_tensor,
        Config.FC_SIZE,
        regularizer
    )
    fc1 = batch_norm(fc1)
    fc2 = FC_layer(
        'fc2',
        fc1,
        Config.OUTPUT_NODE,
        regularizer
    )
    return fc2


def test_unit():
    input_tensor = tf.placeholder(
        tf.float32,
        [
            20,
            54,
            54,
            1
        ]
    )
    y = inference(input_tensor)
    if y.get_shape().as_list() == [20, 5]:
        print 'Success'
        return
    print 'Error'


if __name__ == '__main__':
    test_unit()