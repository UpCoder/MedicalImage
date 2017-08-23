# -*- coding: utf-8 -*-
import tensorflow as tf
from Net.tools import do_conv, pool, FC_layer, batch_norm
from Config import Config


# 实现ＬｅＮｅｔ网络结构
def inference(input_tensor):
    # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, dropout=False, regularizer=None):
    conv1_1 = do_conv(
        'conv1_1',
        input_tensor,
        Config.CONV1_1_DEEP,
        ksize=[Config.CONV1_1_SIZE,Config.CONV1_1_SIZE],
        dropout=Config.DROP_OUT
    )
    # pool(layer_name, x, kernel=[1,2,2,1], stride=[1, 2, 2, 1], is_max_pool=True):
    pooling1 = pool(
        'pooling1',
        conv1_1,
    )
    conv2_1 = do_conv(
        'conv2_1',
        pooling1,
        Config.CONV2_1_DEEP,
        ksize=[Config.CONV2_1_SIZE, Config.CONV2_1_SIZE],
        dropout=Config.DROP_OUT
    )
    pooling2 = pool(
        'pooling2',
        conv2_1
    )
    # FC_layer(layer_name, x, out_nodes, regularizer=None):
    fc1 = FC_layer(
        'fc1',
        pooling2,
        Config.FC_SIZE
    )
    fc1 = batch_norm(fc1)
    fc2 = FC_layer(
        'fc2',
        fc1,
        Config.OUTPUT_NODE
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