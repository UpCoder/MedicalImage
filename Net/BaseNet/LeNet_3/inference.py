# -*- coding: utf-8 -*-
import tensorflow as tf
from Net.tools import do_conv, pool, FC_layer, batch_norm, convert_two_dim
from Config import Config


# 实现ＬｅＮｅｔ网络结构
def inference(input_tensors, phase_names, regularizer=None, return_feature=False):
    print input_tensors.get_shape().as_list()
    # input_tensor 包含了三个期项，ｓｈａｐｅ是batchsize, image_w, image_h, 3
    # do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, dropout=False, regularizer=None):
    keys = list(Config.CONV_LAYERS_CONFIG.keys())
    keys.sort()
    CONV_OUTPUT = None
    for phase_index, phase in enumerate(phase_names):
        phase_config = Config.CONV_LAYERS_CONFIG[phase]
        conv_names = list(phase_config.keys())
        conv_names.sort()
        input_tensor = input_tensors[:, :, :, phase_index]
        for conv_name in conv_names:
            conv_layer_config = phase_config[conv_name]
            conv_res = do_conv(
                conv_name,
                input_tensor,
                conv_layer_config['deep'],
                [conv_layer_config['size'], conv_layer_config['size']],
                dropout=conv_layer_config['dropout'],
                is_pretrain=conv_layer_config['trainable'],
                batch_normalization=conv_layer_config['batch_norm']
            )
            input_tensor = conv_res
            if conv_layer_config['pooling']['exists']:
                pooling = pool(
                    conv_layer_config['pooling']['name'],
                    input_tensor
                )
                input_tensor = pooling
        input_tensor = convert_two_dim(input_tensor)
        if CONV_OUTPUT is not None:
            CONV_OUTPUT = tf.concat([CONV_OUTPUT, input_tensor], axis=1)
        else:
            CONV_OUTPUT = input_tensor

    # FC_layer(layer_name, x, out_nodes, regularizer=None):
    fc1 = FC_layer(
        'fc1',
        CONV_OUTPUT,
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
    if return_feature:
        return fc2, fc1
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
    phase_names = ['NC']
    y = inference(input_tensor, regularizer=None, phase_names=phase_names)
    print y.get_shape()


if __name__ == '__main__':
    test_unit()