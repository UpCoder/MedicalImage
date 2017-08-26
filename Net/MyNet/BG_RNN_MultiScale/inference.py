# -*- coding: utf-8 -*-
from Net.tools import do_conv, pool, FC_layer, convert_two_dim, batch_norm
import tensorflow as tf
from Config import Config

def inference(input_tensors, bg_tensors, regularizer):
    CONV_OUTPUT = None
    for index, input_tensor in enumerate(input_tensors):
        # 同一个尺度
        bg_tensor = bg_tensors[index]
        shape = input_tensor.get_shape().as_list()
        shape_bg = bg_tensor.get_shape().as_list()
        state = tf.zeros(
            shape=[
                shape[0],
                Config.STATE_FEATURE_DIM
            ],
            dtype=tf.float32
        )
        with tf.variable_scope('rnn-part-' + str(index)):
            for i in range(shape[3]):
                cur_input_tensor = input_tensor[:, :, :, i] # 获取某一层的数据
                cur_input_tensor = tf.reshape(
                    cur_input_tensor,
                    shape=[
                        shape[0],
                        shape[1],
                        shape[2],
                        1
                    ]
                )
                cur_bg_tensor = bg_tensor[:, :, :, i]
                cur_bg_tensor = tf.reshape(
                    cur_bg_tensor,
                    shape=[
                        shape_bg[0],
                        shape_bg[1],
                        shape_bg[2],
                        1
                    ]
                )
                if i == 0:
                    reuse = False
                else:
                    reuse = True
                # 针对该ｐｈａｓｅ 计算roi的特征
                for key in Config.CONV_LAYERS_CONFIG:
                    layer_config = Config.CONV_LAYERS_CONFIG[key]
                    # def do_conv(name, input_tensor, out_channel, ksize, stride=[1, 1, 1, 1], is_pretrain=True, dropout=False, regularizer=None, reuse=False):
                    conv_res = do_conv(
                        key,
                        cur_input_tensor,
                        layer_config['deep'],
                        [layer_config['size'], layer_config['size']],
                        dropout=layer_config['dropout'],
                        reuse=reuse
                    )
                    cur_input_tensor = conv_res
                    if layer_config['pooling']['exists']:
                        pooling = pool(
                            layer_config['pooling']['name'],
                            cur_input_tensor
                        )
                        cur_input_tensor = pooling
                # 针对该ｐｈａｓｅ计算ｂｇ的特征
                for key in Config.CONV_LAYERS_CONFIG_BG:
                    layer_config = Config.CONV_LAYERS_CONFIG_BG[key]
                    conv_res = do_conv(
                        key,
                        cur_bg_tensor,
                        layer_config['deep'],
                        [layer_config['size'], layer_config['size']],
                        dropout=layer_config['dropout'],
                        reuse=reuse
                    )
                    cur_bg_tensor = conv_res
                    if layer_config['pooling']['exists']:
                        pooling = pool(
                            layer_config['pooling']['name'],
                            cur_bg_tensor
                        )
                        cur_bg_tensor = pooling
                cur_input_tensor = convert_two_dim(cur_input_tensor)
                cur_bg_tensor = convert_two_dim(cur_bg_tensor)
                fc_input = tf.concat([cur_input_tensor, cur_bg_tensor, state], axis=1)
                state = FC_layer('extract_state', fc_input, Config.STATE_FEATURE_DIM, regularizer, reuse)  # 更新状态
            if CONV_OUTPUT is None:
                CONV_OUTPUT = state
            else:
                CONV_OUTPUT = tf.concat([CONV_OUTPUT, state], axis=1)

    input_tensor = CONV_OUTPUT
    for key in Config.FC_LAYERS_CONFIG:
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

if __name__ == '__main__':
    x1 = tf.placeholder(
        tf.float32,
        [20, 45, 45, 3]
    )
    x2 = tf.placeholder(
        tf.float32,
        [20, 100, 100, 3]
    )
    y1 = tf.placeholder(
        tf.float32,
        [20, 35, 35, 3]
    )
    y2 = tf.placeholder(
        tf.float32,
        [20, 120, 120, 3]
    )
    y = inference([x1, x2], [y1, y2], None)
    print y