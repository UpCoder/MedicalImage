import tensorflow as tf
from Net.tools import do_conv, FC_layer, batch_norm, pool, convert_two_dim
from Config import Config


def inference(input_tensors, regularizer=None):
    conv_laerys_output = None
    for index, input_tensor in enumerate(input_tensors):
        if index == 0:
            reuse = None
        else:
            reuse = True
        for key in Config.CONV_LAYERS_CONFIG:
            layer_config = Config.CONV_LAYERS_CONFIG[key]
            conv_res = do_conv(
                key,
                input_tensor,
                layer_config['deep'],
                [layer_config['size'], layer_config['size']],
                dropout=layer_config['dropout'],
                reuse=reuse

            )
            input_tensor = conv_res
            if layer_config['pooling']['exists']:
                pooling = pool(
                    layer_config['pooling']['name'],
                    input_tensor
                )
                input_tensor = pooling
        if conv_laerys_output is None:
            conv_laerys_output = convert_two_dim(input_tensor)
        else:
            conv_laerys_output = tf.concat([conv_laerys_output, convert_two_dim(input_tensor)], 1)
    input_tensor = conv_laerys_output
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


def test_unit():
    x1 = tf.placeholder(
        tf.float32,
        [
            10,
            45,
            45,
            3
        ]
    )
    x2 = tf.placeholder(
        tf.float32,
        [
            10,
            100,
            100,
            3
        ]
    )
    y = inference([x1, x2], None)
    print y

if __name__ == '__main__':
    test_unit()
