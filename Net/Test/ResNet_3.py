# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from Net.BaseNet.ResNet_3.resnet import inference_small, loss
import tensorflow as tf
from Net.BaseNet.ResNet_3.Config import Config as sub_Config
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValDataMultiPhase import ValDataSetMultiPhase


def val(val_data_set, load_model_path, phases_names):
    x = tf.placeholder(
        tf.float32,
        shape=[
            None,
            sub_Config.IMAGE_W,
            sub_Config.IMAGE_H,
            sub_Config.IMAGE_CHANNEL*len(phases_names)
        ],
        name='input_x'
    )
    y_ = tf.placeholder(
        tf.float32,
        shape=[
            None,
        ]
    )
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder('bool', [], name='is_training')
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar-data',
                               'where to store the dataset')
    tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
    y = inference_small(x, is_training=is_training,
                        num_classes=sub_Config.OUTPUT_NODE,
                        use_bias=FLAGS.use_bn,
                        phase_names=phases_names,
                        num_blocks=3)

    with tf.variable_scope('accuracy'):
        accuracy_tensor = tf.reduce_mean(
            tf.cast(
                tf.equal(x=tf.argmax(y, 1), y=tf.cast(y_, tf.int64)),
                tf.float32
            )
        )
        tf.summary.scalar(
            'accuracy',
            accuracy_tensor
        )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load_model_path:
            saver.restore(sess, load_model_path)

        validation_images, validation_labels = val_data_set.get_next_batch(None, None)

        validation_accuracy, logits = sess.run(
            [accuracy_tensor, y],
            feed_dict={
                x: validation_images,
                y_: validation_labels
            }
        )
        calculate_acc_error(
            logits=np.argmax(logits, 1),
            label=validation_labels,
            show=True
        )
        binary_acc = acc_binary_acc(
            logits=np.argmax(logits, 1),
            label=validation_labels,
        )
        print 'accuracy is %g, binary_acc is %g' % \
              (validation_accuracy, binary_acc)
if __name__ == '__main__':
    phase_names = ['NC', 'ART', 'PV']
    state = ''
    val_dataset = ValDataSetMultiPhase(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H],
                                       phases=phase_names,
                                       shuffle=False,
                                       category_number=sub_Config.OUTPUT_NODE,
                                       data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI/val'
                                       )
    val(
        val_dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet_3/model/fine_tuning/5-128/14500/',
        phases_names=phase_names
    )