# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from Net.BaseNet.ResNet.resnet import inference_small, loss
import tensorflow as tf
from Net.BaseNet.ResNet.Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValData import ValDataSet
from Patch.Config import Config as patch_config


def val(val_data_set, load_model_path):
    x = tf.placeholder(
        tf.float32,
        shape=[
            None,
            sub_Config.IMAGE_W,
            sub_Config.IMAGE_H,
            sub_Config.IMAGE_CHANNEL
        ],
        name='input_x'
    )
    y_ = tf.placeholder(
        tf.float32,
        shape=[
            None,
        ]
    )
    tf.summary.histogram(
        'label',
        y_
    )
    # global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder('bool', [], name='is_training')
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar-data',
                               'where to store the dataset')
    tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
    y = inference_small(x, is_training=is_training,
                        num_classes=sub_Config.OUTPUT_NODE,
                        use_bias=FLAGS.use_bn,
                        num_blocks=3)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    loss_ = loss(
        logits=y,
        labels=tf.cast(y_, np.int32)
    )
    tf.summary.scalar(
        'loss',
        loss_
    )
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
        validation_images, validation_labels = val_data_set.images, val_data_set.labels
        validation_images = changed_shape(
            validation_images,
            [
                len(validation_images),
                sub_Config.IMAGE_W,
                sub_Config.IMAGE_W,
                1
            ]
        )
        validation_accuracy, validation_loss, logits = sess.run(
            [accuracy_tensor, loss_, y],
            feed_dict={
                x: validation_images,
                y_: validation_labels
            }
        )
        _, _, _, error_indexs, error_record = calculate_acc_error(
            logits=np.argmax(logits, 1),
            label=validation_labels,
            show=True
        )
        print 'validation loss value is %g, accuracy is %g' % \
              (validation_loss, validation_accuracy)
        return error_indexs, error_record
if __name__ == '__main__':
    phase_name = 'ART'
    state = ''
    val_dataset = ValDataSet(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H],
                             phase=phase_name,
                             category_number=sub_Config.OUTPUT_NODE,
                             data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROIMulti/val')
    error_indexs, error_record = val(
        val_dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet/models/fine_tuning/5-128/'
    )
    val_dataset.show_error_name(error_indexs, error_record, copy=False)