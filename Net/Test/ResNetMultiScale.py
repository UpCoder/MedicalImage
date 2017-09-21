# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from Net.BaseNet.ResNetMultiScale.resnet import inference_small, loss
import tensorflow as tf
from Net.BaseNet.ResNetMultiScale.Config import Config as sub_Config
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValDataMultiSize import ValDataSet


def train(val_data_set, load_model_path):
    xs = []
    for index in range(len(sub_Config.SIZES)):
        xs.append(
            tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    sub_Config.SIZES[index][0],
                    sub_Config.SIZES[index][1]
                ]
            )
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
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder('bool', [], name='is_training')
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar-data',
                               'where to store the dataset')
    tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
    y = inference_small(xs, is_training=is_training,
                        num_classes=sub_Config.OUTPUT_NODE,
                        use_bias=FLAGS.use_bn,
                        num_blocks=3)
    tf.summary.histogram(
        'logits',
        tf.argmax(y, 1)
    )
    # with tf.control_dependencies([train_step, vaeriable_average_op]):
    #     train_op = tf.no_op(name='train')

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
        feed_dict = {}
        for index, _ in enumerate(sub_Config.SIZES):
            feed_dict[xs[index]] = validation_images[index]
        feed_dict[y_] = validation_labels
        validation_accuracy, logits = sess.run(
            [accuracy_tensor, y],
            feed_dict=feed_dict
        )
        _, _, _, error_index, error_record = calculate_acc_error(
            logits=np.argmax(logits, 1),
            label=validation_labels,
            show=True
        )
        print 'accuracy is %g' % \
              (validation_accuracy)
        return error_index, error_record
if __name__ == '__main__':
    phase_name = 'ART'
    state = ''
    val_dataset = ValDataSet(new_sizes=sub_Config.SIZES,
                             phase=phase_name,
                             category_number=sub_Config.OUTPUT_NODE,
                             shuffle=False,
                             data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI/val')
    error_index, error_record = train(
        val_dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNetMultiScale/models/fine_tuning/5/21000/'
    )
    val_dataset.show_error_name(error_index, error_record)