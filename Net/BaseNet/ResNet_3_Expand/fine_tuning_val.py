# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from resnet import inference_small, loss
import tensorflow as tf
from Config import Config as net_config
from Tools import changed_shape, calculate_acc_error, acc_binary_acc
import numpy as np
from Patch.ValDataMultiPhaseExpand import ValDataSetMultiPhase


def val(val_data_set, load_model_path, phases_names):

    x_ROI = tf.placeholder(
        tf.float32,
        shape=[
            None,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL*len(phases_names)
        ],
        name='input_x'
    )

    x_EXPAND = tf.placeholder(
        tf.float32,
        shape=[
            None,
            net_config.EXPAND_SIZE_W,
            net_config.EXPAND_SIZE_H,
            net_config.IMAGE_CHANNEL * len(phases_names)
        ]
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
    # variable_average = tf.train.ExponentialMovingAverage(
    #     sub_Config.MOVING_AVERAGE_DECAY,
    #     global_step
    # )
    # vaeriable_average_op = variable_average.apply(tf.trainable_variables())
    # regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    is_training = tf.placeholder('bool', [], name='is_training')
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar-data',
                               'where to store the dataset')
    tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
    y = inference_small([x_ROI, x_EXPAND], is_training=is_training,
                        num_classes=net_config.OUTPUT_NODE,
                        use_bias=FLAGS.use_bn,
                        phase_names=phases_names,
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
    merge_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load_model_path:
            saver.restore(sess, load_model_path)

        validation_images, validation_images_expand, validation_labels = val_data_set.get_next_batch()

        validation_accuracy, validation_loss, summary, logits = sess.run(
            [accuracy_tensor, loss_, merge_op, y],
            feed_dict={
                x_ROI: validation_images,
                x_EXPAND: validation_images_expand,
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

        print 'validation loss value is %g, accuracy is %g, binary_acc is %g' % \
              (validation_loss, validation_accuracy, binary_acc)
if __name__ == '__main__':
    phase_names = ['NC', 'ART', 'PV']
    state = ''
    val_dataset = ValDataSetMultiPhase(new_sizes=[
        [net_config.ROI_SIZE_W, net_config.ROI_SIZE_H],
        [net_config.EXPAND_SIZE_W, net_config.EXPAND_SIZE_H],
    ],
                                       phases=phase_names,
                                       shuffle=False,
                                       category_number=net_config.OUTPUT_NODE,
                                       data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROIMultiExpand1/val'
                                       )
    val(
        val_dataset,
        # load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet_3/model/fine_tuning/2-128/5000/',
        # load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet_3/model/fine_tuning/2-128/9901/c',
        # load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet_3_Expand/models/5/12901/',
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/ResNet_3_Expand/models/5/single_val/9701/',
        phases_names=phase_names
    )