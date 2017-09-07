# -*- coding: utf-8 -*-
from Net.BaseNet.LeNet.inference import inference
import tensorflow as tf
from Net.BaseNet.LeNet.Config import Config as sub_Config
from Tools import changed_shape, calculate_acc_error
import numpy as np
from Patch.ValData import ValDataSet
from Patch.Config import Config as patch_config


def val(dataset, load_model_path, save_model_path):
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
    regularizer = tf.contrib.layers.l2_regularizer(sub_Config.REGULARIZTION_RATE)
    y = inference(x, regularizer)

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
        validation_images, validation_labels = dataset.images, dataset.labels
        validation_images = changed_shape(
            validation_images,
            [
                len(validation_images),
                sub_Config.IMAGE_W,
                sub_Config.IMAGE_W,
                1
            ]
        )
        validation_accuracy, logits = sess.run(
            [accuracy_tensor, y],
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
        print 'accuracy is %g' % \
              (validation_accuracy)
        return error_indexs, error_record
if __name__ == '__main__':
    dataset = ValDataSet(data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI/val',
                         new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H], shuffle=False)
    error_indexs, error_record = val(
        dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model_finetuing/',
        save_model_path=None
    )
    dataset.show_error_name(error_indexs, error_record)