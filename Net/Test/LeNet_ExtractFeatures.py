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
    y, features = inference(x, regularizer, return_feature=True)

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
        validation_accuracy, features_value = sess.run(
            [accuracy_tensor, features],
            feed_dict={
                x: validation_images,
                y_: validation_labels
            }
        )
        print validation_accuracy
        return features_value
if __name__ == '__main__':
    phase = 'pv'
    state = 'val'
    dataset = ValDataSet(data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI/' + state,
                         phase=phase.upper(),
                         new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H], shuffle=False)
    features = val(
        dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model_finetuing/model_' + phase +'/',
        save_model_path=None
    )
    np.save(
        '/home/give/PycharmProjects/MedicalImage/Net/data/' + state + '_' + phase + '.npy',
        features
    )
    np.save(
        '/home/give/PycharmProjects/MedicalImage/Net/data/' + state + '_' + phase + '_label.npy',
        dataset.labels
    )
    print np.shape(features)
