# -*- coding: utf-8 -*-
# 使用ｐａｔｃｈ训练好的模型，来对ＲＯＩ进行微调
from Net.BaseNet.LeNet.inference import inference
import tensorflow as tf
from Net.BaseNet.LeNet.Config import Config as sub_Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data
from Tools import changed_shape, calculate_acc_error, get_game_evaluate
import numpy as np
from Patch.ValData import ValDataSet
from Patch.Config import Config as patch_config
from Net.tools import save_weights, load


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
    global_step = tf.Variable(0, trainable=False)
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
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        if load_model_path:
            # load(load_model_path, sess)
            # with tf.variable_scope('conv1_1', reuse=True):
            #     weights1 = tf.get_variable('weights')
            #     print weights1.eval(sess)
            saver.restore(sess, load_model_path)
        else:
            sess.run(tf.global_variables_initializer())

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
        # validation_labels[validation_labels == 1] = 0
        # validation_labels[validation_labels == 3] = 0
        # validation_labels[validation_labels == 4] = 1
        # validation_labels[validation_labels == 2] = 1
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
        recall, precision, f1_score = get_game_evaluate(
            np.argmax(logits, 1),
            validation_labels
        )
        validation_labels = np.array(validation_labels)
        print 'label=0 %d, label=1 %d' % (np.sum(validation_labels == 0), np.sum(validation_labels == 1))
        print 'recall is %g, precision is %g, f1_score is %g' % (recall, precision, f1_score)
        print 'accuracy is %g' % \
              (validation_accuracy)
        return error_indexs, error_record
if __name__ == '__main__':
    phase_name = 'ART'
    # state = '_Expand'
    state = ''
    val_dataset = ValDataSet(new_size=[sub_Config.IMAGE_W, sub_Config.IMAGE_H],
                             phase=phase_name,
                             shuffle=False,
                             category_number=sub_Config.OUTPUT_NODE,
                             data_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/ROI' + state + '/val')

    error_indexs, error_record = val(
        val_dataset,
        load_model_path='/home/give/PycharmProjects/MedicalImage/Net/BaseNet/LeNet/model_finetuing/2/art/',
    )
    val_dataset.show_error_name(error_indexs, error_record, copy=False)